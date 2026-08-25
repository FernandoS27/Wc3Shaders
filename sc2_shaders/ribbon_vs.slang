// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026, Fernando Sahmkow
// See LICENSE in the repository root for full terms.
//
// Ribbon family vertex shader (ribbon.fx : RibbonVertexMain).
//
// A ribbon is a strip of quads swept along a trail of positions.  The vertex
// shader carries the whole simulation: it ages each vertex, interpolates its
// size and colour from three keyframes, derives a tangent frame from the
// neighbouring trail samples (or from a Bezier hodograph, or from the drag
// integrator's instantaneous velocity), spins that frame about the tangent, and
// finally offsets the vertex sideways by the ribbon's half-width.
//
// `b_iRibbonSimTech` picks WHERE the trail comes from, and it is also the axis
// that reshapes the vertex declaration:
//
//   GPUONLY(0)                    fully GPU-simulated from initial velocity
//   SPLINE_RIBBON(1)              4 control points in constants; the whole
//                                 stream collapses to position/uv/offset/batch
//   MIXED(2)                      CPU trail + GPU procedural offset
//   MIXED_PRECOMPUTED_TANGENT(3)  ...with the tangent supplied in vPositionPrev
//   LEGACY(4)                     CPU trail, tangent from prev/next neighbours
//
// The family's own permutation axes (this one included) live in a 23-bit header
// block at key bit 104, read straight out of `Ribbon_BuildFamily` in the retail
// binary — see tools/sc2_family_decode.FAMILY_OWN_AXES.  Everything downstream
// of the simulation (interpolant transport, vertex lighting, GenUV, fog/shadow,
// the parallax vector) is shared with the Model root.

implementing sc2_shaders;

#include "vs/model_vs_defines.slangh"
#include "types/perm_cb.slangh"

// Axes read as CODE need a file-local default: only NONZERO axes are passed
// as -D, so a zero one would otherwise be an undefined identifier.
#ifndef b_clampTail
#define b_clampTail 0
#endif
#ifndef b_computeUFromAge
#define b_computeUFromAge 0
#endif
#ifndef b_gpuSplineRibbon
#define b_gpuSplineRibbon 0
#endif
#ifndef b_iRibbonColorInterpolation
#define b_iRibbonColorInterpolation 0
#endif
#ifndef b_iRibbonSizeInterpolation
#define b_iRibbonSizeInterpolation 0
#endif
#ifndef b_precomputedAge
#define b_precomputedAge 0
#endif
#ifndef b_precomputedTangent
#define b_precomputedTangent 0
#endif
#ifndef b_proceduralPosition
#define b_proceduralPosition 0
#endif
#ifndef b_ribbonLocalSpace
#define b_ribbonLocalSpace 0
#endif
#ifndef b_splineRibbon
#define b_splineRibbon 0
#endif

// ---------------------------------------------------------------------------
// ribbon.fx : VertDecl — the vertex declaration, reshaped by b_iRibbonSimTech.
//
// `BatchIndexType` is `uint4` off the OpenGL path: iBatchIndex is a true integer
// attribute, not a float one.  HALF2/3/4 resolve to float2/3/4 at the SM4+
// PIXEL_SHADER_VERSION the engine always compiles these with.
// ---------------------------------------------------------------------------
#if b_iRibbonSimTech == RIBBON_SIM_SPLINE_RIBBON
public struct RibbonVSStreamIn
{
    float4 vPosition   : POSITION;
    float2 vUV         : TEXCOORD0;
    float2 vOffset     : NORMAL;
    uint4  iBatchIndex : TEXCOORD6;
};
#else
public struct RibbonVSStreamIn
{
    float4 vPosition : POSITION;
#if b_iRibbonSimTech == RIBBON_SIM_LEGACY
    float2 vUV : TEXCOORD0;
#endif
    float4 vSize            : TEXCOORD1;
    float4 cColor0          : COLOR0;
    float4 cColor1          : COLOR1;
    float4 cColor2          : TEXCOORD2;
    float3 vRotation        : TEXCOORD3;
    float2 vOffset          : NORMAL;
    float3 vUp              : TEXCOORD4;
    float4 vInitialVelocity : TEXCOORD5;
    uint4  iBatchIndex      : TEXCOORD6;
#if (b_iRibbonSimTech == RIBBON_SIM_MIXED_PRECOMPUTED_TANGENT || b_iRibbonSimTech == RIBBON_SIM_LEGACY)
    float3 vPositionPrev : TEXCOORD7;
#endif
#if b_iRibbonSimTech == RIBBON_SIM_LEGACY
    float3 vPositionNext : TEXCOORD8;
#endif
};
#endif

// ---------------------------------------------------------------------------
// ribbon.fx : Input — the internal (unpermuted) vertex.  TranslateVertexLayout
// zero-fills everything the declaration did not carry, so the body can read any
// field unconditionally.
// ---------------------------------------------------------------------------
public struct RibbonVSVertex
{
    float4 vPosition;
    float2 vUV;
    float4 vSize;
    float4 cColor0;
    float4 cColor1;
    float4 cColor2;
    float3 vRotation;
    float2 vOffset;
    float3 vUp;
    float4 vInitialVelocity;
    uint4  iBatchIndex;
    float3 vPositionPrev;
    float3 vPositionNext;
};

RibbonVSVertex vsTranslateRibbonVert(RibbonVSStreamIn layout)
{
    RibbonVSVertex input;
    input.vPosition        = layout.vPosition;
    input.iBatchIndex      = layout.iBatchIndex;
    input.vOffset          = layout.vOffset;
    input.vUV              = float2(0, 0);
    input.vSize            = float4(0, 0, 0, 0);
    input.cColor0          = float4(0, 0, 0, 0);
    input.cColor1          = float4(0, 0, 0, 0);
    input.cColor2          = float4(0, 0, 0, 0);
    input.vRotation        = float3(0, 0, 0);
    input.vUp              = float3(0, 0, 0);
    input.vInitialVelocity = float4(0, 0, 0, 0);
    input.vPositionPrev    = float3(0, 0, 0);
    input.vPositionNext    = float3(0, 0, 0);
#if (b_iRibbonSimTech == RIBBON_SIM_SPLINE_RIBBON || b_iRibbonSimTech == RIBBON_SIM_LEGACY)
    input.vUV = layout.vUV;
#endif
#if b_iRibbonSimTech != RIBBON_SIM_SPLINE_RIBBON
    input.vSize            = layout.vSize;
    input.cColor0          = layout.cColor0;
    input.cColor1          = layout.cColor1;
    input.cColor2          = layout.cColor2;
    input.vRotation        = layout.vRotation;
    input.vUp              = layout.vUp;
    input.vInitialVelocity = layout.vInitialVelocity;
  #if (b_iRibbonSimTech == RIBBON_SIM_MIXED_PRECOMPUTED_TANGENT || b_iRibbonSimTech == RIBBON_SIM_LEGACY)
    input.vPositionPrev = layout.vPositionPrev;
  #endif
  #if b_iRibbonSimTech == RIBBON_SIM_LEGACY
    input.vPositionNext = layout.vPositionNext;
  #endif
#endif
    return input;
}

// ---------------------------------------------------------------------------
// Per-batch matrix accessors.  Matrix ARRAYS are stored as their float4 rows
// (VS_MATROWS) because slangc does not apply its row-major storage wrapper to
// array members — see the note in types/model_vs_io.slang.
// ---------------------------------------------------------------------------
float4x4 vsRibbonInstanceMatrix(int i)
{
    int b = i * 4;
    return float4x4(vsGlobals.mRibbonInstanceTransform[b + 0],
                    vsGlobals.mRibbonInstanceTransform[b + 1],
                    vsGlobals.mRibbonInstanceTransform[b + 2],
                    vsGlobals.mRibbonInstanceTransform[b + 3]);
}

float4x4 vsInvPRWorldMatrix(int i)
{
    int b = i * 4;
    return float4x4(vsGlobals.mInvPRWorldTransform[b + 0],
                    vsGlobals.mInvPRWorldTransform[b + 1],
                    vsGlobals.mInvPRWorldTransform[b + 2],
                    vsGlobals.mInvPRWorldTransform[b + 3]);
}

// ---------------------------------------------------------------------------
// ribbon.fx : VERTEX SHADER EMITTERS
// ---------------------------------------------------------------------------
float4 vsEmitRibbonHPos(RibbonVSVertex vertIn)
{
    return mul(float4(vertIn.vPosition.xyz, 1), vsGlobals.mVPTransform);
}

float3 vsEmitRibbonViewPos(RibbonVSVertex vertIn)
{
    return mul(vsGlobals.mViewTransform, float4(vertIn.vPosition.xyz, 1));   // as 3x4
}

// UVs.  NOTE the deliberate x/y swap: CRibbon::RenderBatches() builds the vertex
// buffer with the components transposed, so UV0/UV1 read `vUV.yx` while UV2/UV3
// read `vUV.xy`.  There is no local space and no second UV set on a ribbon, so
// the same values are passed for both slots of each pair.
float4 vsEmitRibbonUV<let M : int, let I : int>(RibbonVSVertex vertIn, float3 vVertexNormal)
{
    return vsGenUV<M, I>(
        vertIn.vPosition.xyz,        // no local space available
        vertIn.vPosition.xyz,
        vVertexNormal,
        vVertexNormal,
        vertIn.vUV.yx,
        vertIn.vUV.yx,               // no vUV 1
        vertIn.vUV.xy,
        vertIn.vUV.xy,
        0,
        float3(0, 0, 0));
}

// Always right-handed.
float4 vsEmitRibbonNormal(float3 vVertexNormal)
{
    return float4(vVertexNormal, 1);
}

float4 vsEmitRibbonTangent(float3 vVertexTangent)
{
    return float4(vVertexTangent, 1);
}

float4 vsEmitRibbonVertexColor(RibbonVSVertex vertIn, float fAge, int iBatchIndex)
{
    float4 cColor;

    cColor.rgb = vsInterpolateValue(
        fAge,
        vertIn.cColor0.rgb, vertIn.cColor1.rgb, vertIn.cColor2.rgb,
        vsGlobals.vMidKeyTimes[iBatchIndex].y,
        vsGlobals.vInversedMidKeyTimes[iBatchIndex].y,
        vsGlobals.vMidKeyHoldTimes[iBatchIndex].y,
        SC2_AXIS(b_iRibbonColorInterpolation));

    cColor.a = vsInterpolateValue(
        fAge,
        vertIn.cColor0.a, vertIn.cColor1.a, vertIn.cColor2.a,
        vsGlobals.vMidKeyTimes[iBatchIndex].z,
        vsGlobals.vInversedMidKeyTimes[iBatchIndex].z,
        vsGlobals.vMidKeyHoldTimes[iBatchIndex].z,
        SC2_AXIS(b_iRibbonColorInterpolation));

    if (b_tintRibbonRed)
        cColor = float4(1.0f, 0.0f, 0.0f, 1.0f);

    return cColor;
}

float3 vsEmitRibbonEyeToVertex(RibbonVSVertex vertIn)
{
    return normalize(vertIn.vPosition.xyz - vsGlobals.vEyePos);
}

float4 vsEmitRibbonFOWUV(RibbonVSVertex vertIn)
{
    return mul(float4(vertIn.vPosition.xyz, 1), vsGlobals.mFOWMatrix);
}

// ---------------------------------------------------------------------------
// UV-emitter unroll (same shape as the Model / Foliage roots').
// ---------------------------------------------------------------------------
#if SC2_HAS_UV
  #define SC2_RIB_UV(dst, i)                                                        \
    vertOut.m_UV[dst] = vsEmitRibbonUV<SC2_UVMAPPING##i, i>(vertIn, vVertexNormal)
  #define SC2_RIB_UV_PAIR(dst, i, j)                                                \
    vertOut.m_UV[dst] =                                                             \
        float4(vsEmitRibbonUV<SC2_UVMAPPING##i, i>(vertIn, vVertexNormal).xy,       \
               vsEmitRibbonUV<SC2_UVMAPPING##j, j>(vertIn, vVertexNormal).xy)
#else
  #define SC2_RIB_UV(dst, i)
  #define SC2_RIB_UV_PAIR(dst, i, j)
#endif

// ---------------------------------------------------------------------------
// Main vertex shader body.
// ---------------------------------------------------------------------------
[shader("vertex")]
public DefaultVSOutput ribbon_vs_main(RibbonVSStreamIn vertLayout)
{
    DefaultVSOutput vertOut;

    float fAge = 0.0f;
    float3 vVertexNormal = float3(0, 0, 0);
    float3 vVertexTangent = float3(0, 0, 0);
    float3 vVertexBinormal = float3(0, 0, 0);
    int iBatchIndex = 0;

    RibbonVSVertex vertIn = vsTranslateRibbonVert(vertLayout);

    // Remap to the proper batch slot — GPU spline ribbons are already direct.
    iBatchIndex = (int)vertIn.iBatchIndex.x;
    if (!SC2_AXIS(b_gpuSplineRibbon))
        iBatchIndex = (int)vsGlobals.fRibbonBatchIndexRemappingTable[iBatchIndex];

    if (b_compressedVertex)
    {
        vertIn.vSize = (vertIn.vSize + 1.0f) * 0.5f;
        vertIn.vRotation = (vertIn.vRotation + 1.0f) * 0.5f;

        vertIn.vUV.y = vertIn.vSize.x;

        // ryantodo (original): pull these from constants
        float3 sizeScalar = float3(10.0f, 10.0f, 10.0f);
        vertIn.vSize.xyz = vertIn.vSize.yzw * sizeScalar;

        float pi = 3.1415926535897932384626433832795f;
        float3 rotationScalar = float3(400 * pi, 400 * pi, 400 * pi);
        float3 rotationOffset = float3(-200 * pi, -200 * pi, -200 * pi);
        vertIn.vRotation.xyz = vertIn.vRotation.xyz * rotationScalar + rotationOffset;
    }

    float fSize;

    float3 vNormal = float3(0.0f, 0.0f, 1.0f);
    float3 vTangent = float3(0, 0, 0);
    float3 vBinormal = float3(0, 0, 0);
    float3 vOffset = float3(0, 0, 0);

    // --- spline ribbon path ------------------------------------------------
    if (SC2_AXIS(b_gpuSplineRibbon))
    {
        fAge = vertIn.vPosition.x;

        float age = saturate(vertIn.vPosition.x);
        float ageSquared = age * age;
        float invAge = saturate(vertIn.vPosition.z);
        float invAgeSquared = invAge * invAge;

        float contractedAge = saturate(vertIn.vPosition.y);
        float contractedAgeSquared = contractedAge * contractedAge;
        float contractedInvAge = saturate(vertIn.vPosition.w);
        float contractedInvAgeSquared = contractedInvAge * contractedInvAge;

        float3 cp0 = vsGlobals.mSplineRibbonControlPoints[iBatchIndex * 4 + 0];
        float3 cp1 = vsGlobals.mSplineRibbonControlPoints[iBatchIndex * 4 + 1];
        float3 cp2 = vsGlobals.mSplineRibbonControlPoints[iBatchIndex * 4 + 2];
        float3 cp3 = vsGlobals.mSplineRibbonControlPoints[iBatchIndex * 4 + 3];

        // cubic Bezier position
        vertIn.vPosition.xyz = invAge * invAgeSquared * cp0;
        vertIn.vPosition.xyz += 3 * age * invAgeSquared * cp1;
        vertIn.vPosition.xyz += 3 * invAge * ageSquared * cp2;
        vertIn.vPosition.xyz += age * ageSquared * cp3;

        // ...and its hodograph, which is the tangent
        vTangent = (cp1 - cp0) * contractedInvAgeSquared;
        vTangent += 2 * (cp2 - cp1) * contractedAge * contractedInvAge;
        vTangent += (cp3 - cp2) * contractedAgeSquared;
        vTangent = normalize(vTangent);

        // guess a plane that can serve as the up vector for the whole spline
        float3 n0 = cross(cp1 - cp0, cp3 - cp0);
        float3 n1 = cross(cp2 - cp3, cp0 - cp3);
        n0 *= (dot(n0, n1) < 0 ? -1.0f : 1.0f);
        n0 += n1;
        n0.x += 1;   // bogus value to force stabilisation of near-colinear endpoints
        vertIn.vUp.xyz = normalize(n0);

        vertIn.vSize.xyz = vsGlobals.vSplineRibbonSize[iBatchIndex];
        vertIn.cColor0 = vsGlobals.vSplineRibbonColor0[iBatchIndex];
        vertIn.cColor1 = vsGlobals.vSplineRibbonColor1[iBatchIndex];
        vertIn.cColor2 = vsGlobals.vSplineRibbonColor2[iBatchIndex];
    }
    // --- normal ribbon path ------------------------------------------------
    else
    {
        float proceduralScalar = 1.0;    // set to 0 if clamped

        if (SC2_AXIS(b_precomputedAge))
        {
            fAge = vertIn.vPosition.w;
        }
        else
        {
            float ageScalar = 1.0;

            if (!SC2_AXIS(b_ribbonLocalSpace))
            {
                // clamp the head
                {
                    float ageDelta = vsGlobals.vLockedStartPositionAndTime[iBatchIndex].w
                                   - vertIn.vPosition.w;
                    float4 posChange;
                    posChange.w = vsGlobals.vGPURibbonValues[iBatchIndex].x - vertIn.vPosition.w;
                    posChange.xyz = vsGlobals.vLockedStartPositionAndTime[iBatchIndex].xyz
                                  - vertIn.vPosition.xyz;
                    proceduralScalar = (ageDelta > 0.001) ? 1.0 : 0.0;
                    posChange *= (1.0f - proceduralScalar);
                    vertIn.vPosition += posChange;
                }

                // clamp the tail
                if (SC2_AXIS(b_clampTail))
                {
                    float ageDelta = vsGlobals.vLockedTailPositionAndTime[iBatchIndex].w
                                   - vertIn.vPosition.w;
                    float3 posChange;
                    posChange.xyz = vsGlobals.vLockedTailPositionAndTime[iBatchIndex].xyz
                                  - vertIn.vPosition.xyz;
                    float localProceduralScalar = (ageDelta < -0.001) ? 1.0 : 0.0;

                    posChange *= (1.0f - localProceduralScalar);
                    vertIn.vPosition.xyz += posChange;
                    proceduralScalar *= localProceduralScalar;
                }
            }

            if (SC2_AXIS(b_clampTail))
                ageScalar = vsGlobals.vGPURibbonValues[iBatchIndex].w;

            fAge = (vsGlobals.vGPURibbonValues[iBatchIndex].x - vertIn.vPosition.w)
                 / (vertIn.vInitialVelocity.w - vertIn.vPosition.w);
            fAge = saturate(fAge * ageScalar);
        }

        if (SC2_AXIS(b_proceduralPosition))
        {
            float fMass = vsGlobals.vSystemPhysicsConstants[iBatchIndex].x;
            float fInvMass = vsGlobals.vSystemPhysicsConstants[iBatchIndex].y;
            float fGravity = -vsGlobals.vSystemPhysicsConstants[iBatchIndex].z;
            float fDrag = vsGlobals.vSystemPhysicsConstants[iBatchIndex].w;
            float fInvDrag = 1.0f / vsGlobals.vSystemPhysicsConstants[iBatchIndex].w;
            float3 vOffsetFromOrigin;
            float3 vInstantaneousVelocity;

            float fTime = (vsGlobals.vGPURibbonValues[iBatchIndex].x - vertIn.vPosition.w);
            float3 vInitialVelocity = vertIn.vInitialVelocity.xyz;
            vsCalculateDisplacmentAndVelocity(fTime, vInitialVelocity, fMass, fInvMass,
                                              fDrag, fInvDrag, fGravity,
                                              vOffsetFromOrigin, vInstantaneousVelocity);
            vertIn.vPosition.xyz += (proceduralScalar * vOffsetFromOrigin);

            if (SC2_AXIS(b_precomputedTangent))
                vTangent = vertIn.vPositionPrev.xyz;
            else
                vTangent = -vInstantaneousVelocity;
        }
    }

    // transform instanced ribbons
    if (b_useModelInstancing)
        vertIn.vPosition.xyz = mul(float4(vertIn.vPosition.xyz, 1),
                                   vsRibbonInstanceMatrix(iBatchIndex)).xyz;

    if (SC2_AXIS(b_computeUFromAge))
    {
        if (SC2_AXIS(b_gpuSplineRibbon))
        {
            vertIn.vUV.x = 1.0f - fAge;
        }
        else
        {
            vertIn.vUV.x = 1.0f - (vsGlobals.vGPURibbonValues[iBatchIndex].y * fAge
                                 + vsGlobals.vGPURibbonValues[iBatchIndex].z);
        }
    }

    fSize = vsInterpolateValue(
        fAge,
        vertIn.vSize.x, vertIn.vSize.y, vertIn.vSize.z,
        vsGlobals.vMidKeyTimes[iBatchIndex].x,
        vsGlobals.vInversedMidKeyTimes[iBatchIndex].x,
        vsGlobals.vMidKeyHoldTimes[iBatchIndex].x,
        SC2_AXIS(b_iRibbonSizeInterpolation));

    fSize = fSize * 0.5f;

    if (!SC2_AXIS(b_proceduralPosition) && !SC2_AXIS(b_gpuSplineRibbon))
    {
        float3 vPrevSegmentDir = vertIn.vPosition.xyz - vertIn.vPositionPrev.xyz;
        float3 vNextSegmentDir = vertIn.vPositionNext.xyz - vertIn.vPosition.xyz;

        // Average direction of both segments — not the "real" tangent, but it
        // produces smoother results.
        vTangent = vPrevSegmentDir + vNextSegmentDir;
    }

    vTangent = vsSafeNormalize(vTangent, float3(1, 0, 0));

    // compute binormal
    if ((!SC2_AXIS(b_splineRibbon)) && (!SC2_AXIS(b_proceduralPosition) || SC2_AXIS(b_precomputedTangent)
                              || b_iRibbonSimTech == RIBBON_SIM_MIXED))
    {
        if (b_iRibbonType == RIBBON_BILLBOARD)
        {
            // put the camera forward into the ribbon's local space first
            float3 cameraDir = vsGlobals.vCameraDirection.xyz;
            if (SC2_AXIS(b_ribbonLocalSpace))
                cameraDir.xyz = mul(vsGlobals.vCameraDirection.xyz,
                                    (float3x3)(vsInvPRWorldMatrix(iBatchIndex)));

            vNormal = -cameraDir.xyz;
            vBinormal = vsSafeNormalize(cross(vNormal, vTangent), float3(0, 1, 0));
        }
        else
        {
            vBinormal = vertIn.vUp.xyz;
            vNormal = vsSafeNormalize(cross(vTangent, vBinormal), float3(0, 0, 1));
        }
    }
    else
    {
        if (b_iRibbonType == RIBBON_BILLBOARD)
        {
            float3 cameraDir = vsGlobals.vCameraDirection.xyz;
            if (SC2_AXIS(b_ribbonLocalSpace))
                cameraDir.xyz = mul(vsGlobals.vCameraDirection.xyz,
                                    (float3x3)(vsInvPRWorldMatrix(iBatchIndex)));

            // Flatten onto the camera plane; the billboard direction is
            // perpendicular to the flattened segment.
            float3 vSegment = normalize(vTangent - cameraDir
                                        * dot(vTangent, vsGlobals.vCameraDirection));
            vBinormal = cross(vSegment, cameraDir);
        }
        else
        {
            vBinormal = vsSafeNormalize(cross(vTangent.xyz, vertIn.vUp.xyz),
                                        float3(0, 1, 0));
        }

        vNormal = vsSafeNormalize(cross(vTangent, vBinormal), float3(0, 0, 1));
    }

    float fAngle;
    if (fAge < vsGlobals.vMidKeyTimes[iBatchIndex].w)
        fAngle = vertIn.vRotation.x + (vertIn.vRotation.y - vertIn.vRotation.x)
               * fAge / vsGlobals.vMidKeyTimes[iBatchIndex].w;
    else
        fAngle = vertIn.vRotation.y + (vertIn.vRotation.z - vertIn.vRotation.y)
               * (fAge - vsGlobals.vMidKeyTimes[iBatchIndex].w)
               / (1.0f - vsGlobals.vMidKeyTimes[iBatchIndex].w);

    float3x3 mRotation = vsMakeRotation(fAngle, vTangent);
    vBinormal = mul(vBinormal, mRotation);
    vNormal = mul(vNormal, mRotation);

    if (b_iRibbonType == RIBBON_BILLBOARD || b_iRibbonType == RIBBON_PLANAR)
    {
        vOffset = vertIn.vOffset.y * vBinormal;
        vVertexNormal = vNormal;
        vVertexTangent = -vBinormal;
        vVertexBinormal = -vTangent;
    }
    else if (b_iRibbonType == RIBBON_CYLINDER || b_iRibbonType == RIBBON_STAR)
    {
        // Basis around the segment (rows: normal, tangent, binormal).
        float3x3 basis = float3x3(vNormal.x,   vNormal.y,   vNormal.z,
                                  vTangent.x,  vTangent.y,  vTangent.z,
                                  vBinormal.x, vBinormal.y, vBinormal.z);
        vOffset = mul(float3(vertIn.vOffset.x, 0.0f, vertIn.vOffset.y), basis);

        if (b_iRibbonType == RIBBON_CYLINDER)
            vOffset = vsSafeNormalize(vOffset, float3(0, 0, 1));

        // :TODO: (original) Planar ribbons are dual-sided and would need normals
        // on both sides, which means duplicating vertices.
        vVertexNormal = vOffset;
        vVertexTangent = vTangent;
        vVertexBinormal = cross(vVertexTangent, vVertexNormal);
    }

    vertIn.vPosition.xyz = vertIn.vPosition.xyz + vOffset * fSize;

    if (SC2_AXIS(b_ribbonLocalSpace))
    {
        float4x4 mPRWorld = vsPRWorldMatrix(iBatchIndex);
        vertIn.vPosition.xyz = mul(float4(vertIn.vPosition.xyz, 1.0f), mPRWorld).xyz;
        vVertexNormal = mul(vVertexNormal, (float3x3)mPRWorld);
        vVertexTangent = mul(vVertexTangent, (float3x3)mPRWorld);
        vVertexBinormal = mul(vVertexBinormal, (float3x3)mPRWorld);
    }

    if (b_enableVertexWarps == 1)
        vertIn.vPosition.xyz = vsApplyWarps(float4(vertIn.vPosition.xyz, 1.0)).xyz;

    vertOut.HPos = vsEmitRibbonHPos(vertIn);

    float4 interpNormal   = vsEmitRibbonNormal(vVertexNormal);
    float4 interpTangent  = vsEmitRibbonTangent(vVertexTangent);
    float3 interpBinormal = vVertexBinormal;

#if SC2_HAS_WorldPos
    vertOut.m_WorldPos = float4(vertIn.vPosition.xyz, 1);
#endif
#if SC2_HAS_HPosAsUV
    vertOut.m_HPosAsUV = mul(float4(vertIn.vPosition.xyz, 1), vsGlobals.mVPTransform);
#endif
#if SC2_HAS_ViewPos
    vertOut.m_ViewPos = vsEmitRibbonViewPos(vertIn);
#endif
#if SC2_HAS_Normal
    vertOut.m_Normal = interpNormal;
#endif
#if SC2_HAS_Tangent
    vertOut.m_Tangent = interpTangent;
#endif
#if SC2_HAS_Binormal
    vertOut.m_Binormal = interpBinormal;
#endif
#if SC2_HAS_VertexColor
    vertOut.m_VertexColor = vsEmitRibbonVertexColor(vertIn, fAge, iBatchIndex);
#endif

    // Vertex lighting — a ribbon passes no lighting regions (literal 0 vector).
#if (SC2_HAS_Diffuse || SC2_HAS_Specular || SC2_HAS_ShadowDiffuse || SC2_HAS_ShadowSpecular)
    float3 cVertexSpecularLighting = float3(0, 0, 0);
    float3 cVertexShadowSpecularLighting = float3(0, 0, 0);
    float3 cVertexShadowSpecularLighting2 = float3(0, 0, 0);
  #if SC2_HAS_Diffuse
    vertOut.m_Diffuse = vsEmitDiffuse(vertOut, vertIn.vPosition.xyz, vVertexNormal,
                                      float4(0, 0, 0, 0), cVertexSpecularLighting,
                                      cVertexShadowSpecularLighting,
                                      cVertexShadowSpecularLighting2);
  #endif
  #if SC2_HAS_Specular
    vertOut.m_Specular = vsEmitSpecular(cVertexSpecularLighting);
  #endif
  #if SC2_HAS_ShadowDiffuse
    vertOut.m_ShadowDiffuse = vsEmitShadowDiffuse(vVertexNormal, float4(0, 0, 0, 0),
                                                  cVertexShadowSpecularLighting,
                                                  cVertexShadowSpecularLighting2);
  #endif
  #if SC2_HAS_ShadowSpecular
    vertOut.m_ShadowSpecular = vsEmitShadowSpecular(cVertexShadowSpecularLighting);
  #endif
#endif

#if SC2_HAS_ShadowMapUV
    vertOut.m_ShadowMapUV = vsEmitShadowMapUV(vertIn.vPosition.xyz);
#endif
#if SC2_HAS_FogColor
    vertOut.m_FogColor = vsEmitFogColor(vertIn.vPosition.xyz, float3(0, 0, 0));
#endif
#if SC2_HAS_HalfVec
    vertOut.m_HalfVec = vsComputeHalfVector(vertIn.vPosition.xyz);
#endif
#if SC2_HAS_EyeToVertex
    vertOut.m_EyeToVertex = vsEmitRibbonEyeToVertex(vertIn);
#endif
#if SC2_HAS_EyeToVertexFresnel
    vertOut.m_EyeToVertexFresnel = vsEmitRibbonEyeToVertex(vertIn);
#endif
#if SC2_HAS_FOWUV
    vertOut.m_FOWUV = vsEmitRibbonFOWUV(vertIn);
#endif

#if b_usePackedUVEmitter
  #if (b_iUVEmitterCount / 2 > 0)
    SC2_RIB_UV_PAIR(0, 0, 1);
  #endif
  #if (b_iUVEmitterCount / 2 > 1)
    SC2_RIB_UV_PAIR(1, 2, 3);
  #endif
  #if (b_iUVEmitterCount / 2 > 2)
    SC2_RIB_UV_PAIR(2, 4, 5);
  #endif
  #if (b_iUVEmitterCount / 2 > 3)
    SC2_RIB_UV_PAIR(3, 6, 7);
  #endif
  #if (b_iUVEmitterCount == 1)
    SC2_RIB_UV(0, 0);
  #elif (b_iUVEmitterCount == 3)
    SC2_RIB_UV(1, 2);
  #elif (b_iUVEmitterCount == 5)
    SC2_RIB_UV(2, 4);
  #elif (b_iUVEmitterCount == 7)
    SC2_RIB_UV(3, 6);
  #endif
#else
  #if (b_iUVEmitterCount > 0)
    SC2_RIB_UV(0, 0);
  #endif
  #if (b_iUVEmitterCount > 1)
    SC2_RIB_UV(1, 1);
  #endif
  #if (b_iUVEmitterCount > 2)
    SC2_RIB_UV(2, 2);
  #endif
  #if (b_iUVEmitterCount > 3)
    SC2_RIB_UV(3, 3);
  #endif
  #if (b_iUVEmitterCount > 4)
    SC2_RIB_UV(4, 4);
  #endif
  #if (b_iUVEmitterCount > 5)
    SC2_RIB_UV(5, 5);
  #endif
  #if (b_iUVEmitterCount > 6)
    SC2_RIB_UV(6, 6);
  #endif
  #if (b_iUVEmitterCount > 7)
    SC2_RIB_UV(7, 7);
  #endif
#endif

#if SC2_HAS_ParallaxVector
    // ribbon.fx reads the interpolants BACK here (INTERPOLANT_Normal/Tangent/
    // Binormal), it does not reuse the locals the way foliage.fx does — so a
    // varying that was packed out contributes its safety-net value (0, or the
    // Normal×Tangent reconstruction for Binormal), not the value just computed.
    vertOut.m_ParallaxVector = vsEmitParallaxVector(
        vertIn.vPosition.xyz, vsGlobals.vEyePos,
        vertOut.Normal().xyz, vertOut.Tangent().xyz, vertOut.Binormal());
#endif

    return vertOut;
}
