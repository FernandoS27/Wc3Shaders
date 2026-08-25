// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026, Fernando Sahmkow
// See LICENSE in the repository root for full terms.
//
// Particle family vertex shader (particle.fx : ParticleVertexMain).
//
// Every particle in a system is one quad expanded here from a single point.  The
// vertex shader ages it, interpolates size / colour / rotation from three
// keyframes, optionally integrates its position under gravity and drag, then
// builds the quad's basis — and `b_iInstanceType` is what decides how that basis
// is oriented:
//
//   BILLBOARD(0)             camera-facing, spun about the view direction
//   TAIL(1) / TRAIL(10)      stretched along the velocity, length ∝ speed
//   FACE_TRAVEL_DIR(2)       facing the instantaneous velocity
//   FACE_WORLD_DIR(3)        facing a fixed world direction
//   SINGLE_AXIS(4)           spun about one supplied axis
//   TERRAIN_ORIENTED(5)      flattened onto the terrain normal
//   TERRAIN_DIR_ORIENTED(6)  ...and stretched along the projected velocity
//   EMITTER_ORIENTED(7)      basis unpacked from the emitter's rotation
//   PHYSICS_ORIENTED(8)      same unpack, driven by the physics body
//   PINNED(9)                quad spanning spawn point to current point
//
// Unlike Ribbon, the vertex declaration is FIXED — `Input` is unpermuted — but it
// is unusual in carrying true INTEGER attributes (`int4 vSize`, `int4 vRotation`,
// `int2 vOffset`, `uint4 iBatchIndex`), which the body rescales by 1/256 and 1/32.
//
// Particle is also the only root that reads `b_UVRandomOffsetEnable[]` (the other
// three merely forward it to InitShader), and the only one with a flipbook UV
// mode that walks an atlas by cell index.

implementing sc2_shaders;

#include "vs/model_vs_defines.slangh"
#include "types/perm_cb.slangh"

// Axes read as CODE need a file-local default: only NONZERO axes are passed
// as -D, so a zero one would otherwise be an undefined identifier.
#ifndef b_clampedTailLength
#define b_clampedTailLength 0
#endif
#ifndef b_fixedTailLength
#define b_fixedTailLength 0
#endif
#ifndef b_iParticleColorInterpolation
#define b_iParticleColorInterpolation 0
#endif
#ifndef b_iParticleRotationInterpolation
#define b_iParticleRotationInterpolation 0
#endif
#ifndef b_iParticleSizeInterpolation
#define b_iParticleSizeInterpolation 0
#endif
#ifndef b_localSpace
#define b_localSpace 0
#endif
#ifndef b_randomFlipBookStart
#define b_randomFlipBookStart 0
#endif
#ifndef b_useProceduralPosition
#define b_useProceduralPosition 0
#endif

// ---------------------------------------------------------------------------
// particle.fx : Input — the (unpermuted) vertex declaration.
// ---------------------------------------------------------------------------
public struct ParticleVSStreamIn
{
    float4 vPosition          : POSITION;
    int4   vSize              : TEXCOORD0;
    float4 cColor0            : COLOR0;
    float4 cColor1            : COLOR1;
    float4 cColor2            : TEXCOORD1;
    int4   vRotation          : TEXCOORD2;
    float4 vBirthDeathAndDrag : TEXCOORD3;
    uint4  iBatchIndex        : TEXCOORD4;
    float4 vInterpolator1     : TEXCOORD5;
    float4 vInterpolator2     : TEXCOORD6;
    float4 vNoiseVector       : TEXCOORD7;
    int2   vOffset            : NORMAL;
};

float4x4 vsParticleInstanceMatrix(int i)
{
    int b = i * 4;
    return float4x4(vsGlobals.mParticleInstanceTransform[b + 0],
                    vsGlobals.mParticleInstanceTransform[b + 1],
                    vsGlobals.mParticleInstanceTransform[b + 2],
                    vsGlobals.mParticleInstanceTransform[b + 3]);
}

// ---------------------------------------------------------------------------
// particle.fx : VERTEX SHADER EMITTERS
// ---------------------------------------------------------------------------
float4 vsEmitParticleHPos(ParticleVSStreamIn vertIn)
{
    return mul(float4(vertIn.vPosition.xyz, 1.0), vsGlobals.mVPTransform);
}

float3 vsEmitParticleViewPos(ParticleVSStreamIn vertIn)
{
    return mul(vsGlobals.mViewTransform, float4(vertIn.vPosition.xyz, 1.0));   // as 3x4
}

float3 vsEmitParticleEyeToVertex(ParticleVSStreamIn vertIn)
{
    return normalize(vertIn.vPosition.xyz - vsGlobals.vEyePos);
}

float4 vsEmitParticleFOWUV(ParticleVSStreamIn vertIn)
{
    return mul(float4(vertIn.vPosition.xyz, 1.0), vsGlobals.mFOWMatrix);
}

float4 vsEmitParticleVertexColor(ParticleVSStreamIn vertIn, float4 vSizeAndAge,
                                 int iBatchIndex)
{
    float4 cColor;

    cColor.rgb = vsInterpolateValue(
        vSizeAndAge.w,
        vertIn.cColor0.rgb, vertIn.cColor1.rgb, vertIn.cColor2.rgb,
        vsGlobals.vMidKeyTimes[iBatchIndex].y,
        vsGlobals.vInversedMidKeyTimes[iBatchIndex].y,
        vsGlobals.vMidKeyHoldTimes[iBatchIndex].y,
        SC2_AXIS(b_iParticleColorInterpolation));

    cColor.a = vsInterpolateValue(
        vSizeAndAge.w,
        vertIn.cColor0.a, vertIn.cColor1.a, vertIn.cColor2.a,
        vsGlobals.vMidKeyTimes[iBatchIndex].z,
        vsGlobals.vInversedMidKeyTimes[iBatchIndex].z,
        vsGlobals.vMidKeyHoldTimes[iBatchIndex].z,
        SC2_AXIS(b_iParticleColorInterpolation));

    return cColor;
}

// UVs.  The quad corner (vOffset, -1/+1) becomes the base [0,1] coordinate, then
// either walks the flipbook atlas or goes through the shared GenUV.
//
// `M` is b_iUVMapping[index] and `R` is b_UVRandomOffsetEnable[index]; both are
// InitShader-filled arrays that the engine fills with per-instance literals, so
// they are compile-time constants here exactly as fxc sees them after folding.
float4 vsEmitParticleUV<let M : int, let I : int, let R : int>(
    ParticleVSStreamIn vertIn, float4 vSizeAndAge, int iBatchIndex)
{
    float2 vUV = (float2)vertIn.vOffset * float2(0.5f, -0.5f) + float2(0.5f, 0.5f);

    if (M == UVMAP_PARTICLE_FLIPBOOK)
    {
        int iCell;
        float fbColumns =
            vsGlobals.vSystemTime_ElementScale_FlipbookMidKeyTime_FlipbookColumnCount[iBatchIndex].w;
        float fbMidKey =
            vsGlobals.vSystemTime_ElementScale_FlipbookMidKeyTime_FlipbookColumnCount[iBatchIndex].z;

        if (vSizeAndAge.w <= fbMidKey)
        {
            float fRange = vsGlobals.fFlipbookFrames[iBatchIndex].y
                         - vsGlobals.fFlipbookFrames[iBatchIndex].x;
            float fOffset = fRange * (vSizeAndAge.w / fbMidKey);
            iCell = (int)(vsGlobals.fFlipbookFrames[iBatchIndex].x + floor(fOffset + 0.5f));
        }
        else
        {
            float fRange = vsGlobals.fFlipbookFrames[iBatchIndex].z
                         - vsGlobals.fFlipbookFrames[iBatchIndex].y;
            float fOffset = fRange * ((vSizeAndAge.w - fbMidKey) / (1.0 - fbMidKey));
            iCell = (int)(vsGlobals.fFlipbookFrames[iBatchIndex].y + floor(fOffset + 0.5f));
        }
        // NOTE: gated on b_randomFlipBookStart ALONE — the flipbook arm never
        // consults b_UVRandomOffsetEnable (that axis only drives the else arm).
        if (SC2_AXIS(b_randomFlipBookStart))
            iCell += (int)floor(vertIn.vNoiseVector.w);

        // NOTE the asymmetry, transcribed as-is: the COLUMN is a true integer
        // modulus against `(int)columnCount`, but the ROW divides the int by the
        // FLOAT column count (so it is a float divide, truncated on assignment).
        int iCellX = iCell % (int)fbColumns;
        int iCellY = (int)(iCell / fbColumns);
        float2 vOffs = float2(iCellX * vsGlobals.vFlipbookCellSize[iBatchIndex].x,
                              iCellY * vsGlobals.vFlipbookCellSize[iBatchIndex].y);
        vUV = vUV * vsGlobals.vFlipbookCellSize[iBatchIndex] + vOffs;
    }
    else
    {
        // offset with a random uv (the two bytes packed into vRotation.w)
        if (R != 0)
        {
            float r = vertIn.vRotation.w;
            float x = floor(r / 256);      // hi byte
            float y = r - x * 256;         // lo byte
            vUV += float2(x, y) / 255;
        }

        vUV = vsGenUV<M, I>(
            vertIn.vPosition.xyz,
            vertIn.vPosition.xyz,
            float3(0, 0, 0),               // no normal
            float3(0, 0, 0),
            vUV,
            vUV,                           // no vUV slot 1
            vUV,
            vUV,
            0,
            float3(0, 0, 0)).xy;
    }

    return float4(vUV, 0, 1);
}

float vsGenerateRotation(float fAge, float3 vRotation, int iBatchIndex)
{
    return vsInterpolateValue(
        fAge, vRotation.x, vRotation.y, vRotation.z,
        vsGlobals.vMidKeyTimes[iBatchIndex].w,
        vsGlobals.vInversedMidKeyTimes[iBatchIndex].w,
        vsGlobals.vMidKeyHoldTimes[iBatchIndex].w,
        SC2_AXIS(b_iParticleRotationInterpolation));
}

// ---------------------------------------------------------------------------
// particle.fx : CalculatePositionAndVelocity — integrate the procedural motion
// and write the instantaneous velocity back into the interpolators the basis
// code will read.
// ---------------------------------------------------------------------------
void vsCalculatePositionAndVelocity(inout ParticleVSStreamIn vertIn,
                                    float4 vSizeAndAge, int iBatchIndex)
{
    float3 vOffsetFromOrigin;
    float3 vInstantaneousVelocity;

    float fTime = vsGlobals.vSystemTime_ElementScale_FlipbookMidKeyTime_FlipbookColumnCount[iBatchIndex].x
                - vertIn.vBirthDeathAndDrag.x;
    float3 vInitialVelocity = vertIn.vInterpolator1.xyz;
    float fInvMass = vertIn.vInterpolator1.w;
    float fMass = 1.0f / fInvMass;
    float fDrag = vertIn.vBirthDeathAndDrag.z;
    float fInvDrag = vertIn.vBirthDeathAndDrag.w;
    float fGravity = -vertIn.vInterpolator2.w;
    vsCalculateDisplacmentAndVelocity(fTime, vInitialVelocity, fMass, fInvMass,
                                      fDrag, fInvDrag, fGravity,
                                      vOffsetFromOrigin, vInstantaneousVelocity);

    vertIn.vPosition.xyz += vOffsetFromOrigin;

    if (b_iInstanceType == PARTICLE_FACE_TRAVEL_DIR)
    {
        vertIn.vInterpolator2.xyz = vInstantaneousVelocity;
    }
    else if (b_iInstanceType == PARTICLE_TAIL || b_iInstanceType == PARTICLE_PINNED
             || b_iInstanceType == PARTICLE_TRAIL)
    {
        vertIn.vInterpolator1.xyz = vInstantaneousVelocity;

        if (SC2_AXIS(b_clampedTailLength))
        {
            float fSpeed = length(vInstantaneousVelocity);
            float fTailLength;
            if (SC2_AXIS(b_fixedTailLength))
                fTailLength = vertIn.vInterpolator2.x * fSpeed * vSizeAndAge.xyz.x;
            else
                fTailLength = max(vertIn.vInterpolator2.x,
                                  vertIn.vInterpolator2.x * fSpeed) * vSizeAndAge.xyz.x;

            float fOffsetDistance = length(vOffsetFromOrigin);
            if (fOffsetDistance < fTailLength)
                vertIn.vInterpolator2.x = min(fOffsetDistance / (fSpeed * vSizeAndAge.xyz.x),
                                              vertIn.vInterpolator2.x);
        }
    }
}

// ---------------------------------------------------------------------------
// particle.fx : UnpackNormals — the basis for the four "oriented" instance types.
// ---------------------------------------------------------------------------
void vsUnpackNormals(ParticleVSStreamIn vertIn, out float3 vRight, out float3 vUp,
                     out float3 vForward, int iBatchIndex)
{
    if (b_iInstanceType == PARTICLE_EMITTER_ORIENTED)
    {
        if (SC2_AXIS(b_localSpace))
        {
            // every particle takes the emitter's current rotation
            float4x4 mPRWorld = vsPRWorldMatrix(iBatchIndex);
            vRight = normalize(mPRWorld[0].xyz);
            vUp = normalize(mPRWorld[1].xyz);
            vForward = normalize(cross(vRight, vUp));
        }
        else
        {
            // vInterpolator2 packs right and up two-per-channel:
            //   x => right.y | right.x     y => right.z | up.x     z => up.y | up.z
            vRight.y = (trunc(vertIn.vInterpolator2.x / 65536.0f));
            vRight.x = vertIn.vInterpolator2.x - (vRight.y * 65536.0f);
            vRight.z = (trunc(vertIn.vInterpolator2.y / 65536.0f));

            vUp.x = vertIn.vInterpolator2.y - (vRight.z * 65536.0f);
            vUp.y = (trunc(vertIn.vInterpolator2.z / 65536.0f));
            vUp.z = vertIn.vInterpolator2.z - (vUp.y * 65536.0f);

            vRight = ((vRight / 255.0f) * 2) - 1;
            vUp = ((vUp / 255.0f) * 2) - 1;

            vRight = normalize(vRight);
            vUp = normalize(vUp);
            vForward = normalize(cross(vRight, vUp));
        }
    }
    else
    {
        vForward = vertIn.vInterpolator2.xyz;
        vForward = normalize(vForward + float3(0.0f, 0.0001f, 0.0f));
        vRight = normalize(cross(float3(0.0f, 0.0f, 1.0f), vForward));
        vUp = normalize(cross(vForward, vRight));
    }
}

// ---------------------------------------------------------------------------
// UV-emitter unroll.  Particle needs the RANDOM-OFFSET axis alongside the
// mapping, so its macros take one more generic argument than the other roots'.
// ---------------------------------------------------------------------------
#if SC2_HAS_UV
  #define SC2_PAR_UV(dst, i)                                                        \
    vertOut.m_UV[dst] = vsEmitParticleUV<SC2_UVMAPPING##i, i, SC2_UVRANDOM##i>(     \
        vertIn, vSizeAndAge, iBatchIndex)
  #define SC2_PAR_UV_PAIR(dst, i, j)                                                \
    vertOut.m_UV[dst] = float4(                                                     \
        vsEmitParticleUV<SC2_UVMAPPING##i, i, SC2_UVRANDOM##i>(                     \
            vertIn, vSizeAndAge, iBatchIndex).xy,                                   \
        vsEmitParticleUV<SC2_UVMAPPING##j, j, SC2_UVRANDOM##j>(                     \
            vertIn, vSizeAndAge, iBatchIndex).xy)
#else
  #define SC2_PAR_UV(dst, i)
  #define SC2_PAR_UV_PAIR(dst, i, j)
#endif

// ---------------------------------------------------------------------------
// Main vertex shader body.
// ---------------------------------------------------------------------------
[shader("vertex")]
public DefaultVSOutput particle_vs_main(ParticleVSStreamIn streamIn)
{
    DefaultVSOutput vertOut;
    ParticleVSStreamIn vertIn = streamIn;    // the original mutates its `in` parameter

    float4 vSizeAndAge;
    float3 vParticleNormal = float3(0, 0, 0);
    float3 vParticleTangent = float3(0, 0, 0);
    float3 vParticleBinormal = float3(0, 0, 0);
    int iBatchIndex = 0;

    // decompress vertex
    float4 vInputSize = vertIn.vSize * (1.f / 256.f);
    float3 vInputRotation = (vertIn.vRotation * (1.f / 32.f)).xyz;

    iBatchIndex = (int)(vertIn.iBatchIndex.x + 0.1);
    iBatchIndex = (int)(vsGlobals.fParticleBatchIndexRemappingTable[iBatchIndex] + 0.1);

    vSizeAndAge.w = (vsGlobals.vSystemTime_ElementScale_FlipbookMidKeyTime_FlipbookColumnCount[iBatchIndex].x
                     - vertIn.vBirthDeathAndDrag.x)
                  / (vertIn.vBirthDeathAndDrag.y - vertIn.vBirthDeathAndDrag.x);
    // dead particles can still be submitted, hence the saturate
    vSizeAndAge.w = saturate(vSizeAndAge.w);

    // Compute size.  The arguments are scalars, so this picks the SCALAR overload
    // of InterpolateValue and the single result is broadcast across xyz (which is
    // also why the flipbook/tail code can use `vSizeAndAge.xyz.x` interchangeably).
    float fSizeInterp = vsInterpolateValue(
        vSizeAndAge.w,
        vInputSize.x, vInputSize.y, vInputSize.z,
        vsGlobals.vMidKeyTimes[iBatchIndex].x,
        vsGlobals.vInversedMidKeyTimes[iBatchIndex].x,
        vsGlobals.vMidKeyHoldTimes[iBatchIndex].x,
        SC2_AXIS(b_iParticleSizeInterpolation));
    vSizeAndAge.xyz = float3(fSizeInterp, fSizeInterp, fSizeInterp);

    if (SC2_AXIS(b_useProceduralPosition))
        vsCalculatePositionAndVelocity(vertIn, vSizeAndAge, iBatchIndex);

    if (SC2_AXIS(b_localSpace))
        vertIn.vPosition.xyz = mul(float4(vertIn.vPosition.xyz, 1),
                                   vsPRWorldMatrix(iBatchIndex)).xyz;

    vertIn.vPosition.xyz += vertIn.vNoiseVector.xyz;

    if (b_iInstanceType == PARTICLE_SINGLE_AXIS)
    {
        // fixed up direction
        float3 vRight = normalize(cross(vertIn.vInterpolator2.xyz, vsGlobals.vCameraDirection));

        float fAngle = vsGenerateRotation(vSizeAndAge.w, vInputRotation, iBatchIndex);

        float3 vForward = normalize(cross(vRight, vertIn.vInterpolator2.xyz));
        float3x3 mRotation = vsMakeRotation(fAngle, vForward);
        float3 vOffset = vertIn.vOffset.x * vRight
                       + vertIn.vOffset.y * vertIn.vInterpolator2.xyz;
        vOffset *= vsGlobals.vSystemTime_ElementScale_FlipbookMidKeyTime_FlipbookColumnCount[iBatchIndex].y;
        vertIn.vPosition.xyz = vertIn.vPosition.xyz + mul(vSizeAndAge.xyz * vOffset, mRotation);

        vParticleNormal = normalize(cross(vRight, vForward));
        vParticleTangent = vRight;
        vParticleBinormal = vForward;
    }
    else if (b_iInstanceType == PARTICLE_FACE_TRAVEL_DIR
          || b_iInstanceType == PARTICLE_FACE_WORLD_DIR
          || b_iInstanceType == PARTICLE_EMITTER_ORIENTED
          || b_iInstanceType == PARTICLE_PHYSICS_ORIENTED)
    {
        float3 vRight, vUp, vDirectionVector;
        vsUnpackNormals(vertIn, vRight, vUp, vDirectionVector, iBatchIndex);

        float4 vOffsetAngle;
        vOffsetAngle.xyz = vertIn.vOffset.x * vRight + vertIn.vOffset.y * vUp;
        vOffsetAngle.xyz *= vsGlobals.vSystemTime_ElementScale_FlipbookMidKeyTime_FlipbookColumnCount[iBatchIndex].y;
        vOffsetAngle.w = vsGenerateRotation(vSizeAndAge.w, vInputRotation, iBatchIndex);

        float3x3 mRotation = vsMakeRotation(vOffsetAngle.w, vDirectionVector);
        vertIn.vPosition.xyz = vertIn.vPosition.xyz
                             + mul(vSizeAndAge.xyz * vOffsetAngle.xyz, mRotation);
    }
    else if (b_iInstanceType == PARTICLE_TERRAIN_ORIENTED)
    {
        // vInterpolator1 = particle direction, vInterpolator2 = terrain normal
        float3 vDirectionVector = vertIn.vInterpolator1.xyz;
        float3 vProjectedDir = vDirectionVector
                             - dot(vDirectionVector, vertIn.vInterpolator2.xyz)
                               * vertIn.vInterpolator2.xyz;

        float fAngle = vsGenerateRotation(vSizeAndAge.w, vInputRotation, iBatchIndex);
        float3x3 mRotation = vsMakeRotation(fAngle, vertIn.vInterpolator2.xyz);

        // :TODO: (original) Find a better way.
        if (dot(vProjectedDir, vProjectedDir) < 0.01f)
            vProjectedDir = float3(1.0f, 0.0f, 0.0f);
        vProjectedDir = normalize(vProjectedDir);
        float3 vRight = (cross(vProjectedDir, vertIn.vInterpolator2.xyz));
        vProjectedDir = mul(vProjectedDir, mRotation);
        vRight = mul(vRight, mRotation);
        float3 vOffset = vertIn.vOffset.x * vRight + vertIn.vOffset.y * vProjectedDir;
        vOffset *= vsGlobals.vSystemTime_ElementScale_FlipbookMidKeyTime_FlipbookColumnCount[iBatchIndex].y;
        vertIn.vPosition.xyz = vertIn.vPosition.xyz + (vSizeAndAge.xyz * vOffset);

        vParticleNormal = normalize(cross(vRight, vProjectedDir));
        vParticleTangent = vRight;
        vParticleBinormal = vProjectedDir;

        // flip the basis if the particle faces away from the camera
        float d = -dot(vertIn.vPosition.xyz, vParticleNormal);
        float cameraDistFromParticlePlane = dot(float4(vsGlobals.vEyePos.xyz, 1),
                                                float4(vParticleNormal, d));
        vParticleNormal = (cameraDistFromParticlePlane > 0) ? vParticleNormal : -vParticleNormal;
        vParticleTangent = (cameraDistFromParticlePlane > 0) ? vParticleTangent : -vParticleTangent;
        vParticleBinormal = (cameraDistFromParticlePlane > 0) ? vParticleBinormal : -vParticleBinormal;
    }
    else if (b_iInstanceType == PARTICLE_TERRAIN_DIR_ORIENTED)
    {
        // vInterpolator1 = particle velocity, vInterpolator2 = terrain normal
        float fMag = length(vertIn.vInterpolator1.xyz);

        float3 vDirectionVector = normalize(vertIn.vInterpolator1.xyz);
        float3 vProjectedDir = normalize(vDirectionVector
                                         - dot(vDirectionVector, vertIn.vInterpolator2.xyz)
                                           * vertIn.vInterpolator2.xyz);
        float3 vRight = (cross(vProjectedDir, vertIn.vInterpolator2.xyz));

        // scale by the tail length, so it grows as we go faster
        vProjectedDir = vProjectedDir * max(vInputRotation.x, (fMag * vInputRotation.x));
        float3 vOffset = vertIn.vOffset.x * vRight + vertIn.vOffset.y * vProjectedDir;

        vOffset *= vsGlobals.vSystemTime_ElementScale_FlipbookMidKeyTime_FlipbookColumnCount[iBatchIndex].y;
        vertIn.vPosition.xyz = vertIn.vPosition.xyz + (vSizeAndAge.xyz * vOffset);

        vParticleNormal = normalize(cross(vRight, vProjectedDir));
        vParticleTangent = vRight;
        vParticleBinormal = vProjectedDir;

        float d = -dot(vertIn.vPosition.xyz, vParticleNormal);
        float cameraDistFromParticlePlane = dot(float4(vsGlobals.vEyePos.xyz, 1),
                                                float4(vParticleNormal, d));
        vParticleNormal = (cameraDistFromParticlePlane > 0) ? vParticleNormal : -vParticleNormal;
        vParticleTangent = (cameraDistFromParticlePlane > 0) ? vParticleTangent : -vParticleTangent;
        vParticleBinormal = (cameraDistFromParticlePlane > 0) ? vParticleBinormal : -vParticleBinormal;
    }
    else if (b_iInstanceType == PARTICLE_TAIL || b_iInstanceType == PARTICLE_TRAIL)
    {
        vertIn.vInterpolator1.xyz += vertIn.vNoiseVector.xyz;

        float fMag = length(vertIn.vInterpolator1.xyz);
        if (SC2_AXIS(b_localSpace))
            vertIn.vInterpolator1.xyz = mul(vertIn.vInterpolator1.xyz,
                                            (float3x3)vsPRWorldMatrix(iBatchIndex));

        float3 vDirectionVector = normalize(vertIn.vInterpolator1.xyz);
        float3 vRight = normalize((cross(vsGlobals.vCameraDirection, vDirectionVector)));

        float fTailLength;
        if (SC2_AXIS(b_fixedTailLength))
            fTailLength = vertIn.vInterpolator2.x;
        else
            fTailLength = max(vertIn.vInterpolator2.x, (fMag * vertIn.vInterpolator2.x));
        vDirectionVector = vDirectionVector * fTailLength;

        float3 vOffset = vertIn.vOffset.x * vRight + vertIn.vOffset.y * vDirectionVector;
        float3 vSize = vsGlobals.vSystemTime_ElementScale_FlipbookMidKeyTime_FlipbookColumnCount[iBatchIndex].y
                     * vSizeAndAge.xyz;
        vertIn.vPosition.xyz = vertIn.vPosition.xyz + vOffset * vSize;

        if (b_iInstanceType == PARTICLE_TRAIL)
            vertIn.vPosition.xyz -= vSize * vDirectionVector;

        vParticleNormal = normalize(cross(vRight, vDirectionVector));
        vParticleTangent = vRight;
        vParticleBinormal = -vDirectionVector;
    }
    else if (b_iInstanceType == PARTICLE_PINNED)
    {
        if (SC2_AXIS(b_localSpace))
            vertIn.vInterpolator2.xyz = mul(float4(vertIn.vInterpolator2.xyz, 1),
                                            vsPRWorldMatrix(iBatchIndex)).xyz;

        // delta from the spawn position to the current position
        float3 vDelta = vertIn.vPosition.xyz - vertIn.vInterpolator2.xyz;
        float norm = length(vDelta);

        float3 vForward = vDelta / norm;
        float3 vRight = normalize(cross(vsGlobals.vCameraDirection, vForward));
        float endScale = vertIn.vOffset.y * .5f + .5f;
        endScale = lerp(1.f, vInputSize.w, endScale);

        float3 vCenter = 0.5f * (vertIn.vPosition.xyz + vertIn.vInterpolator2.xyz);
        float3 vOffset = vertIn.vOffset.x * vSizeAndAge.xyz * vRight * endScale
                       + 0.5f * vertIn.vOffset.y * norm * vForward;

        vertIn.vPosition.xyz = vCenter + vOffset;

        vParticleNormal = cross(vRight, vForward);
        vParticleTangent = vRight;
        vParticleBinormal = -vForward;
    }
    else
    {
        // regular billboard
        float fAngle = vsGenerateRotation(vSizeAndAge.w, vInputRotation, iBatchIndex);

        float3 forward = vsGlobals.vCameraDirection.xyz;
        float3 right = vsGlobals.vBillboardRight.xyz;
        float3 up = vsGlobals.vBillboardUp.xyz;
        if (b_useModelInstancing)
        {
            // NOTE: indexes with the RAW attribute and a bare (int) cast — NOT the
            // `+ 0.1`-rounded iBatchIndex computed above.  Transcribed as-is.
            float4x4 mInstanceTransfrom = vsParticleInstanceMatrix(
                (int)vsGlobals.fParticleBatchIndexRemappingTable[vertIn.iBatchIndex.x]);

            vertIn.vPosition.xyz = mul(float4(vertIn.vPosition.xyz, 1), mInstanceTransfrom).xyz;
        }

        float3x3 mRotation = vsMakeRotation(fAngle, forward);
        float3 vOffset = vertIn.vOffset.x * right + vertIn.vOffset.y * up;
        vOffset *= vsGlobals.vSystemTime_ElementScale_FlipbookMidKeyTime_FlipbookColumnCount[iBatchIndex].y;
        vertIn.vPosition.xyz = vertIn.vPosition.xyz + mul(vSizeAndAge.xyz * vOffset, mRotation);

        right = mul(right, mRotation);
        up = mul(up, mRotation);

        vParticleNormal = normalize(cross(right, up));
        vParticleTangent = right;
        vParticleBinormal = -up;
    }

    // transform instanced particles; billboards were handled above
    if (b_useModelInstancing && b_iInstanceType != PARTICLE_BILLBOARD)
        vertIn.vPosition.xyz = mul(
            float4(vertIn.vPosition.xyz, 1),
            vsParticleInstanceMatrix(
                (int)vsGlobals.fParticleBatchIndexRemappingTable[vertIn.iBatchIndex.x])).xyz;

    if (b_enableVertexWarps == 1)
        vertIn.vPosition.xyz = vsApplyWarps(float4(vertIn.vPosition.xyz, 1.0)).xyz;

    vertOut.HPos = vsEmitParticleHPos(vertIn);

    // NOTE: particle.fx writes WorldPos only on the Metal path, so the D3D vertex
    // shader never emits it.
#if SC2_HAS_HPosAsUV
    vertOut.m_HPosAsUV = mul(float4(vertIn.vPosition.xyz, 1.0), vsGlobals.mVPTransform);
#endif
#if SC2_HAS_ViewPos
    vertOut.m_ViewPos = vsEmitParticleViewPos(vertIn);
#endif
#if SC2_HAS_VertexColor
    vertOut.m_VertexColor = vsEmitParticleVertexColor(vertIn, vSizeAndAge, iBatchIndex);
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
    vertOut.m_EyeToVertex = vsEmitParticleEyeToVertex(vertIn);
#endif
#if SC2_HAS_EyeToVertexFresnel
    vertOut.m_EyeToVertexFresnel = vsEmitParticleEyeToVertex(vertIn);
#endif
#if SC2_HAS_FOWUV
    vertOut.m_FOWUV = vsEmitParticleFOWUV(vertIn);
#endif

#if b_usePackedUVEmitter
  #if (b_iUVEmitterCount / 2 > 0)
    SC2_PAR_UV_PAIR(0, 0, 1);
  #endif
  #if (b_iUVEmitterCount / 2 > 1)
    SC2_PAR_UV_PAIR(1, 2, 3);
  #endif
  #if (b_iUVEmitterCount / 2 > 2)
    SC2_PAR_UV_PAIR(2, 4, 5);
  #endif
  #if (b_iUVEmitterCount / 2 > 3)
    SC2_PAR_UV_PAIR(3, 6, 7);
  #endif
  #if (b_iUVEmitterCount == 1)
    SC2_PAR_UV(0, 0);
  #elif (b_iUVEmitterCount == 3)
    SC2_PAR_UV(1, 2);
  #elif (b_iUVEmitterCount == 5)
    SC2_PAR_UV(2, 4);
  #elif (b_iUVEmitterCount == 7)
    SC2_PAR_UV(3, 6);
  #endif
#else
  #if (b_iUVEmitterCount > 0)
    SC2_PAR_UV(0, 0);
  #endif
  #if (b_iUVEmitterCount > 1)
    SC2_PAR_UV(1, 1);
  #endif
  #if (b_iUVEmitterCount > 2)
    SC2_PAR_UV(2, 2);
  #endif
  #if (b_iUVEmitterCount > 3)
    SC2_PAR_UV(3, 3);
  #endif
  #if (b_iUVEmitterCount > 4)
    SC2_PAR_UV(4, 4);
  #endif
  #if (b_iUVEmitterCount > 5)
    SC2_PAR_UV(5, 5);
  #endif
  #if (b_iUVEmitterCount > 6)
    SC2_PAR_UV(6, 6);
  #endif
  #if (b_iUVEmitterCount > 7)
    SC2_PAR_UV(7, 7);
  #endif
#endif

    // The particle root emits a CONSTANT parallax vector, not EmitParallaxVector.
#if SC2_HAS_ParallaxVector
    vertOut.m_ParallaxVector = float3(0, 0, 0);
#endif

    // The basis is recomputed in a second pass for the oriented types (the
    // original does this to keep the SM2.0 temp-register count down).
    if (b_iInstanceType == PARTICLE_FACE_TRAVEL_DIR || b_iInstanceType == PARTICLE_FACE_WORLD_DIR
        || b_iInstanceType == PARTICLE_EMITTER_ORIENTED
        || b_iInstanceType == PARTICLE_PHYSICS_ORIENTED)
    {
        float3 vRight, vUp, vDirectionVector;
        vsUnpackNormals(vertIn, vRight, vUp, vDirectionVector, iBatchIndex);

        vParticleNormal = vDirectionVector;
        vParticleTangent = vRight;
        vParticleBinormal = vUp;

        // flip if the particle is oriented away from the camera.  NOTE this arm
        // uses `* sign(d)` where the terrain arms use a `> 0 ? :` select — they
        // differ at exactly d == 0 (sign gives 0, the select gives +1).
        float d = -dot(vertIn.vPosition.xyz, vParticleNormal);
        float cameraDistFromParticlePlane = dot(float4(vsGlobals.vEyePos.xyz, 1),
                                                float4(vParticleNormal, d));
        vParticleNormal = vParticleNormal * sign(cameraDistFromParticlePlane);
        vParticleTangent = vParticleTangent * sign(cameraDistFromParticlePlane);
        vParticleBinormal = vParticleBinormal * sign(cameraDistFromParticlePlane);
    }

#if SC2_HAS_Normal
    vertOut.m_Normal = float4(vParticleNormal, 1);   // always right-handed
#endif
#if SC2_HAS_Tangent
    vertOut.m_Tangent = float4(vParticleTangent, 1);
#endif
#if SC2_HAS_Binormal
    vertOut.m_Binormal = vParticleBinormal;
#endif

#if (SC2_HAS_Diffuse || SC2_HAS_Specular || SC2_HAS_ShadowDiffuse || SC2_HAS_ShadowSpecular)
    float3 cVertexSpecularLighting = float3(0, 0, 0);
    float3 cVertexShadowSpecularLighting = float3(0, 0, 0);
    float3 cVertexShadowSpecularLighting2 = float3(0, 0, 0);
  #if SC2_HAS_Diffuse
    vertOut.m_Diffuse = vsEmitDiffuse(vertOut, vertIn.vPosition.xyz, vParticleNormal,
                                      float4(0, 0, 0, 0), cVertexSpecularLighting,
                                      cVertexShadowSpecularLighting,
                                      cVertexShadowSpecularLighting2);
  #endif
  #if SC2_HAS_ShadowDiffuse
    vertOut.m_ShadowDiffuse = vsEmitShadowDiffuse(vParticleNormal, float4(0, 0, 0, 0),
                                                  cVertexShadowSpecularLighting,
                                                  cVertexShadowSpecularLighting2);
  #endif
  #if SC2_HAS_Specular
    vertOut.m_Specular = vsEmitSpecular(cVertexSpecularLighting);
  #endif
  #if SC2_HAS_ShadowSpecular
    vertOut.m_ShadowSpecular = vsEmitShadowSpecular(cVertexShadowSpecularLighting);
  #endif
#endif

    return vertOut;
}
