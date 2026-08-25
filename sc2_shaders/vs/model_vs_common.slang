// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026, Fernando Sahmkow
// See LICENSE in the repository root for full terms.
//
// Model-family vertex-shader support bodies — the `VS*.fx` include closure that
// model.fx's ModelVertexMain pulls in:
//
//   vsmodelvertexformat.fx : TranslateVert
//   vsskinning.fx          : Skin
//   vsuvmapping.fx         : GenUV (+ the triplanar / splat-projector helpers)
//   vsvertexwarp.fx        : ApplyWarp / ApplyWarps (+ vsutils.fx MakeRotation)
//   vsutils.fx             : ComputeGridTangent
//   vsfog.fx / common.fx   : FogDensity, Ndc2Tex, ComputeTriPlanarBlendingWeights,
//                            SplatAttenuationProjector, ComputeHalfVector,
//                            EmitParallaxVector
//
// Every free function is `vs`-prefixed: the module is ONE translation unit shared
// with the pixel root, which already defines `decodeNormal` / `computeHalfVector` /
// `ndc2Tex` / `lighting` with different (pixel-stage) specialisations.

implementing sc2_shaders;

// Compile-time axes + .fx enums (see the header: `__include` merges declarations
// but not #defines, so each VS translation unit includes this itself).
#include "vs/model_vs_defines.slangh"
#include "types/perm_cb.slangh"

// Axes read as CODE need a file-local default: only NONZERO axes are passed
// as -D, so a zero one would otherwise be an undefined identifier.
#ifndef b_iBlendWeightCount
#define b_iBlendWeightCount 0
#endif
#ifndef b_iUsePOMNew
#define b_iUsePOMNew 0
#endif
#ifndef b_splatAttenuationEnabled
#define b_splatAttenuationEnabled 0
#endif
#ifndef b_splatTextureProjectorEnabled
#define b_splatTextureProjectorEnabled 0
#endif
#ifndef b_useIdentityBasis
#define b_useIdentityBasis 0
#endif
#ifndef b_useNormalMapping
#define b_useNormalMapping 0
#endif
#ifndef b_useParallaxMapping
#define b_useParallaxMapping 0
#endif

// ---------------------------------------------------------------------------
// common.fx helpers
// ---------------------------------------------------------------------------

// common.fx : Ndc2Tex — normalised device coords -> texture space.
float2 vsNdc2Tex(float2 vNDC)
{
    return vNDC * float2(0.5f, -0.5f) + 0.5f;
}

// common.fx : FogDensity — the analytic height-fog integral along the eye ray.
float vsFogDensity(float3 vPosWorld)
{
    float3 vViewVect = vPosWorld - vsGlobals.vEyePos;
    float fDist = length(vViewVect);
    float fFogIntegration = vsGlobals.vFogDensity_Falloff_MinHeight.x * fDist
        * exp((vsGlobals.vFogDensity_Falloff_MinHeight.z - vsGlobals.vEyePos.z)
              * vsGlobals.vFogDensity_Falloff_MinHeight.y);

    const float c_fSlopeEps = 0.01f;
    if (abs(vViewVect.z) > c_fSlopeEps)
    {
        float fT = vsGlobals.vFogDensity_Falloff_MinHeight.y * vViewVect.z;
        if (fT == 0.0f)
        {
            fFogIntegration = 0.0f;
        }
        else
        {
            // Keep exp(-fT) inside FLOAT_MAX (common.fx:53).
            fT = clamp(fT, -88.0f, 88.0f);
            fFogIntegration *= (1.0f - exp(-fT)) / fT;
        }
    }

    float fOutDensity = fFogIntegration;
    if (vsGlobals.vFogDistance.z > 0.0f)
    {
        if (vsGlobals.vFogDistance.y != vsGlobals.vFogDistance.x)
        {
            fOutDensity = fFogIntegration
                * saturate(vsGlobals.vFogDistance.z
                           * (fDist - vsGlobals.vFogDistance.x)
                           * vsGlobals.vFogDistance.w);
        }
    }
    return saturate(fOutDensity);
}

// common.fx : ComputeTriPlanarBlendingWeights.
float3 vsComputeTriPlanarBlendingWeights(float3 vNormal)
{
    float3 w = abs(vNormal);
    w -= float3(0.2f, 0.2f, 0.2f);
    w *= float3(7.0f, 7.0f, 7.0f);
    w = saturate(w);
    w = pow(w, 3);
    w /= dot(w, float3(1, 1, 1));
    return w;
}

// common.fx : SplatAttenuationProjector — 1 unless the splat attenuation axis is on.
float vsSplatAttenuationProjector(float3 worldPos, float4 vPlane, float halfHeight,
                                  float minHeight)
{
    [branch] if (SC2_AXIS(b_splatAttenuationEnabled))
    {
        float d = abs(dot(float4(worldPos, 1.0f), vPlane));
        return saturate((halfHeight - d) / (halfHeight - minHeight));
    }
    return 1;
}

// common.fx : ComputeHalfVector.  (The `if (false)` vertex-lighting variant in the
// original is dead source — b_useVertexLighting is commented out at common.fx:95.)
float3 vsComputeHalfVector(float3 vWorldPos, float3 vWorldLightDirection)
{
    float3 vEyeVec = normalize(vsGlobals.vEyePos - vWorldPos);
    return normalize(vEyeVec - vWorldLightDirection);
}

// vslighting.fx : ComputeHalfVector( vPosWS ) — the key light's half vector.
float3 vsComputeHalfVector(float3 vPosWS)
{
    return vsComputeHalfVector(vPosWS, vsGlobals.directionalLights[0].vDirection.xyz);
}

// common.fx : EmitParallaxVector.  Compiled out (returns 0) unless the parallax
// axis is set — the `#if (b_useParallaxMapping == 1 || CPP_SHADER)` guard.
float3 vsEmitParallaxVector(float3 vVertWS, float3 vEyeWS, float3 vNormal,
                            float3 vTangent, float3 vBinormal)
{
    [branch] if (SC2_AXIS(b_useParallaxMapping) != 1)
        return float3(0, 0, 0);

    float3 vPos = vEyeWS - vVertWS;
    float3 vEyeTS;
    [branch] if (SC2_AXIS(b_useIdentityBasis))
    {
        vEyeTS = vPos * float3(1.0f, -1.0f, 1.0f);
    }
    else
    {
        float3x3 mWorldToTangent;
        mWorldToTangent[0] = normalize(vTangent);
        mWorldToTangent[1] = normalize(vBinormal);
        mWorldToTangent[2] = vNormal;
        vEyeTS = normalize(mul(mWorldToTangent, vPos));
    }

    [branch] if (SC2_AXIS(b_iUsePOMNew))
        return vEyeTS.xyz;

    float2 vParallaxDirection = normalize(vEyeTS.xy);
    float fLength = length(vEyeTS);
    // Parallax length grows with the divergent viewing angle.
    float fParallaxLength = sqrt(fLength * fLength - vEyeTS.z * vEyeTS.z) / vEyeTS.z;
    return float3(vParallaxDirection * fParallaxLength
        * vsGlobals.vSpecularMultiplier_DepthBlendThreshold_HeightMapScale_AlphaFactor.z,
        0);
}

// ---------------------------------------------------------------------------
// vsshadow.fx / vsfog.fx — shared by every Default-family vertex root.
// ---------------------------------------------------------------------------
// Shadow map UV (vsshadow.fx : EmitShadowMapUV).
float4 vsEmitShadowMapUV(float3 position)
{
    return mul(float4(position, 1.0), vsGlobals.mShadowTransform);
}

// Fog colour (vsfog.fx : EmitFogColor) — .rgb carries the world triplanar weights,
// .a the fog density.
float4 vsEmitFogColor(float3 vPosition, float3 triPlanarWeightsWorld)
{
    return float4(triPlanarWeightsWorld, vsFogDensity(vPosition));
}

// ---------------------------------------------------------------------------
// vsutils.fx
// ---------------------------------------------------------------------------

// vsutils.fx : MakeRotation — axis/angle rotation matrix, written element-wise
// exactly as the original does (it minimises temp registers, and the element order
// is what fxc's codegen follows).
float3x3 vsMakeRotation(float fAngle, float3 vAxis)
{
    float fS, fC;
    sincos(fAngle, fS, fC);
    float fOneC = 1.0f - fC;
    float3x3 result;

    result[0][0] = fOneC * vAxis.x * vAxis.x + fC;
    result[1][1] = fOneC * vAxis.y * vAxis.y + fC;
    result[2][2] = fOneC * vAxis.z * vAxis.z + fC;

    float fXY = vAxis.x * vAxis.y;
    float fZS = vAxis.z * fS;
    result[1][0] = fOneC * fXY + fZS;
    result[0][1] = fOneC * fXY - fZS;

    float fZX = vAxis.z * vAxis.x;
    float fYS = vAxis.y * fS;
    result[2][0] = fOneC * fZX - fYS;
    result[0][2] = fOneC * fZX + fYS;

    float fYZ = vAxis.y * vAxis.z;
    float fXS = vAxis.x * fS;
    result[2][1] = fOneC * fYZ + fXS;
    result[1][2] = fOneC * fYZ - fXS;
    return result;
}

// vsutils.fx : ComputeGridTangent — terrain's right-handed grid tangent.
float3 vsComputeGridTangent(float3 normal)
{
    const float3 c_forward = float3(0, 1, 0);
    float3 tangent = cross(c_forward, normal);
    float m = dot(tangent, tangent);
    // if m == 0 then tangent = float3(1,0,0)
    tangent += (1 - sign(m)) * float3(1, 0, 0);
    tangent = normalize(tangent);
    return tangent;
}

// ---------------------------------------------------------------------------
// vsmodelvertexformat.fx : TranslateVert
// ---------------------------------------------------------------------------
// The tangent basis arrives in compressed ubyte4n form (2x-1), the binormal is
// rebuilt from it, and the UV sets go through the p_uvRangeOffset scale/bias.
ModelVSVertex vsTranslateVert(ModelVSStreamIn src)
{
    ModelVSVertex dst;
    dst.vPosition = src.vPosition;
#if !b_iLayoutHasNoVertexBlendWeights
    dst.vBlendWeights = src.vBlendWeights;
#endif
#if !b_iLayoutHasNoVertexBlendIndices
    dst.vBlendIndices = src.vBlendIndices;
#endif
#if (b_iUVCoordCount >= 1)
    dst.vUV0 = float2(src.vUV0) * vsGlobals.uvRangeOffset.x + vsGlobals.uvRangeOffset.y;
#endif
#if (b_iUVCoordCount >= 2)
    dst.vUV1 = float2(src.vUV1) * vsGlobals.uvRangeOffset.x + vsGlobals.uvRangeOffset.y;
#endif
#if (b_iUVCoordCount >= 3)
    dst.vUV2 = float2(src.vUV2) * vsGlobals.uvRangeOffset.x + vsGlobals.uvRangeOffset.y;
#endif
#if (b_iUVCoordCount >= 4)
    dst.vUV3 = float2(src.vUV3) * vsGlobals.uvRangeOffset.x + vsGlobals.uvRangeOffset.y;
#endif
    dst.vNormal  = 2.0f * src.vNormal - 1.0f;
    dst.vTangent = 2.0f * src.vTangent - 1.0f;
    dst.vBinormal = normalize(cross(dst.vTangent.xyz, dst.vNormal.xyz))
                  * sign(dst.vNormal.w);
#if b_iLayoutHasColor
    dst.cColor = src.cColor;
#endif
#if b_iLayoutHasCustomUB4N1
    dst.cCustomUB4N1 = src.cCustomUB4N1;
#endif
    return dst;
}

// ---------------------------------------------------------------------------
// vsskinning.fx : Skin
// ---------------------------------------------------------------------------

// `(float3x3)m` on a 3x4 keeps each row's first three columns.
float3x3 vsMat3(float3x4 m)
{
    return float3x3(m[0].xyz, m[1].xyz, m[2].xyz);
}

// p_mBlendMatrices[bone] — rebuilt from its three stored rows (see the
// VS_MATROWS note in types/model_vs_io.slang).  The row constructor makes the
// matrix row-major in the *value* domain, so mul(m, v) is `dot(row_i, v)` exactly
// as the reference's `dp4 cb0[bone*3 + i], v` does.
float3x4 vsBlendMatrix(int bone)
{
    int b = bone * 3;
    return float3x4(vsGlobals.mBlendMatrices[b + 0],
                    vsGlobals.mBlendMatrices[b + 1],
                    vsGlobals.mBlendMatrices[b + 2]);
}

// p_mSplatTextureProjMatrix[i] — likewise, four stored rows.
float4x4 vsSplatProjMatrix(int i)
{
    int b = i * 4;
    return float4x4(vsGlobals.mSplatTextureProjMatrix[b + 0],
                    vsGlobals.mSplatTextureProjMatrix[b + 1],
                    vsGlobals.mSplatTextureProjMatrix[b + 2],
                    vsGlobals.mSplatTextureProjMatrix[b + 3]);
}

// HLSL's D3DCOLORtoUBYTE4 intrinsic: swizzle to zyxw, scale by 255.001953, truncate.
int4 vsD3DCOLORtoUBYTE4(float4 v)
{
    return int4(v.zyxw * 255.001953f);
}

// Bone-blend the position/normal and — under normal mapping — the tangent basis.
// The count was a compile-time literal, so exactly one arm survived.  It is a
// payload field now: the BLENDWEIGHT attribute is always a float4 and the arms
// already read only as much of it as the count says, so nothing but the folding
// changes (design section 3).
void vsSkin(inout float4 vPosition, inout float3 vNormal, inout float3 vTangent,
            inout float3 vBinormal, float4 vBlendWeights, float4 vBlendIndices)
{
    int4 iBoneIndices = vsD3DCOLORtoUBYTE4(vBlendIndices);
    float4 vSwizzledBlendWeights = vBlendWeights.zyxw;
    float3x4 mBlendMatrix = float3x4(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    const int iBlendWeightCount = SC2_AXIS(b_iBlendWeightCount);

    // Skin position and normal.
    if (iBlendWeightCount > 0)
    {
        if (iBlendWeightCount == 1)
        {
            vPosition.xyz = mul(vsBlendMatrix(iBoneIndices[0]), vPosition);
            vNormal = mul(vsMat3(vsBlendMatrix(iBoneIndices[0])), vNormal);
        }
        else
        {
            mBlendMatrix = vsBlendMatrix(iBoneIndices[0]) * vSwizzledBlendWeights[0];
            if (iBlendWeightCount >= 2)
            {
                mBlendMatrix += vsBlendMatrix(iBoneIndices[1]) * vSwizzledBlendWeights[1];
                mBlendMatrix += vsBlendMatrix(iBoneIndices[2]) * vSwizzledBlendWeights[2];
                mBlendMatrix += vsBlendMatrix(iBoneIndices[3]) * vSwizzledBlendWeights[3];
            }
            vPosition.xyz = mul(mBlendMatrix, vPosition);
            vNormal = mul(vsMat3(mBlendMatrix), vNormal);
        }
        vNormal = normalize(vNormal);
    }

    // Skin tangents and binormals (the original leaves them un-normalised).
    if (SC2_AXIS(b_useNormalMapping) && iBlendWeightCount > 0)
    {
        if (iBlendWeightCount == 1)
        {
            vTangent  = mul(vsBlendMatrix(iBoneIndices[0]), float4(vTangent, 0.0f));
            vBinormal = mul(vsBlendMatrix(iBoneIndices[0]), float4(vBinormal, 0.0f));
        }
        else
        {
            vTangent  = mul(mBlendMatrix, float4(vTangent, 0.0f));
            vBinormal = mul(mBlendMatrix, float4(vBinormal, 0.0f));
        }
    }
}

// ---------------------------------------------------------------------------
// vsvertexwarp.fx : ApplyWarp / ApplyWarps
// ---------------------------------------------------------------------------
// Takes the warp INDEX, and reads the warp's FIELDS — never the VertexWarp
// as a whole.  slangc's GLSL emitter aborts with "internal error 99999: unexpected IR
// opcode during code emit" the moment a cbuffer struct holding a `row_major`
// matrix becomes a value: passing it to a function trips it, and so does copying
// it into a local (slangc 2026.1 / 2026.5.2 / 2026.8.1 — the D3D, SPIR-V, Metal
// and WGSL backends all emit either form fine).  Reading the three members separately
// is the shape that survives.  MAX_WARPS is 1 and the index is a compile-time
// constant, so the D3D bytecode is identical to the by-value version's.
float4 vsApplyWarp(float4 vVertexWS, int iWarp)
{
    row_major float3x4 mWarpTransform    = vsGlobals.warps[iWarp].vTransform;
    row_major float3x4 mWarpInvTransform = vsGlobals.warps[iWarp].vInvTransform;
    float2             vStrengthRadius   = vsGlobals.warps[iWarp].vStrengthRadius;

    // Put in warp space.
    float3 vVertex = mul(mWarpInvTransform, vVertexWS);
    float3 vModelPos = mul(mWarpInvTransform,
                           float4(vsGlobals.vVertexWarpModelPosition, 1.0f));
    vModelPos.z = 0;

    float fModelDistance = saturate(length(vModelPos) / vStrengthRadius.y);
    float fOriginalLength = length(vVertex.xy);

    // Gradient from 0 to 1.
    float fGradient = saturate(fOriginalLength / vStrengthRadius.y);
    float fSquaredGradient = 1.0f - fGradient * fGradient;

    float fScalar = vStrengthRadius.x;

    // Put R from -5 to 1.
    float fR = (1.0f - fGradient)
             * max(0.0f, pow(1.5f, fSquaredGradient * 6.0f - 5.0f)) * fScalar / 3.0f;

    // Apply swirl.
    float3x3 rotation = vsMakeRotation(fR * fR * 3.1616f * 2.0f, float3(0.0f, 0.0f, 1.0f));
    vVertex = mul(rotation, vVertex);

    // Squash to a disc.
    vVertex.z = vVertex.z * pow(fGradient, 6.0f);

    // Compress towards centre.
    vVertex = vVertex * fModelDistance;

    // Back in world space.
    vVertexWS.xyz = mul(mWarpTransform, float4(vVertex, 1.0f));
    return vVertexWS;
}

// `for (i = 0; i < b_iWarpCount; i++)` with MAX_WARPS == 1 and a compile-time
// count — unrolled here so no dynamic loop survives.
float4 vsApplyWarps(float4 vVertexWS)
{
#if (b_iWarpCount >= 1)
    vVertexWS = vsApplyWarp(vVertexWS, 0);
#endif
    return vVertexWS;
}

// ---------------------------------------------------------------------------
// vsuvmapping.fx
// ---------------------------------------------------------------------------
float4 vsGenSplatProjectiveTexture(float3 vWorldPos, int iIndex)
{
    return mul(float4(vWorldPos, 1.0f), vsSplatProjMatrix(iIndex));
}

float4 vsComputeTriPlanarUVs(float3 vPos, float3 vOffset, float3 vScale)
{
    float3 t = (vPos.xyz + vOffset.xyz) * vScale.xyz;
    return float4(t, 1);
}

float3 vsGetTriPlanarPos(int iMapping, float3 vLocalPos, float3 vWorldPos)
{
    if (iMapping == UVMAP_TRIPLANAR_LOCAL)
        return vLocalPos;
    else if (iMapping == UVMAP_TRIPLANAR_WORLD)
        return vWorldPos;
    else if (iMapping == UVMAP_TRIPLANAR_WORLD_LOCAL_Z)
        return float3(vWorldPos.x, vWorldPos.y, vLocalPos.z);
    return vLocalPos;
}

// vsuvmapping.fx : GenUV.  `iMapping` is `b_iUVMapping[iIndex]`, which the engine
// fills with per-instance literals through InitShader — so it is a compile-time
// constant here (a generic parameter), exactly as fxc sees it after folding.
// `iIndex` is likewise the literal emitter slot.
float4 vsGenUV<let iMapping : int, let iIndex : int>(
    float3 vLocalPos,
    float3 vWorldPos,
    float3 vLocalNormal,
    float3 vWorldNormal,
    float2 vUV0,
    float2 vUV1,
    float2 vUV2,
    float2 vUV3,
    int iSplatIndex,
    float3 vTriPlanarWeightsWorld)
{
    float4 vResult = float4(1, 1, 1, 1);

    if (iMapping == UVMAP_EXPLICITUV0)
        vResult = float4(vUV0, 0, 1);
    else if (iMapping == UVMAP_EXPLICITUV1)
        vResult = float4(vUV1, 0, 1);
    else if (iMapping == UVMAP_EXPLICITUV2)
        vResult = float4(vUV2, 0, 1);
    else if (iMapping == UVMAP_EXPLICITUV3)
        vResult = float4(vUV3, 0, 1);
    else if (iMapping == UVMAP_CUBICENVIO)
        vResult = float4(0, 0, 0, 0);
    else if (iMapping == UVMAP_SPHERICALENVIO)
        vResult = float4(0, 0, 0, 0);
    else if (iMapping == UVMAP_REFLECT_CUBICENVIO)
        vResult = float4(0, 0, 0, 0);
    else if (iMapping == UVMAP_REFLECT_SPHERICALENVIO)
        vResult = float4(0, 0, 0, 0);
    else if (iMapping == UVMAP_PLANARLOCALX)
    {
        vResult.xy = vLocalPos.yz * c_planarLocalUVMapping;
        vResult.y = 1 - vResult.y;
    }
    else if (iMapping == UVMAP_PLANARLOCALY)
    {
        vResult.xy = vLocalPos.xz * c_planarLocalUVMapping;
        vResult.y = 1 - vResult.y;
    }
    else if (iMapping == UVMAP_PLANARLOCALZ)
    {
        vResult.xy = vLocalPos.xy * c_planarLocalUVMapping;
        vResult.y = 1 - vResult.y;
    }
    else if (iMapping == UVMAP_PLANARWORLDX)
    {
        vResult.xy = vWorldPos.yz * vsGlobals.vTerrainTiling.xy + vsGlobals.vTerrainTiling.zw;
        vResult.y = 1 - vResult.y;
    }
    else if (iMapping == UVMAP_PLANARWORLDY)
    {
        vResult.xy = vWorldPos.xz * vsGlobals.vTerrainTiling.xy + vsGlobals.vTerrainTiling.zw;
        vResult.y = 1 - vResult.y;
    }
    else if (iMapping == UVMAP_PLANARWORLDZ)
    {
        vResult.xy = vWorldPos.xy * vsGlobals.vTerrainTiling.xy + vsGlobals.vTerrainTiling.zw;
        vResult.y = 1 - vResult.y;
    }
    else if (iMapping == UVMAP_SCREENSPACE)
    {
        vResult = mul(float4(vWorldPos.x, vWorldPos.y, vWorldPos.z, 1.0f),
                      vsGlobals.mVPTransform);
        vResult = vResult / vResult.w;
        vResult.xy = vResult.xy * float2(0.5f, -0.5f) + 0.5f;    // To [0..1]
    }
    else if (VS_IsTriPlanarMappingFX(iMapping))
    {
        float3 vNormal;
        if (VS_TriPlanarIsWorldFX(iMapping))
            vNormal = vWorldNormal.xyz;
        else
            vNormal = vLocalNormal.xyz;

        float3 vPos = vsGetTriPlanarPos(iMapping, vLocalPos, vWorldPos);
        vResult = vsComputeTriPlanarUVs(vPos, vsGlobals.vTriPlanarOffset[iIndex],
                                        vsGlobals.vTriPlanarScale[iIndex]);
        vResult.w = vTriPlanarWeightsWorld.z;
    }

    // Replace uv0/uv1 with the auto-generated splat projector UVs.
    bool isProjectorChannel = (iMapping == UVMAP_EXPLICITUV0) || (iMapping == UVMAP_EXPLICITUV1);
    if (SC2_AXIS(b_splatTextureProjectorEnabled) && isProjectorChannel)
    {
        vResult = vsGenSplatProjectiveTexture(vWorldPos, iSplatIndex);
        vResult.xy = vsNdc2Tex(vResult.xy / vResult.w);
    }

    return vResult;
}
