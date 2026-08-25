// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026, Fernando Sahmkow
// See LICENSE in the repository root for full terms.
//
// Model-family normal decode (psmaterial.fx : DecodeNormal / DecodeNormalInternal),
// the value MainShading feeds to the material fresnel term and to lighting.
//
// Covers the vertex-based arm (normalize(INTERPOLANT_Normal)) and the pixel-based
// normal-mapping arm (b_useNormalMapping + Normal layer: DecodeTextureNormal →
// TangentToWorld with the Tangent/Binormal basis), plus the b_iTwoSided backface
// flip.  The tri-planar normal sampling (DecodeTriTextureNormal) lands with the
// tri-planar increment; the explicit-UV normal map is here.

implementing sc2_shaders;
#include "types/perm_cb.slangh"

#ifndef b_iTwoSided
#define b_iTwoSided 0
#endif
#ifndef b_useNormalMapping
#define b_useNormalMapping 0
#endif
#ifndef b_DXNStyleNormalMaps
#define b_DXNStyleNormalMaps 0
#endif
#ifndef b_useIdentityBasis
#define b_useIdentityBasis 0
#endif
#ifndef b_iNormalUVMapping
#define b_iNormalUVMapping 0
#endif
#ifndef b_iNormalBlending
#define b_iNormalBlending 0
#endif
#ifndef b_iNormalBlendingEnableMask2
#define b_iNormalBlendingEnableMask2 0
#endif
#ifndef b_iNormalBlendNormalLayerEnable
#define b_iNormalBlendNormalLayerEnable 0
#endif

// materialdefines.fx triplanar selectors (redeclared: slang __include doesn't
// propagate #defines across files).
#define IsTriPlanarMappingFX(m) ((m) >= 16 && (m) <= 18)
#define TriPlanarIsLocalFX(m)   ((m) == 16)

// psmateriallayer.fx : DecodeNormalSample — unpack a tangent-space normal from the
// sampled texel (DXN two-channel form or the plain xyz*2-1 form).
float3 decodeNormalSample(float4 cTextureSample, out float fExtraValue)
{
    [branch] if (SC2_AXIS(b_DXNStyleNormalMaps))
    {
        float3 vPixelNormal;
        vPixelNormal.xy = 2.0f * cTextureSample.wy - 1.0f;
        vPixelNormal.z = sqrt(max(0.0f, 1.0f - dot(vPixelNormal.xy, vPixelNormal.xy)));
        fExtraValue = 1.0f;             // w was consumed by the normal
        return vPixelNormal;
    }
    fExtraValue = cTextureSample.w;
    return 2.0f * cTextureSample.xyz - 1.0f;
}

// psmateriallayer.fx : DecodeTextureNormal.
float3 decodeTextureNormal(Texture2D<float4> tNormalMap, SamplerState sNormalMap,
                           float4 vNormalMapUV, out float fExtraValue)
{
    float4 cSample = tNormalMap.Sample(sNormalMap, vNormalMapUV.xy);
    return decodeNormalSample(cSample, fExtraValue);
}

// psmateriallayer.fx : DecodeTriTextureNormal — triplanar normal sample (computes the
// atlas UVs inline; the shared mTriPlanarUVs table is not used for normals).
float3 decodeTriTextureNormal(Texture2D<float4> tNormalMap, SamplerState sNormalMap,
                              float4 vUV, float4 vAtlasParams, float4 vWeights,
                              float3 vNormal, out float fExtraValue)
{
    float2 uvZ, uvY, uvX;
    triplanarAtlas(vAtlasParams, vNormal, vUV.xyz, uvZ, uvY, uvX);
    float4 vBlended = sampleTriPlanarTexture(tNormalMap, sNormalMap, uvZ, uvY, uvX, vWeights.xyz);
    return decodeNormalSample(vBlended, fExtraValue);
}

// psmateriallayer.fx : DecodeTextureNormalGrad — explicit-gradient normal sample
// (SampleGrad truncates the float4 gradients to .xy for the 2D texture).
float3 decodeTextureNormalGrad(Texture2D<float4> tNormalMap, SamplerState sNormalMap,
                               float4 vNormalMapUV, out float fExtraValue,
                               float4 ddxv, float4 ddyv)
{
    float4 cSample = tNormalMap.SampleGrad(sNormalMap, vNormalMapUV.xy, ddxv.xy, ddyv.xy);
    return decodeNormalSample(cSample, fExtraValue);
}

// pslighting.fx : TangentToWorld — rotate the tangent-space normal into world space
// by the [tangent, binormal, normal] basis (or the identity-basis fast path).
float3 tangentToWorld(float3 vTextureNormal, float3 vNormal, float3 vTangent, float3 vBinormal)
{
    [branch] if (SC2_AXIS(b_useIdentityBasis))
        return normalize(vTextureNormal * float3(1.0f, -1.0f, 1.0f));

    float3x3 mTangentToWorld;
    mTangentToWorld[0] = vTangent;
    mTangentToWorld[1] = vBinormal;
    mTangentToWorld[2] = vNormal;
    return normalize(mul(vTextureNormal, mTangentToWorld));
}

// psmaterial.fx : DecodeNormalInternal — the general (layer-parameterised) normal
// decode.  decodeNormal(vertOut) below is the Normal-layer specialisation; this
// variant takes an arbitrary layer config + texture (cloaking's CloakingNormal layer).
// NOTE (.fx quirk): the triplanar atlas params come from the NORMAL layer
// (p_vNormalAtlasParams), not the passed layer — the .fx's SETUP_LAYER(Normal) inside
// DecodeNormalInternal makes p_vNormalAtlasParams the atlas source regardless.
float3 decodeNormalInternal(ModelPSInput vertOut, SLayerConfig cfg,
                            Texture2D<float4> tLayer, SamplerState sLayer)
{
    float3 vNormalWS;
    float fExtra = 0.0f;

#if b_useNormalMapping
    if (cfg.b_iLayerEnable)
    {
        float4 vUV = getUVEmitter(vertOut, cfg.b_iUVEmitter,
                                  cfg.p_vMultiplyAddAlphaTrans.w, cfg.p_mUVTransform);
        float3 vNormalTS;
        if (!IsTriPlanarMappingFX(cfg.b_iUVMapping))
            vNormalTS = decodeTextureNormal(tLayer, sLayer, vUV, fExtra);
        else if (TriPlanarIsLocalFX(cfg.b_iUVMapping))
            vNormalTS = decodeTriTextureNormal(tLayer, sLayer, vUV, psGlobals.vNormalAtlasParams,
                            vertOut.TriPlanarWeights(), vertOut.VectorUI0().xyz, fExtra);
        else
        {
            float4 weights = float4(vertOut.FogColor().x, vertOut.FogColor().y, vUV.w, 0.0f);
            vNormalTS = decodeTriTextureNormal(tLayer, sLayer, vUV, psGlobals.vNormalAtlasParams,
                            weights, vertOut.Normal().xyz, fExtra);
        }
        vNormalWS = tangentToWorld(vNormalTS, vertOut.Normal().xyz,
                                   vertOut.Tangent().xyz, vertOut.Binormal());
    }
    else
        vNormalWS = normalize(vertOut.Normal().xyz);
#else
    vNormalWS = normalize(vertOut.Normal().xyz);
#endif

    if (SC2_AXIS(b_iTwoSided) && vertOut.FrontFace())
        vNormalWS = -vNormalWS;
    return vNormalWS;
}

// psmaterial.fx : DecodeNormal(vertOut) -> DecodeNormalInternal.
float3 decodeNormal(ModelPSInput vertOut)
{
    float3 vNormalWS;

#if b_useNormalMapping && b_iNormalLayerEnable
    // Pixel-based: sample the Normal layer's tangent-space normal (explicit-UV arm;
    // the tri-planar arm lands with the tri-planar increment) and rotate to world.
    {
        SLayerConfig cfg = setupNormalLayer();
        float4 vUV = getUVEmitter(vertOut, cfg.b_iUVEmitter,
                                  cfg.p_vMultiplyAddAlphaTrans.w, cfg.p_mUVTransform);
        float fExtra;
        float3 vNormalTS;
    #if IsTriPlanarMappingFX(b_iNormalUVMapping)
      #if TriPlanarIsLocalFX(b_iNormalUVMapping)
        // Local: blend weights + local normal from the VectorUI0 / TriPlanarWeights
        // interpolants.
        vNormalTS = decodeTriTextureNormal(p_sNormalSampler_t, p_sSamp, vUV,
                        cfg.p_vAtlasParams, vertOut.TriPlanarWeights(),
                        vertOut.VectorUI0().xyz, fExtra);
      #else
        // World: weights from FogColor.xy + vUV.w, world normal from INTERPOLANT_Normal.
        float4 weights = float4(vertOut.FogColor().x, vertOut.FogColor().y, vUV.w, 0.0f);
        vNormalTS = decodeTriTextureNormal(p_sNormalSampler_t, p_sSamp, vUV,
                        cfg.p_vAtlasParams, weights, vertOut.Normal().xyz, fExtra);
      #endif
    #else
        vNormalTS = decodeTextureNormal(p_sNormalSampler_t, p_sSamp, vUV, fExtra);
    #endif
        vNormalWS = tangentToWorld(vNormalTS, vertOut.Normal().xyz,
                                   vertOut.Tangent().xyz, vertOut.Binormal());
    }
#else
    // Vertex-based (psmaterial.fx:465).
    vNormalWS = normalize(vertOut.Normal().xyz);
#endif

    // PS_40+ backface flip (psmaterial.fx:479-487): SV_IsFrontFace nonzero flips the
    // normal so two-sided surfaces light correctly from the back.
    if (SC2_AXIS(b_iTwoSided) && vertOut.FrontFace())
        vNormalWS = -vNormalWS;

    return vNormalWS;
}

// psmaterial.fx : DecodeNormalGradInternal — normal decode with explicit gradients
// (the normal-blend path samples the blend-normal maps with the mask UV gradients).
// Uses the NormalBlendNormal layer config for the UV, the passed texture + gradients.
float3 decodeNormalGradInternal(ModelPSInput vertOut, SLayerConfig cfg,
                                Texture2D<float4> tLayer, SamplerState sLayer,
                                float4 ddxv, float4 ddyv)
{
    float3 vNormalWS;
#if b_useNormalMapping
    if (cfg.b_iLayerEnable)
    {
        float4 vUV = getUVEmitter(vertOut, cfg.b_iUVEmitter,
                                  cfg.p_vMultiplyAddAlphaTrans.w, cfg.p_mUVTransform);
        float fExtra;
        float3 vNormalTS = decodeTextureNormalGrad(tLayer, sLayer, vUV, fExtra, ddxv, ddyv);
        vNormalWS = tangentToWorld(vNormalTS, vertOut.Normal().xyz,
                                   vertOut.Tangent().xyz, vertOut.Binormal());
    }
    else
        vNormalWS = normalize(vertOut.Normal().xyz);
#else
    vNormalWS = normalize(vertOut.Normal().xyz);
#endif

    if (SC2_AXIS(b_iTwoSided) && vertOut.FrontFace())
        vNormalWS = -vNormalWS;
    return vNormalWS;
}

// psmaterial.fx : NormalBlending — blend the base normal toward a second normal map
// selected by a mask-driven factor (two mask layers, two blend-normal maps).
float3 normalBlending(ModelPSInput vertOut, float3 iBaseNormal, TriPlanarUVs tp)
{
    SLayerConfig cfgMask  = setupNormalBlendMaskLayer();
    SLayerConfig cfgMask2 = setupNormalBlendMask2Layer();
    SLayerConfig cfgNorm  = setupNormalBlendNormalLayer();

    float4 mask = getLayerColor2D(vertOut, cfgMask, p_sNormalBlendMaskSampler_t,
                                  p_sSamp, 1.0f, iBaseNormal, tp);
    float blendFactor = dot(mask, psGlobals.vNormalBlendingFactors);

#if b_iNormalBlendingEnableMask2
    {
        float4 mask2 = getLayerColor2D(vertOut, cfgMask2, p_sNormalBlendMask2Sampler_t,
                                       p_sSamp, 1.0f, iBaseNormal, tp);
        blendFactor += dot(mask2, psGlobals.vNormalBlendingFactors2);
    }
#endif
    blendFactor = clamp(blendFactor, -1.0f, 1.0f);
    blendFactor *= abs(blendFactor);

    float4 norm1UV = getUVEmitter(vertOut, cfgMask.b_iUVEmitter,
                                  cfgMask.p_vMultiplyAddAlphaTrans.w, cfgMask.p_mUVTransform);
    float4 norm2UV = getUVEmitter(vertOut, cfgMask2.b_iUVEmitter,
                                  cfgMask2.p_vMultiplyAddAlphaTrans.w, cfgMask2.p_mUVTransform);

    // Gradients computed outside the branch (psmaterial.fx:626-629).
    float4 ddx1 = ddx(norm1UV), ddy1 = ddy(norm1UV);
    float4 ddx2 = ddx(norm2UV), ddy2 = ddy(norm2UV);

    float3 blendNormal;
    if (blendFactor < 0.0f)
        blendNormal = decodeNormalGradInternal(vertOut, cfgNorm, p_sNormalBlendNormalSampler_t,
                                               p_sSamp, ddx1, ddy1);
    else
        blendNormal = decodeNormalGradInternal(vertOut, cfgNorm, p_sNormalBlendNormal2Sampler_t,
                                               p_sSamp, ddx2, ddy2);

    return normalize(lerp(iBaseNormal, blendNormal, abs(blendFactor)));
}
