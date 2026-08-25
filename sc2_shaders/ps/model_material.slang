// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026, Fernando Sahmkow
// See LICENSE in the repository root for full terms.
//
// Model-family material layer system (psmateriallayer.fx / psmateriallayercolor.fx
// / psmaterialdefines.fx), transcribed faithfully.
//
// M1 increment 2 scope: the CONSTANT-COLOUR layer path (SampleLayer's early
// `if (b_iUseConstantColor) return p_cConstantColor`) + the full per-layer colour
// pipeline (ComputeLayerColorInternal + CombineLayerColor) + the diffuse layer
// setup.  Texture sampling (GetUVEmitter / atlas / triplanar / RTT) and the other
// layers (Decal / Specular / Emissive / Envio / Normal / …) are added in the next
// increments — each behind its own `b_i<Layer>*` axis so green perms stay green.
//
// Per-layer `b_i<Layer>*` axes default to 0 when the engine didn't set them (the
// driver only passes nonzero defines); the `#ifndef … #define … 0` block below
// supplies those defaults so the C-expression setup code always resolves.

implementing sc2_shaders;

// SC2_LAYER_<Prop>: a material-layer property reads from the permutation
// constant buffer once its switch is on, and from the `#define` until then.
#include "ps/model_layers_axes.slangh"
#include "types/perm_cb.slangh"

// ---- enums (psmaterialdefines.fx) ------------------------------------------
#define COLORSRC_CONSTANT   2
#define LAYEROP_MOD                     0
#define LAYEROP_MOD2X                   1
#define LAYEROP_ADD                     2
#define LAYEROP_LERP                    3
#define LAYEROP_TEAMCOLOR_EMISSIVE_ADD  4
#define LAYEROP_TEAMCOLOR_DIFFUSE_ADD   5
#define LAYEROP_ADD_NO_ALPHA            6
#define CHANNELSELECT_RGB   0
#define CHANNELSELECT_RGBA  1
#define CHANNELSELECT_A     2
#define CHANNELSELECT_R     3
#define CHANNELSELECT_G     4
#define CHANNELSELECT_B     5
#define TEAMCOLOR_NONE      0
#define TEAMCOLOR_DIFFUSE   1
#define TEAMCOLOR_EMISSIVE  2
#define FRESNELMODE_NONE        0
#define FRESNELMODE_STANDARD    1
#define FRESNELMODE_INVERTED    2
#define FRESNELTRANSFORM_NONE       0
#define FRESNELTRANSFORM_SIMPLE     1
#define FRESNELTRANSFORM_NORMALIZED 2
// psmaterialdefines.fx specular-type selector.
#define SPECULAR_RGB        0
#define SPECULAR_A_ONLY     1
// materialdefines.fx UV-mapping selectors (only what the transcribed path needs).
#define UVMAP_TRIPLANAR_LOCAL         16
#define UVMAP_TRIPLANAR_WORLD         17
#define UVMAP_TRIPLANAR_WORLD_LOCAL_Z 18
#define c_maxNumLayers                18
#define IsTriPlanarMappingFX(m) ((m) >= UVMAP_TRIPLANAR_LOCAL && (m) <= UVMAP_TRIPLANAR_WORLD_LOCAL_Z)
#define TriPlanarIsWorldFX(m)   ((m) >= UVMAP_TRIPLANAR_WORLD && (m) <= UVMAP_TRIPLANAR_WORLD_LOCAL_Z)
#define TriPlanarIsLocalFX(m)   ((m) == UVMAP_TRIPLANAR_LOCAL)

// The 18-slot triplanar UV table (psmaterial.fx : mTriPlanarUVs[c_maxNumLayers]),
// wrapped so it threads through the material path as one value.  Every index used to
// read/write it is a compile-time layer id (a b_* define), so the array scalarises and
// no dynamic-index / immediate-constant-buffer is emitted.
public struct TriPlanarUVs { float3x2 m[c_maxNumLayers]; };

// ---- axis defaults (0 = engine default) ------------------------------------
#ifndef b_useSimplifiedFresnel
#define b_useSimplifiedFresnel 0
#endif
#ifndef b_litOnly
#define b_litOnly 0
#endif
#ifndef b_useVertexColors
#define b_useVertexColors 0
#endif
#ifndef b_usePackedUVEmitter
#define b_usePackedUVEmitter 0
#endif
#ifndef b_iDiffuseLayerEnable
#define b_iDiffuseLayerEnable 0
#endif
#ifndef b_iDiffuseUseConstantColor
#define b_iDiffuseUseConstantColor 0
#endif
// SLayerConfig completeness axes (used by every setup*Layer, incl. Diffuse/Specular).
#ifndef b_iDiffuseLayerId
#define b_iDiffuseLayerId 0
#endif
#ifndef b_iDiffuseTextureType
#define b_iDiffuseTextureType 0
#endif
#ifndef b_iDiffuseTriPlanarUvId
#define b_iDiffuseTriPlanarUvId 0
#endif
#ifndef b_iSpecularLayerId
#define b_iSpecularLayerId 0
#endif
#ifndef b_iSpecularTextureType
#define b_iSpecularTextureType 0
#endif
#ifndef b_iSpecularTriPlanarUvId
#define b_iSpecularTriPlanarUvId 0
#endif
#ifndef b_iDiffuseUVEmitter
#define b_iDiffuseUVEmitter 0
#endif
#ifndef b_iDiffuseUVMapping
#define b_iDiffuseUVMapping 0
#endif
#ifndef b_iDiffuseIsRTTTexture
#define b_iDiffuseIsRTTTexture 0
#endif
#ifndef b_iDiffuseUseMask
#define b_iDiffuseUseMask 0
#endif
#ifndef b_iDiffuseChannelSelect
#define b_iDiffuseChannelSelect 0
#endif
#ifndef b_iDiffuseTeamColorMode
#define b_iDiffuseTeamColorMode 0
#endif
#ifndef b_iDiffuseInvert
#define b_iDiffuseInvert 0
#endif
#ifndef b_iDiffuseClamp
#define b_iDiffuseClamp 0
#endif
#ifndef b_iDiffuseFresnelMode
#define b_iDiffuseFresnelMode 0
#endif
#ifndef b_iDiffuseFresnelTransformMode
#define b_iDiffuseFresnelTransformMode 0
#endif
#ifndef b_iDiffuseFresnelSaturate
#define b_iDiffuseFresnelSaturate 0
#endif
#ifndef b_iDiffuseOp
#define b_iDiffuseOp 0
#endif
// Specular layer axes (psmaterial.fx SETUP_LAYER(Specular)).
#ifndef b_iSpecularUseConstantColor
#define b_iSpecularUseConstantColor 0
#endif
#ifndef b_iSpecularUseMask
#define b_iSpecularUseMask 0
#endif
#ifndef b_iSpecularChannelSelect
#define b_iSpecularChannelSelect 0
#endif
#ifndef b_iSpecularTeamColorMode
#define b_iSpecularTeamColorMode 0
#endif
#ifndef b_iSpecularInvert
#define b_iSpecularInvert 0
#endif
#ifndef b_iSpecularClamp
#define b_iSpecularClamp 0
#endif
#ifndef b_iSpecularFresnelMode
#define b_iSpecularFresnelMode 0
#endif
#ifndef b_iSpecularFresnelTransformMode
#define b_iSpecularFresnelTransformMode 0
#endif
#ifndef b_iSpecularFresnelSaturate
#define b_iSpecularFresnelSaturate 0
#endif
#ifndef b_iSpecularOp
#define b_iSpecularOp 0
#endif
#ifndef b_iSpecularUVEmitter
#define b_iSpecularUVEmitter 0
#endif
#ifndef b_iSpecularUVMapping
#define b_iSpecularUVMapping 0
#endif
#ifndef b_iSpecularIsRTTTexture
#define b_iSpecularIsRTTTexture 0
#endif
#ifndef b_iSpecularLayerEnable
#define b_iSpecularLayerEnable 0
#endif
#ifndef b_iSpecularType
#define b_iSpecularType 0
#endif
#ifndef b_useSpecular
#define b_useSpecular 0
#endif
#ifndef b_useTeamColorSpecular
#define b_useTeamColorSpecular 0
#endif
// Layer-id defaults for the triplanar-UV table fill (initTriPlanarUVs writes
// tp.m[b_i<Layer>LayerId], so every layer's LayerId must resolve even when the perm
// left it at 0).  Diffuse/Specular are declared above.
#ifndef b_iDecalLayerId
#define b_iDecalLayerId 0
#endif
#ifndef b_iEmissiveLayerId
#define b_iEmissiveLayerId 0
#endif
#ifndef b_iEmissive2LayerId
#define b_iEmissive2LayerId 0
#endif
#ifndef b_iAlphaMaskLayerId
#define b_iAlphaMaskLayerId 0
#endif
#ifndef b_iAlphaMask2LayerId
#define b_iAlphaMask2LayerId 0
#endif
#ifndef b_iLightmapLayerId
#define b_iLightmapLayerId 0
#endif
#ifndef b_iAmbientOcclusionLayerId
#define b_iAmbientOcclusionLayerId 0
#endif
#ifndef b_iSpecularExponentLayerId
#define b_iSpecularExponentLayerId 0
#endif
#ifndef b_iNormalLayerId
#define b_iNormalLayerId 0
#endif

// ---- SLayerConfig (psmateriallayer.fx : SLayerConfig) ----------------------
public struct SLayerConfig
{
    int    b_iLayerEnable;
    int    b_iLayerId;
    int    b_iTextureType;
    int    b_iUseConstantColor;
    int    b_iUseMask;
    int    b_iChannelSelect;
    int    b_iTeamColorMode;
    int    b_iInvert;
    int    b_iClamp;
    int    b_iFresnelMode;
    int    b_iFresnelTransformMode;
    int    b_iFresnelSaturate;
    int    b_iOp;
    int    b_iUVEmitter;
    int    b_iUVMapping;
    int    b_iIsRTTTexture;
    int    b_iTriPlanarUvId;
    float4 p_vMultiplyAddAlphaTrans;
    float4 p_vAtlasParams;
    float3 p_vFresnelExponentBiasScale;
    float4 p_cConstantColor;
    float4 p_vRTTTextureOffsetScale;
    float3x4 p_mFresnelTransform;
    float2x4 p_mUVTransform;
};

// psmaterialdefines.fx : ChooseChannel.
float4 chooseChannel(int iChannelSelect, float4 cResult)
{
    if (iChannelSelect == CHANNELSELECT_R)      cResult = cResult.rrrr;
    else if (iChannelSelect == CHANNELSELECT_G) cResult = cResult.gggg;
    else if (iChannelSelect == CHANNELSELECT_B) cResult = cResult.bbbb;
    else if (iChannelSelect == CHANNELSELECT_A) cResult = cResult.aaaa;
    else if (iChannelSelect == CHANNELSELECT_RGB) cResult.a = 1;
    return cResult;
}

// psmateriallayer.fx : CalcFresnelTerm (the PS_14 early-out is unreachable at
// SM4+, so it is dropped).
float calcFresnelTerm(float3 vNormal, float3 vEyetoVertex, float exponent,
                      float3x4 mFresnelTransform, int iFresnelTransformMode,
                      int bFresnelSaturate)
{
    float3 vFresnelDir = vEyetoVertex;
    float fResult;
    if (iFresnelTransformMode != FRESNELTRANSFORM_NONE)
    {
        vFresnelDir = mul(mFresnelTransform, float4(vFresnelDir, 1.0)).xyz;
        if (iFresnelTransformMode == FRESNELTRANSFORM_NORMALIZED)
            vFresnelDir = normalize(vFresnelDir);
    }
    if (bFresnelSaturate)
        fResult = 1.0f - saturate(-dot(vNormal, vFresnelDir));
    else
        fResult = 1.0f - abs(dot(vNormal, vFresnelDir));
    fResult = pow(max(fResult, 0.0001), exponent);
    return fResult;
}

// psmateriallayer.fx : ComputeLayerColorInternal.  The fresnel term is computed
// only when it can affect the result (mode != NONE) — behaviorally identical to
// the .fx (which computes-then-conditionally-applies) and it avoids the fresnel
// interpolant/constant dependency on non-fresnel perms.
float4 computeLayerColorInternal(ModelPSInput vertOut, SLayerConfig cfg,
                                 float fMask, float4 cColor, float3 vNormal)
{
    float4 cResult = cColor;

    if (cfg.b_iUseMask)
        cResult = cResult * fMask;

    cResult = chooseChannel(cfg.b_iChannelSelect, cResult);

    // Alpha factor.
    cResult.a *= cfg.p_vMultiplyAddAlphaTrans.z;

    // Team colour.
    if (cfg.b_iChannelSelect != CHANNELSELECT_RGB)
    {
        if (cfg.b_iTeamColorMode == TEAMCOLOR_DIFFUSE)
            cResult = float4(lerp(psGlobals.cDiffuseTeamColor.rgb, cResult.rgb,
                             saturate(pow(cColor.a, psGlobals.fTeamColorIntensity.x))), 1.0f);
        else if (cfg.b_iTeamColorMode == TEAMCOLOR_EMISSIVE)
            cResult = float4(lerp(psGlobals.cEmissiveTeamColor.rgb, cResult.rgb, cColor.a), 1.0f);
    }

    if (cfg.b_iInvert)
        cResult = float4(1.0f, 1.0f, 1.0f, 1.0f) - cResult;

    // RGB multiply + add.
    cResult = (cResult * cfg.p_vMultiplyAddAlphaTrans.x) + cfg.p_vMultiplyAddAlphaTrans.y;

    if (cfg.b_iClamp)
        cResult = saturate(cResult);

    // Fresnel term.
    if (cfg.b_iFresnelMode != FRESNELMODE_NONE)
    {
        float fFresnelTerm;
        // Both arms differ only in the view vector, and EyeToVertexFresnel() is a
        // total accessor, so this is an ordinary value selection.
        float3 vFresnelView = SC2_AXIS(b_useSimplifiedFresnel)
                            ? psGlobals.vCameraDirection : vertOut.EyeToVertexFresnel();
        fFresnelTerm = calcFresnelTerm(vNormal, vFresnelView,
            cfg.p_vFresnelExponentBiasScale.x, cfg.p_mFresnelTransform,
            cfg.b_iFresnelTransformMode, cfg.b_iFresnelSaturate);
        if (cfg.b_iFresnelMode == FRESNELMODE_INVERTED)
            fFresnelTerm = 1.0f - fFresnelTerm;
        fFresnelTerm = saturate(fFresnelTerm * cfg.p_vFresnelExponentBiasScale.z
                                + cfg.p_vFresnelExponentBiasScale.y);
        cResult = cResult * fFresnelTerm;
    }

    return cResult;
}

// psmateriallayer.fx : CombineLayerColor.
float3 combineLayerColor(float4 cLayerColor, float3 cResult, SLayerConfig cfg)
{
    if (cfg.b_iOp == LAYEROP_MOD)
        cResult = cResult * cLayerColor.rgb;
    else if (cfg.b_iOp == LAYEROP_MOD2X)
        cResult = cResult * cLayerColor.rgb * 2.0f;
    else if (cfg.b_iOp == LAYEROP_ADD)
        cResult = cResult + cLayerColor.rgb * cLayerColor.a;
    else if (cfg.b_iOp == LAYEROP_ADD_NO_ALPHA)
        cResult = cResult + cLayerColor.rgb;
    else if (cfg.b_iOp == LAYEROP_LERP)
        cResult = lerp(cResult, cLayerColor.rgb, cLayerColor.a);
    else if (cfg.b_iOp == LAYEROP_TEAMCOLOR_EMISSIVE_ADD)
        cResult = cResult + saturate(pow(cLayerColor.a, psGlobals.fTeamColorIntensity.y)) * psGlobals.cEmissiveTeamColor.rgb;
    else if (cfg.b_iOp == LAYEROP_TEAMCOLOR_DIFFUSE_ADD)
        cResult = cResult + cLayerColor.a * psGlobals.cDiffuseTeamColor.rgb;
    return cResult;
}

// pscommon.fx : GetUVEmitter — read the layer's per-emitter UV slot, optionally
// through the layer's 2×4 UV transform.
float4 getUVEmitter(ModelPSInput vertOut, int index, float iUVTransform, float2x4 iUVMatrix)
{
    float4 rvalue;
#if b_usePackedUVEmitter
    if (index % 2 == 0)
        rvalue = float4(vertOut.UV(index / 2).xy, 0, 1);
    else
        rvalue = float4(vertOut.UV(index / 2).zw, 0, 1);
#else
    rvalue = vertOut.UV(index);
#endif
    float4 transform = float4(mul(iUVMatrix, float4(rvalue.x, rvalue.y, 0.0f, 1.0f)).xy, 0, 1);
    return (iUVTransform != 0.0f) ? transform : rvalue;
}

// psuvmapping.fx : TriplanarAtlas — the per-axis atlas UVs for triplanar sampling.
void triplanarAtlas(float4 vAtlasParams, float3 normal, float3 uvSource,
                    out float2 uvZ, out float2 uvY, out float2 uvX)
{
    float2 uvCalc = frac(uvSource.xy);
    float4 uvReverse = frac(-uvSource.xyzy);

    float4 atlasScale = vAtlasParams.xyyx;
    float4 atlasOffset = vAtlasParams.wzzw;

    uvCalc = uvCalc * atlasScale.x + atlasOffset.x;
    uvReverse = uvReverse * atlasScale + atlasOffset;

    uvZ.x = (normal.z >= 0) ? uvCalc.x : uvReverse.x;
    uvZ.y = uvReverse.y;
    uvY.x = (normal.y >= 0) ? uvReverse.x : uvCalc.x;
    uvY.y = uvReverse.z;
    uvX.x = (normal.x >= 0) ? uvCalc.y : uvReverse.w;
    uvX.y = uvReverse.z;

    uvY.x += 0.5f;
    uvX.x += 0.5f;
}

// psutils.fx : SampleTriPlanarTexture — three bias(-20) samples blended by weights.
float4 sampleTriPlanarTexture(Texture2D<float4> tTex, SamplerState sTex,
                              float2 uvZ, float2 uvY, float2 uvX, float3 weights)
{
    float4 result  = tTex.SampleBias(sTex, uvZ, -20.0f) * weights.z;
           result += tTex.SampleBias(sTex, uvY, -20.0f) * weights.y;
           result += tTex.SampleBias(sTex, uvX, -20.0f) * weights.x;
    return result;
}

// psmateriallayer.fx : InitializeTriPlanarUVsSingleImpl — this layer's triplanar UVs.
float3x2 initTriPlanarUVsSingleImpl(ModelPSInput vertOut, SLayerConfig cfg)
{
    float4 vUV = getUVEmitter(vertOut, cfg.b_iUVEmitter,
                              cfg.p_vMultiplyAddAlphaTrans.w, cfg.p_mUVTransform);
    float3 normal;
    if (TriPlanarIsWorldFX(cfg.b_iUVMapping))
        normal = vertOut.Normal().xyz;
    else
        normal = vertOut.VectorUI0().xyz;

    float2 uvZ, uvY, uvX;
    triplanarAtlas(cfg.p_vAtlasParams, normal, vUV.xyz, uvZ, uvY, uvX);

    float3x2 outMat;
    outMat[0] = uvZ;
    outMat[1] = uvY;
    outMat[2] = uvX;
    return outMat;
}

// psmateriallayercolor.fx : SampleLayer.  Constant-colour early-out, the triplanar
// path (2D textures: shared table slot or inline atlas), and the explicit 2D sample.
// Envio/cube/volume arms land in their own paths (getLayerColorEnvio); the diffuse
// layer passes allowEnvio=false.
float4 sampleLayer(ModelPSInput vertOut, SLayerConfig cfg,
                   Texture2D<float4> tLayer, SamplerState sLayer, TriPlanarUVs tp)
{
    if (cfg.b_iUseConstantColor)
        return cfg.p_cConstantColor;

    float4 vUV = getUVEmitter(vertOut, cfg.b_iUVEmitter,
                              cfg.p_vMultiplyAddAlphaTrans.w, cfg.p_mUVTransform);

    // Triplanar mapping (2D textures only — textureEnum is 2D/UNDEFINED on these layers).
    if (IsTriPlanarMappingFX(cfg.b_iUVMapping))
    {
        float3 normal, weights;
        if (TriPlanarIsWorldFX(cfg.b_iUVMapping))
        {
            normal = vertOut.Normal().xyz;
            weights = float3(vertOut.FogColor().x, vertOut.FogColor().y, vUV.w);
        }
        else
        {
            normal = vertOut.VectorUI0().xyz;
            weights = vertOut.TriPlanarWeights().xyz;
        }

        float2 uvZ, uvY, uvX;
        if (cfg.b_iTriPlanarUvId >= c_maxNumLayers)
        {
            if (cfg.b_iLayerId < c_maxNumLayers)
            {
                uvZ = tp.m[cfg.b_iLayerId][0];
                uvY = tp.m[cfg.b_iLayerId][1];
                uvX = tp.m[cfg.b_iLayerId][2];
            }
            else
                triplanarAtlas(cfg.p_vAtlasParams, normal, vUV.xyz, uvZ, uvY, uvX);
        }
        else
        {
            uvZ = tp.m[cfg.b_iTriPlanarUvId][0];
            uvY = tp.m[cfg.b_iTriPlanarUvId][1];
            uvX = tp.m[cfg.b_iTriPlanarUvId][2];
        }
        return sampleTriPlanarTexture(tLayer, sLayer, uvZ, uvY, uvX, weights);
    }

    if (cfg.b_iIsRTTTexture)
        vUV.xy = vUV.xy * cfg.p_vRTTTextureOffsetScale.zw + cfg.p_vRTTTextureOffsetScale.xy;

    return tLayer.Sample(sLayer, vUV.xy);
}

// psmateriallayer.fx : GetLayerColor -> ComputeLayerColor = SampleLayer +
// ComputeLayerColorInternal.  The 2D (non-envio) path; envio sampling is in
// getLayerColorEnvio.  mTriPlanarUVs threads the shared triplanar UV table.
float4 getLayerColor2D(ModelPSInput vertOut, SLayerConfig cfg,
                       Texture2D<float4> tLayer, SamplerState sLayer,
                       float fMask, float3 vNormal, TriPlanarUVs tp)
{
    float4 cColor = sampleLayer(vertOut, cfg, tLayer, sLayer, tp);
    return computeLayerColorInternal(vertOut, cfg, fMask, cColor, vNormal);
}

// SETUP_LAYER(Diffuse) — build the diffuse layer's SLayerConfig from its axes +
// constants.
SLayerConfig setupDiffuseLayer()
{
    SLayerConfig c;
    c.b_iLayerEnable            = SC2_CORE_ENABLE(Diffuse, b_iDiffuseLayerEnable);
    c.b_iLayerId                = SC2_LAYER_LayerId(b_iDiffuseLayerId);
    c.b_iTextureType            = b_iDiffuseTextureType;
    c.b_iTriPlanarUvId          = SC2_LAYER_TriPlanarUvId(b_iDiffuseTriPlanarUvId);
    c.p_vAtlasParams            = psGlobals.vDiffuseAtlasParams;
    c.b_iUseConstantColor       = SC2_CORE_ENABLE(Diffuse, b_iDiffuseUseConstantColor);
    c.b_iUseMask                = SC2_LAYER_UseMask(b_iDiffuseUseMask);
    c.b_iChannelSelect          = SC2_LAYER_ChannelSelect(b_iDiffuseChannelSelect);
    c.b_iTeamColorMode          = SC2_LAYER_TeamColorMode(b_iDiffuseTeamColorMode);
    c.b_iInvert                 = SC2_LAYER_Invert(b_iDiffuseInvert);
    c.b_iClamp                  = SC2_LAYER_Clamp(b_iDiffuseClamp);
    c.b_iFresnelMode            = SC2_LAYER_FresnelMode(b_iDiffuseFresnelMode);
    c.b_iFresnelTransformMode   = SC2_LAYER_FresnelTransformMode(b_iDiffuseFresnelTransformMode);
    c.b_iFresnelSaturate        = SC2_LAYER_FresnelSaturate(b_iDiffuseFresnelSaturate);
    c.b_iOp                     = SC2_LAYER_Op(b_iDiffuseOp);
    c.b_iUVEmitter              = SC2_LAYER_UVEmitter(b_iDiffuseUVEmitter);
    c.b_iUVMapping              = SC2_LAYER_UVMapping(b_iDiffuseUVMapping);
    c.b_iIsRTTTexture           = SC2_LAYER_IsRTTTexture(b_iDiffuseIsRTTTexture);
    c.p_vMultiplyAddAlphaTrans  = psGlobals.vDiffuseMultiplyAddAlphaTrans;
    c.p_vFresnelExponentBiasScale = psGlobals.vDiffuseFresnelExponentBiasScale;
    c.p_cConstantColor          = psGlobals.cDiffuseConstantColor;
    c.p_vRTTTextureOffsetScale   = psGlobals.vDiffuseRTTTextureOffsetScale;
    c.p_mFresnelTransform       = psGlobals.mDiffuseFresnelTransform;
    c.p_mUVTransform            = psGlobals.mDiffuseUVTransform;
    return c;
}

// SETUP_LAYER(Specular) — the specular material layer's SLayerConfig.
SLayerConfig setupSpecularLayer()
{
    SLayerConfig c;
    c.b_iLayerEnable            = SC2_CORE_ENABLE(Specular, b_iSpecularLayerEnable);
    c.b_iLayerId                = SC2_LAYER_LayerId(b_iSpecularLayerId);
    c.b_iTextureType            = b_iSpecularTextureType;
    c.b_iTriPlanarUvId          = SC2_LAYER_TriPlanarUvId(b_iSpecularTriPlanarUvId);
    c.p_vAtlasParams            = psGlobals.vSpecularAtlasParams;
    c.b_iUseConstantColor       = SC2_CORE_ENABLE(Specular, b_iSpecularUseConstantColor);
    c.b_iUseMask                = SC2_LAYER_UseMask(b_iSpecularUseMask);
    c.b_iChannelSelect          = SC2_LAYER_ChannelSelect(b_iSpecularChannelSelect);
    c.b_iTeamColorMode          = SC2_LAYER_TeamColorMode(b_iSpecularTeamColorMode);
    c.b_iInvert                 = SC2_LAYER_Invert(b_iSpecularInvert);
    c.b_iClamp                  = SC2_LAYER_Clamp(b_iSpecularClamp);
    c.b_iFresnelMode            = SC2_LAYER_FresnelMode(b_iSpecularFresnelMode);
    c.b_iFresnelTransformMode   = SC2_LAYER_FresnelTransformMode(b_iSpecularFresnelTransformMode);
    c.b_iFresnelSaturate        = SC2_LAYER_FresnelSaturate(b_iSpecularFresnelSaturate);
    c.b_iOp                     = SC2_LAYER_Op(b_iSpecularOp);
    c.b_iUVEmitter              = SC2_LAYER_UVEmitter(b_iSpecularUVEmitter);
    c.b_iUVMapping              = SC2_LAYER_UVMapping(b_iSpecularUVMapping);
    c.b_iIsRTTTexture           = SC2_LAYER_IsRTTTexture(b_iSpecularIsRTTTexture);
    c.p_vMultiplyAddAlphaTrans  = psGlobals.vSpecularMultiplyAddAlphaTrans;
    c.p_vFresnelExponentBiasScale = psGlobals.vSpecularFresnelExponentBiasScale;
    c.p_cConstantColor          = psGlobals.cSpecularConstantColor;
    c.p_vRTTTextureOffsetScale   = psGlobals.vSpecularRTTTextureOffsetScale;
    c.p_mFresnelTransform       = psGlobals.mSpecularFresnelTransform;
    c.p_mUVTransform            = psGlobals.mSpecularUVTransform;
    return c;
}

// psmaterial.fx : InitializeTriPlanarUVs — precompute the shared triplanar UV table.
// Each layer that uses triplanar mapping with an implicit id (TriPlanarUvId >=
// c_maxNumLayers) writes its atlas UVs into slot LayerId; other layers can then read a
// slot by TriPlanarUvId.  Every write/read index is a compile-time layer id.  The
// per-layer fills are #if-gated on the (compile-time) InitializeTriPlanarUVsSingle
// condition so no dead out-of-range write is emitted.  Envio/EnvioMask (cube/envio)
// don't participate in the 2D triplanar table.
TriPlanarUVs initTriPlanarUVs(ModelPSInput vertOut)
{
    TriPlanarUVs tp;
    float3x2 ident = float3x2(1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    tp.m[0]  = ident; tp.m[1]  = ident; tp.m[2]  = ident; tp.m[3]  = ident;
    tp.m[4]  = ident; tp.m[5]  = ident; tp.m[6]  = ident; tp.m[7]  = ident;
    tp.m[8]  = ident; tp.m[9]  = ident; tp.m[10] = ident; tp.m[11] = ident;
    tp.m[12] = ident; tp.m[13] = ident; tp.m[14] = ident; tp.m[15] = ident;
    tp.m[16] = ident; tp.m[17] = ident;

#if (IsTriPlanarMappingFX(b_iDiffuseUVMapping) && b_iDiffuseTriPlanarUvId >= c_maxNumLayers && b_iDiffuseLayerId < c_maxNumLayers)
    tp.m[b_iDiffuseLayerId] = initTriPlanarUVsSingleImpl(vertOut, setupDiffuseLayer());
#endif
#if (IsTriPlanarMappingFX(b_iSpecularUVMapping) && b_iSpecularTriPlanarUvId >= c_maxNumLayers && b_iSpecularLayerId < c_maxNumLayers)
    tp.m[b_iSpecularLayerId] = initTriPlanarUVsSingleImpl(vertOut, setupSpecularLayer());
#endif
#if (IsTriPlanarMappingFX(b_iDecalUVMapping) && b_iDecalTriPlanarUvId >= c_maxNumLayers && b_iDecalLayerId < c_maxNumLayers)
    tp.m[b_iDecalLayerId] = initTriPlanarUVsSingleImpl(vertOut, setupDecalLayer());
#endif
#if (IsTriPlanarMappingFX(b_iEmissiveUVMapping) && b_iEmissiveTriPlanarUvId >= c_maxNumLayers && b_iEmissiveLayerId < c_maxNumLayers)
    tp.m[b_iEmissiveLayerId] = initTriPlanarUVsSingleImpl(vertOut, setupEmissiveLayer());
#endif
#if (IsTriPlanarMappingFX(b_iEmissive2UVMapping) && b_iEmissive2TriPlanarUvId >= c_maxNumLayers && b_iEmissive2LayerId < c_maxNumLayers)
    tp.m[b_iEmissive2LayerId] = initTriPlanarUVsSingleImpl(vertOut, setupEmissive2Layer());
#endif
#if (IsTriPlanarMappingFX(b_iAlphaMaskUVMapping) && b_iAlphaMaskTriPlanarUvId >= c_maxNumLayers && b_iAlphaMaskLayerId < c_maxNumLayers)
    tp.m[b_iAlphaMaskLayerId] = initTriPlanarUVsSingleImpl(vertOut, setupAlphaMaskLayer());
#endif
#if (IsTriPlanarMappingFX(b_iAlphaMask2UVMapping) && b_iAlphaMask2TriPlanarUvId >= c_maxNumLayers && b_iAlphaMask2LayerId < c_maxNumLayers)
    tp.m[b_iAlphaMask2LayerId] = initTriPlanarUVsSingleImpl(vertOut, setupAlphaMask2Layer());
#endif
#if (IsTriPlanarMappingFX(b_iLightmapUVMapping) && b_iLightmapTriPlanarUvId >= c_maxNumLayers && b_iLightmapLayerId < c_maxNumLayers)
    tp.m[b_iLightmapLayerId] = initTriPlanarUVsSingleImpl(vertOut, setupLightmapLayer());
#endif
#if (IsTriPlanarMappingFX(b_iAmbientOcclusionUVMapping) && b_iAmbientOcclusionTriPlanarUvId >= c_maxNumLayers && b_iAmbientOcclusionLayerId < c_maxNumLayers)
    tp.m[b_iAmbientOcclusionLayerId] = initTriPlanarUVsSingleImpl(vertOut, setupAmbientOcclusionLayer());
#endif
#if (IsTriPlanarMappingFX(b_iSpecularExponentUVMapping) && b_iSpecularExponentTriPlanarUvId >= c_maxNumLayers && b_iSpecularExponentLayerId < c_maxNumLayers)
    tp.m[b_iSpecularExponentLayerId] = initTriPlanarUVsSingleImpl(vertOut, setupSpecularExponentLayer());
#endif
#if (IsTriPlanarMappingFX(b_iNormalUVMapping) && b_iNormalTriPlanarUvId >= c_maxNumLayers && b_iNormalLayerId < c_maxNumLayers)
    tp.m[b_iNormalLayerId] = initTriPlanarUVsSingleImpl(vertOut, setupNormalLayer());
#endif
    return tp;
}
