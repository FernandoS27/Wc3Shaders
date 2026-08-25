// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026, Fernando Sahmkow
// See LICENSE in the repository root for full terms.
//
// Model-family lighting (pslighting.fx : MainLighting), the stage that produces
// cLightDiffuse for the MaterialColor blend (cFinal.rgb = cDeferredDiffuse *
// cLightDiffuse, psmaterial.fx:234).
//
// INCREMENT 3b-i SCOPE: the two cheapest MainLighting arms —
//   * b_useLighting == 0            -> cLightDiffuse = 1  (the unlit path every
//                                      earlier increment already relied on).
//   * VERTEXLIGHTING_FULL           -> cLightDiffuse = INTERPOLANT_Diffuse, the
//                                      VS-accumulated vertex lighting read through
//                                      an interpolant (no per-pixel Lighting()).
// The per-pixel arm (VERTEXLIGHTING_NONE -> Lighting(): ambient + 3 directional
// lights + specular, needs the p_directionalLights[]/p_cAmbientColor constants)
// and the PARTIAL arm land in the following lit increments; cLightSpecular /
// shadows / AO arrive with them.  Measured: the FULL/no-specular/no-shadow class
// this covers is ~2383 Model perms.

implementing sc2_shaders;

// pslighting.fx / lighting.fx vertex-lighting-mode enum.
#define VERTEXLIGHTING_NONE     0
#define VERTEXLIGHTING_FULL     1
#define VERTEXLIGHTING_PARTIAL  2

// lighting.fx light-filter + light-mode enums (the PS path is always ALL_LIGHTS,
// isPerPixel=true; point/spot only run in PIXEL_SINGLEPASS_LIGHTING_MODE).
#define ALL_LIGHTS                          0
#define EXCLUDE_SHADOWLIGHT                 1
#define SHADOWLIGHT_ONLY                    2
#define NO_LOCAL_LIGHTS_MODE                0
#define DEFERRED_LIGHTING_MODE             1
#define VERTEX_SINGLEPASS_LIGHTING_MODE    2
#define PIXEL_SINGLEPASS_LIGHTING_MODE     3
#define MAX_DIRECTIONAL_LIGHTS             3
#define MAX_POINT_LIGHTS                   3
#define MAX_SPOTLIGHTS                     7

#include "types/perm_cb.slangh"
// SC2_CORE_LAYER_* / SC2_LAYER_GATE_* for the two layers evaluated here.  The
// `-D` a core class carries would define them anyway, but a `#define` does NOT
// cross an `__include`, so without this the DEFAULTS would be missing and a
// future `SC2_LAYER_GATE_` here would expand to an undeclared function call.
#include "ps/model_layers_axes.slangh"

#ifndef b_useLighting
#define b_useLighting 0
#endif
#ifndef b_iLightingMode
#define b_iLightingMode 0
#endif
#ifndef b_UseLightingRegions
#define b_UseLightingRegions 0
#endif
#ifndef b_useShadows
#define b_useShadows 0
#endif
#ifndef b_iVertexLightingMode
#define b_iVertexLightingMode 0
#endif
#ifndef b_iDirectionalLightCount
#define b_iDirectionalLightCount 0
#endif
#ifndef b_useDblLambert
#define b_useDblLambert 0
#endif
#ifndef b_UseAmbientEnvironment
#define b_UseAmbientEnvironment 0
#endif
#ifndef b_iUseDynamicAO
#define b_iUseDynamicAO 0
#endif
#ifndef b_iUseSpecular
#define b_iUseSpecular 0
#endif
#ifndef b_iNormalizeHalfVector
#define b_iNormalizeHalfVector 0
#endif
#ifndef b_iSpecularExponentLayerEnable
#define b_iSpecularExponentLayerEnable 0
#endif
#ifndef b_UseVertexBasedAmbientOcclusion
#define b_UseVertexBasedAmbientOcclusion 0
#endif

// Lighting() result — diffuse + specular light colour + the shadow colour
// (MainLighting out params; cShadowColor feeds the Lightmap-shadow blend and fog).
struct LightResult
{
    float3 cLightDiffuse;
    float3 cLightSpecular;
    float4 cShadowColor;
    float4 cLightingRegionMap;   // region weights (region texture / vertex colour); fog reads it
};

// psmaterial.fx : MaterialSpecularity — the specular power/gloss fed to Blinn, plus
// the deferred glossiness out-param.  The SpecularExponent material layer squares
// its sampled alpha into the exponent; else the constant p_vSpecColorSpecularity.w.
float materialSpecularity(ModelPSInput vertOut, float3 vDeferredNormal,
                          out float cDeferredGlossiness, TriPlanarUVs tp)
{
#if (SC2_CORE_LAYER_SpecularExponent || b_iSpecularExponentLayerEnable)
    SLayerConfig cfg = setupSpecularExponentLayer();
    float4 cClr = getLayerColor2D(vertOut, cfg, p_sSpecularExponentSampler_t,
                                  p_sSamp, 1.0f, vDeferredNormal, tp);
    // Written as an EARLY-OUT under `#if SC2_CORE_LAYER_`, not as an `if/else`
    // around the whole body: with the layer non-core the gate is absent and the
    // remaining text is character-for-character what it was, so the non-core blob
    // stays bit-identical.  Wrapping the body instead cost nothing semantically and
    // still moved two instructions in Ribbon_ps -- a scheduler artifact, but one
    // that makes the bit-identity gate report a difference it cannot explain.
  #if SC2_CORE_LAYER_SpecularExponent
    // An early-OUT rather than a gate around the body, for the same reason the gate
    // macro exists: what follows must stay textually untouched when non-core.
    [branch] if (!cfg.b_iLayerEnable)
    {
        cDeferredGlossiness = 1;
        return psGlobals.vSpecColorSpecularity.w;
    }
  #endif
    cDeferredGlossiness = cClr.a;
    return cClr.a * cClr.a * psGlobals.vSpecColorSpecularity.w;
#else
    cDeferredGlossiness = 1;
    return psGlobals.vSpecColorSpecularity.w;
#endif
}

// psutils.fx : GetBackBufferUV, VPOS_SEMANTIC path (PS_40+ reads SV_Position as
// INTERPOLANT_ScreenPos): ScreenPos.xy * recipRTSize.xy + recipRTSize.zw.
float2 getBackBufferUV(ModelPSInput vertOut)
{
    return vertOut.HPos.xy * psGlobals.vRecipRenderTargetSizeOffset.xy
         + psGlobals.vRecipRenderTargetSizeOffset.zw;
}

// psmaterial.fx : DecodeAOValue — the AO material layer (samples its alpha and also
// writes the deferred AO output) then the dynamic-AO (SSAO) multiply.
float decodeAOValue(ModelPSInput vertOut, float3 vDeferredNormal, inout float cDeferredAO,
                    TriPlanarUVs tp)
{
    float fAmbientOcclusion;
#if (SC2_CORE_LAYER_AmbientOcclusion || b_iAmbientOcclusionLayerEnable)
    SLayerConfig cfg = setupAmbientOcclusionLayer();
    fAmbientOcclusion = getLayerColor2D(vertOut, cfg, p_sAmbientOcclusionSampler_t,
                                        p_sSamp, 1.0f, vDeferredNormal, tp).a;
  #if SC2_CORE_LAYER_AmbientOcclusion
    // Same early-out shape as materialSpecularity above, and for the same reason.
    [branch] if (!cfg.b_iLayerEnable)
        fAmbientOcclusion = 1.0f;
    else
  #endif
    cDeferredAO = fAmbientOcclusion;
#else
    fAmbientOcclusion = 1.0f;
#endif
#if b_iUseDynamicAO
    fAmbientOcclusion = saturate(fAmbientOcclusion
        * p_sSSAO_t.Sample(p_sSamp, getBackBufferUV(vertOut)).a);
#endif
    return fAmbientOcclusion;
}

// lightingmodels.fx : BlinnShading — diffuse in .x, specular in .y, .z = 0.
// b_iLightingModel is Blinn (0) on every retail Model perm (KajiyaKay hair is
// deferred).  ComputeDirectionalLight passes bSpecifyHalfVector = true, so the
// half-vector is used directly (no ComputeHalfVector).
float3 blinnShading(float3 vNormalWS, float3 vLightDirWS, bool bUseSpecular,
                    float3 vHalfVectorWS, float fSpecularPower)
{
    float3 ds;
    ds.x = dot(vNormalWS, vLightDirWS);
    [branch] if (SC2_AXIS(b_useDblLambert))
        ds.x = -1.0f + ds.x * 2.0f;
    if (bUseSpecular)
    {
        ds.y = saturate(dot(vNormalWS, vHalfVectorWS));
        ds.y = pow(ds.y, fSpecularPower);
        ds.y = saturate(ds.y * sign(ds.x));
    }
    else
        ds.y = 0.0f;

    ds.x = saturate(ds.x);
    ds.z = 0.0f;
    return ds;
}

// common.fx : ComputeHalfVector — eye-to-vertex half vector for specular.  The
// b_useVertexLighting NaN-workaround branch is dead (its guard is `false`), so only
// the normalize form survives.
float3 computeHalfVector(float3 vWorldPos, float3 vWorldLightDirection)
{
    float3 vEyeVec = normalize(psGlobals.vEyePos - vWorldPos);
    return normalize(vEyeVec - vWorldLightDirection);
}

// lighttypes.fx : ComputeLightAttenuation.
float computeLightAttenuation(float fAttenMultiplier, float fRecipRange, float fRecipDist)
{
    float fAttenuation = 1.0f - saturate(fRecipRange / fRecipDist);
    fAttenuation = saturate(fAttenuation * fAttenMultiplier);
    fAttenuation = fAttenuation * fAttenuation;
    return fAttenuation;
}

// lighttypes.fx : ComputePointLight — Blinn shading with distance attenuation (the
// half vector is recomputed per-light from eye + light, bSpecifyHalfVector=false).
float3 computePointLight(float3 vNormalWS, float3 vVertexPosWS, float3 vLightPosWS,
                         float fRecipRange, float fAttenMultiplier, float fSpecularPower)
{
    float3 vLightDir = vLightPosWS - vVertexPosWS;
    float fRecipDist = rsqrt(dot(vLightDir, vLightDir));
    vLightDir *= fRecipDist;
    float3 vHalf = computeHalfVector(vVertexPosWS, normalize(vVertexPosWS - vLightPosWS));
    float3 ds = blinnShading(vNormalWS, vLightDir, true, vHalf, fSpecularPower);
    return ds * computeLightAttenuation(fAttenMultiplier, fRecipRange, fRecipDist);
}

// lighttypes.fx : ComputeSpotLight — point light + the cone falloff (dot-product form).
float3 computeSpotLight(float3 vNormalWS, float3 vVertexPosWS, float3 vLightPosWS,
                        float3 vSpotDir, float fRecipRange, float fAttenMultiplier,
                        float fFalloffBias, float fFalloffScale, float fHotspotMultiplier,
                        float fSpecularPower)
{
    float3 vLightDir = vLightPosWS - vVertexPosWS;
    float fRecipDist = rsqrt(dot(vLightDir, vLightDir));
    vLightDir *= fRecipDist;
    float3 vHalf = computeHalfVector(vVertexPosWS, normalize(vVertexPosWS - vLightPosWS));
    float3 ds = blinnShading(vNormalWS, vLightDir, true, vHalf, fSpecularPower);

    float fFalloff = saturate(dot(-vLightDir, vSpotDir));
    fFalloff = saturate((fFalloff + fFalloffBias) * fFalloffScale);
    fFalloff = saturate(fFalloff * fHotspotMultiplier);
    fFalloff = fFalloff * fFalloff;

    return ds * computeLightAttenuation(fAttenMultiplier, fRecipRange, fRecipDist) * fFalloff;
}

// lighting.fx : BlendLightSetup — accumulate one region's contribution.  Called
// with constant indices (manual unroll) so no dynamic cbuffer indexing / dynamic
// loop is emitted (the dxbc interpreter has no icb/dynamic-index support).
void blendLightSetup(LightSetup ls, float frac, inout float3 amb, inout float3 spec,
                     inout float3 d0, inout float3 d1, inout float3 d2)
{
    amb += ls.cAmbientIntensity * frac;
    spec += ls.cLight0Specular * frac;
    d0 += ls.cLight0Diffuse * frac;
    d1 += ls.cLight1Diffuse * frac;
    d2 += ls.cLight2Diffuse * frac;
}

// One point light's diffuse+specular AddLightContribution (useSpecular=true, shadow=1).
void accumPointLight(PointLight pl, float3 vNormalWS, float3 vPositionWS,
                     float fSpecularity, inout float3 dif, inout float3 spec)
{
    float3 ds = computePointLight(vNormalWS, vPositionWS, pl.vPositionRecipRadius.xyz,
                    pl.vPositionRecipRadius.w, pl.cColorAttenMultiplier.a, fSpecularity);
    dif += pl.cColorAttenMultiplier.rgb * ds.x;
    spec += pl.cSpecular.rgb * ds.y;
}

// One spot light's diffuse+specular AddLightContribution.
void accumSpotLight(SpotLight sl, float3 vNormalWS, float3 vPositionWS,
                    float fSpecularity, inout float3 dif, inout float3 spec)
{
    float3 ds = computeSpotLight(vNormalWS, vPositionWS, sl.vPositionRecipRadius.xyz,
                    sl.vDirection.xyz, sl.vPositionRecipRadius.w,
                    sl.cColorAttenMultiplier.a,
                    sl.vSpotFalloffBiasScaleAndHotSpotMultiplier.x,
                    sl.vSpotFalloffBiasScaleAndHotSpotMultiplier.y,
                    sl.vSpotFalloffBiasScaleAndHotSpotMultiplier.z, fSpecularity);
    dif += sl.cColorAttenMultiplier.rgb * ds.x;
    spec += sl.cSpecular.rgb * ds.y;
}

// lighting.fx : Lighting — the per-pixel arm (isPerPixel=true, mode=ALL_LIGHTS, so
// bShadowLight is always true and bSecondaryLights == count>1).  Reproduces the
// region-setup blend (BlendLightSetup ×4), the ambient environment cube, the key
// shadow light, the ambient term, the secondary directional loop, and — under
// PIXEL_SINGLEPASS_LIGHTING_MODE — the single-pass point + spot lights.  Accumulation
// order matches the .fx; cLightSpecular2 (ds.z) is always 0 (Blinn) and MaterialColor
// never reads it, so it is dropped.
LightResult lighting(float3 vNormalWS, float3 vPositionWS, float fAmbientOcclusion,
                     float4 cShadowColor, float fSpecularity, float4 vLightingRegions,
                     float3 vHalfVecWS)
{
    LightResult r;
    float3 curTotalDiffuse = float3(0, 0, 0);
    float3 curTotalSpecular = float3(0, 0, 0);

    float3 cAmbient;
    float3 cDirectional[3];
    float3 cSpecular;

#if b_UseLightingRegions
    // bShadowLight || bSecondaryLights is always true here (bShadowLight==true).
    // Manual unroll over the 4 setups (fractions from vLightingRegions.xyzw).
    cAmbient = cSpecular = float3(0, 0, 0);
    cDirectional[0] = cDirectional[1] = cDirectional[2] = float3(0, 0, 0);
    blendLightSetup(psGlobals.lightSetups[0], vLightingRegions.x, cAmbient, cSpecular,
                    cDirectional[0], cDirectional[1], cDirectional[2]);
    blendLightSetup(psGlobals.lightSetups[1], vLightingRegions.y, cAmbient, cSpecular,
                    cDirectional[0], cDirectional[1], cDirectional[2]);
    blendLightSetup(psGlobals.lightSetups[2], vLightingRegions.z, cAmbient, cSpecular,
                    cDirectional[0], cDirectional[1], cDirectional[2]);
    blendLightSetup(psGlobals.lightSetups[3], vLightingRegions.w, cAmbient, cSpecular,
                    cDirectional[0], cDirectional[1], cDirectional[2]);
#else
    cAmbient = psGlobals.cAmbientColor;
    cSpecular = psGlobals.vSpecColorSpecularity.rgb;
    cDirectional[0] = psGlobals.directionalLights[0].cColor.rgb;
    cDirectional[1] = psGlobals.directionalLights[1].cColor.rgb;
    cDirectional[2] = psGlobals.directionalLights[2].cColor.rgb;
#endif

    // Per-pixel ambient environment cube (lighting.fx:172).
#if b_UseAmbientEnvironment
    cAmbient *= p_sAmbientEnvironmentTexture_t.Sample(p_sSamp, vNormalWS).xyz;
#endif

    // Key shadow light (directional 0), AddLightContribution(useSpecular=true).  ds.y
    // is 0 when b_iUseSpecular is off (BlinnShading), so the specular add folds away.
    float3 ds0 = blinnShading(vNormalWS, -psGlobals.directionalLights[0].vDirection.xyz,
                              SC2_AXIS(b_iUseSpecular) != 0, vHalfVecWS, fSpecularity);
    curTotalDiffuse += cDirectional[0] * ds0.x * cShadowColor.rgb;
    curTotalSpecular += cSpecular * ds0.y * cShadowColor.rgb;

    // Ambient term (isPerPixel -> always added).
    curTotalDiffuse += cAmbient * fAmbientOcclusion;

    // Secondary directional lights (1 .. count-1); AddLightContribution's 4th arg is
    // fAmbientOcclusion (a scalar broadcast), diffuse only, no specular.  Unrolled by
    // the compile-time light count (constant indices — no dynamic local-array index).
    const int iDirectionalLightCount = SC2_AXIS(b_iDirectionalLightCount);
    [branch] if (iDirectionalLightCount > 1)
    {
        float3 ds = blinnShading(vNormalWS, -psGlobals.directionalLights[1].vDirection.xyz,
                                 false, vHalfVecWS, fSpecularity);
        curTotalDiffuse += cDirectional[1] * ds.x * fAmbientOcclusion;
    }
    [branch] if (iDirectionalLightCount > 2)
    {
        float3 ds = blinnShading(vNormalWS, -psGlobals.directionalLights[2].vDirection.xyz,
                                 false, vHalfVecWS, fSpecularity);
        curTotalDiffuse += cDirectional[2] * ds.x * fAmbientOcclusion;
    }

    // Single-pass point + spot lights (PS side only in PIXEL_SINGLEPASS_LIGHTING_MODE);
    // AddLightContribution(useSpecular=isPerPixel=true, cShadowColor=1).
#if (b_iLightingMode == PIXEL_SINGLEPASS_LIGHTING_MODE)
    // Manual unroll (MAX_POINT_LIGHTS = 3, MAX_SPOTLIGHTS = 7) — constant cbuffer
    // indices, no dynamic loop.
    accumPointLight(psGlobals.pointLights[0], vNormalWS, vPositionWS, fSpecularity, curTotalDiffuse, curTotalSpecular);
    accumPointLight(psGlobals.pointLights[1], vNormalWS, vPositionWS, fSpecularity, curTotalDiffuse, curTotalSpecular);
    accumPointLight(psGlobals.pointLights[2], vNormalWS, vPositionWS, fSpecularity, curTotalDiffuse, curTotalSpecular);
    accumSpotLight(psGlobals.spotLights[0], vNormalWS, vPositionWS, fSpecularity, curTotalDiffuse, curTotalSpecular);
    accumSpotLight(psGlobals.spotLights[1], vNormalWS, vPositionWS, fSpecularity, curTotalDiffuse, curTotalSpecular);
    accumSpotLight(psGlobals.spotLights[2], vNormalWS, vPositionWS, fSpecularity, curTotalDiffuse, curTotalSpecular);
    accumSpotLight(psGlobals.spotLights[3], vNormalWS, vPositionWS, fSpecularity, curTotalDiffuse, curTotalSpecular);
    accumSpotLight(psGlobals.spotLights[4], vNormalWS, vPositionWS, fSpecularity, curTotalDiffuse, curTotalSpecular);
    accumSpotLight(psGlobals.spotLights[5], vNormalWS, vPositionWS, fSpecularity, curTotalDiffuse, curTotalSpecular);
    accumSpotLight(psGlobals.spotLights[6], vNormalWS, vPositionWS, fSpecularity, curTotalDiffuse, curTotalSpecular);
#endif

    r.cLightDiffuse = curTotalDiffuse;
    r.cLightSpecular = curTotalSpecular;
    r.cShadowColor = cShadowColor;
    r.cLightingRegionMap = vLightingRegions;
    return r;
}

// pslighting.fx : MainLighting — returns the diffuse + specular light colour the
// MaterialColor blend consumes, plus cShadowColor and the lighting-region map (fog
// reads it).  cShadowColor is computed unconditionally (1 on the no-shadow path).
// Standard entry (Model / splat / terrain): the shadow UV is the VS-emitted
// INTERPOLANT_ShadowMapUV.  Delegates to the explicit-shadowUV form below.
LightResult mainLighting(ModelPSInput vertOut, float3 vDeferredNormal,
                         float fSpecularity, float fAmbientOcclusion, float2 softShadowUV)
{
    return mainLighting(vertOut, vDeferredNormal, fSpecularity, fAmbientOcclusion,
                        softShadowUV, vertOut.ShadowMapUV());
}

// Explicit-shadowUV entry: Water recomputes a distorted shadow UV in the PS
// (mul(worldPos + p_fShadowDistortion*rnd, p_mShadowTransform)) and passes it in,
// rather than reading the interpolant.  Everything else is identical.
LightResult mainLighting(ModelPSInput vertOut, float3 vDeferredNormal,
                         float fSpecularity, float fAmbientOcclusion, float2 softShadowUV,
                         float4 shadowUV)
{
    LightResult r;
    r.cLightDiffuse = float3(0, 0, 0);
    r.cLightSpecular = float3(0, 0, 0);
    r.cLightingRegionMap = float4(0, 0, 0, 0);

    // Shadow value first (pslighting.fx:90) — always computed, 1 when b_useShadows off.
    // softShadowUV is passed in (the standard arm feeds decodeSoftShadowUV; terrain /
    // creep feed their own terrain/creep-space noise UV).  The POM-shadow heightmap
    // params are omitted: no reachable terrain/creep perm sets b_useParallaxMapping,
    // so the height-map depth-bias path in ShadowIntensity DCEs.
    r.cShadowColor = shadowIntensity(vertOut, shadowUV, softShadowUV);

#if b_useLighting == 0
    // Unlit: full diffuse pass-through (b_iDebugMode==LIGHTING_ONLY -> 0 lands with
    // the debug increment).
    r.cLightDiffuse = float3(1, 1, 1);
    return r;
#else
  #if (b_iVertexLightingMode == VERTEXLIGHTING_NONE)
    // Per-pixel lighting.  Region weights sampled from the region map (b_UseLightingRegions).
    float4 vLightingRegions = float4(0, 0, 0, 0);
    #if b_UseLightingRegions
    {
        float2 vUV = vertOut.WorldPos().xy * psGlobals.vWorldToLightingRegionTransform.zw
                   + psGlobals.vWorldToLightingRegionTransform.xy;
        vLightingRegions = p_sLightingRegions_t.Sample(p_sSamp, vUV);
    }
    #endif
    r.cLightingRegionMap = vLightingRegions;

    float3 vHalfVecWS = vertOut.HalfVec();
    [branch] if (SC2_AXIS(b_iNormalizeHalfVector))
        vHalfVecWS = normalize(vHalfVecWS);
    // Vertex-based AO override (pslighting.fx:112-114): replaces the AO used for
    // lighting only (cDeferredAO from DecodeAOValue is unchanged).
    #if b_UseVertexBasedAmbientOcclusion
    fAmbientOcclusion = vertOut.VectorUI1().a;
    #endif

    LightResult lr = lighting(vDeferredNormal, vertOut.WorldPos().xyz, fAmbientOcclusion,
                              r.cShadowColor, fSpecularity, vLightingRegions, vHalfVecWS);
    r.cLightDiffuse = lr.cLightDiffuse;
    r.cLightSpecular = lr.cLightSpecular;
    return r;
  #else
    // FULL / PARTIAL: region weights come from the vertex colour instead.
    r.cLightingRegionMap.xyz = vertOut.VertexColor().xyz;
    r.cLightingRegionMap.a = 1.0f - (r.cLightingRegionMap.x + r.cLightingRegionMap.y
                                     + r.cLightingRegionMap.z);

    #if (b_iVertexLightingMode == VERTEXLIGHTING_PARTIAL)
    {
        float3 vHalfVecWS = computeHalfVector(vertOut.WorldPos().xyz,
                                psGlobals.directionalLights[0].vDirection.xyz);
        LightResult lr = lighting(vDeferredNormal, vertOut.WorldPos().xyz, fAmbientOcclusion,
                                  r.cShadowColor, fSpecularity, r.cLightingRegionMap, vHalfVecWS);
        r.cLightDiffuse = lr.cLightDiffuse + vertOut.Diffuse();
        r.cLightSpecular = lr.cLightSpecular + vertOut.Specular();
    }
    #else
    // VERTEXLIGHTING_FULL: precomputed per-vertex diffuse/specular, plus the shadowed
    // vertex contribution scaled by cShadowColor.
    r.cLightDiffuse = vertOut.Diffuse();
    r.cLightSpecular = vertOut.Specular();
    #if b_useShadows
    r.cLightDiffuse += vertOut.ShadowDiffuse() * r.cShadowColor.rgb;
    r.cLightSpecular += vertOut.ShadowSpecular() * r.cShadowColor.rgb;
    #endif
    #endif
    return r;
  #endif
#endif
}
