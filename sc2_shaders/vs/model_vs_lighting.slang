// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026, Fernando Sahmkow
// See LICENSE in the repository root for full terms.
//
// Model-family VERTEX lighting — lighting.fx's `Lighting()` at `isPerPixel = false`,
// plus vslighting.fx's four Emit* wrappers that turn it into the Diffuse /
// Specular / ShadowDiffuse / ShadowSpecular varyings.
//
// This is a separate transcription from ps/model_lighting.slang on purpose: that
// one is specialised to `isPerPixel = true, mode = ALL_LIGHTS`, which flips every
// branch in `Lighting()`.  Here `isPerPixel` is false, so
//
//   * bShadowLight       = mode != EXCLUDE_SHADOWLIGHT && vlm == FULL
//   * bSecondaryLights   = mode != SHADOWLIGHT_ONLY && count > 1 && vlm == FULL
//   * the ambient term is skipped for vlm == PARTIAL
//   * the ambient-environment cube is skipped entirely (the VS samples no textures)
//   * single-pass point/spot lights run under VERTEX_SINGLEPASS_LIGHTING_MODE
//     (not PIXEL_), and with useSpecular = isPerPixel = false
//
// and `mode` differs per call site (EXCLUDE_SHADOWLIGHT / ALL_LIGHTS /
// SHADOWLIGHT_ONLY), so it is a generic parameter and every arm folds away.
//
// KajiyaKay is pixel-only: lightingmodels.fx's non-PIXEL_SHADER `KajiyaKayShading`
// tail-calls `BlinnShading`, so both `b_iLightingModel` arms are Blinn in a vertex
// shader and `LightShading` collapses to it.
//
// Every loop is unrolled with CONSTANT indices — a dynamic index into a local
// array makes slangc emit a `dcl_immediateConstantBuffer` the dxbc-interp parser
// cannot read (the M1 lesson, docs/SC2_SHADERS_IMPLEMENTATION.md).

implementing sc2_shaders;

#include "vs/model_vs_defines.slangh"
#include "types/perm_cb.slangh"

// Axes read as CODE need a file-local default: only NONZERO axes are passed
// as -D, so a zero one would otherwise be an undefined identifier.
#ifndef b_UseLightingRegions
#define b_UseLightingRegions 0
#endif
#ifndef b_UseVertexBasedAmbientOcclusion
#define b_UseVertexBasedAmbientOcclusion 0
#endif
#ifndef b_iDirectionalLightCount
#define b_iDirectionalLightCount 0
#endif
#ifndef b_iUseSpecular
#define b_iUseSpecular 0
#endif
#ifndef b_iVertexLightingMode
#define b_iVertexLightingMode 0
#endif
#ifndef b_useDblLambert
#define b_useDblLambert 0
#endif
#ifndef b_useLighting
#define b_useLighting 0
#endif

// ---------------------------------------------------------------------------
// lightingmodels.fx : BlinnShading  (the only model reachable from a VS)
// ---------------------------------------------------------------------------
// Returns diffuse in .x, specular in .y, the second (hair) specular in .z.
float3 vsBlinnShading(float3 vNormalWS, float3 vLightDirWS, bool bUseSpecular,
                      bool bSpecifyHalfVector, float3 vHalfVectorWS,
                      float3 vVertexPosWS, float3 vLightPosWS, float fSpecularPower)
{
    float3 vLightDiffuseSpecular;
    vLightDiffuseSpecular.x = dot(vNormalWS, vLightDirWS);
    if (SC2_AXIS(b_useDblLambert))
        vLightDiffuseSpecular.x = -1.0f + vLightDiffuseSpecular.x * 2.0f;

    if (bUseSpecular)
    {
        if (!bSpecifyHalfVector)
            vHalfVectorWS = vsComputeHalfVector(vVertexPosWS,
                                                normalize(vVertexPosWS - vLightPosWS));
        vLightDiffuseSpecular.y = saturate(dot(vNormalWS, vHalfVectorWS));
        vLightDiffuseSpecular.y = pow(vLightDiffuseSpecular.y, fSpecularPower);
        vLightDiffuseSpecular.y = saturate(vLightDiffuseSpecular.y
                                           * sign(vLightDiffuseSpecular.x));
    }
    else
    {
        vLightDiffuseSpecular.y = 0;
    }

    vLightDiffuseSpecular.x = saturate(vLightDiffuseSpecular.x);
    vLightDiffuseSpecular.z = 0;
    return vLightDiffuseSpecular;
}

// lightingmodels.fx : LightShading — both b_iLightingModel arms reach Blinn in a
// vertex shader (KajiyaKay's non-PIXEL_SHADER definition forwards to it).
float3 vsLightShading(float3 vNormalWS, float3 vLightDirWS, bool bUseSpecular,
                      bool bSpecifyHalfVector, float3 vHalfVectorWS,
                      float3 vVertexPosWS, float3 vLightPosWS, float fSpecularPower)
{
    if (b_iLightingModel == VS_BLINN_SHADING || b_iLightingModel == VS_KAJIYAKAY_SHADING)
        return vsBlinnShading(vNormalWS, vLightDirWS, bUseSpecular, bSpecifyHalfVector,
                              vHalfVectorWS, vVertexPosWS, vLightPosWS, fSpecularPower);
    else
        return float3(0, 0, 0);
}

// ---------------------------------------------------------------------------
// lighttypes.fx
// ---------------------------------------------------------------------------
float vsComputeLightAttenuation(float fAttenMultiplier, float fRecipRange, float fRecipDist)
{
    float fAttenuation = 1.0f - saturate(fRecipRange / fRecipDist);
    fAttenuation = saturate(fAttenuation * fAttenMultiplier);
    fAttenuation = fAttenuation * fAttenuation;
    return fAttenuation;
}

float3 vsComputeDirectionalLight(float3 vNormalWS, float3 vLightDirWS,
                                 float3 vLightHalfVecWS, bool bUseSpecular,
                                 float fSpecularity)
{
    return vsLightShading(vNormalWS, -vLightDirWS, bUseSpecular, true, vLightHalfVecWS,
                          float3(0, 0, 0), float3(0, 0, 0), fSpecularity);
}

float3 vsComputePointLight(float3 vNormalWS, float3 vVertexPosWS, float3 vLightPosWS,
                           float fRecipRange, float fAttenMultiplier, bool bUseSpecular,
                           float fSpecularPower)
{
    float3 vLightDir = vLightPosWS - vVertexPosWS;
    float fRecipDist = rsqrt(dot(vLightDir, vLightDir));   // reuse the squared distance
    vLightDir *= fRecipDist;

    float3 ds = vsLightShading(vNormalWS, vLightDir, bUseSpecular, false, float3(0, 0, 0),
                               vVertexPosWS, vLightPosWS, fSpecularPower);
    return ds * vsComputeLightAttenuation(fAttenMultiplier, fRecipRange, fRecipDist);
}

float3 vsComputeSpotLight(float3 vNormalWS, float3 vVertexPosWS, float3 vLightPosWS,
                          float3 vSpotDir, float fRecipRange, float fAttenMultiplier,
                          float fFalloffBias, float fFalloffScale,
                          float fHotspotMultiplier, bool bUseSpecular,
                          float fSpecularPower)
{
    float3 vLightDir = vLightPosWS - vVertexPosWS;
    float fRecipDist = rsqrt(dot(vLightDir, vLightDir));
    vLightDir *= fRecipDist;

    float3 ds = vsLightShading(vNormalWS, vLightDir, bUseSpecular, false, float3(0, 0, 0),
                               vVertexPosWS, vLightPosWS, fSpecularPower);

    float fFalloff = saturate(dot(-vLightDir, vSpotDir));                 // dot product
    fFalloff = saturate((fFalloff + fFalloffBias) * fFalloffScale);       // mad_sat
    fFalloff = saturate(fFalloff * fHotspotMultiplier);                   // mul_sat
    fFalloff = fFalloff * fFalloff;

    return ds * vsComputeLightAttenuation(fAttenMultiplier, fRecipRange, fRecipDist)
              * fFalloff;
}

// ---------------------------------------------------------------------------
// lighting.fx : BlendLightSetup / AddLightContribution
// ---------------------------------------------------------------------------
void vsBlendLightSetup(LightSetup ls, float fraction, inout float3 ambient,
                       inout float3 specular, inout float3 light0, inout float3 light1,
                       inout float3 light2)
{
    ambient  += ls.cAmbientIntensity * fraction;
    specular += ls.cLight0Specular * fraction;
    light0   += ls.cLight0Diffuse * fraction;
    light1   += ls.cLight1Diffuse * fraction;
    light2   += ls.cLight2Diffuse * fraction;
}

void vsAddLightContribution(bool useSpecular, float3 cLightColor,
                            float3 cLightDiffuseSpecular, float4 cShadowColor,
                            float3 cSpecular, inout float3 curTotalDiffuse,
                            inout float3 curTotalSpecular, inout float3 curTotalSpecular2)
{
    // Colour by light (the `if (b_useShadows)` guards are commented out upstream —
    // the shadow colour always multiplies in).
    float3 cLightDiffuseColor = cLightColor * cLightDiffuseSpecular.x;
    cLightDiffuseColor *= cShadowColor.rgb;
    curTotalDiffuse += cLightDiffuseColor;

    if (useSpecular)
    {
        float3 cLightSpecularColor = cSpecular * cLightDiffuseSpecular.y;
        cLightSpecularColor *= cShadowColor.rgb;
        curTotalSpecular += cLightSpecularColor;

        float3 cLightSpecularColor2 = cSpecular * cLightDiffuseSpecular.z;
        cLightSpecularColor2 *= cShadowColor.rgb;
        curTotalSpecular2 += cLightSpecularColor2;
    }
}

// ---------------------------------------------------------------------------
// lighting.fx : Lighting  (isPerPixel == false)
// ---------------------------------------------------------------------------
void vsLighting<let MODE : int>(float3 vPositionWS, float3 vNormalWS,
                                float4 cShadowColor, float fAmbientOcclusion,
                                float fSpecularity, float4 vLightingRegions,
                                out float3 cTotalDiffuse, out float3 cTotalSpecular,
                                out float3 cTotalSpecular2, float3 vHalfVecWS)
{
    float3 curTotalDiffuse = float3(0, 0, 0);
    float3 curTotalSpecular = float3(0, 0, 0);
    float3 curTotalSpecular2 = float3(0, 0, 0);
    float3 cLightDiffuseSpecular = float3(0, 0, 0);

    float3 cAmbient;
    float3 cDirectional0, cDirectional1, cDirectional2;
    float3 cSpecular;

    // isPerPixel == false, so both gates reduce to "vertex lighting is FULL".
    const int iVertexLightingMode = SC2_AXIS(b_iVertexLightingMode);
    const int iDirectionalLightCount = SC2_AXIS(b_iDirectionalLightCount);
    const bool bShadowLight = (MODE != VS_EXCLUDE_SHADOWLIGHT)
                           && (iVertexLightingMode == VS_VERTEXLIGHTING_FULL);
    const bool bSecondaryLights = (MODE != VS_SHADOWLIGHT_ONLY)
                               && (iDirectionalLightCount > 1)
                               && (iVertexLightingMode == VS_VERTEXLIGHTING_FULL);

    if (SC2_AXIS(b_UseLightingRegions) && (bShadowLight || bSecondaryLights))
    {
        cAmbient = float3(0, 0, 0);
        cSpecular = float3(0, 0, 0);
        cDirectional0 = float3(0, 0, 0);
        cDirectional1 = float3(0, 0, 0);
        cDirectional2 = float3(0, 0, 0);
        // Unrolled with constant setup indices (MAX_LIGHT_SETUPS == 4).
        vsBlendLightSetup(vsGlobals.lightSetups[0], vLightingRegions.x, cAmbient,
                          cSpecular, cDirectional0, cDirectional1, cDirectional2);
        vsBlendLightSetup(vsGlobals.lightSetups[1], vLightingRegions.y, cAmbient,
                          cSpecular, cDirectional0, cDirectional1, cDirectional2);
        vsBlendLightSetup(vsGlobals.lightSetups[2], vLightingRegions.z, cAmbient,
                          cSpecular, cDirectional0, cDirectional1, cDirectional2);
        vsBlendLightSetup(vsGlobals.lightSetups[3], vLightingRegions.a, cAmbient,
                          cSpecular, cDirectional0, cDirectional1, cDirectional2);
    }
    else
    {
        cAmbient = vsGlobals.cAmbientColor;
        cSpecular = vsGlobals.vSpecColorSpecularity.rgb;
        cDirectional0 = vsGlobals.directionalLights[0].cColor.rgb;
        cDirectional1 = vsGlobals.directionalLights[1].cColor.rgb;
        cDirectional2 = vsGlobals.directionalLights[2].cColor.rgb;
    }

    // (lighting.fx:172's ambient-environment cube is `isPerPixel &&` gated — a
    // vertex shader never samples it.)

    if (bShadowLight)
    {
        // Key shadow light.
        cLightDiffuseSpecular = vsComputeDirectionalLight(
            vNormalWS, vsGlobals.directionalLights[0].vDirection.xyz, vHalfVecWS,
            SC2_AXIS(b_iUseSpecular) != 0, fSpecularity);
        vsAddLightContribution(true, cDirectional0, cLightDiffuseSpecular, cShadowColor,
                               cSpecular, curTotalDiffuse, curTotalSpecular,
                               curTotalSpecular2);
    }

    if (MODE != VS_SHADOWLIGHT_ONLY)
    {
        // Ambient term (skipped for PARTIAL on the vertex side).
        if (iVertexLightingMode != VS_VERTEXLIGHTING_PARTIAL)
            curTotalDiffuse += cAmbient * fAmbientOcclusion;

        // Secondary directional lights, i = 1 .. count-1.  AddLightContribution's
        // shadow argument here is the scalar fAmbientOcclusion, broadcast.
        if (bSecondaryLights)
        {
            cLightDiffuseSpecular = vsComputeDirectionalLight(
                vNormalWS, vsGlobals.directionalLights[1].vDirection.xyz, vHalfVecWS,
                false, fSpecularity);
            vsAddLightContribution(false, cDirectional1, cLightDiffuseSpecular,
                                   float4(fAmbientOcclusion, fAmbientOcclusion,
                                          fAmbientOcclusion, fAmbientOcclusion),
                                   cSpecular, curTotalDiffuse, curTotalSpecular,
                                   curTotalSpecular2);
            [branch] if (iDirectionalLightCount > 2)
            {
                cLightDiffuseSpecular = vsComputeDirectionalLight(
                    vNormalWS, vsGlobals.directionalLights[2].vDirection.xyz, vHalfVecWS,
                    false, fSpecularity);
                vsAddLightContribution(false, cDirectional2, cLightDiffuseSpecular,
                                       float4(fAmbientOcclusion, fAmbientOcclusion,
                                              fAmbientOcclusion, fAmbientOcclusion),
                                       cSpecular, curTotalDiffuse, curTotalSpecular,
                                       curTotalSpecular2);
            }
        }

        // Single-pass point + spot lights — the VERTEX variant of the mode gate.
        // useSpecular == isPerPixel == false, and the shadow argument is 1.
#if (b_iLightingMode == VS_VERTEX_SINGLEPASS_LIGHTING_MODE)
        {
            const float4 kNoShadow = float4(1, 1, 1, 1);
#define VS_ACCUM_POINT(i)                                                            \
            cLightDiffuseSpecular = vsComputePointLight(                             \
                vNormalWS, vPositionWS, vsGlobals.pointLights[i].vPositionRecipRadius.xyz, \
                vsGlobals.pointLights[i].vPositionRecipRadius.w,                     \
                vsGlobals.pointLights[i].cColorAttenMultiplier.a, false, fSpecularity); \
            vsAddLightContribution(false,                                            \
                vsGlobals.pointLights[i].cColorAttenMultiplier.rgb,                  \
                cLightDiffuseSpecular, kNoShadow,                                    \
                vsGlobals.pointLights[i].cSpecular.rgb,                              \
                curTotalDiffuse, curTotalSpecular, curTotalSpecular2)

            VS_ACCUM_POINT(0);
            VS_ACCUM_POINT(1);
            VS_ACCUM_POINT(2);
#undef VS_ACCUM_POINT

#define VS_ACCUM_SPOT(i)                                                             \
            cLightDiffuseSpecular = vsComputeSpotLight(                              \
                vNormalWS, vPositionWS, vsGlobals.spotLights[i].vPositionRecipRadius.xyz, \
                vsGlobals.spotLights[i].vDirection.xyz,                              \
                vsGlobals.spotLights[i].vPositionRecipRadius.w,                      \
                vsGlobals.spotLights[i].cColorAttenMultiplier.a,                     \
                vsGlobals.spotLights[i].vSpotFalloffBiasScaleAndHotSpotMultiplier.x, \
                vsGlobals.spotLights[i].vSpotFalloffBiasScaleAndHotSpotMultiplier.y, \
                vsGlobals.spotLights[i].vSpotFalloffBiasScaleAndHotSpotMultiplier.z, \
                false, fSpecularity);                                                \
            vsAddLightContribution(false,                                            \
                vsGlobals.spotLights[i].cColorAttenMultiplier.rgb,                   \
                cLightDiffuseSpecular, kNoShadow,                                    \
                vsGlobals.spotLights[i].cSpecular.rgb,                               \
                curTotalDiffuse, curTotalSpecular, curTotalSpecular2)

            VS_ACCUM_SPOT(0);
            VS_ACCUM_SPOT(1);
            VS_ACCUM_SPOT(2);
            VS_ACCUM_SPOT(3);
            VS_ACCUM_SPOT(4);
            VS_ACCUM_SPOT(5);
            VS_ACCUM_SPOT(6);
#undef VS_ACCUM_SPOT
        }
#endif
    }

    cTotalDiffuse = curTotalDiffuse;
    cTotalSpecular = curTotalSpecular;
    cTotalSpecular2 = curTotalSpecular2;
}

// ---------------------------------------------------------------------------
// vslighting.fx : the four vertex-lighting emitters
// ---------------------------------------------------------------------------

// Vertex-stage diffuse lighting.  `vertOut` is passed so the vertex-AO variant can
// read back INTERPOLANT_VectorUI1 (written earlier in the body, as in the original).
float3 vsEmitDiffuse(DefaultVSOutput vertOut, float3 vPosition, float3 vNormalWS,
                     float4 vLightingRegions, inout float3 cVertexSpecularLighting,
                     inout float3 cVertexShadowSpecularLighting,
                     inout float3 cVertexShadowSpecularLighting2)
{
    cVertexShadowSpecularLighting = float3(0, 0, 0);
    float3 cVertexDiffuseLighting;
    if (SC2_AXIS(b_useLighting) == 0)
    {
        cVertexDiffuseLighting = float3(1, 1, 1);
        cVertexSpecularLighting = float3(0, 0, 0);
        cVertexShadowSpecularLighting2 = float3(0, 0, 0);
    }
    else
    {
        float3 vHalfVecWS = vsComputeHalfVector(vPosition);
#if b_useShadows
        // The shadow-casting key light is emitted separately (EmitShadowDiffuse) so
        // the pixel stage can apply the shadow term to it alone.
        vsLighting<VS_EXCLUDE_SHADOWLIGHT>(
            vPosition, vNormalWS, float4(1, 1, 1, 1), 1.0f,
            vsGlobals.vSpecColorSpecularity.w, vLightingRegions,
            cVertexDiffuseLighting, cVertexSpecularLighting,
            cVertexShadowSpecularLighting2, vHalfVecWS);
#else
        float fAmbientOcclusion;
        if (SC2_AXIS(b_UseVertexBasedAmbientOcclusion))
            fAmbientOcclusion = vertOut.VectorUI1().a;
        else
            fAmbientOcclusion = 1.0f;
        vsLighting<VS_ALL_LIGHTS>(
            vPosition, vNormalWS, float4(1, 1, 1, 1), fAmbientOcclusion,
            vsGlobals.vSpecColorSpecularity.w, vLightingRegions,
            cVertexDiffuseLighting, cVertexSpecularLighting,
            cVertexShadowSpecularLighting2, vHalfVecWS);
#endif
    }
    return cVertexDiffuseLighting;
}

// Vertex-stage specular lighting (pass-through of EmitDiffuse's out value).
float3 vsEmitSpecular(float3 cVertexSpecularLighting)
{
    return cVertexSpecularLighting;
}

// Vertex-stage shadow-light diffuse.
float3 vsEmitShadowDiffuse(float3 vNormalWS, float4 vLightingRegions,
                           inout float3 cVertexShadowSpecularLighting,
                           inout float3 cVertexShadowSpecularLighting2)
{
    float3 cVertexDiffuseLighting;
    if (SC2_AXIS(b_useLighting) == 0)
    {
        cVertexDiffuseLighting = float3(1, 1, 1);
    }
    else
    {
        vsLighting<VS_SHADOWLIGHT_ONLY>(
            float3(0, 0, 0), vNormalWS, float4(1, 1, 1, 1), 1.0f,
            vsGlobals.vSpecColorSpecularity.w, vLightingRegions,
            cVertexDiffuseLighting, cVertexShadowSpecularLighting,
            cVertexShadowSpecularLighting2, float3(0, 0, 0));
    }
    return cVertexDiffuseLighting;
}

float3 vsEmitShadowSpecular(float3 cVertexShadowSpecularLighting)
{
    return cVertexShadowSpecularLighting;
}
