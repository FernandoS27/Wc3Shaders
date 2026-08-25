// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026, Fernando Sahmkow
// See LICENSE in the repository root for full terms.
//
// Model-family creep (pscreep.fx) — the dominant terrain sub-path (b_creepEnabled).
// PixelCreepGeneric samples the creep textures (planar or tri-planar), blends the
// interior/edge normals, lights, and produces one of four output modes; three pass
// wrappers (opaque / translucent / additive) post-process the result.  DebugColor is
// dead (no reachable perm sets b_iDebugMode), so it is omitted.

implementing sc2_shaders;
#include "types/perm_cb.slangh"

#define CREEP_OUTPUT_DIFFUSE_ONLY 0
#define CREEP_OUTPUT_NORMALS_ONLY 1
#define CREEP_OUTPUT_COMPLETE     2
#define CREEP_OUTPUT_STENCIL_ONLY 3
#define CREEP_OPAQUE      0
#define CREEP_TRANSLUCENT 1
#define CREEP_ADDITIVE_COLOR 2

#ifndef b_iCreepPass
#define b_iCreepPass 0
#endif
#ifndef b_iCreepOutputMode
#define b_iCreepOutputMode 0
#endif
#ifndef b_iTriPlanarCreep
#define b_iTriPlanarCreep 0
#endif
#ifndef b_iCreepUseGroundNormalMap
#define b_iCreepUseGroundNormalMap 0
#endif
#ifndef b_useNormalMapping
#define b_useNormalMapping 0
#endif
#ifndef b_useVertexColors
#define b_useVertexColors 0
#endif
#ifndef b_iTerrainAlphaMaskLayerEnable
#define b_iTerrainAlphaMaskLayerEnable 0
#endif
#ifndef b_emitFinalColor
#define b_emitFinalColor 0
#endif

// common.fx : DecodeNormal(HALF3) — the raw (v*2-1) decode, distinct from the material
// DecodeNormal(vertOut).
float3 decodeNormalRaw(float3 vNormal)
{
    return vNormal * 2.0f - 1.0f;
}

// pscreep.fx : GetCreepTextureUV — three tri-planar UV sets, or the planar TerrainUV.zw.
void getCreepTextureUV(ModelPSInput vertOut, float2 tiling, out float2 uvs[3])
{
    [branch] if (SC2_AXIS(b_iTriPlanarCreep))
    {
        float3 vScaledPos = vertOut.WorldPos().xyz * psGlobals.vCreepTriPlanarScale;
        float3 vNormal = vertOut.Normal().xyz;
        float3 s = sign(vNormal);
        uvs[0] = vScaledPos.xy * float2(s.z, -1.0f) * tiling;
        uvs[1] = vScaledPos.xz * float2(-s.y, -1.0f) * tiling.x;
        uvs[2] = vScaledPos.yz * float2(s.x, -1.0f) * tiling.yx;
        // Non-BSL: rotate each UV by the 2x2 built from p_vCreepTextureRotation.
        uvs[0] = mul(float2x2(psGlobals.vCreepTextureRotation), uvs[0]);
        uvs[1] = mul(float2x2(psGlobals.vCreepTextureRotation), uvs[1]);
        uvs[2] = mul(float2x2(psGlobals.vCreepTextureRotation), uvs[2]);
    }
    else
    {
        uvs[0] = vertOut.TerrainUV().zw * tiling;
        uvs[1] = float2(0, 0);
        uvs[2] = float2(0, 0);
    }
}

// pscreep.fx : SampleCreepTexture.
float4 sampleCreepTexture(Texture2D<float4> tCreep, SamplerState sCreep, float2 uvs[3],
                          ModelPSInput vertOut)
{
    [branch] if (SC2_AXIS(b_iTriPlanarCreep))
        return sampleTriPlanarTexture(tCreep, sCreep, uvs[0], uvs[1], uvs[2],
                                      vertOut.TriPlanarWeights().xyz);
    return tCreep.Sample(sCreep, uvs[0].xy);
}

// pscreep.fx : SampleGroundTexture.
float4 sampleGroundTexture(Texture2D<float4> tGround, SamplerState sGround, ModelPSInput vertOut)
{
    float2 vUV = mul(float4(vertOut.WorldPos().xyz, 1.0f), psGlobals.mCreepGroundTextureMatrix).xy;
    vUV = ndc2Tex(vUV);
    return tGround.Sample(sGround, vUV);
}

// pscreep.fx : PixelCreepGeneric.
float4 pixelCreepGeneric(ModelPSInput vertOut, bool useSpecular, out float4 cLightingRegionMap,
                         inout DeferredOutputs def)
{
    float4 output;
    float4 coverageUV = float4(vertOut.TerrainUV().xy, 0, 0);

    float2 baseUV[3], normalUV[3], edgeUV[3];
    getCreepTextureUV(vertOut, psGlobals.vCreepBaseTextureTiling, baseUV);
    getCreepTextureUV(vertOut, psGlobals.vCreepBaseNormalMapTiling, normalUV);
    getCreepTextureUV(vertOut, psGlobals.vCreepEdgeNormalMapTiling, edgeUV);

    // Optional material-system coverage mask.
    float4 creepGameCoverageSample = float4(1, 1, 1, 1);
#if b_iTerrainAlphaMaskLayerEnable
    {
        TriPlanarUVs tp = initTriPlanarUVs(vertOut);
        SLayerConfig cfg = setupTerrainAlphaMaskLayer();
        creepGameCoverageSample = getLayerColor2D(vertOut, cfg, p_sTerrainAlphaMaskSampler_t,
                                                  p_sSamp, 1.0f, def.vNormal, tp);
    }
#endif
    creepGameCoverageSample *= p_sCreepGameCoverage_t.Sample(p_sSamp, coverageUV.xy);

    float4 diffuseColorSample = sampleCreepTexture(p_sCreepDiffuseColor_t, p_sSamp, baseUV, vertOut);
    diffuseColorSample.xyz *= psGlobals.cCreepDiffuseTintColor.xyz;
    creepGameCoverageSample.a *= diffuseColorSample.w;

    float4 edgeNormalMapAndAlphaMultiplierSample =
        sampleCreepTexture(p_sCreepEdgeNormalMapAndAlphaMultiplier_t, p_sSamp, edgeUV, vertOut);
    edgeNormalMapAndAlphaMultiplierSample.xyz = decodeNormalRaw(edgeNormalMapAndAlphaMultiplierSample.xyz);

    // Height edge factor (tri-planar) / slope factor (planar).
    float heightFactor = 1.0f;
    float slopeFactor = 1.0f;
    [branch] if (SC2_AXIS(b_iTriPlanarCreep))
        heightFactor = 1.0f - saturate((vertOut.WorldPos().z - psGlobals.vCreepHeightRamp.x)
                                       * psGlobals.vCreepHeightRamp.y);
    else
        slopeFactor = vertOut.FOWUV().z;

    float edgeFactor = saturate((creepGameCoverageSample.a - psGlobals.vEdgeNormalMapRampConstants.x)
                                * psGlobals.vEdgeNormalMapRampConstants.y);

    float3 worldNormal;
    float3 pixelNormalSample;
#if b_useNormalMapping
    pixelNormalSample = sampleCreepTexture(p_sCreepInteriorNormalMap_t, p_sSamp, normalUV, vertOut).xyz;
    pixelNormalSample.xyz = decodeNormalRaw(pixelNormalSample.xyz);
    [branch] if (SC2_AXIS(b_iCreepUseGroundNormalMap))
    {
        float2 groundNormalSample = decodeNormalRaw(
            sampleGroundTexture(p_sCreepGroundNormalMap_t, p_sSamp, vertOut).xyy).xy;
        pixelNormalSample.xy = pixelNormalSample.xy + groundNormalSample * psGlobals.vfCreepGroundNormalBlend;
    }
    pixelNormalSample.xyz = normalize(lerp(edgeNormalMapAndAlphaMultiplierSample.xyz,
                                           pixelNormalSample.xyz, edgeFactor * heightFactor));
    [branch] if (SC2_AXIS(b_iCreepOutputMode) == CREEP_OUTPUT_NORMALS_ONLY)
        worldNormal = (pixelNormalSample * 0.5f) + 0.5f;
    else
        worldNormal = tangentToWorld(pixelNormalSample, vertOut.Normal().xyz,
                                     vertOut.Tangent().xyz, vertOut.Binormal());
#else
    pixelNormalSample.xyz = vertOut.Normal().xyz;
    worldNormal = vertOut.Normal().xyz;
#endif

    // Lighting (creep feeds its own soft-shadow noise UV = baseUV[0]*128).
    LightResult light = mainLighting(vertOut, worldNormal, psGlobals.vSpecColorSpecularity.w,
                                     1.0f, baseUV[0].xy * 128.0f);
    cLightingRegionMap = light.cLightingRegionMap;
    float3 diff = light.cLightDiffuse;
    float3 spec = light.cLightSpecular;

    const int iCreepOutputMode = SC2_AXIS(b_iCreepOutputMode);
    [branch] if (iCreepOutputMode == CREEP_OUTPUT_COMPLETE)
    {
        output.xyz = diff * diffuseColorSample.xyz;
        [branch] if (SC2_AXIS(b_useVertexColors))
            output.rgb *= saturate(vertOut.VertexColor().xyz);
    }
    else [branch] if (iCreepOutputMode == CREEP_OUTPUT_DIFFUSE_ONLY)
        output.xyz = diffuseColorSample.xyz;
    else [branch] if (iCreepOutputMode == CREEP_OUTPUT_NORMALS_ONLY)
        output.xyz = (pixelNormalSample.xyz * 0.5f) + 0.5f;
    else    // CREEP_OUTPUT_STENCIL_ONLY
        output.xyz = float3(1, 1, 1);

    output.w = creepGameCoverageSample.a * edgeNormalMapAndAlphaMultiplierSample.a
             * heightFactor * slopeFactor;

    float3 specularMaskSample = float3(0, 0, 0);
    [branch] if (iCreepOutputMode != CREEP_OUTPUT_NORMALS_ONLY)
    {
        float specularMask = saturate(edgeNormalMapAndAlphaMultiplierSample.a * 5.0f - 0.5f);
        specularMaskSample = sampleCreepTexture(p_sCreepSpecularMask_t, p_sSamp, baseUV, vertOut).xyz
                           * specularMask;
        [branch] if (iCreepOutputMode == CREEP_OUTPUT_COMPLETE)
        {
            if (useSpecular)
                output.xyz += spec * specularMaskSample;
        }
    }

    // Cheesy luminance for the deferred specular power.
    def.cSpecularPower = dot(specularMaskSample, float3(0.3f, 0.59f, 0.11f));
    def.cDiffuse = diffuseColorSample.xyz;
    def.vNormal = worldNormal;
    def.cSpecular = specularMaskSample.xyz;

    // DebugColor is dead (b_iDebugMode always NO_DEBUG).
    return output;
}

// pscreep.fx : PixelCreepOpaque.
float4 pixelCreepOpaque(ModelPSInput vertOut, inout DeferredOutputs def)
{
    float4 cLightingRegionMap;
    float4 output = pixelCreepGeneric(vertOut, true, cLightingRegionMap, def);

    const int iCreepOutputMode = SC2_AXIS(b_iCreepOutputMode);
    [branch] if (iCreepOutputMode == CREEP_OUTPUT_DIFFUSE_ONLY)
        output.a = step(psGlobals.fCreepOpaqueAlphaThreshold, output.a)
                 * (def.cSpecularPower + (1.0f / 255.0f));
    else [branch] if (iCreepOutputMode == CREEP_OUTPUT_STENCIL_ONLY)
    { float s = step(psGlobals.fCreepOpaqueAlphaThreshold, output.a); output = float4(s, s, s, s); }
    else    // NORMALS_ONLY or COMPLETE
        output.a = step(psGlobals.fCreepOpaqueAlphaThreshold, output.a);

    [branch] if (iCreepOutputMode != CREEP_OUTPUT_STENCIL_ONLY
                 && iCreepOutputMode != CREEP_OUTPUT_NORMALS_ONLY)
    {
        output = float4(applyFog(vertOut, output, 1.0f, cLightingRegionMap).xyz, output.a);
    #if b_emitFinalColor
        output = applyFogOfWar(ndc2Tex(vertOut.FOWUV().xy), output, def.cDiffuse);
    #endif
    }
    output *= psGlobals.vTerrainTint;
    return output;
}

// pscreep.fx : PixelCreepTranslucent.
float4 pixelCreepTranslucent(ModelPSInput vertOut, inout DeferredOutputs def)
{
    float4 cLightingRegionMap;
    float4 output = pixelCreepGeneric(vertOut, true, cLightingRegionMap, def);
    output.rgb = (output.rgb * psGlobals.cTransparentTintColor) + psGlobals.cTransparentEmissiveColor;
    output.a = saturate((output.a - psGlobals.vCreepTransparentRampConstants.x)
                        * psGlobals.vCreepTransparentRampConstants.y);

    [branch] if (SC2_AXIS(b_iCreepOutputMode) != CREEP_OUTPUT_NORMALS_ONLY)
    {
        output = float4(applyFog(vertOut, output, 1.0f, cLightingRegionMap).xyz, output.a);
    #if b_emitFinalColor
        output = applyFogOfWar(ndc2Tex(vertOut.FOWUV().xy), output, def.cDiffuse);
    #endif
    }
    output *= psGlobals.vTerrainTint;
    def.cDiffuse = def.cDiffuse * psGlobals.cTransparentTintColor;
    def.cSpecular = def.cSpecular * psGlobals.cTransparentTintColor;
    return output;
}

// pscreep.fx : PixelCreepAdditiveColor.
float4 pixelCreepAdditiveColor(ModelPSInput vertOut, inout DeferredOutputs def)
{
    float4 cLightingRegionMap;
    float4 output = pixelCreepGeneric(vertOut, false, cLightingRegionMap, def);
    output.rgb = psGlobals.cCreepAdditiveColor.rgb;
    output.a = saturate((output.a - psGlobals.vCreepTransparentRampConstants.x)
                        * psGlobals.vCreepTransparentRampConstants.y) * psGlobals.cCreepAdditiveColor.a;
    output *= psGlobals.vTerrainTint;
    def.cDiffuse = def.cDiffuse * psGlobals.cTransparentTintColor;
    def.cSpecular = def.cSpecular * psGlobals.cTransparentTintColor;
    return output;
}

// psterraindoodad.fx : ShadeTerrain_Creep.
float4 shadeTerrainCreep(ModelPSInput vertOut, inout DeferredOutputs def)
{
#if (b_iCreepPass == CREEP_OPAQUE)
    return pixelCreepOpaque(vertOut, def);
#elif (b_iCreepPass == CREEP_TRANSLUCENT)
    return pixelCreepTranslucent(vertOut, def);
#else
    return pixelCreepAdditiveColor(vertOut, def);
#endif
}
