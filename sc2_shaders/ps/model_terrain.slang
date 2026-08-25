// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026, Fernando Sahmkow
// See LICENSE in the repository root for full terms.
//
// Model-family SHADINGMODE_TERRAIN arm (psterraindoodad.fx : ShadeTerrain) — models
// rendered with terrain textures.  Four sub-paths: solid-colour, height-encode, creep
// (pscreep.fx, in model_creep.slang), and the terrain base.  No reachable terrain perm
// sets b_useParallaxMapping (POM dead) or b_iDebugMode/b_iDebugRenderMode (DebugColor /
// ApplyDebugRenderMode dead) or b_iUseVisualization/b_iBlockVisualization, so those
// branches are omitted.

implementing sc2_shaders;
#include "types/perm_cb.slangh"

// common.fx terrain height range (assumes 16-bit height precision).
#define c_terrainMinHeight (-100.0f)
#define c_terrainMaxHeight (100.0f)
#define c_terrainHeightScale (65535.0f / (c_terrainMaxHeight - c_terrainMinHeight))

#ifndef b_iSolidColor
#define b_iSolidColor 0
#endif
#ifndef b_computeHeight
#define b_computeHeight 0
#endif
#ifndef b_creepEnabled
#define b_creepEnabled 0
#endif
#ifndef b_useNormalMapping
#define b_useNormalMapping 0
#endif
#ifndef b_iUseDynamicAO
#define b_iUseDynamicAO 0
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

// forward: pscreep.fx creep entry (model_creep.slang).
float4 shadeTerrainCreep(ModelPSInput vertOut, inout DeferredOutputs def);

// psterraindoodad.fx : TerrainTextures — the terrain diffuse + normal sample.  POM is
// dead (no reachable perm sets b_useParallaxMapping), so the height-map offset is
// omitted.
void terrainTextures(ModelPSInput vertOut, out float2 vUV, out float4 cDiffuse, out float3 vNormal)
{
    vUV = ndc2Tex(vertOut.TerrainUV().xy);
    cDiffuse = p_sTerrainDiffuse_t.Sample(p_sSamp, vUV);
#if b_useNormalMapping
    float4 vNormalSample = p_sTerrainNormal_t.Sample(p_sSamp, vUV);
    vNormal = vNormalSample.xyz * 2.0f - 1.0f;
#else
    vNormal = vertOut.Normal().xyz;
#endif
}

// psterraindoodad.fx : ShadeTerrain_TerrainBase.
float4 shadeTerrainBase(ModelPSInput vertOut, inout DeferredOutputs def)
{
    float2 vUV;
    float4 cDiffuse;
    float3 vNormal;
    terrainTextures(vertOut, vUV, cDiffuse, vNormal);

    float3 vWorldNormal;
#if b_useNormalMapping
    vWorldNormal = tangentToWorld(vNormal, vertOut.Normal().xyz,
                                  vertOut.Tangent().xyz, vertOut.Binormal());
#else
    vWorldNormal = vNormal;
#endif

    float fAO = 1.0f;
#if b_iUseDynamicAO
    fAO = saturate(p_sSSAO_t.Sample(p_sSamp, getBackBufferUV(vertOut)).a);
#endif

    // Terrain feeds its own soft-shadow noise UV (TerrainUV.xy * 512) and uses the
    // material specular exponent p_vSpecColorSpecularity.w.
    float2 softShadowUV = vertOut.TerrainUV().xy * 512.0f;
    LightResult light = mainLighting(vertOut, vWorldNormal, psGlobals.vSpecColorSpecularity.w,
                                     fAO, softShadowUV);

    def.vNormal = vWorldNormal;
    def.cDiffuse = cDiffuse.xyz;
    float fSpec = cDiffuse.w
        * psGlobals.vSpecularMultiplier_DepthBlendThreshold_HeightMapScale_AlphaFactor.x;
    def.cSpecular = float3(fSpec, fSpec, fSpec);

    float4 cResult;
    cResult.xyz = light.cLightDiffuse * cDiffuse.rgb + light.cLightSpecular * cDiffuse.a;
    cResult.w = 1.0f;

    [branch] if (SC2_AXIS(b_useVertexColors))
        cResult.rgb *= saturate(vertOut.VertexColor().xyz);

#if b_iTerrainAlphaMaskLayerEnable
    {
        TriPlanarUVs tp = initTriPlanarUVs(vertOut);
        SLayerConfig cfg = setupTerrainAlphaMaskLayer();
        float4 cColor = getLayerColor2D(vertOut, cfg, p_sTerrainAlphaMaskSampler_t,
                                        p_sSamp, 1.0f, def.vNormal, tp);
        cResult.w *= cColor.w;
    }
#endif

    // ApplyDebugRenderMode is the identity here (debug axes are always off).
    // Fog (b_iDebugMode == NO_DEBUG always) — applyFog preserves .a.
    cResult = applyFog(vertOut, cResult, 1.0f, light.cLightingRegionMap);

#if b_emitFinalColor
    cResult = applyFogOfWar(ndc2Tex(vertOut.FOWUV().xy), cResult, def.cDiffuse);
#endif

    cResult *= psGlobals.vTerrainTint;
    return cResult;
}

// psterraindoodad.fx : ShadeTerrain — the four sub-paths, gated on the compile-time
// axes (the .fx if-ladder in solid > height > creep > base order).
float4 shadeTerrain(ModelPSInput vertOut, inout DeferredOutputs def)
{
    // ShadeTerrain re-inits the g-buffer to 1 (psterraindoodad.fx:172-175).
    def.vNormal = float3(1, 1, 1);
    def.cDiffuse = float3(1, 1, 1);
    def.cSpecular = float3(1, 1, 1);
    def.cSpecularPower = 1.0f;

    [branch] if (SC2_AXIS(b_iSolidColor))
    {
        def.vNormal = normalize(vertOut.Normal().xyz);
        return float4(1, 0, 0, 1);
    }
    [branch] if (SC2_AXIS(b_computeHeight))
    {
        float h = (vertOut.TerrainUV().x - c_terrainMinHeight) * c_terrainHeightScale;
        // Output format assumed A8R8G8B8.
        float g = floor(h / 256.0f);
        float b = floor(h - g * 256.0f);
        return float4(0.0f, g / 255.0f, b / 255.0f, 1.0f);
    }
    [branch] if (SC2_AXIS(b_creepEnabled))
        return shadeTerrainCreep(vertOut, def);
    return shadeTerrainBase(vertOut, def);
}
