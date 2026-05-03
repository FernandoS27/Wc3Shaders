// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026, Fernando Sahmkow
// See LICENSE in the repository root for full terms.
//
// Foliage pixel shader body.
//
// Pipeline:
//   1. Single-texture material sample (albedo, normal, mask, emissive,
//      lightmap) — no splat layers like terrain.
//   2. Optional TINT_OVERRIDE substitution of albedo + mask UV.
//   3. Decode tangent-space normal (RG-A pack — same as terrain).
//   4. TBN: build the world normal from the blended TS normal and the
//      VS-supplied geometry tangent / bitangent / normal.
//   5. Geometric specular AA: roughness from screen-space normal
//      variance — `math/brdf.specularAA` clamped to 0.18 and √-folded
//      with the mask's smoothed roughness.
//   6. Cook-Torrance partials (F0 = metallic·albedo, diffuse = (1 -
//      metallic)·albedo·(1/π)).
//   7. Cascaded shadow visibility (reuses
//      ps/terrain_ps_shadow.computeTerrainShadow).
//   8. Dominant directional light + IBL combine, multi-light loop —
//      reuses ps/terrain_ps_lighting.evaluate{Dominant,Additional}.
//   9. Compose: π · totalLight + emissive, vertex sRGB→linear tint,
//      AO modulation, optional fog (TFog policy).
//  10. Output: HAS_TINT routes through the debug-viz router; the
//      release path returns the lit colour.
//
// Alpha-test discard and MRT outputs are handled in the entry shader
// after the body returns its result.

implementing wc3_shaders;

public struct FoliageShadeResult
{
    public float3 color;
    public float  alpha;             // pre-multiplied vertColor.w · texAlpha
    public float  texAlpha;          // raw albedo.w (used for ALPHA_TEST recompute)
    public float3 worldNormal;       // for MRT normal pack
    public float  viewZ;             // for MRT depth feedback
};

public FoliageShadeResult foliagePSBody<
    TFog            : IFogMode,
    let HAS_SHADOWS : bool,
    let HAS_TINT    : bool>(FoliagePSInput input)
{
    FoliageShadeResult out_;

    // ---- Material sample ----
    float4 albedo   = t_foliageAlbedo  .Sample(s_foliageAlbedo,   input.texCoord.xy);
    float3 nSample  = t_foliageNormal  .Sample(s_foliageNormal,   input.texCoord.xy).xyw;
    float2 mask     = t_foliageMask    .Sample(s_foliageMask,     input.texCoord.xy).yz;
    float3 emissive = t_foliageEmissive.Sample(s_foliageEmissive, input.texCoord.xy).xyz;
    float  occRaw   = t_foliageLightmap.Sample(s_foliageLightmap, input.texCoord.zw).w;

    // Save raw albedo alpha — the entry shader needs it for
    // ALPHA_TEST's `vertColor.w · texAlpha < alphaRef` recompute,
    // which has to use the un-multiplied value.
    out_.texAlpha = albedo.w;

    // ---- Optional TINT_OVERRIDE substitution ----
    if (HAS_TINT)
    {
        uint2 tintBits = asuint(psCB3.enabledShaders.xx) & uint2(1, 2);
        albedo.xyz = (tintBits.x != 0) ? psCB3.overrideAlbedo : albedo.xyz;
        mask       = (tintBits.y != 0) ? psCB3.overrideOrm.yz : mask;
    }

    // ---- Decode TS normal (RG-A pack — same as terrain) ----
    float3 tsN = decodeSplatNormal(nSample);

    // ---- TBN ----
    // Bitangent = cross(geomNormal, geomTangent) · handedness.
    float3 tangentWS = input.worldTangent.xyz;
    float3 normalWS  = input.worldNormal;
    float  hSign     = input.worldTangent.w;
    float3 bitangent = normalWS.yzx * tangentWS.zxy - tangentWS.yzx * normalWS.zxy;
    bitangent = bitangent * hSign;

    float3 worldN = tsN.x * tangentWS + tsN.y * bitangent + tsN.z * normalWS;
    worldN = worldN * rsqrt(dot(worldN, worldN));

    // ---- Mask channels: roughness (.x) and metallic (.y) ----
    // .x carries the raw roughness, biased into the [1/256, 1] range
    // (the engine-fixed half-texel quantization step that matches the
    // mask LUT's storage convention). .y carries the F0 / albedo
    // weight (treated as a pseudo-metallic factor).
    float roughnessRaw = mask.x * 0.99609375 + 0.00390625;
    float metallic     = mask.y;

    // ---- Specular AA: screen-space normal variance → ndfAlpha ----
    // Foliage has no splat-layer pre-smoothing, so it does its own
    // variance inflation here. Shared helper in math/brdf.slang.
    float ndfAlphaSmooth = specularAA(roughnessRaw, worldN);
    float roughness = (asuint(psCB2.useNdf) != 0) ? ndfAlphaSmooth : roughnessRaw;

    // ---- View dir ----
    float3 viewPos = input.worldPos.xyz;
    float3 viewDir = -viewPos * rsqrt(dot(-viewPos, -viewPos));

    // ---- Cook-Torrance partials ----
    float3 F0            = metallic.xxx * albedo.xyz;
    float3 oneMinus      = -metallic.xxx * albedo.xyz + albedo.xyz;
    float3 diffuseScaled = INV_PI.xxx * oneMinus;

    // ---- Shadow visibility ----
    float shadowVis = 1.0;
    if (HAS_SHADOWS)
    {
        shadowVis = computeTerrainShadow(
            input.shadowClip0, input.shadowClip1, input.shadowClip2,
            asuint(sdPsCB1.numCascades));
    }

    float NdotV = dot(worldN, viewDir);

    // ---- Dominant directional light + IBL ----
    // Reuses the terrain helper — shares the same `psCB2` / IBL probe
    // / Blizzard ambient blend convention as the foliage pass.
    TerrainDominantLight dom = evaluateDominantLight(
        viewPos, worldN, viewDir,
        diffuseScaled, F0, roughness, NdotV, shadowVis);

    // ---- Multi-light loop ----
    float3 totalLit = evaluateAdditionalLights(
        viewPos, worldN, viewDir,
        dom.lit,
        diffuseScaled, F0, roughness, NdotV, dom.firstLight);

    // ---- Compose: π · totalLight + emissive, then vertex tint ----
    float3 lit = PI.xxx * totalLit + emissive;
    float3 vertLinear = srgbToLinear(input.color.xyz);
    lit = vertLinear * lit;

    // ---- AO modulation ----
    float ao = 1.0 - occRaw;
    lit = lit * ao;

    // ---- Optional fog ----
    // TFog.apply runs the policy's fog blend toward `psCB2.fogColor`
    // using `viewPos.z` as the depth scalar. Wraps the same Hermite-
    // smoothed falloff curve used by the HD path.
    lit = TFog.apply(lit, psCB2.fogColor.xyz,
                     viewPos.z, psCB2.fogStart, psCB2.fogEnd, psCB2.fogDensity);

    // ---- Output ----
    if (HAS_TINT)
    {
        // Debug-viz router. Mode 0 returns the lit colour; modes 1..8
        // emit one of the engine's dev-mode buffers. Reuses the
        // terrain debug helper — modes / palette / dispatch are
        // identical, the only minor difference is that mode 1 is
        // engine-undefined in both shaders so a fixed value is fine.
        // LOD for mode 3 comes from the foliage albedo. (`queryTextureLOD`
        // in ps/ps_post.slang already handles WGSL fallback.)
        float lodRaw = queryTextureLOD(t_foliageAlbedo, s_foliageAlbedo,
                                       input.texCoord.xy);
        out_.color = terrainDebugVisualize(
            lit, PI.xxx * totalLit, worldN, lodRaw,
            dom.domAmbient, dom.domSpec);
    }
    else
    {
        out_.color = lit;
    }

    out_.alpha       = input.color.w * albedo.w;
    out_.worldNormal = worldN;
    out_.viewZ       = input.worldPos.z;
    return out_;
}
