// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026, Fernando Sahmkow
// See LICENSE in the repository root for full terms.
//
// Terrain pixel shader main body.
//
// Pipeline:
//   1. Splat layer 0..3 sampling and accumulation (albedo, TS-normal,
//      mask UV) — see ps/terrain_ps_splat.slang.
//   2. TBN: build the world-space normal from the blended splat normal
//      and the geometry tangent / bitangent / normal in the PS input.
//   3. Vertex sRGB → linear (cubic poly) × accumulated albedo.
//   4. Pre-bake the layer-weighted diffuse and the F0/Schlick partials.
//   5. Cascaded shadow visibility (ps/terrain_ps_shadow.slang).
//   6. Dominant directional light + IBL combine
//      (ps/terrain_ps_lighting.slang).
//   7. Multi-light Cook-Torrance loop (same file).
//   8. Output: π scale, AO from t_lightmap.w, and either the post
//      colour matrix (release) or the debug-viz router (TINT_OVERRIDE).
//   9. MRT outputs (.xyz packed normal + view-Z feedback) when the
//      MRT axis is set.

implementing wc3_shaders;

public struct TerrainShadeResult
{
    public float4 color;
    public float3 worldNormal;       // for MRT normal pack
    public float  viewZ;             // for MRT depth feedback
};

public TerrainShadeResult terrainPSBody<
    let HAS_SHADOWS : bool,
    let HAS_TINT    : bool>(
    TerrainPSInput input)
{
    TerrainShadeResult out_;

    // Per-slot mask: a bone-id of 0xFFFF flags the splat layer as
    // inactive; everything inside that layer's `if (active)` block is
    // skipped.
    bool4 active = (int4)input.boneIDs != int4(0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF);

    // ---- Splat accumulators ----
    // After the four-layer blend, `maskUVAccum` carries the two scalar
    // material parameters baked into the splat-mask texture:
    //   .x — layer-blended `mask.z` (no bias) — drives F0 / albedo split
    //         (treated as a pseudo-metallic factor).
    //   .y — layer-blended biased `mask.y` (255/256 expand) — smoothed
    //         roughness consumed by the GGX D term and the IBL mip
    //         selection.
    float3 albedoAccum;
    float3 normalAccum;
    float2 maskUVAccum;

    initSplatAccum<HAS_TINT>(active.x, input.boneIDs.x, input.texCoord.xy,
                             input.color.w,
                             albedoAccum, normalAccum, maskUVAccum);

    lerpSplatLayer<HAS_TINT>(active.y, input.boneIDs.y, input.texCoord.xy,
                             input.color.w,
                             albedoAccum, normalAccum, maskUVAccum);
    lerpSplatLayer<HAS_TINT>(active.z, input.boneIDs.z, input.texCoord.xy,
                             input.color.w,
                             albedoAccum, normalAccum, maskUVAccum);
    lerpSplatLayer<HAS_TINT>(active.w, input.boneIDs.w, input.texCoord.xy,
                             input.color.w,
                             albedoAccum, normalAccum, maskUVAccum);

    // ---- TBN ----
    // Bitangent = cross(geomNormal, geomTangent) · tangent.w (handedness).
    // Build the world normal from the blended TS normal.
    float3 tangentWS = input.worldTangent.xyz;
    float3 normalWS  = input.worldNormal;
    float  hSign     = input.worldTangent.w;
    float3 bitangent = normalWS.yzx * tangentWS.zxy - tangentWS.yzx * normalWS.zxy;
    bitangent = bitangent * hSign;

    float3 worldN = normalAccum.x * tangentWS
                  + normalAccum.y * bitangent
                  + normalAccum.z * normalWS;
    worldN = worldN * rsqrt(dot(worldN, worldN));

    // ---- Vertex sRGB → linear (Hable cubic) × accumulated albedo ----
    float3 vertLinear = srgbToLinear(input.color.xyz);
    float3 albedoLit  = vertLinear * albedoAccum;

    // ---- View dir ----
    float3 viewPos = input.worldPos.xyz;
    float3 viewDir = -viewPos * rsqrt(dot(-viewPos, -viewPos));

    // ---- Cook-Torrance partials (layer-weighted) ----
    // F0       = layerMask · albedo (treated as Schlick base)
    // diffuse  = (1 - layerMask) · albedo · (1/π)
    float metallic = maskUVAccum.x;
    float roughness = maskUVAccum.y;
    float3 F0       = albedoLit * metallic.xxx;
    float3 oneMinus = -albedoLit * metallic.xxx + albedoLit;
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
    TerrainDominantLight dom = evaluateDominantLight(
        viewPos, worldN, viewDir,
        diffuseScaled, F0, roughness,
        NdotV, shadowVis);

    // ---- Multi-light loop ----
    float3 totalLit = evaluateAdditionalLights(
        viewPos, worldN, viewDir,
        dom.lit,
        diffuseScaled, F0, roughness,
        NdotV, dom.firstLight);

    // ---- Output finishing ----
    float3 piScaled = PI.xxx * totalLit;
    float  aoFactor = 1.0 - t_lightmap.Sample(s_lightmap, input.texCoord.zw).w;
    float3 preGrade = piScaled * aoFactor.xxx;

    if (HAS_TINT)
    {
        // TINT_OVERRIDE: bypass the post-grade colour matrix and route
        // through the debug-viz router instead. LOD comes from the
        // splat-color sample (mode 3 visualises this directly). WGSL
        // has no `CalculateLevelOfDetail` intrinsic so we reconstruct
        // the LOD from scaled-UV derivatives in that target.
        float lodRaw;
        __target_switch
        {
        case wgsl:
            {
                uint w, h, slices;
                t_splatColor.GetDimensions(w, h, slices);
                float2 scaledUV = input.texCoord.xy * float2(w, h);
                float2 dxv = ddx(scaledUV);
                float2 dyv = ddy(scaledUV);
                lodRaw = 0.5 * log2(max(dot(dxv, dxv), dot(dyv, dyv)));
                break;
            }
        default:
            lodRaw = t_splatColor.CalculateLevelOfDetail(s_splatColor,
                                                         input.texCoord.xy);
            break;
        }

        float3 viz = terrainDebugVisualize(
            preGrade, piScaled, worldN, lodRaw,
            dom.domAmbient, dom.domSpec);
        out_.color = float4(viz, 1.0);
    }
    else
    {
        // Release path: apply the standard HD colour-grading 3×3.
        out_.color = float4(applyColorMatrix(preGrade), 1.0);
    }

    out_.worldNormal = worldN;
    out_.viewZ       = input.worldPos.z;
    return out_;
}
