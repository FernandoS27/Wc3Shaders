// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026, Fernando Sahmkow
// See LICENSE in the repository root for full terms.
//
// SD highspec (classic) vertex shader entry point.
//
// This is the retail SD pipeline: per-vertex Blinn-Phong-ish lighting,
// no shadows, no tangent frame. The VS is the only place where
// lighting is evaluated — the PS (sd_classic_ps.slang) just modulates
// the lit color with up to two fixed-function texture stages.
//
// `NUM_LIGHTS` is a compile-time count so the light loop unrolls; the
// engine instantiates variants for 0..8 active lights.

implementing wc3_shaders;

[shader("vertex")]
public SDVSOutput sd_highspec_vs_main<
    TSkin : ISkinning,
    TVF   : IVertexFormat,
    let NUM_LIGHTS : int
>(VSInput<TVF> input)
{
    SDVSOutput output;

    // Skinning (rigid / four-bone), identical to the HD entry.
    float3 skinnedPos, skinnedNormal, skinnedTangent;
    TSkin.transform(
        input.position, input.normal, input.tangent,
        input.weights, input.boneIndices, vsCB3,
        skinnedPos, skinnedNormal, skinnedTangent);

    // World and clip transforms come from the SD highspec CB (b0)
    // rather than the shared vsCB2 — the SD highspec pipeline has a
    // different constant buffer layout.
    float3 worldPos = transformByWorld(skinnedPos, sdHighspecVsCB0.world);
    output.position = transformClipPos(skinnedPos, sdHighspecVsCB0.worldViewProj);
    output.worldPos = worldPos;

    // Material tint × optional per-vertex color.
    float4 material = sdHighspecVsCB0.diffuseColor;
    if (TVF.hasVertexColor)
        material = material * input.vertColor;

    // When at least one light is active, run the per-vertex light
    // loop and emit its output as the PS color; otherwise emit the
    // raw material so the PS still gets a tint to modulate.
    if (NUM_LIGHTS > 0)
    {
        float3 worldNormal = transformNormal(skinnedNormal, sdHighspecVsCB0.world);
        float3 lit = sdComputePerVertexLighting<NUM_LIGHTS>(
            worldNormal, worldPos, material, sdHighspecVsCB0.lights);
        output.color = float4(lit, material.w);
    }
    else
    {
        output.color = material;
    }

    // UV transforms — same as the HD path.
    output.texCoord = transformUVs<TVF>(
        input.uv0, input.uv1,
        sdHighspecVsCB0.texMtx0Row0, sdHighspecVsCB0.texMtx0Row1,
        sdHighspecVsCB0.texMtx1Row0, sdHighspecVsCB0.texMtx1Row1);

    return output;
}
