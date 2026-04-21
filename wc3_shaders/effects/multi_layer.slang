// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026, Fernando Sahmkow
// See LICENSE in the repository root for full terms.
//
// Team-color multi-layer blend.
//
// Every "team-tintable" asset in HD ships a second team-color albedo
// (a recolored copy of the mesh) plus a blend mask in ORM.w. This file
// implements the engine's bespoke hue-preserving blend between the two
// layers: the base and team albedos are each factored into
// (hue, luma); the hues are lerped by sqrt(w) (softer onset), while the
// luma is driven toward min(baseLuma, teamLuma) by a smoothed w² ramp.
// That formulation keeps team-painted regions from washing out their
// neighbouring details when the mask has soft edges.
//
// The emissive layer gets its own blend against the team hue so hot
// emissive pixels also pick up the team tint proportionally.

implementing wc3_shaders;

// How much the luma drift saturates with the mask weight. 0.95 is the
// authored value from the stock shader — keeps the blended region
// slightly darker than the base to hint at a recess / decal feel.
static const float ML_BLEND_SMOOTH   = 0.949999988;
// Emissive pickup of the team tint. 0.75 so glowing team decals still
// read as glowing rather than as flat team paint.
static const float ML_EMISSIVE_BLEND = 0.75;

public struct MultiLayerResult
{
    public float3 albedo;
    public float3 emissive;
};

public MultiLayerResult multiLayerBlend(
    float3 baseAlbedo, float3 teamAlbedo, float blendWeight,
    float3 baseEmissive)
{
    // Two separate mask shapes: hue uses sqrt(w) (faster onset at low
    // weights), luma uses w² (slower onset → avoids a hard seam at the
    // mask edge where w is just barely > 0).
    float blendFactor = sqrt(blendWeight);
    float smoothBlend = blendWeight * blendWeight * ML_BLEND_SMOOTH;

    // Decompose each albedo into (peak luma, normalised hue).
    float baseLuma = max(max(baseAlbedo.x, baseAlbedo.y), baseAlbedo.z);
    float3 baseHue = baseAlbedo / (baseLuma + SAFE_EPS);

    float teamLuma = max(max(teamAlbedo.x, teamAlbedo.y), teamAlbedo.z);
    float3 teamHue = teamAlbedo / (teamLuma + SAFE_EPS);

    // Hue lerp: swap toward the team hue.
    float3 blendedHue = blendFactor * (teamHue - baseHue) + baseHue;

    // Luma target is the darker of the two — prevents the team paint
    // from ever brightening the base surface.
    float targetLuma = min(teamLuma, baseLuma);
    float finalLuma = smoothBlend * (targetLuma - baseLuma) + baseLuma;

    // Emissive inherits the team hue at reduced strength. The emissive
    // luma itself is preserved so glow intensity is unchanged.
    float emissiveBlendFactor = ML_EMISSIVE_BLEND * blendWeight;
    float emLuma = max(max(baseEmissive.x, baseEmissive.y), baseEmissive.z);
    float3 emHue = baseEmissive / (emLuma + SAFE_EPS);
    float3 blendedEmHue = emissiveBlendFactor * (teamHue - emHue) + emHue;

    MultiLayerResult result;
    result.albedo = blendedHue * finalLuma;
    result.emissive = blendedEmHue * emLuma;
    return result;
}
