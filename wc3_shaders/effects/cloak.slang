// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026, Fernando Sahmkow
// See LICENSE in the repository root for full terms.
//
// "Cloak" overlay used when a unit is in stealth / invisibility.
//
// Visually it blends the lit color toward a pale cyan tint weighted by
// a squared-fresnel edge term, so silhouettes remain visible while flat
// surfaces fade almost completely. The alpha output is modulated by the
// same edge factor so the cloaked mesh punches out of the scene with a
// gently feathered rim.
//
// Two variants are provided:
//   - applyCloak:    single-layer (standard) materials.
//   - applyCloakML:  multi-layer / team-color materials, where the
//                    team-mask blend weight also modulates the cloak
//                    strength so team-tinted regions fade differently
//                    than the base albedo.

implementing wc3_shaders;

static const float3 CLOAK_TINT       = float3(0.312999994, 0.853999972, 0.787999988);
static const float  CLOAK_ALPHA_BIAS = 0.200000003;  // Floor alpha so silhouette stays visible.
static const float  CLOAK_EDGE_SCALE = 0.5;          // Edge vs. flat-area weighting.
static const float  EDGE_DETECT_SCALE = 10.0;        // Sharpness of the silhouette mask.

// Used by the standard cloak path as well as by both depth-prepass
// variants when the cloak amount is non-zero — hence the struct wrapper
// so the return value (color + alpha) can flow through without losing
// the packed shape.
public struct CloakResult
{
    float4 colorAlpha;
};

// Standard cloak: tint the lit color toward the cyan CLOAK_TINT based
// on a squared-fresnel factor, and push alpha toward a feathered edge
// shape. `cloakAmount` is a 0..1 animation knob that ramps the effect
// in and out; at cloakAmount == 0 the output is identical to the input.
public CloakResult applyCloak(
    float3 color, float alpha, float albedoAlpha,
    float NdotV, float cloakAmount, float vertAlpha)
{
    // Squared fresnel: strong at grazing angles (silhouettes) and zero
    // at facing-the-camera.
    float oneMinusNdotV = 1.0 - NdotV;
    float fresnelSq = oneMinusNdotV * oneMinusNdotV;

    // `edgeMask` is 1 in the body of the mesh and tapers to 0 along the
    // silhouette — this is the inverse of fresnelSq scaled up then
    // clamped so the transition band is narrow.
    float edgeMask = min(1.0, (1.0 - fresnelSq) * EDGE_DETECT_SCALE);

    // Blend the per-vertex alpha toward an edge-shaped cloak alpha. The
    // bias keeps a visible silhouette even at full cloak.
    float cloakAlpha = edgeMask * CLOAK_EDGE_SCALE + CLOAK_ALPHA_BIAS;
    cloakAlpha = cloakAmount * (cloakAlpha - vertAlpha) + vertAlpha;
    float finalAlpha = albedoAlpha * cloakAlpha;

    // Color tint is strongest at the silhouette (fresnelSq) and ramps
    // with cloakAmount.
    float3 tinted = cloakAmount * fresnelSq * (CLOAK_TINT - color) + color;

    CloakResult result;
    result.colorAlpha = float4(tinted, finalAlpha);
    return result;
}

// Multi-layer cloak tuning constants: team-color / emissive regions on
// a cloaked unit fade faster in alpha and tint less aggressively in
// color so hero emblems / runes remain legible while stealthed.
static const float ML_ALPHA_RAMP_SCALE = 0.199999988;
static const float ML_ALPHA_RAMP_BIAS  = 0.5;
static const float ML_EMISSIVE_BLEND_CLOAK = 0.75;

// Multi-layer cloak: same pair of outputs as applyCloak, but the alpha
// ramp and the color tint strength both depend on the team-color /
// emissive blend weight so team-painted regions cloak at a different
// rate than the base material.
public CloakResult applyCloakML(
    float3 color, float alpha, float albedoAlpha,
    float NdotV, float cloakAmount, float vertAlpha,
    float blendWeight)
{
    float oneMinusNdotV = 1.0 - NdotV;
    float fresnelSq = oneMinusNdotV * oneMinusNdotV;

    float edgeMask = min(1.0, (1.0 - fresnelSq) * EDGE_DETECT_SCALE);

    // `alphaRamp` replaces the constant CLOAK_EDGE_SCALE: team-colored
    // pixels see a bigger ramp (so runes remain visible) while plain
    // pixels fade toward the smaller baseline.
    float alphaRamp = blendWeight * ML_ALPHA_RAMP_SCALE + ML_ALPHA_RAMP_BIAS;
    float cloakAlpha = edgeMask * alphaRamp + CLOAK_ALPHA_BIAS;
    cloakAlpha = cloakAmount * (cloakAlpha - vertAlpha) + vertAlpha;
    float finalAlpha = albedoAlpha * cloakAlpha;

    // Tint strength decays with emissive/team weight so hot pixels are
    // not washed into cyan — keeps glyph silhouettes readable.
    float tintStrength = fresnelSq * (1.0 - blendWeight * ML_EMISSIVE_BLEND_CLOAK);

    float3 tinted = cloakAmount * tintStrength * (CLOAK_TINT - color) + color;

    CloakResult result;
    result.colorAlpha = float4(tinted, finalAlpha);
    return result;
}
