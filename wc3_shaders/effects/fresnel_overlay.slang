// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026, Fernando Sahmkow
// See LICENSE in the repository root for full terms.
//
// "Fresnel overlay" — the rim-lighting / hero-glow effect triggered
// when a selectable unit is highlighted. The overlay tints the lit
// color toward an artist-authored `fresnelTint` (or, for multi-layer
// materials, a team-color-derived tint) using a squared-fresnel ramp.
//
// The team-color blend is implemented as a hue-preserving swap: each
// tint is separated into a hue vector (tint / max(tint)) and a peak
// intensity, then the two hues are lerped by `fresnelTeamColor` while
// the intensity is driven toward the authored peak. This avoids the
// desaturation that a plain RGB lerp would produce when the two tints
// have very different chroma.

implementing wc3_shaders;

// Fallback hue (pure blue) used by the standard variant when no team
// color is bound — matches the classic blue selection ring.
static const float3 FRESNEL_TARGET = float3(0, 0, 0.999000967);

public struct FresnelOverlayResult
{
    public float3 color;
    public float  alpha;
};

// Standard fresnel overlay. Used by every single-layer HD material.
public FresnelOverlayResult applyFresnelOverlay(
    float3 litColor, float albedoAlpha, float NdotV,
    float3 fresnelTint, float fresnelAlpha, float fresnelTeamColor,
    float vertAlpha)
{
    // (1 − NdotV)² — strongest at grazing angles.
    float oneMinusNdotV = 1.0 - saturate(NdotV);
    float fresnelSq = oneMinusNdotV * oneMinusNdotV;
    float fAlpha = fresnelAlpha * fresnelSq;

    // teamSq drives the intensity pull toward the team color.
    float teamSq = fresnelTeamColor * fresnelTeamColor;
    float maxFresnel = max(max(fresnelTint.x, fresnelTint.y), fresnelTint.z);
    float safeMax = 1.0 / (maxFresnel + SAFE_EPS);
    float3 normTint = fresnelTint * safeMax; // Hue-only tint direction.

    // Hue-lerp toward pure blue selection ring.
    float3 blendedTint = fresnelTeamColor * (FRESNEL_TARGET - normTint) + normTint;

    // When the tint is very bright (> 1) the intensity keeps its peak;
    // when it's dark we lift it toward 1 as the team factor rises.
    float intensity = maxFresnel + teamSq * (min(1.0, maxFresnel) - maxFresnel);

    float3 fresnelColor = blendedTint * intensity;
    float3 result = fAlpha * (fresnelColor - litColor) + litColor;
    float  alphaResult = saturate(vertAlpha * albedoAlpha + fAlpha * albedoAlpha);

    FresnelOverlayResult r;
    r.color = result;
    r.alpha = alphaResult;
    return r;
}

// Multi-layer variant. Differs in two ways:
//   1. The tint target is the team-color albedo (normalised to its own
//      hue vector) rather than the constant FRESNEL_TARGET.
//   2. The intensity envelope is pulled toward the lesser of the tint's
//      and the team albedo's peak luma, preventing the rim from blowing
//      past what the diffuse layer itself produces.
public FresnelOverlayResult applyFresnelOverlayML(
    float3 litColor, float albedoAlpha, float NdotV,
    float3 fresnelTint, float fresnelAlpha, float fresnelTeamColor,
    float vertAlpha,
    float3 teamAlbedo, float teamLumaMax)
{
    float oneMinusNdotV = 1.0 - saturate(NdotV);
    float fresnelSq = oneMinusNdotV * oneMinusNdotV;
    float fAlpha = fresnelAlpha * fresnelSq;

    float teamSq = fresnelTeamColor * fresnelTeamColor;
    float maxFresnel = max(max(fresnelTint.x, fresnelTint.y), fresnelTint.z);
    float safeMax = 1.0 / (maxFresnel + SAFE_EPS);
    float3 normTint = fresnelTint * safeMax;

    float teamSafeMax = 1.0 / (teamLumaMax + SAFE_EPS);
    float3 teamHueNorm = teamAlbedo * teamSafeMax;
    float3 blendedTint = fresnelTeamColor * (teamHueNorm - normTint) + normTint;

    float intensity = maxFresnel + teamSq * (min(maxFresnel, teamLumaMax) - maxFresnel);

    float3 fresnelColor = blendedTint * intensity;
    float3 result = fAlpha * (fresnelColor - litColor) + litColor;
    float  alphaResult = saturate(vertAlpha * albedoAlpha + fAlpha * albedoAlpha);

    FresnelOverlayResult r;
    r.color = result;
    r.alpha = alphaResult;
    return r;
}
