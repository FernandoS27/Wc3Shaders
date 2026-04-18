// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026, Fernando Sahmkow
// See LICENSE in the repository root for full terms.
//
// Analytic BRDF building blocks for the HD PBR path: GGX normal
// distribution, Schlick-Fresnel approximation, and the Schlick-GGX
// visibility term (i.e. the G/(4·NdotL·NdotV) combined factor from the
// Unreal/Epic formulation). Also hosts a screen-space-derivative based
// specular-AA roughness inflation and a trivial inverse-square point
// light falloff.
//
// The constants below are the tiny-but-meaningful epsilons chosen by the
// original engine — they are exposed here rather than inlined so that
// every shading path sees the exact same values.

implementing wc3_shaders;

static const float PI              = 3.14159274;
static const float INV_PI          = 0.318309873;
static const float LOG2E           = 1.44269502;   // 1 / ln(2), used by exp2-based fog.
static const float SAFE_EPS        = 0.00100000005; // Divide-by-zero guard in hue normalization.
static const float POINT_EPS       = 0.000500000024; // Avoids infinite gain at distance 0.
static const float NDF_VAR_CLAMP   = 0.180000007;  // Max geometric roughness inflation per pixel.
static const float SCHLICK_K_FACTOR = 0.125;       // k = (roughness+1)² / 8 → (r+1)²·0.125.

// GGX (Trowbridge-Reitz) normal distribution function.
// D(h) = α⁴ / (π · (NdotH² · (α⁴ − 1) + 1)²), with α = roughness².
// Using α⁴ (instead of the more common α²) is the disney-remap form.
public float ggxNDF(float NdotH, float roughness)
{
    float a2 = roughness * roughness;
    float a4 = a2 * a2;
    float d  = NdotH * NdotH * (a4 - 1.0) + 1.0;
    return a4 / (PI * d * d);
}

// Schlick's Fresnel: F0 + (1 − F0)·(1 − VdotH)⁵.
// t5 = t·t·t·t·t is expanded manually so the compiler keeps a minimum
// dependency chain.
public float3 schlickFresnel(float3 F0, float VdotH)
{
    float t = 1.0 - VdotH;
    float t2 = t * t;
    float t5 = t2 * t2 * t;
    return F0 + (1.0 - F0) * t5;
}

// Schlick-GGX visibility term used in the Cook-Torrance numerator:
// V(l,v) = 1 / (4 · (NdotL(1−k) + k) · (NdotV(1−k) + k))
// with k = (α+1)² / 8. The saturate() guards against under-roughed
// surfaces whose denominator can dip below 0.25 and blow up the term.
public float schlickGGXVisibility(float NdotL, float NdotV, float roughness)
{
    float a1  = 1.0 + roughness;
    float a12 = a1 * a1;
    float k   = a12 * SCHLICK_K_FACTOR;
    float oneMinusK = 1.0 - k;
    float gl = NdotL * oneMinusK + k;
    float gv = NdotV * oneMinusK + k;
    return saturate(1.0 / (4.0 * gl * gv));
}

// Convenience wrapper that assembles the three classic Cook-Torrance
// factors. Callers that need finer control (e.g. the light-loop hot
// path) inline the components instead of calling this.
public float3 cookTorranceSpecular(
    float3 F0, float NdotH, float NdotL, float NdotV, float VdotH, float roughness)
{
    float  D = ggxNDF(NdotH, roughness);
    float3 F = schlickFresnel(F0, VdotH);
    float  V = schlickGGXVisibility(NdotL, NdotV, roughness);
    return D * F * V;
}

// Specular-antialiasing: a Kaplanyan-style variance estimate based on
// screen-space derivatives of the world normal is added to the squared
// roughness so glancing, high-frequency normals don't alias into
// flickering specular speckles. The variance is clamped so extremely
// tangent surfaces (silhouettes) don't go fully diffuse.
public float specularAA(float roughnessMip, float3 worldNormal)
{
    float3 dnx = ddx_coarse(worldNormal);
    float3 dny = ddy_coarse(worldNormal);
    float variance = 2.0 * (dot(dnx, dnx) + dot(dny, dny));
    variance = min(NDF_VAR_CLAMP, variance);
    float a2 = roughnessMip * roughnessMip + variance;
    return sqrt(min(1.0, a2));
}

// Remap the authored 8-bit roughness value (0..255 / 255) into the
// slightly biased form expected by the cubemap-array mip-addressing
// convention used by the IBL sampler. Matches what the original runtime
// does when populating its roughness-to-mip LUT.
public float roughnessToMip(float roughness)
{
    return (roughness * 255.0 + 1.0) / 256.0;
}

// Inverse-square point-light attenuation with a tiny radial bias so
// lights don't produce unbounded intensity when a vertex lands exactly
// on the light position.
public float pointLightAttenuation(float distSq, float dist)
{
    return 1.0 / (distSq + POINT_EPS * dist);
}
