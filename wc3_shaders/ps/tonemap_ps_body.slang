// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026, Fernando Sahmkow
// See LICENSE in the repository root for full terms.
//
// Tonemap pixel shader body — final HDR→LDR resolve of the HD pipeline.
//
// The HD lighting passes render into a linear HDR off-screen target; this
// full-screen pass samples that target, applies an exposure multiplier
// uploaded in `tonemapPsCB1.exposure`, and runs the result through
// Krzysztof Narkowicz's analytical ACES filmic fit before writing the
// LDR back-buffer. Alpha is passed through unchanged so downstream
// composition (UI, post-cloak overlay) can still blend against it.
//
// Reference: Krzysztof Narkowicz, "ACES Filmic Tone Mapping Curve" (2015)
//   https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/
//
//   f(x) = saturate((x · (A·x + B)) / (x · (C·x + D) + E))
//
// The coefficients (A..E) below are the standard values from the fit —
// they approximate the reference ACES RRT+ODT output curve in a single
// per-pixel evaluation with no LUTs.

implementing wc3_shaders;

// Narkowicz ACES filmic fit coefficients.
static const float ACES_A = 2.51;
static const float ACES_B = 0.03;
static const float ACES_C = 2.43;
static const float ACES_D = 0.59;
static const float ACES_E = 0.14;

// Saturated per-component ACES tonemap. Takes a linear HDR colour and
// returns the LDR colour in [0, 1]^3.
float3 tonemapACES(float3 color)
{
    float3 numerator   = color * (ACES_A * color + ACES_B);
    float3 denominator = color * (ACES_C * color + ACES_D) + ACES_E;
    return saturate(numerator / denominator);
}

public float4 tonemapPSBody(TonemapPSInput input)
{
    float4 hdr     = t_sceneHdr.Sample(s_sceneHdr, input.uv);
    float3 exposed = hdr.rgb * tonemapPsCB1.exposure;
    return float4(tonemapACES(exposed), hdr.a);
}
