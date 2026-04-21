// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026, Fernando Sahmkow
// See LICENSE in the repository root for full terms.
//
// Tonemap pixel shader entry point.
//
// Single-permutation full-screen pass — no fog, material, or alpha-test
// specialisation. All the work lives in `tonemapPSBody`
// (ps/tonemap_ps_body.slang); this file just binds the entry point.

implementing wc3_shaders;

[shader("fragment")]
public float4 tonemap_ps_main(TonemapPSInput input) : SV_Target0
{
    return tonemapPSBody(input);
}
