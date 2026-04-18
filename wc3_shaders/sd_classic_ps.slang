// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026, Fernando Sahmkow
// See LICENSE in the repository root for full terms.
//
// SD classic pixel shader entry point.
//
// Thin wrapper around `sdClassicPsBody` (in ps/sd_classic_ps_body.slang)
// that just forwards the four compile-time policies (two texture-stage
// policies, fog mode, alpha test) into the body and packs the result
// into the SD single-target PS output. The classic pipeline has no
// MRT / IBL / debug variants — this is the simplest entry in the
// module.

implementing wc3_shaders;

[shader("fragment")]
public SDPSOutput sd_classic_ps_main<
    T0     : ITextureStage,
    T1     : ITextureStage,
    TFog   : IFogMode,
    TAlpha : IAlphaTest
>(SDPSInput input)
{
    SDPSOutput output;
    output.color = sdClassicPsBody<T0, T1, TFog, TAlpha>(input);
    return output;
}
