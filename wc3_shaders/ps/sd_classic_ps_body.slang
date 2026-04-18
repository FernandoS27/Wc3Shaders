// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026, Fernando Sahmkow
// See LICENSE in the repository root for full terms.
//
// SD classic pixel shader body.
//
// Recreates the D3D9 fixed-function texture-stage chain the classic
// Warcraft III renderer used. Two texture stages are chained through
// the compile-time policies `T0`, `T1`; lighting is already baked into
// `input.vertColor` by the vertex shader, so the body just applies the
// combiner chain to it, runs the alpha test, and applies fog.

implementing wc3_shaders;

public float4 sdClassicPsBody<
    T0    : ITextureStage,
    T1    : ITextureStage,
    TFog  : IFogMode,
    TAlpha: IAlphaTest
>(SDPSInput input)
{
    // The "current color" starts as the pre-lit vertex color.
    float4 current = input.vertColor;

    if (T0.isEnabled)
    {
        // Stage 0: sample albedo (t0) at UV0 and combine.
        float4 tex0 = t_albedo.Sample(s_albedo, input.uv.xy);
        current = T0.applyOp(current, tex0);

        if (T1.isEnabled)
        {
            // Stage 1: sample the second texture (bound at t_normal /
            // t1 in the classic pipeline, despite the name — the
            // classic renderer uses t1 as the detail / lightmap slot)
            // at UV1 and combine.
            float4 tex1 = t_normal.Sample(s_normal, input.uv.zw);
            current = T1.applyOp(current, tex1);
        }
    }

    // Alpha test after combining so cut-out materials see the right
    // alpha including any modulation from the texture stages.
    TAlpha.test(current.w, sdClassicPsCB0.alphaRef);

    // Fog is applied on view-space Z (packed in worldPos.z by the VS).
    current.xyz = TFog.apply(current.xyz, sdClassicPsCB0.fogColor.xyz,
                             input.worldPos.z,
                             sdClassicPsCB0.fogStart,
                             sdClassicPsCB0.fogEnd,
                             sdClassicPsCB0.fogDensity);

    return current;
}
