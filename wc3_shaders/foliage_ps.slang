// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026, Fernando Sahmkow
// See LICENSE in the repository root for full terms.
//
// Foliage pixel shader entry point.
//
// Specialisation space:
//
//   HAS_NULL_PASS    — empty / discard-only stub (depth-only or shadow
//                      caster pass). When ALPHA_TEST is also active
//                      the stub samples the albedo alpha and runs the
//                      alpha-test discard, otherwise it just returns.
//                      Overrides every other axis below.
//   HAS_MRT          — emit SV_Target1 (depth-feedback in .x = view-Z)
//                      and SV_Target2 (encoded world normal + alpha)
//                      in addition to SV_Target0.
//   HAS_SHADOWS      — sample 3 cascaded shadow maps with PCF.
//   TFog             — fog-mode policy (`FogNone` / `FogLinear` /
//                      `FogExponential` / `FogExp2`).
//   HAS_ALPHA_TEST   — `discard` when `vertColor.w · texAlpha < cb2[0].x`.
//   HAS_TINT         — debug visualisation: per-splat tint substitution
//                      + 9-mode debug-viz router. Bypasses the lit
//                      colour output path.
//
// The engine's 128 perms collapse to many functionally distinct outputs
// (NULL_PASS dominates 64 of them); the perm-mapper handles the
// dispatching.

implementing wc3_shaders;

[shader("fragment")]
public FoliagePSOutput<!HAS_NULL_PASS, !HAS_NULL_PASS && HAS_MRT> foliage_ps_main<
    let HAS_NULL_PASS  : bool,
    let HAS_MRT        : bool,
    let HAS_SHADOWS    : bool,
    TFog               : IFogMode,
    let HAS_ALPHA_TEST : bool,
    let HAS_TINT       : bool>(
    FoliagePSInput input)
{
    FoliagePSOutput<!HAS_NULL_PASS, !HAS_NULL_PASS && HAS_MRT> output;

    if (HAS_NULL_PASS)
    {
        // 64-perm "skip" stub. With alpha-test enabled the stub still
        // performs the discard so the depth buffer punches the right
        // silhouette through cut-out leaves; otherwise it's a bare
        // `ret` with zero output dcls (the Conditional gates collapse
        // every SV_Target away, matching the shipped 312-byte perm).
        if (HAS_ALPHA_TEST)
        {
            float alphaTex = t_foliageAlbedo.Sample(s_foliageAlbedo,
                                                    input.texCoord.xy).w;
            if (input.color.w * alphaTex - psCB2.alphaRef < 0.0)
                discard;
        }
        return output;
    }

    FoliageShadeResult shaded =
        foliagePSBody<TFog, HAS_SHADOWS, HAS_TINT>(input);

    // Alpha-test discard: recompute against the raw texAlpha (the body
    // saved it on the result struct because the multiply-into-color
    // path consumes it elsewhere).
    if (HAS_ALPHA_TEST)
    {
        if (input.color.w * shaded.texAlpha - psCB2.alphaRef < 0.0)
            discard;
    }

    output.color = float4(shaded.color, shaded.alpha);

    if (HAS_MRT)
    {
        // SV_Target1 — depth-feedback (view-space Z + sentinel .w = 1).
        output.depthFeed = float4(shaded.viewZ, 0, 0, 1);
        // SV_Target2 — packed world normal `(n + 1) · 0.5` in RGB,
        // pre-fog alpha in .w. The debug-viz path keeps this output
        // intact (the engine quirk is to substitute *only* SV_Target0
        // under TINT_OVERRIDE).
        output.normalEnc = float4(shaded.worldNormal * 0.5 + 0.5,
                                  shaded.alpha);
    }

    return output;
}
