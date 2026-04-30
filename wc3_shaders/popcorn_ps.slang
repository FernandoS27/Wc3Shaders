// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026, Fernando Sahmkow
// See LICENSE in the repository root for full terms.
//
// PopcornFX pixel shader entry point.
//
// The popcorn pipeline emits the same particle TWICE per draw — once
// per pass — and selects the matching VS+PS pair via the engine's
// permutation index. The two passes are wholly different shaders:
//
//   • COLOR pass   — full lit / diffuse / fog / soft-particles /
//                    alpha-LUT / gbuffer surface shading.
//   • MOTION pass  — encodes 2D screen-space motion vectors into
//                    RT0.xy for TAA / motion-blur reconstruction.
//
// Specialisation space (1152 perms = 9 outer × 128 inner; see
// re_shaders/popcornfx/popcorn_ps.hlsl for the encoding):
//
//   TMode             — IPopcornMode (NoUV, BasicUV, Billboard,
//                        Atlas) — drives the per-mode UV resolution
//                        in both passes.
//   TFog              — IFogMode (None / Linear / Exp / Exp²) — only
//                        consulted in the COLOR pass.
//   IS_MOTION_PASS    — route to the motion-vector body. Flips off
//                        most of the per-feature flags below at the
//                        math level (the engine still binds them so
//                        the input signature stays aligned across
//                        pass switches).
//   HAS_GBUFFER       — also write SV_TARGET1 + SV_TARGET2 (deferred
//                        depth + material payloads). COLOR pass only.
//   HAS_SOFT_PARTICLES — depth-fade against the scene depth buffer
//                        (alpha modulation in COLOR; motion-magnitude
//                        modulation in MOTION).
//   HAS_ALPHA_LUT     — t1 LUT lookup for alpha (COLOR pass only;
//                        the engine never binds the random scalar
//                        without a UV stream to associate it with).
//   HAS_VC            — modulate by per-vertex color (COLOR pass:
//                        full RGBA; MOTION pass: vc.xy tints motion
//                        and vc.w becomes the output alpha).
//   HAS_LIT           — full Schlick-GGX BRDF + IBL (cubemap
//                        diffuse / spec + BRDF LUT). Requires the
//                        VS to have HAS_NT enabled so TEXCOORD4/5
//                        carry the per-vertex normal + tangent.
//                        COLOR pass only.

implementing wc3_shaders;

[shader("fragment")]
public PopcornPSOutput<HAS_GBUFFER && !IS_MOTION_PASS> popcorn_ps_main<
    TMode  : IPopcornMode,
    TFog   : IFogMode,
    let IS_MOTION_PASS     : bool,
    let HAS_GBUFFER        : bool,
    let HAS_SOFT_PARTICLES : bool,
    let HAS_ALPHA_LUT      : bool,
    let HAS_VC             : bool,
    let HAS_LIT            : bool
>(PopcornPSInput input, uint frontFace : SV_IsFrontFace)
{
    PopcornPSOutput<HAS_GBUFFER && !IS_MOTION_PASS> output;

    if (IS_MOTION_PASS)
    {
        // Motion-vector pass — most "color" features are dead at the
        // math level here (the engine still binds them so the input
        // signature stays aligned with the matching VS).
        output.color = popcornMotionPassBody<TMode, HAS_VC, HAS_SOFT_PARTICLES>(input);
        return output;
    }

    // COLOR pass — full surface shading.
    bool isFrontFace = frontFace != 0;
    PopcornColorOut shaded = popcornColorPassBody<
        TMode, TFog, HAS_VC, HAS_ALPHA_LUT, HAS_SOFT_PARTICLES, HAS_LIT>(
            input, isFrontFace);

    output.color = shaded.color;

    // Deferred g-buffer. RT1 carries the view-space depth + a packed
    // normal hint; RT2 the default material slot + the pre-SP /
    // pre-fog alpha (the deferred resolver re-applies its own
    // alpha-blend pass).
    if (HAS_GBUFFER)
    {
        output.depthRT    = float4(input.worldPos.z, 0.0, 0.0, 1.0);
        output.materialRT = float4(GBUFFER_DEFAULT_MAT, shaded.gbufferAlpha);
    }

    return output;
}
