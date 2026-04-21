// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026, Fernando Sahmkow
// See LICENSE in the repository root for full terms.
//
// Module root for custom / mod shaders that sit on top of the
// reconstructed Warcraft III Reforged HD pipeline.
//
// Everything in `wc3_shaders` is the faithful reconstruction of the
// shipped shaders — same entry points, same resource layout, same
// binding slots. This module pulls that in via `import wc3_shaders;`
// and provides drop-in alternative pixel/vertex shader bodies that
// match the HD pipeline's VS/PS contract but replace its shading
// model. Consumers compile `custom_shaders.slang` directly (not
// `wc3_shaders.slang`) when building the variant BLS files.
//
// Current bodies:
//
//   toon_hd_ps   — Breath-of-the-Wild style 2-band cel shading.
//   toon_hd_vs   — VS paired with toon_hd_ps (same interpolants as hd_vs).

module custom_shaders;

// Pull in the entire HD pipeline (constant buffers, texture/sampler
// bindings, PSInput/VSOutput, BRDF helpers, IBL sampling, fresnel /
// cloak / multilayer effects, depth-prepass body, debug visualisation,
// shadow cascade PCF, etc.). Everything marked `public` inside
// wc3_shaders becomes visible here.
import wc3_shaders;

// Toon shading body + entry points.
__include "ps_toon";
__include "toon_hd_ps";
__include "toon_hd_vs";
