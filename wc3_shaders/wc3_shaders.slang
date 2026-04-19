// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026, Fernando Sahmkow
// See LICENSE in the repository root for full terms.
//
// Module root for the Warcraft III Reforged shader reconstruction.
//
// This file is the single unit of compilation: every other `.slang`
// source in this tree declares `implementing wc3_shaders;` and is
// pulled in through one of the `__include` directives below. Each
// permutation the engine asks for instantiates one of the entry points
// in the bottom section (hd_ps / hd_vs / sd_* / sd_classic_*) with the
// appropriate policy types + compile-time flags.
//
// File layout:
//
//   types/        Constant buffer / texture / sampler bindings and the
//                 VS/PS I/O struct layouts.
//   math/         Stateless math primitives: color-space transforms,
//                 normal-map decode, analytic BRDF building blocks.
//   interfaces/   Compile-time policy interfaces (traits): fog mode,
//                 alpha test, skinning, vertex format, texture stages,
//                 material layer. These drive specialisation.
//   effects/      Higher-level "visual effects" that the HD body stack
//                 composes: cloak, fresnel overlay, multi-layer team
//                 blend.
//   ps/           Pixel-shader bodies, shared across entry points.
//   vs/           Vertex-shader helpers and per-vertex lighting.
//
// The entry points at the bottom map to:
//
//   hd_ps        — HD pipeline pixel shader (PBR / IBL).
//   hd_vs        — HD pipeline vertex shader.
//   crystal_ps   — HD refractive / transparent variant (gem / glass props).
//   sd_on_hd_ps  — SD assets rendered through the HD pipeline (PS).
//   sd_on_hd_vs  — SD assets rendered through the HD pipeline (VS).
//   sd_highspec_vs — SD pipeline per-vertex-lit VS.
//   sd_classic_ps — SD pipeline fixed-function texture-stage PS.

module wc3_shaders;

// Data types: constant buffers, bindings, VS/PS I/O structs.
__include "types/cb_structs";
__include "types/vs_io";
__include "types/ps_io";

// Math primitives.
__include "math/color";
__include "math/normal";
__include "math/brdf";

// Compile-time policy interfaces.
__include "interfaces/fog";
__include "interfaces/material";
__include "interfaces/alpha_test";
__include "interfaces/skinning";
__include "interfaces/vertex_format";
__include "interfaces/texture_stage";

// Higher-level visual effects used by the PS bodies.
__include "effects/cloak";
__include "effects/fresnel_overlay";
__include "effects/multi_layer";
__include "effects/crystal_refract";

// HD pixel-shader bodies (shared among ps_main permutations).
__include "ps/ps_depth_prepass";
__include "ps/ps_ibl";
__include "ps/ps_standard";
__include "ps/ps_post";

// Crystal pixel-shader bodies (refractive variant of the HD pipeline).
__include "ps/crystal_ps_depth_prepass";
__include "ps/crystal_ps_ibl";
__include "ps/crystal_ps_standard";

// SD-on-HD pixel-shader bodies.
__include "ps/sd_ps_depth_prepass";
__include "ps/sd_ps_shadow";
__include "ps/sd_ps_ibl";
__include "ps/sd_ps_standard";
__include "ps/sd_ps_post";

// SD classic pixel-shader body (fixed-function texture stages).
__include "ps/sd_classic_ps_body";

// Vertex-shader helpers.
__include "vs/vs_body";
__include "vs/sd_vs_lighting";

// Entry points.
__include "hd_ps";
__include "hd_vs";
__include "crystal_ps";
__include "sd_on_hd_ps";
__include "sd_on_hd_vs";
__include "sd_highspec_vs";
__include "sd_classic_ps";
