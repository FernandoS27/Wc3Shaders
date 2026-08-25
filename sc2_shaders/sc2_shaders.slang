// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026, Fernando Sahmkow
// See LICENSE in the repository root for full terms.
//
// Module root for the StarCraft II shader reconstruction (`sc2_shaders`).
//
// This is the SC2 analogue of `wc3_shaders`: the original shipped `.fx`
// shader families, rewritten as one idiomatic Slang translation unit and
// compiled per-family into BLS bundles whose permutation slots reproduce
// each family's retail permutation set, in retail cache order (design:
// docs/SC2_SHADERS_MODULE_DESIGN.md; plan: docs/SC2_SHADERS_IMPLEMENTATION.md).
//
// Every other `.slang` in this tree declares `implementing sc2_shaders;`
// and is pulled in through one of the `__include` directives below; each
// permutation the engine asks for instantiates one entry point with the
// appropriate compile-time `#if b_*` / specialisation axes.
//
// Validation: each entry is checked against the ORIGINAL `.fx` compile of
// the same decoded permutation (fxc ps_5_0 / vs_5_0 reference vs slangc
// candidate, both run through dxbc_interp on identical random inputs) by
// tools/sc2_validate_family.py — no Blizzard-fxc drift, purely our own
// toolchain.
//
// File layout (grows milestone by milestone):
//   types/   Constant buffers, texture/sampler bindings, VS/PS IO structs.
//   (ps/, vs/, math/, interfaces/, effects/ added as families land.)

module sc2_shaders;

// --- Cross-backend intrinsic shims ------------------------------------------
__include "portable";

// --- Data types -------------------------------------------------------------
// The permutation constant buffer (b2) and the SC2_AXIS accessor.  Inert until
// SC2_USE_PERM_CB is set; see types/perm_cb.slang.
__include "types/perm_cb";
__include "types/simple_io";
__include "types/model_io";
__include "types/model_vs_io";

// --- Vertex-shader body stack (Model family) --------------------------------
__include "vs/model_vs_common";
__include "vs/model_vs_lighting";
// Shared Ribbon/Particle simulation helpers (ribbonparticlecommon.fx).
__include "vs/ribbon_particle_common";

// --- Pixel-shader body stack (Model/DefaultPixelMain families) --------------
// pscommon.fx depth-prepass decode shared by the screen-space roots.
__include "ps/ps_common";
__include "ps/model_material";
__include "ps/model_layers";
__include "ps/model_normal";
__include "ps/model_shadow";
__include "ps/model_lighting";
__include "ps/model_fog";
__include "ps/model_fow";
__include "ps/model_gbuffer";
__include "ps/model_envio";
__include "ps/model_parallax";
// --- Non-standard shading-mode arms (M2.1) ----------------------------------
__include "ps/model_layers_m2";
__include "ps/model_displace";
__include "ps/model_vectorui";
__include "ps/model_volume";
__include "ps/model_reflection";
__include "ps/model_cloaking";
__include "ps/model_creep";
__include "ps/model_terrain";
__include "ps/model_main_shading";

// --- Entry points -----------------------------------------------------------
// Simple family (simple.fx) — the walking-skeleton family (M0).
__include "simple_vs";
__include "simple_ps";

// Model family (model.fx / psmainshading.fx : DefaultPixelMain) — the shared
// material/shading pixel root (M1). Standard shading mode, growing by feature.
__include "model_ps";

// SplatDirect family (splatdirect.fx : SplatDirectPixelMain) — stencil-fill wrapper
// over DefaultPixelMain; Model PS schema + own axis b_stencilFillPass (M2.3).
__include "ps/splatdirect_ps";

// SplatDeferred family (splatdeferred.fx : SplatDeferredPixelMain) — depth-reconstruction
// box-render preamble over DefaultPixelMain; Model PS schema + 3 own axes (M2.3).
__include "ps/splatdeferred_ps";

// TerrainBlend family (terrainblend.fx : TerrainBlendPixelMain) — own-IO terrain/projector
// texture blending; own key schema (3 axes + Common), 9 blend modes (M2.3).
__include "ps/terrainblend_ps";

// Water family (water.fx : WaterPixelMain) — own-root PS reusing the shared transport +
// body stack; own key schema (5 axes + Common/Lighting/FOW), env/cheap/pretty (M2).
__include "ps/water_ps";

// Model family (model.fx : ModelVertexMain) — the Model-specific vertex root (M4).
__include "model_vs";

// Foliage family (foliage.fx : FoliageVertexMain) — compressed-stream vertex root,
// sharing the Default-family transport / lighting / GenUV bodies (M4).
__include "foliage_vs";

// Ribbon family (ribbon.fx : RibbonVertexMain) — GPU ribbon simulation (spline /
// mixed / legacy), batched through a per-vertex batch index (M4).
__include "ribbon_vs";

// Particle family (particle.fx : ParticleVertexMain) — point-to-quad expansion
// with 11 instance orientations, flipbook UVs and procedural motion (M4).
__include "particle_vs";

// ---------------------------------------------------------------------------
// M5 - the own-ROOT vertex stages of the four families whose PIXEL roots landed in
// M2.  Each decodes from the SAME family key schema its pixel stage already uses.
// ---------------------------------------------------------------------------
// SplatDirect (splatdirect.fx) - the legacy batched splat, on the shared transport.
__include "vs/splatdirect_vs";
// SplatDeferred (splatdeferred.fx) - the single-splat deferred variant.
__include "vs/splatdeferred_vs";
// TerrainBlend (terrainblend.fx) - own-IO terrain/projector blend quad.
__include "vs/terrainblend_vs";
// Water (water.fx) - the Gerstner wave surface, on the shared transport.
__include "vs/water_vs";

// ---------------------------------------------------------------------------
// M4 - self-contained families (own IO structs, own small key schemas).
// ---------------------------------------------------------------------------
// RenderPlane (renderplane.fx) - the FOW black-mask plane.
__include "renderplane";

// MinimapTerrain (minimapterrain.fx) - minimap terrain/water/creep/FOW composite.
__include "minimapterrain";

// LensFlare (lensflare.fx) - screen-space flare sprites + dirty lens.
__include "lensflare";

// Flash (flash.fx) - the Scaleform UI backend; 5 vertex layouts x 14 colour modes.
__include "flash";

// HDR (hdr.fx + pstonemap.fx) - the tonemap resolve; 5 tone curves + 3D LUT grade.
__include "hdr";

// Image (image.fx) - the UI compositor; up to 4 premultiplied layers + 5 adjust modes.
__include "image";

// Bokeh (bokeh.fx + psbokeh*.fx) - the 7-pass bokeh depth-of-field chain.
__include "bokeh";

// DeferredLight (deferredlight.fx) - the deferred point/spot light volume pass.
// Own root, but it rides the SHARED interpolant transport.
__include "deferredlight";

// PostProcessQuad (postprocessquad.fx + pspostprocess/psdof/psssao/psantialiasing/
// pshalo) - the full-screen post-process quad; 15 modes behind 36 own axes.  Also an
// own root on the SHARED transport (and the only user of GaussianBlurSample).
__include "postprocessquad";
