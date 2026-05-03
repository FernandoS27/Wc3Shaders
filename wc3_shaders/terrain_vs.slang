// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026, Fernando Sahmkow
// See LICENSE in the repository root for full terms.
//
// Terrain vertex shader entry point.
//
// Specialisation space — two compile-time bools that map onto the
// engine's three-bit terrain VS perm encoding:
//
//   HAS_VERTEX_COLOR — read COLOR from the per-vertex stream (ATTR2)
//                      instead of the constant tint at cb2[13].
//   HAS_SHADOWS      — RECEIVE_SHADOWS && !SHADOW_PASS in engine terms;
//                      emits the three cascade-clip outputs the PS uses
//                      for PCF lookup. The `SHADOW_PASS` flag in the
//                      engine acts as a cascade-output suppression
//                      override — a draw rendering its own shadow caster
//                      pass never emits cascade UVs, even when the
//                      receive flag is also set, so perm_003 / perm_007
//                      collapse onto the no-shadow output.
//
// All eight engine perms collapse to four functionally distinct outputs;
// the perm-mapper in compile_all_slang.py handles the redundant slots.

implementing wc3_shaders;

[shader("vertex")]
public TerrainVSOutput<HAS_SHADOWS> terrain_vs_main<
    let HAS_VERTEX_COLOR : bool,
    let HAS_SHADOWS      : bool>(TerrainVSInput input)
{
    return terrainVSBody<HAS_VERTEX_COLOR, HAS_SHADOWS>(input);
}
