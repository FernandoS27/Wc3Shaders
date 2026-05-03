// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026, Fernando Sahmkow
// See LICENSE in the repository root for full terms.
//
// Terrain pixel shader entry point.
//
// Specialisation space — four functional axes; the engine ships 128
// perms (7 raw bits) but bits 3/4/5 are reserved (collapse to the same
// output) and bit 1 (`HAS_NULL_PASS`) overrides every other axis with
// an empty stub.
//
//   HAS_NULL_PASS  — empty shader, no outputs (engine "skip" stub).
//                    Overrides every other axis: 64 of 128 perms map
//                    here. The output struct's color/MRT slots are
//                    gated out via `Conditional` so the dxbc emits no
//                    SV_Target entries, matching the shipped layout.
//   HAS_MRT        — emit SV_Target1 (depth-feedback in .x = view-Z)
//                    and SV_Target2 (encoded world normal + 1) in
//                    addition to SV_Target0.
//   HAS_SHADOWS    — sample the 3 cascaded shadow maps with PCF and
//                    multiply visibility into the dominant light.
//   HAS_TINT       — `TINT_OVERRIDE` in engine terms: per-splat layer
//                    color/UV substitution + the debug-viz output
//                    router. Bypasses the post-grade colour matrix.
//
// All of the heavy lifting lives in `terrainPSBody`
// (ps/terrain_ps_body.slang); this file just dispatches to it under
// the right specialisation and packs the result into the output
// payload.

implementing wc3_shaders;

[shader("fragment")]
public TerrainPSOutput<!HAS_NULL_PASS, !HAS_NULL_PASS && HAS_MRT> terrain_ps_main<
    let HAS_NULL_PASS : bool,
    let HAS_MRT       : bool,
    let HAS_SHADOWS   : bool,
    let HAS_TINT      : bool>(
    TerrainPSInput input)
{
    TerrainPSOutput<!HAS_NULL_PASS, !HAS_NULL_PASS && HAS_MRT> output;

    if (HAS_NULL_PASS)
    {
        // 64-perm "skip" stub — all output fields are compiled out via
        // the false `HAS_COLOR` / `HAS_MRT` Conditional gates, leaving
        // a bare `ret` that matches the shipped DXBC byte-for-byte.
        return output;
    }

    TerrainShadeResult shaded = terrainPSBody<HAS_SHADOWS, HAS_TINT>(input);
    output.color = shaded.color;

    if (HAS_MRT)
    {
        // Depth-feedback target: world-space Z carried through the VS
        // payload.
        output.depthFeed = float4(shaded.viewZ, 0, 0, 1);

        if (HAS_TINT)
        {
            // Engine quirk: in TINT_OVERRIDE / debug-viz mode the
            // normal MRT is intentionally left uninitialised; only .w
            // gets a constant write so the depth-feedback resolver
            // still sees a valid alpha.
            output.normalEnc = float4(0, 0, 0, 1);
        }
        else
        {
            // Encoded world normal: (n + 1) · 0.5 in RGB, .w = 1.
            output.normalEnc = float4(shaded.worldNormal * 0.5 + 0.5, 1.0);
        }
    }

    return output;
}
