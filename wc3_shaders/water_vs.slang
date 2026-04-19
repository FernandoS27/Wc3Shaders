// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026, Fernando Sahmkow
// See LICENSE in the repository root for full terms.
//
// Water vertex shader entry point.
//
// The water pipeline compiles a single permutation — there's no
// skinning, shadow, or vertex-format specialisation here. The engine
// always binds the same 4-attribute stream (position, normal, color,
// uv) and the same `vsCB2` layout (shared with HD_VS), so this shader
// is just a small linear transform stack.
//
// What the shader emits (consumed by water_ps):
//   clipPos      — object → clip via `vsCB2.worldViewProj`.
//   color.rgb    — raw vertex color.
//   color.a      — shore-opacity remap α' = 0.5·(α² + α) (see below).
//   texCoord.xy  — UV run through the 2×3 matrix in `texMtx0Row0/Row1`.
//                  The PS re-scales this into two octave-specific UVs
//                  and animates them with `effectTime`.
//   worldPos     — object → world via `vsCB2.world`.
//   normalWS     — renormalised world-space geometry normal.
//   upAxisWS.xyz — world Y column of `vsCB2.world` (per-object "up").
//                  Used by the water PS as the TBN tangent so the
//                  tangent frame stays aligned with the water surface.
//   upAxisWS.w   — 1 (consumed by the PS as the bitangent-handedness
//                  scalar; it's always 1 for water).

implementing wc3_shaders;

// Shore-opacity remap: α' = 0.5·α² + 0.5·α.
//   α=0 → 0, α=1 → 1, α=0.5 → 0.375 (slightly concave).
// Biases low-α vertices further toward transparent so shore lines fade
// more aggressively than a straight pass-through.
float remapShoreAlpha(float a)
{
    return (a * a - a) * 0.5 + a;
}

[shader("vertex")]
public WaterVSOutput water_vs_main(WaterVSInput input)
{
    WaterVSOutput output;

    // Step 1: clip-space position (computed directly from object space
    // via the combined worldViewProj — avoids the precision loss of an
    // intermediate world-space round-trip).
    output.position = transformClipPos(input.position, vsCB2.worldViewProj);

    // Step 2: vertex color pass-through + alpha remap.
    output.color.rgb = input.vertColor.rgb;
    output.color.a   = remapShoreAlpha(input.vertColor.a);

    // Step 3: transformed UV. The water VS only uses UV0; the 2×3
    // affine matrix in texMtx0Row0/Row1 is shared with the HD path.
    output.texCoord = transformUV(
        input.uv, vsCB2.texMtx0Row0, vsCB2.texMtx0Row1);

    // Step 4: view-space position (emitted as the PS's "worldPos" — the
    // PS transforms it back out through `invView` when sampling probes).
    output.worldPos = transformByWorldView(input.position, vsCB2.worldView);

    // Step 5: view-space normal (renormalised). Same space as the
    // position above; the water PS's TBN frame and Snell refraction
    // operate in this space.
    output.normalWS = transformNormal(input.normal, vsCB2.worldView);

    // Step 6: per-object up axis. The water PS's TBN frame uses this as
    // the tangent direction (rather than a per-vertex tangent attr) —
    // it's the world Y axis rotated by the object's worldView (so it
    // stays co-located with `normalWS` above in view space). Extracted
    // via `mul(worldView, (0,1,0,0))` rather than `worldView[1]`
    // because the latter returns row 1 in HLSL/Slang matrix indexing,
    // not column 1. .w = 1 is the bitangent handedness (always 1).
    output.upAxisWS.xyz = mul(vsCB2.worldView, float4(0, 1, 0, 0)).xyz;
    output.upAxisWS.w   = 1;

    return output;
}
