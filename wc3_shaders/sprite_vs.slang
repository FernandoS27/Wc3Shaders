// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026, Fernando Sahmkow
// See LICENSE in the repository root for full terms.
//
// Sprite vertex shader entry point.
//
// Single permutation, no specialisation axes. Sprite draws ship pre-
// projected vertices: ATTR0.xyz is already in clip space, ATTR3.xy is
// the texture coordinate. The VS just packs xyz → SV_Position.xyz, sets
// the homogeneous w to 1, and forwards the UV.
//
// The engine's sprite vertex buffer reuses the HD vertex layout slot
// numbers (ATTR3 for uv0) so the same input element layout can drive
// HD or sprite draws — the unused HD attributes simply aren't bound.

implementing wc3_shaders;

[shader("vertex")]
public SpriteVSOutput sprite_vs_main(SpriteVSInput input)
{
    SpriteVSOutput output;
    output.position = float4(input.position, 1);
    output.uv       = input.uv;
    return output;
}
