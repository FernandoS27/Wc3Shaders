// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026, Fernando Sahmkow
// See LICENSE in the repository root for full terms.
//
// Sprite pixel shader entry point.
//
// The sprite pipeline samples a single texture and optionally re-encodes
// the colour to/from sRGB so the same shader can drive draws into either
// linear or sRGB-formatted render targets without the caller having to
// pick between two different shader bytecodes ahead of time.
//
// Specialisation space — two independent compile-time bools:
//   HAS_SRGB_ENCODE — apply linear → sRGB OETF before write
//                     (i.e. source is linear, target is non-_SRGB).
//   HAS_SRGB_DECODE — apply sRGB → linear EOTF after sample
//                     (i.e. source is sRGB-encoded, downstream is
//                     linear).
//
// The four resulting permutations match the engine's sprite.bls layout:
//   perm_000  HAS_SRGB_ENCODE=false  HAS_SRGB_DECODE=false  passthrough
//   perm_001  HAS_SRGB_ENCODE=true   HAS_SRGB_DECODE=false  encode
//   perm_002  HAS_SRGB_ENCODE=false  HAS_SRGB_DECODE=true   decode
//   perm_003  HAS_SRGB_ENCODE=true   HAS_SRGB_DECODE=true   passthrough
//                                                          (the two
//                                                          ops cancel)
//
// Alpha is always passed through unchanged — only the RGB lanes are
// gamma-mapped. Reference: IEC 61966-2-1 sRGB transfer function (see
// linearToSrgb / srgbToLinearIEC in math/color.slang).

implementing wc3_shaders;

[shader("fragment")]
public float4 sprite_ps_main<
    let HAS_SRGB_ENCODE : bool,
    let HAS_SRGB_DECODE : bool>(
    SpritePSInput input) : SV_Target0
{
    float4 sample = t_sprite.Sample(s_sprite, input.uv);

    // Both flags set → the encode and decode cancel, matching the
    // shipped perm_003 which is identical to the no-flag perm_000.
    if (HAS_SRGB_ENCODE == HAS_SRGB_DECODE)
        return sample;

    float3 rgb = HAS_SRGB_ENCODE ? linearToSrgb(sample.rgb)
                                 : srgbToLinearIEC(sample.rgb);
    return float4(rgb, sample.a);
}
