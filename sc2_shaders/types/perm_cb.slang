// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026, Fernando Sahmkow
// See LICENSE in the repository root for full terms.
//
// The permutation constant buffer (`b2`) — design section 5.1, plan milestone R3.
//
// Every `b_*` axis that only selects data inside an already-taken code path moves
// out of the `#define` set and into this buffer, so one compiled shader can serve
// every retail permutation that differs only in those axes.  What stays compile-time
// is exactly what a draw call cannot change: resource bindings, the output
// signature, the input signature, discard, and loop trip counts.
//
// TWO PROPERTIES MAKE THIS SAFE TO LAND INCREMENTALLY.
//
// 1. `SC2_USE_PERM_CB` defaults to 0, and `SC2_AXIS(x)` then expands to the plain
//    `#define` — character-identical preprocessor output, hence bit-identical DXBC.
//    That is the R3 exit gate, and it is the safety net for the mechanical rewrite
//    of the layer setup functions that follows.
//
// 2. The BIT LAYOUT lives in the packer, not here.  tools/sc2_perm_reduce.py emits
//    one `SC2_PERMOFF_<axis>` / `SC2_PERMW_<axis>` pair per data axis in the class's
//    define list, so this file never encodes an offset and the shader and the packer
//    cannot drift apart.  Adding an axis to the payload changes no shader code.
//
// Note for the whole-module build: sc2_shaders is ONE translation unit, so the
// `#if` below fires inside every family's file.  That is intentional here — this is
// a shared facility, not a family feature.  Per-family adoption happens at the CALL
// SITES, never by assuming a `#if` in one family's file is invisible to another's
// (see docs/SC2_PERM_REDUCTION_PLAN.md, the one-TU leak trap).

implementing sc2_shaders;

#ifndef SC2_USE_PERM_CB
#define SC2_USE_PERM_CB 0
#endif

// float4 registers in the payload.  The layout is global, so there is exactly one
// size and it comes from the generated header -- never from a guess here, because a
// declaration wider than the packer writes is silently wrong in the other direction
// (the tail reads as zero rather than failing to compile).
#include "types/perm_layout.slangh"

#if SC2_USE_PERM_CB

// b0 stays the retail `psGlobals` block byte-for-byte, so the name-aligned constant
// comparison in the validation harness is untouched; the payload gets its own bank.
struct SC2PermCBT
{
    uint4 w[SC2_PERM_CB_VEC4S];
};
ConstantBuffer<SC2PermCBT> permCB : register(b2);

// One packed field, addressed by absolute BIT offset.  A uint4 is 128 bits, so the
// register index is off/128 and the lane is (off/32)%4.  The packer guarantees no
// field straddles a 32-bit word, which is what keeps this to a shift and a mask.
uint sc2PermBits(uint off, uint width)
{
    uint word = permCB.w[off >> 7u][(off >> 5u) & 3u];
    return (word >> (off & 31u)) & ((1u << width) - 1u);
}

#endif
