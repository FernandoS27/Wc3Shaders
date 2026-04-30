// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026, Fernando Sahmkow
// See LICENSE in the repository root for full terms.
//
// Compile-time policy describing which PopcornFX particle "rendering
// mode" a given permutation belongs to. The four modes mirror the
// engine's authoring axes:
//
//   PopcornNoUV       — particle has no UV stream. The VS just
//                       transforms the pivot-blended position; no
//                       per-mode auxiliary outputs are emitted.
//                       (Engine collapses BILLBOARD/ATLAS perms with
//                        no UV stream onto this same shader.)
//   PopcornBasicUV    — flat UV passed straight through TEXCOORD0.xy.
//                       The simplest "with diffuse texture" particle.
//   PopcornBillboard  — quad UVs derived from SV_VertexID inside the
//                       VS, with two per-vertex axis vectors (ATTR4 /
//                       ATTR5) forwarded to the PS so it can rebuild
//                       the world-space billboard frame.
//   PopcornAtlas      — two-frame atlas blend. UV0 (current frame) +
//                       UV1 (next frame) are packed into TEXCOORD0,
//                       and the integer-truncated life ratio's
//                       fractional part rides on TEXCOORD1.x as the
//                       blend factor.

implementing wc3_shaders;

public interface IPopcornMode
{
    // True when the engine binds a per-vertex UV stream (ATTR3) and
    // the VS therefore needs to emit a TEXCOORD0 the PS can sample.
    static const bool hasUV;

    // Billboard-specific axis stream + SV_VertexID-driven quad UV.
    static const bool isBillboard;

    // Atlas blend stream + life-ratio fractional output.
    static const bool isAtlas;
};

// No UV stream — the engine never binds ATTR3 / ATTR4 / ATTR5 / ATTR6
// in this case, so the VS reduces to "pivot-blend, transform, emit
// worldPos". MODE_BILLBOARD / MODE_ATLAS perms with uv_variant=0
// collapse onto this same shader (the per-mode UV machinery has
// nothing to feed).
public struct PopcornNoUV : IPopcornMode
{
    static const bool hasUV       = false;
    static const bool isBillboard = false;
    static const bool isAtlas     = false;
};

// Plain UV pass-through. The PS reads TEXCOORD0.xy directly.
public struct PopcornBasicUV : IPopcornMode
{
    static const bool hasUV       = true;
    static const bool isBillboard = false;
    static const bool isAtlas     = false;
};

// Billboard particle: the VS computes a (0,0)/(0,1)/(1,0)/(1,1) quad
// UV from SV_VertexID, and forwards the two ATTR4/ATTR5 axis vectors
// raw to the PS for world-space frame reconstruction.
public struct PopcornBillboard : IPopcornMode
{
    static const bool hasUV       = true;
    static const bool isBillboard = true;
    static const bool isAtlas     = false;
};

// Atlas-blended particle: ATTR3 carries UV0, ATTR4.xy carries UV1
// (the next atlas frame), and ATTR5.x carries the life ratio whose
// fractional part is the inter-frame blend factor.
public struct PopcornAtlas : IPopcornMode
{
    static const bool hasUV       = true;
    static const bool isBillboard = false;
    static const bool isAtlas     = true;
};
