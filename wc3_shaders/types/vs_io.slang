// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026, Fernando Sahmkow
// See LICENSE in the repository root for full terms.
//
// Vertex shader input and output layouts.
//
// `VSInput` is generic over an `IVertexFormat` policy so that unused
// attributes (tangent, uv1, vertex color, etc.) can be compiled out
// cleanly at specialization time without changing the input element
// layout — the engine always binds the full attribute set.

implementing wc3_shaders;

// Full HD vertex attribute stream.
// `weights` + `boneIndices` are only consulted when the skinning policy
// is `FourBoneSkinning`; `tangent.w` carries the handedness sign used to
// reconstruct the bitangent in the pixel shader.
public struct VSInput<TVF : IVertexFormat>
{
    public float3 position    : ATTR0;
    public float3 normal      : ATTR1;
    public float4 vertColor   : ATTR2;
    public float2 uv0         : ATTR3;
    public float2 uv1         : ATTR4;
    public float4 weights     : ATTR5;
    public uint4  boneIndices : ATTR6;
    public float4 tangent     : ATTR7;
};

// Full HD vertex-to-pixel payload.
// worldPos.w carries the pre-computed fog-depth scalar (see
// computeFogDepth) rather than a real homogeneous coordinate — the PS
// reads it to evaluate fog without recomputing the view-space Z.
// The three shadowClip slots carry the cascade-space clip positions
// when shadows are enabled; they compile out entirely via `Conditional`
// in the non-shadow permutations — the shipped game shader omits them
// from the VS output signature rather than emitting dead zero writes,
// and we follow that layout so VS→PS linkage matches.
public struct VSOutput<let HAS_SHADOWS : bool>
{
    public float4 position    : SV_Position;
    public float4 color       : COLOR0;
    public float4 texCoord    : TEXCOORD0;
    public float4 worldPos    : TEXCOORD1;
    public float3 normalWS    : TEXCOORD2;
    public float4 tangentWS   : TEXCOORD3;
    public Conditional<float4, HAS_SHADOWS> shadowClip0 : TEXCOORD4;
    public Conditional<float4, HAS_SHADOWS> shadowClip1 : TEXCOORD5;
    public Conditional<float4, HAS_SHADOWS> shadowClip2 : TEXCOORD6;
};

// Compact payload used by the SD classic pipeline, which does all
// lighting in the vertex shader and therefore only needs the lit color,
// the transformed UVs and a world-space position for fog.
public struct SDVSOutput
{
    public float4 position    : SV_Position;
    public float4 color       : COLOR0;
    public float4 texCoord    : TEXCOORD0;
    public float3 worldPos    : TEXCOORD1;
};

// Water vertex stream (subset of the HD stream — no tangent, no UV1,
// no skinning). Kept separate from VSInput so the water VS emits a
// 4-attribute input signature matching the engine's water vertex
// buffer rather than the 8-attribute HD one.
public struct WaterVSInput
{
    public float3 position    : ATTR0;
    public float3 normal      : ATTR1;
    public float4 vertColor   : ATTR2;
    public float2 uv          : ATTR3;
};

// Water VS → PS payload. Differs from VSOutput in two ways:
//   • no TEXCOORD2 slot — the water PS takes its geometry normal on
//     TEXCOORD3 to free TEXCOORD2 for an engine-wide reserved slot.
//   • TEXCOORD4 carries a fixed "up" axis (world Y column), which the
//     water PS uses as the TBN tangent instead of a per-vertex tangent.
// Member types are trimmed to the exact components the PS reads, so
// the compiled VS declares `dcl_output oN.xy/xyz/xyzw` accordingly
// (and doesn't emit dead writes to unused lanes).
public struct WaterVSOutput
{
    public float4 position    : SV_Position;
    public float4 color       : COLOR0;
    public float2 texCoord    : TEXCOORD0;
    public float3 worldPos    : TEXCOORD1;
    public float3 normalWS    : TEXCOORD3;
    public float4 upAxisWS    : TEXCOORD4;
};

// PopcornFX vertex stream. Mirrors the convention used by `VSInput<TVF>`:
// every possible attribute is declared up-front (so the input-element
// layout the engine binds stays fixed across perms), and the entry
// shader gates per-feature processing on the policy types — the dead
// loads compile out cleanly under specialization.
//
// Slot map (recovered from the original BLS input signatures):
//
//   ATTR0  pivot-blended position
//   ATTR1  object-space normal              (live when HAS_NT)
//   ATTR2  per-vertex RGBA color            (live when HAS_VC)
//   ATTR3  primary UV                       (live when TMode.hasUV;
//                                            BILLBOARD declares it but
//                                            ignores it — the engine
//                                            still binds the stream)
//   ATTR4  BILLBOARD: axis0 (.xyzw) /
//          ATLAS:     uv1   (.xy)
//   ATTR5  BILLBOARD: axis1 (.xyzw) /
//          ATLAS:     lifeRatio (.x)
//   ATTR6  per-particle random scalar       (live when HAS_RANDOM
//                                            and TMode.hasUV — the
//                                            engine never binds it
//                                            without a UV stream to
//                                            attach random to)
//   ATTR7  object-space tangent + handedness in .w  (live when HAS_NT)
//   ATTR8  particle pivot (.xyz) + per-vertex blend weight (.w)
//   SV_VertexID                              (read only by BILLBOARD)
//
// ATTR4/ATTR5 are declared as float4 to cover the BILLBOARD axis
// vectors; the ATLAS path uses the .xy / .x lanes of the same slot.
// The two interpretations are unwrapped by `popcornUv1FromSlot4` and
// `popcornLifeFromSlot5` in popcorn_vs.slang to keep the entry body
// readable.
public struct PopcornVSInput
{
    public float3 position    : ATTR0;
    public float3 normal      : ATTR1;
    public float4 vertColor   : ATTR2;
    public float2 uv0         : ATTR3;
    public float4 modeSlot4   : ATTR4;
    public float4 modeSlot5   : ATTR5;
    public float  random      : ATTR6;
    public float4 tangent     : ATTR7;
    public float4 pivot       : ATTR8;
    public uint   vertexID    : SV_VertexID;
};

// PopcornFX VS → PS payload. The conditional fields mirror the live
// per-perm slots so the wire signature only carries what the active
// permutation actually populates.
//
// Slot map:
//
//   SV_Position    clipPos       — always
//   COLOR0         color         — HAS_VC
//   TEXCOORD0      uv (float4)   — TMode.hasUV (basic & billboard
//                                  use only the .xy lanes; ATLAS
//                                  populates the full .xyzw — the
//                                  PS reads .xy / .xyzw as needed)
//   TEXCOORD1      slot1Out      — BILLBOARD (carries axis0,  .xyzw) /
//                                  ATLAS     (carries lifeOut, only
//                                  .x is read by the PS)
//   TEXCOORD2      axis1Out      — BILLBOARD only (.xyzw)
//   TEXCOORD3      randomOut     — HAS_RANDOM (& hasUV)
//   TEXCOORD4      normalWS      — HAS_NT
//   TEXCOORD5      tangentWS     — HAS_NT
//   TEXCOORD7      worldPos      — always
//
// TEXCOORD0 / TEXCOORD1 are always declared as float4 when active —
// the BILLBOARD axis vectors are the widest interpretation, so packing
// every mode through them keeps the slot count at one Conditional
// field per semantic (downstream HLSL emission rejects multiple
// Conditional fields sharing a semantic).
public struct PopcornVSOutput<
    TMode : IPopcornMode,
    let HAS_RANDOM : bool,
    let HAS_VC     : bool,
    let HAS_NT     : bool>
{
    public float4 position : SV_Position;

    public Conditional<float4, HAS_VC> color : COLOR0;

    public Conditional<float4, TMode.hasUV>       uv       : TEXCOORD0;
    public Conditional<float4, TMode.isBillboard> axis0Out : TEXCOORD1;
    public Conditional<float4, TMode.isAtlas>     lifeOut  : TEXCOORD1;
    public Conditional<float4, TMode.isBillboard> axis1Out : TEXCOORD2;

    public Conditional<float4, HAS_RANDOM> randomOut : TEXCOORD3;

    public Conditional<float3, HAS_NT> normalWS  : TEXCOORD4;
    public Conditional<float4, HAS_NT> tangentWS : TEXCOORD5;

    public float3 worldPos : TEXCOORD7;
};
