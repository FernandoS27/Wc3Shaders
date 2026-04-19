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
    float3 position    : ATTR0;
    float3 normal      : ATTR1;
    float4 vertColor   : ATTR2;
    float2 uv0         : ATTR3;
    float2 uv1         : ATTR4;
    float4 weights     : ATTR5;
    uint4  boneIndices : ATTR6;
    float4 tangent     : ATTR7;
};

// Full HD vertex-to-pixel payload.
// worldPos.w carries the pre-computed fog-depth scalar (see
// computeFogDepth) rather than a real homogeneous coordinate — the PS
// reads it to evaluate fog without recomputing the view-space Z.
// The three shadowClip slots are the three cascade-space clip positions;
// they are zero-initialised when the shadow path is disabled.
public struct VSOutput
{
    float4 position    : SV_Position;
    float4 color       : COLOR0;
    float4 texCoord    : TEXCOORD0;
    float4 worldPos    : TEXCOORD1;
    float4 normalWS    : TEXCOORD2;
    float4 tangentWS   : TEXCOORD3;
    float4 shadowClip0 : TEXCOORD4;
    float4 shadowClip1 : TEXCOORD5;
    float4 shadowClip2 : TEXCOORD6;
};

// Compact payload used by the SD classic pipeline, which does all
// lighting in the vertex shader and therefore only needs the lit color,
// the transformed UVs and a world-space position for fog.
public struct SDVSOutput
{
    float4 position    : SV_Position;
    float4 color       : COLOR0;
    float4 texCoord    : TEXCOORD0;
    float3 worldPos    : TEXCOORD1;
};

// Water vertex stream (subset of the HD stream — no tangent, no UV1,
// no skinning). Kept separate from VSInput so the water VS emits a
// 4-attribute input signature matching the engine's water vertex
// buffer rather than the 8-attribute HD one.
public struct WaterVSInput
{
    float3 position    : ATTR0;
    float3 normal      : ATTR1;
    float4 vertColor   : ATTR2;
    float2 uv          : ATTR3;
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
    float4 position    : SV_Position;
    float4 color       : COLOR0;
    float2 texCoord    : TEXCOORD0;
    float3 worldPos    : TEXCOORD1;
    float3 normalWS    : TEXCOORD3;
    float4 upAxisWS    : TEXCOORD4;
};
