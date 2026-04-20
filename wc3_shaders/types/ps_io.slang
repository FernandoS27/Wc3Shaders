// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026, Fernando Sahmkow
// See LICENSE in the repository root for full terms.
//
// Pixel shader input and output layouts. Mirrors the VS output types in
// vs_io.slang — the semantic slot indices must match.

implementing wc3_shaders;

// HD pixel shader input. Mirrors VSOutput; see vs_io.slang for the
// meaning of each slot. Note that worldPos.w carries the fog-depth
// scalar rather than a homogeneous coordinate.
public struct PSInput
{
    float4 svPosition   : SV_Position;
    float4 vertColor    : COLOR0;
    float4 uv           : TEXCOORD0;
    float4 worldPos     : TEXCOORD1;
    float3 normalWS     : TEXCOORD2;
    float4 tangentWS    : TEXCOORD3;
    float4 shadowClip0  : TEXCOORD4;
    float4 shadowClip1  : TEXCOORD5;
    float4 shadowClip2  : TEXCOORD6;
};

// HD pixel shader output. When the MRT path is disabled the specular and
// normal targets are compiled out entirely thanks to `Conditional`, so
// the shader binds a single render target just like the non-MRT builds.
// When the depth-prepass path is enabled, `color` itself is compiled out
// — the prepass pass binds no color render targets and the shipped game
// shader has no SV_Target entries in that signature; emitting one would
// force a dummy color write against a non-existent RTV.
//   color    = final lit + fog + color-matrixed RGBA
//   specular = view-space depth (x) for screen-space resolve effects
//   normal   = packed world-space normal (xyz) + pre-fog alpha (w)
public struct PSOutput<let IS_MRT : bool, let HAS_COLOR : bool>
{
    Conditional<float4, HAS_COLOR> color    : SV_Target0;
    Conditional<float4, IS_MRT>    specular : SV_Target1;
    Conditional<float4, IS_MRT>    normal   : SV_Target2;
};

// SD classic pipeline pixel shader input. No normals or tangents because
// the classic renderer has no per-pixel lighting — the vertex color
// already contains the lit result.
public struct SDPSInput
{
    float4 svPosition : SV_Position;
    float4 vertColor  : COLOR0;
    float4 uv         : TEXCOORD0;
    float3 worldPos   : TEXCOORD1;
};

// Water pixel shader input. Matches `WaterVSOutput` slot-for-slot:
// TEXCOORD2 is skipped (the water pass has no tangent attribute), the
// world-space geometry normal arrives on TEXCOORD3, and TEXCOORD4
// carries the per-object "up" axis used as the TBN tangent.
public struct WaterPSInput
{
    float4 svPosition   : SV_Position;
    float4 vertColor    : COLOR0;
    float2 uv           : TEXCOORD0;
    float3 worldPos     : TEXCOORD1;
    float3 geomNormalWS : TEXCOORD3;
    float4 upAxisWS     : TEXCOORD4;
};

// SD classic pipeline PS output. Single target; no MRT, no debug split.
public struct SDPSOutput
{
    float4 color : SV_Target0;
};
