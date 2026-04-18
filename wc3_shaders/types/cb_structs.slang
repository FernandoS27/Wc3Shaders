// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026, Fernando Sahmkow
// See LICENSE in the repository root for full terms.
//
// Constant buffer layouts, texture bindings, and sampler bindings shared
// across the HD, SD-on-HD, SD highspec, and SD classic pipelines.
//
// The struct member padding (`_padN`, `_pN`) is intentional: these layouts
// must match what the original Warcraft III engine uploads at each slot, so
// explicit padding pins the offsets of every named field to the expected
// 16-byte alignment boundaries imposed by D3D11 constant buffers.

implementing wc3_shaders;

// One analytical light as uploaded by the engine.
// type == 0 => directional (position is actually the light direction).
// type  > 0 => point light (position is a world-space position).
// _pad2.x is repurposed as "ambient weight" in the IBL combine path.
public struct ShaderLight
{
    float3 ambient;    float _pad0;
    float3 diffuse;    float _pad1;
    float3 position;   float  type;
    float4 _pad2;
};

// Per-draw constants for the HD pixel shader (b2).
// Holds fog + alpha-test parameters, view/projection matrices used by IBL,
// effect knobs (cloak, fresnel overlay, emissive gain), IBL blend state
// between the "from" and "to" probe cubemaps, and the active light list.
public struct PSPerDraw
{
    float   alphaRef;         float3 _p0;
    float   fogStart;         float  fogEnd;
    float   fogDensity;       float  _p1;
    float4  fogColor;
    float4x4 worldView;
    float4x4 invView;
    float4x4 projection;
    float4  viewport;
    float   inverseSoftness;  float  cloakAmount;
    float   fresnelTeamColor; float  _p2;
    float4  _pad17;
    float4  fresnelColor;
    float   envFromMipEnd;    float  envToMipEnd;
    float   envTransitionT;   float  _p3;
    float   effectTime;       float  emissiveGain;
    float   lightCount;       float  useNdf;
    ShaderLight lights[8];
};

// The three cascade view-projection matrices used by the shadow map PCF
// sampling path. Uploaded at b1 in the vertex stage.
public struct ShadowCascades
{
    float4x4 cascade0;
    float4x4 cascade1;
    float4x4 cascade2;
};

// Per-draw constants for the HD vertex shader (b2).
// Holds transforms, the fog-depth clip plane (used to pre-bias viewZ when
// underwater), vertex-color tint, and two 2x3 texture-coordinate matrices
// stored as two float4 rows each (the .z column is unused).
public struct VSPerDraw
{
    float4x4 viewProj;
    float4x4 world;
    float4x4 worldViewProj;
    float   effectTime;   float popcornScale;
    float   clipHeight;   float underWater;
    float4  diffuseColor;
    float4  texMtx0Row0;
    float4  texMtx0Row1;
    float4  texMtx1Row0;
    float4  texMtx1Row1;
};

// Skeletal palette at b3: up to 256 bones, each an affine transform.
// Declared as float4x3 so Slang's default column-major CB packing places
// the three columns in three consecutive float4s — preserving the engine
// upload layout (256 bones * 48 bytes = 12288 bytes) while letting
// skinning code treat each entry as a proper matrix. The matrix is the
// transpose of the engine's 3x4 row-major affine, so world-space results
// come from `mul(float4(pos, 1), M)` (row vector on the left).
public struct BonePalette
{
    float4x3 matrices[256];
};

// Debug visualisation controls (b3 in PS).
// `enabledShaders` is a bitfield toggling various debug overrides;
// `debugMode` selects one of the visualisation paths in debugVisualize().
public struct DebugVisCB
{
    uint  enabledShaders;  float debugMode;
    float2 _p0;
    float3 overrideAlbedo; float _p1;
};

public ConstantBuffer<PSPerDraw>      psCB2 : register(b2);
public ConstantBuffer<DebugVisCB>     psCB3 : register(b3);
public ConstantBuffer<ShadowCascades> vsCB1 : register(b1);
public ConstantBuffer<VSPerDraw>      vsCB2 : register(b2);
public ConstantBuffer<BonePalette>    vsCB3 : register(b3);

// HD material textures. Slot ordering matches the engine-side bindings:
//   t0 albedo, t1 normal (XY + AA mask in W), t2 ORM (occ/rough/metal/mask),
//   t3 emissive, t4 team color mask.
//   t13/t14 are the two IBL cubemap arrays we blend between during
//   environment transitions; t15 is the split-sum BRDF LUT.
public Texture2D<float4>         t_albedo    : register(t0);
public Texture2D<float4>         t_normal    : register(t1);
public Texture2D<float4>         t_orm       : register(t2);
public Texture2D<float4>         t_emissive  : register(t3);
public Texture2D<float4>         t_teamColor : register(t4);
public TextureCubeArray<float4>  t_iblFrom   : register(t13);
public TextureCubeArray<float4>  t_iblTo     : register(t14);
public Texture2D<float4>         t_brdfLut   : register(t15);

public SamplerState s_albedo    : register(s0);
public SamplerState s_normal    : register(s1);
public SamplerState s_orm       : register(s2);
public SamplerState s_emissive  : register(s3);
public SamplerState s_teamColor : register(s4);
public SamplerState s_iblFrom   : register(s13);
public SamplerState s_iblTo     : register(s14);
public SamplerState s_brdfLut   : register(s15);

// Per-draw constants for the SD-on-HD pixel shader (b2).
// The layout mirrors the original SD engine CB, so most fields are
// preserved as padding; only the values that are actually consumed by the
// remastered path (fog, alpha ref, inv-view rows, env mip ranges, cloak,
// and lights) are named. invViewRowN are the rows of the transposed
// view-to-world rotation used to orient the IBL cubemap samples.
public struct SDPSPerDraw
{
    float   alphaRef;         float3 _p0;
    float   fogStart;         float  fogEnd;
    float   fogDensity;       float  _p1;
    float4  fogColor;

    float4  _pad3;
    float4  _pad4;
    float4  _pad5;
    float4  _pad6;

    float4  invViewRow0;
    float4  invViewRow1;
    float4  invViewRow2;

    float4  _pad10;
    float4  _pad11;
    float4  _pad12;
    float4  _pad13;
    float4  _pad14;
    float4  _pad15;

    float   inverseSoftness;  float  cloakAmount;
    float2  _p16;

    float4  _pad17;
    float4  _pad18;

    float   envFromMipEnd;    float  envToMipEnd;
    float   envTransitionT;   float  _p19;

    // lightCountSlot.z holds the uint-reinterpreted active light count.
    float4  lightCountSlot;

    ShaderLight lights[8];
};

// Scalar cascade count for the SD-on-HD shadow sampler (b1).
public struct SDShadowCascadeCount
{
    float numCascades;  float3 _pad;
};

public ConstantBuffer<SDPSPerDraw>          sdPsCB2 : register(b2);
public ConstantBuffer<SDShadowCascadeCount> sdPsCB1 : register(b1);

// SD-on-HD shadow cascades use separate textures + comparison samplers so
// each cascade can have its own resolution; PCF is done in sd_ps_shadow.
public Texture2D<float4>      sd_t_shadow0 : register(t10);
public Texture2D<float4>      sd_t_shadow1 : register(t11);
public Texture2D<float4>      sd_t_shadow2 : register(t12);
public SamplerComparisonState sd_s_shadow0 : register(s10);
public SamplerComparisonState sd_s_shadow1 : register(s11);
public SamplerComparisonState sd_s_shadow2 : register(s12);

// Per-draw constants for the SD highspec vertex shader (b0): transforms,
// tint, two texcoord matrices, and the light list — per-vertex lighting
// happens entirely in the VS here, so the PS path doesn't need them.
public struct SDHighspecVSPerDraw
{
    float4x4 world;
    float4x4 worldViewProj;
    float4   diffuseColor;
    float4   texMtx0Row0;
    float4   texMtx0Row1;
    float4   texMtx1Row0;
    float4   texMtx1Row1;
    ShaderLight lights[8];
};

// Per-draw constants for the SD classic pixel shader (b0): the classic
// pipeline only needs fog and alpha-ref; everything else comes from the
// texture-stage ops and the interpolated vertex color.
public struct SDClassicPSPerDraw
{
    float  alphaRef;     float3 _p0;
    float  fogStart;     float  fogEnd;
    float  fogDensity;   float  _p1;
    float4 fogColor;
};

public ConstantBuffer<SDHighspecVSPerDraw>  sdHighspecVsCB0 : register(b0);
public ConstantBuffer<SDClassicPSPerDraw>   sdClassicPsCB0  : register(b0);
