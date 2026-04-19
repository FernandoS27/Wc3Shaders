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
// Holds transforms, the fog-depth clip plane (used to pre-bias world Z when
// underwater), vertex-color tint, and two 2x3 texture-coordinate matrices
// stored as two float4 rows each (the .z column is unused).
//
// Matrix field ordering here matches the engine's upload offsets exactly:
//   world          @ 0x000 — object → world.  Its .z after the transform
//                            is a world-space altitude that the fog apply()
//                            compares against `clipHeight` (the water-line);
//                            its full xyzw feeds the shadow cascade matrices.
//   worldView      @ 0x040 — object → view.   The VS emits this as the PS's
//                            "worldPos" TEXCOORD1 input. Strictly speaking
//                            view-space — the PS's `invView` transform
//                            brings it back out into probe space.
//   worldViewProj  @ 0x080 — object → clip.   Writes SV_POSITION.
public struct VSPerDraw
{
    float4x4 world;
    float4x4 worldView;
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
// `enabledShaders` is a bitfield toggling various debug overrides:
//   bit 0 — replace albedo.rgb with overrideAlbedo.
//   bit 1 — replace the t2 control map with overrideOrm (used by crystal):
//           scalar form (standard / ML) reads .x as a refractMask override,
//           vector form (IBL paths) reads .xyz as a full ORM override.
// `debugMode` selects one of the visualisation paths in debugVisualize().
// The `overrideOrm` slot is unused by the HD pipeline and exists here so
// the crystal pipeline shares the same b3 layout.
public struct DebugVisCB
{
    uint  enabledShaders;  float  debugMode;
    float2 _p0;
    float3 overrideAlbedo; float  _p1;
    float3 overrideOrm;    float  _p2;
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

// Scalar cascade count for the shadow sampler PS path (b1).
// Shared between the HD (IBL) and SD-on-HD pipelines — both consume the
// same cb1[0].x active-cascade count.
public struct ShadowCascadeCount
{
    float numCascades;  float3 _pad;
};

public ConstantBuffer<SDPSPerDraw>        sdPsCB2 : register(b2);
public ConstantBuffer<ShadowCascadeCount> sdPsCB1 : register(b1);

// Shadow cascades at t10-12 / s10-12 are shared between the HD IBL path
// and the SD-on-HD pipeline — both bind the same comparison-sampled
// 2D shadow maps here and run the PCF helper in sd_ps_shadow.
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

// Per-draw constants for the water pixel shader (b2). Shares most of
// its layout with `PSPerDraw` — fog, invView rotation, IBL probe params,
// effect time, and the analytic light list are identical slot-for-slot
// — but repurposes a few fields the HD path doesn't need:
//   • `worldToClip` (cb2[11..14], aka HD's `projection`) is used as the
//     full world-space → clip-space matrix by the SSR march.
//   • `ambientClamp` (cb2[16].xyz) is an HDR ceiling on the emissive
//     probe color (HD uses these 3 scalars as cloak / fresnel knobs).
//   • `refractSoftness` / `litSoftness` (cb2[17].xy) parameterise the
//     two depth-based shore ramps.
//
// `lights[8]` over-declares the last 16 bytes relative to what the
// engine actually uploads (which stops at light[7].position+type), but
// the water shader never reads that slot and D3D tolerates the extra.
public struct WaterPSPerDraw
{
    float4  _pad0;                                 // cb2[0]

    float   fogStart;        float  fogEnd;
    float   fogDensity;      float  _p1;           // cb2[1]
    float4  fogColor;                              // cb2[2]

    float4x4 _padWorldView;                        // cb2[3..6] (unused by water)
    float4x4 invView;                              // cb2[7..10]
    float4x4 worldToClip;                          // cb2[11..14]
    float4   viewport;                             // cb2[15]

    // cb2[16]: HDR clamp applied to the emissive probe color.
    float3   ambientClamp;   float _p16;           // cb2[16]

    // cb2[17]: depth-based shore ramp distances.
    float    refractSoftness;
    float    litSoftness;    float2 _p17;          // cb2[17]

    float4   _pad18;                               // cb2[18]

    float    envFromMipEnd;  float  envToMipEnd;
    float    envTransitionT; float  _p19;          // cb2[19]

    float    effectTime;     float  _p20_y;
    float    lightCount;     float  useNdf;        // cb2[20]

    ShaderLight lights[8];                         // cb2[21..]
};

public ConstantBuffer<WaterPSPerDraw> waterPsCB2 : register(b2);

// Water PS texture bindings. The water draw reuses HD slot numbers but
// binds different resources there — the aliases below make the intent
// explicit in the shader body. Slang only emits bindings that the
// compiled entry actually uses, so these don't collide with the HD
// `t_albedo` / `t_normal` / ... declarations above.
//   t0 = previous-frame scene color (for refraction & SSR hits)
//   t1 = previous-frame depth buffer, sampled as .x
//   t2 = tiling water normal map (DXT5nm-style two-channel packing)
//   t3 = emissive / lightmap probe (sampled at mip 16 ≈ average color)
//
//   s4 = scene-color sampler (note: paired with t0, not s0)
//   s1/s2/s3 match their texture counterparts
public Texture2D<float4> t_waterScene  : register(t0);
public Texture2D<float4> t_waterDepth  : register(t1);
public Texture2D<float4> t_waterNormal : register(t2);
public Texture2D<float4> t_waterProbe  : register(t3);

public SamplerState s_waterScene  : register(s4);
public SamplerState s_waterDepth  : register(s1);
public SamplerState s_waterNormal : register(s2);
public SamplerState s_waterProbe  : register(s3);
