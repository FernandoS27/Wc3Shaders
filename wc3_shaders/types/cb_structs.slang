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
    public float3 ambient;    public float _pad0;
    public float3 diffuse;    public float _pad1;
    public float3 position;   public float  type;
    public float4 _pad2;
};

// Per-draw constants for the HD pixel shader (b2).
// Holds fog + alpha-test parameters, view/projection matrices used by IBL,
// effect knobs (cloak, fresnel overlay, emissive gain), IBL blend state
// between the "from" and "to" probe cubemaps, and the active light list.
public struct PSPerDraw
{
    public float   alphaRef;         public float3 _p0;
    public float   fogStart;         public float  fogEnd;
    public float   fogDensity;       public float  _p1;
    public float4  fogColor;
    public float4x4 worldView;
    public float4x4 view;
    public float4x4 projection;
    public float4  viewport;
    public float   inverseSoftness;  public float  cloakAmount;
    public float   fresnelTeamColor; public float  _p2;
    public float4  _pad17;
    public float4  fresnelColor;
    public float   envFromMipEnd;    public float  envToMipEnd;
    public float   envTransitionT;   public float  _p3;
    public float   effectTime;       public float  emissiveGain;
    public float   lightCount;       public float  useNdf;
    public ShaderLight lights[8];
};

// The three cascade view-projection matrices used by the shadow map PCF
// sampling path. Uploaded at b1 in the vertex stage.
public struct ShadowCascades
{
    public float4x4 cascade0;
    public float4x4 cascade1;
    public float4x4 cascade2;
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
    public float4x4 world;
    public float4x4 worldView;
    public float4x4 worldViewProj;
    public float   effectTime;   public float popcornScale;
    public float   clipHeight;   public float underWater;
    public float4  diffuseColor;
    public float4  texMtx0Row0;
    public float4  texMtx0Row1;
    public float4  texMtx1Row0;
    public float4  texMtx1Row1;
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
    public float4x3 matrices[256];
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
    public uint  enabledShaders;  public float  debugMode;
    public float2 _p0;
    public float3 overrideAlbedo; public float  _p1;
    public float3 overrideOrm;    public float  _p2;
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
    public float   alphaRef;         public float3 _p0;
    public float   fogStart;         public float  fogEnd;
    public float   fogDensity;       public float  _p1;
    public float4  fogColor;

    public float4  _pad3;
    public float4  _pad4;
    public float4  _pad5;
    public float4  _pad6;

    public float4  invViewRow0;
    public float4  invViewRow1;
    public float4  invViewRow2;

    public float4  _pad10;
    public float4  _pad11;
    public float4  _pad12;
    public float4  _pad13;
    public float4  _pad14;
    public float4  _pad15;

    public float   inverseSoftness;  public float  cloakAmount;
    public float2  _p16;

    public float4  _pad17;
    public float4  _pad18;

    public float   envFromMipEnd;    public float  envToMipEnd;
    public float   envTransitionT;   public float  _p19;

    // lightCountSlot.z holds the uint-reinterpreted active light count.
    public float4  lightCountSlot;

    public ShaderLight lights[8];
};

// Scalar cascade count for the shadow sampler PS path (b1).
// Shared between the HD (IBL) and SD-on-HD pipelines — both consume the
// same cb1[0].x active-cascade count.
public struct ShadowCascadeCount
{
    public float numCascades;  public float3 _pad;
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
    public float4x4 world;
    public float4x4 worldViewProj;
    public float4   diffuseColor;
    public float4   texMtx0Row0;
    public float4   texMtx0Row1;
    public float4   texMtx1Row0;
    public float4   texMtx1Row1;
    public ShaderLight lights[8];
};

// Per-draw constants for the SD classic pixel shader (b0): the classic
// pipeline only needs fog and alpha-ref; everything else comes from the
// texture-stage ops and the interpolated vertex color.
public struct SDClassicPSPerDraw
{
    public float  alphaRef;     public float3 _p0;
    public float  fogStart;     public float  fogEnd;
    public float  fogDensity;   public float  _p1;
    public float4 fogColor;
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
    public float4  _pad0;                                 // cb2[0]

    public float   fogStart;        public float  fogEnd;
    public float   fogDensity;      public float  _p1;           // cb2[1]
    public float4  fogColor;                              // cb2[2]

    float4x4 _padWorldView;                        // cb2[3..6] (unused by water)
    public float4x4 view;                                 // cb2[7..10]
    public float4x4 worldToClip;                          // cb2[11..14]
    public float4   viewport;                             // cb2[15]

    // cb2[16]: HDR clamp applied to the emissive probe color.
    public float3   ambientClamp;   public float _p16;           // cb2[16]

    // cb2[17]: depth-based shore ramp distances.
    public float    refractSoftness;
    public float    litSoftness;    public float2 _p17;          // cb2[17]

    public float4   _pad18;                               // cb2[18]

    public float    envFromMipEnd;  public float  envToMipEnd;
    public float    envTransitionT; public float  _p19;          // cb2[19]

    public float    effectTime;     public float  _p20_y;
    public float    lightCount;     public float  useNdf;        // cb2[20]

    public ShaderLight lights[8];                         // cb2[21..]
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

// Per-draw constants for the foliage vertex shader (b2). 16 float4
// slots — superset of the terrain VS layout with two extra float4s for
// the constant tint colour (cb2[13]) and the 2×3 UV0 transform matrix
// (cb2[14..15]). The wind-animation perms additionally read
// `effectTime` from cb2[12].x.
//
//   shadowWorld   @ cb2[0..3]   — full 4×4 model→world used to feed the
//                                 cascade matrices in cb1 (shadow path
//                                 only).
//   worldRow0/1/2 @ cb2[4..6]   — three rows of the model→world rotation
//                                 (.xyz of each are the columns of the
//                                 world 3×3 used for normals/tangents).
//   worldTranslate@ cb2[7]      — model→world translation (.xyz only).
//   worldViewProj @ cb2[8..11]  — model→clip 4×4 for SV_Position.
//   effectTime    @ cb2[12].x   — engine clock used as the cos/sin
//                                 wind-swing phase. .yzw are unused.
//   tintColor     @ cb2[13]     — RGBA constant tint multiplied against
//                                 the per-vertex COLOR.
//   texMtx0Row0   @ cb2[14]     — UV0 affine row 0 (.xyw lanes only).
//   texMtx0Row1   @ cb2[15]     — UV0 affine row 1 (.xyw lanes only).
//
// The vertex-color slot is always read by the body, so the foliage VS
// has no `HAS_VERTEX_COLOR` axis. fxc still shrinks the cb declaration
// based on highest indexed access, so wind-on perms emit `CB2[16]`
// while wind-off perms emit `CB2[16]` as well (the latter still touches
// cb2[14..15] for the UV transform).
public struct FoliageVSPerDraw
{
    public float4x4 shadowWorld;     // cb2[0..3]
    public float4   worldRow0;       // cb2[4]
    public float4   worldRow1;       // cb2[5]
    public float4   worldRow2;       // cb2[6]
    public float4   worldTranslate;  // cb2[7]
    public float4x4 worldViewProj;   // cb2[8..11]
    public float    effectTime;      public float3 _padTime; // cb2[12]
    public float4   tintColor;       // cb2[13]
    public float4   texMtx0Row0;     // cb2[14]
    public float4   texMtx0Row1;     // cb2[15]
};

public ConstantBuffer<FoliageVSPerDraw> foliageVsCB2 : register(b2);

// Wind-area parameters (b3, only bound when WIND_ANIMATION is active).
//   .xy = world-space origin of the wind area (anchor for the gradient
//          texture mapping).
//   .zw = wind area size in world units (used as the UV scale).
public struct FoliageWindCB
{
    public float2 windOrigin;
    public float2 windSize;
};

public ConstantBuffer<FoliageWindCB> foliageVsCB3 : register(b3);

// Foliage VS wind-direction texture (only bound on WIND_ANIMATION
// perms). Single-channel scalar field sampled at three nearby texels
// to recover a 2D gradient.
public Texture2D<float4> t_foliageWind : register(t0);
public SamplerState      s_foliageWind : register(s0);

// Foliage pixel-shader textures. Slot ordering matches the engine
// bindings recovered from the BLS perms:
//   t0 albedo + alpha, t1 packed normal map (xyw lanes), t2 mask
//   (yz = roughness/metallic), t3 emissive, t4 lightmap / AO (.w lane).
// Cascaded shadow + IBL probe + BRDF LUT bindings are reused from the
// shared HD aliases (sd_t_shadow*, t_iblFrom/To, t_brdfLut).
public Texture2D<float4> t_foliageAlbedo   : register(t0);
public Texture2D<float4> t_foliageNormal   : register(t1);
public Texture2D<float4> t_foliageMask     : register(t2);
public Texture2D<float4> t_foliageEmissive : register(t3);
public Texture2D<float4> t_foliageLightmap : register(t4);

public SamplerState s_foliageAlbedo   : register(s0);
public SamplerState s_foliageNormal   : register(s1);
public SamplerState s_foliageMask     : register(s2);
public SamplerState s_foliageEmissive : register(s3);
public SamplerState s_foliageLightmap : register(s4);

// Per-draw constants for the terrain vertex shader (b2). The terrain
// pipeline uses a layout distinct from the HD `VSPerDraw`:
//
//   shadowWorld   @ cb2[0..3]   — full 4×4 model→world used to feed the
//                                 cascade matrices in cb1 (shadow path
//                                 only — fxc DCE'ds it on non-shadow
//                                 perms because nothing reads it).
//   worldRow0/1/2 @ cb2[4..6]   — three rows of the model→world rotation
//                                 (the .xyz of each are taken as the
//                                 columns of the world 3×3); used to
//                                 transform normals + tangents.
//   worldTranslate@ cb2[7]      — model→world translation (.xyz only).
//   worldViewProj @ cb2[8..11]  — model→clip 4×4 for SV_Position.
//   _padReserved  @ cb2[12]     — present in the no-vertex-color layout,
//                                 never read.
//   tintColor     @ cb2[13]     — RGBA constant tint, used only when the
//                                 vertex format does not carry COLOR.
//
// fxc's CB declaration shrinks based on the highest accessed index, so
// the vertex-color-feeding perms (which never read cb2[12..13]) emit
// `dcl_constantbuffer CB2[12]` while the constant-tint perms emit
// `CB2[14]` — even though the slang struct is fixed at 14 float4s.
public struct TerrainVSPerDraw
{
    public float4x4 shadowWorld;     // cb2[0..3]
    public float4   worldRow0;       // cb2[4]
    public float4   worldRow1;       // cb2[5]
    public float4   worldRow2;       // cb2[6]
    public float4   worldTranslate;  // cb2[7]
    public float4x4 worldViewProj;   // cb2[8..11]
    public float4   _padReserved;    // cb2[12]
    public float4   tintColor;       // cb2[13]
};

public ConstantBuffer<TerrainVSPerDraw> terrainVsCB2 : register(b2);

// Terrain pixel-shader splat textures. Slot ordering matches the
// engine bindings: t0 splat color array, t1 splat normal+gloss array
// (rg = nx/ny, w = gloss), t2 splat mask array (.yz = mask UV), t3
// global lightmap / AO (.w = AO factor). The terrain PS shares t10..t15
// (cascade shadows + IBL probes + BRDF LUT) with the HD pipeline via
// the existing aliases declared above.
public Texture2DArray<float4> t_splatColor  : register(t0);
public Texture2DArray<float4> t_splatNormal : register(t1);
public Texture2DArray<float4> t_splatMask   : register(t2);
public Texture2D<float4>      t_lightmap    : register(t3);

public SamplerState s_splatColor  : register(s0);
public SamplerState s_splatNormal : register(s1);
public SamplerState s_splatMask   : register(s2);
public SamplerState s_lightmap    : register(s3);

// Per-draw constants for the PopcornFX vertex shader (b2). Aliases
// the same buffer as `vsCB2` (the engine binds them at the same
// register slot) but exposes only the fields the popcorn pipeline
// actually consumes:
//
//   world         @ 0x040 — particle emitter's object → world transform.
//                           Lives at the same offset as VSPerDraw.worldView,
//                           because the engine binds the world matrix here
//                           when running the particle pipeline (the HD
//                           pipeline binds the worldView product instead).
//   worldViewProj @ 0x080 — particle emitter's object → clip transform.
//   popcornScale  @ 0x0C4 — scalar emitter scale applied to the local-
//                           space position before any matrix transform
//                           (i.e. at the .y slot of cb2[12]).
public struct PopcornVSPerDraw
{
    public float4x4 _padCB0;          // cb2[0..3] — unused by the popcorn VS
    public float4x4 world;            // cb2[4..7]
    public float4x4 worldViewProj;    // cb2[8..11]
    public float    _padTime;         // cb2[12].x  (the HD pipeline's `effectTime`)
    public float    popcornScale;     // cb2[12].y
    public float2   _padHeight;       // cb2[12].zw (HD's clipHeight / underWater)
};

public ConstantBuffer<PopcornVSPerDraw> popcornVsCB2 : register(b2);

// Per-draw constants for the PopcornFX pixel shader (b2). Aliases
// the same register slot as the HD `psCB2` (and `waterPsCB2`) — the
// engine reuses the same buffer slot for whichever pipeline is
// active, and slangc only emits the binding the live entry point
// actually references.
//
// Slot map (recovered from the original BLS PS perms; see
// re_shaders/popcornfx/popcorn_ps.hlsl for the per-feature comments):
//
//   cb2[0]          — unused by the popcorn PS
//   cb2[1].xyz      — fog start / end / density (FOG_*)
//   cb2[2].xyz      — fog color                  (FOG_*)
//   cb2[3..6]       — unused (HD's worldView slot)
//   cb2[7..10]      — per-particle world basis   (LIT IBL)
//   cb2[11..14]     — world → clip matrix        (SOFT_PARTICLES,
//                                                 MOTION_PASS, LIT IBL)
//   cb2[15]         — depth-buffer UV remap (.xy = scale, .zw = bias)
//   cb2[16].x       — depth-fade scale            (SOFT_PARTICLES,
//                                                 MOTION_PASS)
//   cb2[17..18]     — unused
//   cb2[19].x       — IBL `cubeFrom` roughness-to-mip multiplier
//   cb2[19].y       — IBL `cubeTo`   roughness-to-mip multiplier
//   cb2[19].z       — IBL cubemap blend (0 → cubeFrom, 1 → cubeTo)
//   cb2[20].z       — analytic light count       (LIT)
//   cb2[20].w       — "use smoothed normal" flag (LIT specular AA)
//   cb2[21..52]     — analytic light array (8 lights × 4 float4
//                     stride, sharing layout with HD's `ShaderLight`).
//                     lights[0] does double duty as the IBL ambient /
//                     base-material driver — its `.ambient` is added
//                     to the IBL diffuse contribution and its
//                     `.diffuse` is the IBL tint.
public struct PopcornPSPerDraw
{
    public float4   _pad0;                                  // cb2[0]
    public float    fogStart;       public float fogEnd;
    public float    fogDensity;     public float _p1;       // cb2[1]
    public float4   fogColor;                               // cb2[2]
    public float4x4 _padCB3;                                // cb2[3..6]

    // The following two "matrix" slots are declared as raw float4
    // arrays rather than `float4x4` because both are addressed with
    // non-contiguous component swizzles (.xyw / .xzy) and slang's
    // matrix-row accessors lower those to per-scalar loads instead
    // of a single vector load + swizzle. Storing them as raw arrays
    // matches the original engine's `float4 cb2[N]` access pattern.
    //
    //   particleBasis[0..2] — per-particle world basis used by the
    //                          LIT IBL block to rotate worldPos and
    //                          worldNormal into the cubemap-local
    //                          frame (.xzy swizzle).
    //   worldToClip[0..3]   — world → clip (3×4 affine, using the
    //                          .xyw lanes of each row; .z is unused).
    public float4   particleBasis[4];                       // cb2[7..10]
    public float4   worldToClip[4];                         // cb2[11..14]

    public float4   depthUVRemap;                           // cb2[15]
    public float    depthFadeScale; public float3 _p16;     // cb2[16]
    public float4   _pad17;                                 // cb2[17]
    public float4   _pad18;                                 // cb2[18]
    public float    iblMipFrom;     public float iblMipTo;
    public float    iblBlend;       public float _p19;      // cb2[19]
    public float2   _pad20_xy;
    public float    lightCount;     public float useSmoothedNormal; // cb2[20]
    public ShaderLight lights[8];                           // cb2[21..52]
};

public ConstantBuffer<PopcornPSPerDraw> popcornPsCB2 : register(b2);

// Popcorn pixel-shader textures. Slot ordering matches the engine
// bindings recovered from the BLS perms:
//   t0  — diffuse / motion-source (sampled with the per-mode UV);
//         in MOTION_PASS its .xy is the encoded motion vector.
//   t1  — alpha-remap LUT (1D ramp in U, random scalar in V).
//   t2  — packed normal map (LIT only).
//   t3  — smoothness (.y) + metallic (.z) (LIT only).
//   t4  — scene depth (SOFT_PARTICLES color-pass fade and the
//         MOTION_PASS depth-discard test).
//   t13 — IBL diffuse-irradiance + spec cubemap "from" probe (LIT).
//   t14 — IBL spec cubemap "to" probe (LIT) — blended by iblBlend.
//   t15 — split-sum BRDF integration LUT (LIT).
//
// Several of these aliases share register slots with the HD bindings
// declared above (e.g. t13/t14/t15 are also used by the HD IBL path);
// slangc only emits the bindings the active entry actually
// references, so the duplicate declarations are harmless.
public Texture2D<float4>        t_popcornDiffuse  : register(t0);
public Texture2D<float4>        t_popcornAlphaLut : register(t1);
public Texture2D<float4>        t_popcornNormal   : register(t2);
public Texture2D<float4>        t_popcornSmoothMet: register(t3);
public Texture2D<float4>        t_popcornDepth    : register(t4);
public TextureCubeArray<float4> t_popcornCubeFrom : register(t13);
public TextureCubeArray<float4> t_popcornCubeTo   : register(t14);
public Texture2D<float4>        t_popcornBrdfLut  : register(t15);

public SamplerState s_popcornDiffuse  : register(s0);
public SamplerState s_popcornAlphaLut : register(s1);
public SamplerState s_popcornNormal   : register(s2);
public SamplerState s_popcornSmoothMet: register(s3);
public SamplerState s_popcornDepth    : register(s4);
public SamplerState s_popcornCubeFrom : register(s13);
public SamplerState s_popcornCubeTo   : register(s14);
public SamplerState s_popcornBrdfLut  : register(s15);

// Per-draw constants for the tonemap pixel shader (b1). The engine
// only uploads a single 16-byte slot — the linear exposure multiplier
// applied to the HDR scene color before the ACES filmic curve — but
// the CB is still dimensioned as a full float4 to match the wire
// layout fxc emits for `float4 cb1[1]`.
public struct TonemapPSPerDraw
{
    public float  exposure;
    public float3 _p0;
};

public ConstantBuffer<TonemapPSPerDraw> tonemapPsCB1 : register(b1);

// Tonemap PS texture bindings. Aliases the same D3D slots the HD mesh
// PS uses (t0 / s0) — the tonemap pass runs on its own pipeline object
// and slangc only emits bindings the active entry point references, so
// these declarations coexist with the HD material bindings above.
//   t0 = linear-HDR scene target produced by the preceding lighting pass
//   s0 = linear-clamp sampler paired with the scene target
public Texture2D<float4> t_sceneHdr : register(t0);
public SamplerState      s_sceneHdr : register(s0);

// Sprite PS texture bindings. Aliases the same D3D slots the HD mesh
// PS uses (t0 / s0) — sprite draws bind their own source texture here
// and slangc only emits the binding for the active entry point, so the
// declaration coexists with the HD material bindings above.
public Texture2D<float4> t_sprite : register(t0);
public SamplerState      s_sprite : register(s0);

// Distortion PS texture bindings. The full-screen distortion pass
// samples a previous-frame scene texture (t0) at four offset UVs
// driven by an offset map (t1) — see `distortion_ps.slang` for the
// per-channel chromatic blend. Both bindings reuse the engine's
// shared t0/t1 slot numbers; slangc only emits whichever entry-
// point references them, so they coexist with the HD aliases above.
public Texture2D<float4> t_distortionScene : register(t0);
public Texture2D<float4> t_distortionMap   : register(t1);

public SamplerState s_distortionScene : register(s0);
public SamplerState s_distortionMap   : register(s1);
