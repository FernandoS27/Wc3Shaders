// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026, Fernando Sahmkow
// See LICENSE in the repository root for full terms.
//
// Terrain splat-layer sampling and accumulation.
//
// A terrain pixel is the weighted blend of up to four splat "layers" —
// each layer contributing albedo, a tangent-space normal, and a 2-channel
// mask UV. The layer's slot index lives in `boneIDs.{x,y,z,w}`; the
// special value `0xFFFF` flags the slot as inactive and skips the
// fetch. Layer weights are driven by `vertColor.w · albedoSample.w` so
// editor brushes can blend layers per-vertex.
//
// All four layers share the same texture array bindings (t0/t1/t2)
// indexed by the layer slot, so a single set of samplers covers the
// whole splat catalog.

implementing wc3_shaders;

// One layer's three texture fetches plus the optional TINT_OVERRIDE
// substitution. Returns the sampled (albedo.rgba), packed normal map
// xyw lanes (the W lane carries the high-precision X for the normal
// decode), and the splat-mask UV.
public struct SplatLayerSample
{
    public float4 albedo;       // .rgb = base color, .a = layer weight
    public float3 normalXYW;    // raw .xyw of the normal+gloss texture
    public float2 maskUV;       // .yz of the splat mask texture
};

// Sample one splat layer. The `tintMaskBits` argument carries the same
// `cb3[0].x & {1, 2}` packed mask the original used; the entry shader
// passes a zeroed pair on non-tint perms so the swap branches collapse
// out at compile time.
//
// Notes mirror the retail DXBC's choices that the HLSL reconstruction
// preserves: the tint bitmask is read as a uint pair so fxc folds the
// result straight into `movc` predicates, avoiding the `itof + ne`
// round-trip a float-typed ternary would emit.
public SplatLayerSample sampleSplatLayer<let HAS_TINT : bool>(
    uint layerIndex, float2 baseUV)
{
    SplatLayerSample s;

    float3 splatCoord = float3(baseUV, (float)layerIndex);
    s.albedo    = t_splatColor.Sample(s_splatColor,   splatCoord);
    s.normalXYW = t_splatNormal.Sample(s_splatNormal, splatCoord).xyw;
    s.maskUV    = t_splatMask.Sample(s_splatMask,     splatCoord).yz;

    if (HAS_TINT)
    {
        uint2 tintBits = asuint(psCB3.enabledShaders.xx) & uint2(1, 2);
        s.albedo.xyz = (tintBits.x != 0) ? psCB3.overrideAlbedo : s.albedo.xyz;
        s.maskUV     = (tintBits.y != 0) ? psCB3.overrideOrm.yz : s.maskUV;
    }

    return s;
}

// Decode the packed normal sample into a tangent-space unit normal.
// Mirrors `math/normal.decodeNormalMap` but inlined here because the
// terrain encoding is slightly different — `texXYW.x` carries the
// high-byte X factor, `.z` (which is the source texture's W lane) is
// the corresponding low-byte X precision, and `.y` is Y. Reconstructing
// nz from `1 - nx² - ny²` is the standard two-channel decode.
public float3 decodeSplatNormal(float3 nxyw)
{
    // nx = 2 · highX · lowX - 1 (the .xz product reproduces the engine's
    // dual-channel X precision).
    float nx = dot(nxyw.xx, nxyw.zz) - 1.0;
    float yScaled = nxyw.y * 2.0 + (-1.0);
    float ny = -yScaled;

    float nz2 = 1.0 - (nx * nx + yScaled * yScaled);
    nz2 = max(0.0, nz2);
    float nz = sqrt(nz2);

    float3 n = float3(nx, ny, nz);
    return n * rsqrt(dot(n, n));
}

// Bias the splat mask UV from the [0, 1] sample range into the
// [0.5/256, 255.5/256] half-texel-centred range expected by the splat
// mask LUT, then weight by the layer's blend factor.
//   uv' = uv · (255/256, 1) + (0.5/256, 0)
//
// Returns the per-layer (weighted) contribution; the body adds these up
// across layers.
public float2 biasMaskUV(float2 maskUV, float weight)
{
    float2 biased = maskUV * float2(0.99609375, 1.0) + float2(0.00390625, 0.0);
    // The .yx swizzle here matches the engine: stored layout is
    // (mask.x, mask.y) but the shader consumes them as (y, x) — fold
    // the swizzle into the weighted accumulate.
    return weight.xx * biased.yx;
}

// Initialize the splat accumulators from layer 0. When the layer is
// inactive the accumulators take their identity defaults: zero color,
// straight-up TS normal, zero mask UV.
public void initSplatAccum<let HAS_TINT : bool>(
    bool active, uint layerIndex, float2 baseUV, float vertWeight,
    out float3 albedoAccum, out float3 normalAccum,
    out float2 maskUVAccum)
{
    if (active)
    {
        SplatLayerSample s = sampleSplatLayer<HAS_TINT>(layerIndex, baseUV);
        float3 tsN = decodeSplatNormal(s.normalXYW);

        float layerWeight = vertWeight * s.albedo.w;
        albedoAccum = layerWeight.xxx * s.albedo.xyz;

        // Bend the TS normal toward (0,0,1) by `(1 - layerWeight)` —
        // i.e. lerp from straight-up toward the sampled normal so unlit
        // areas stay flat. The math expressed as a single mad chain
        // mirrors the retail DXBC.
        float3 nCentered = tsN + float3(0, 0, -1);
        float3 nWeighted = layerWeight.xxx * nCentered + float3(0, 0, 1);
        normalAccum = nWeighted * rsqrt(dot(nWeighted, nWeighted));

        maskUVAccum = biasMaskUV(s.maskUV, layerWeight);
    }
    else
    {
        albedoAccum = float3(0, 0, 0);
        normalAccum = float3(0, 0, 1);
        maskUVAccum = float2(0, 0);
    }
}

// Lerp one additional splat layer into the running accumulators. Each
// layer's contribution is weighted by `vertWeight · sampledAlpha`.
//
// The mask UV accumulate folds the previous accumulator out of the new
// reading (the `+ -prev` term) so the per-layer contributions add
// linearly — terms cancel perfectly for the inactive case (weight 0)
// and the no-overlap case (mask perfectly aligned).
public void lerpSplatLayer<let HAS_TINT : bool>(
    bool active, uint layerIndex, float2 baseUV, float vertWeight,
    inout float3 albedoAccum, inout float3 normalAccum,
    inout float2 maskUVAccum)
{
    if (!active) return;

    SplatLayerSample s = sampleSplatLayer<HAS_TINT>(layerIndex, baseUV);
    float3 tsN = decodeSplatNormal(s.normalXYW);

    float w = vertWeight * s.albedo.w;

    // Albedo lerp: prev + w · (new - prev).
    albedoAccum = w.xxx * (s.albedo.xyz - albedoAccum) + albedoAccum;

    // TS-normal lerp: prev + w · (new - prev), then renormalise.
    float3 deltaN = tsN - normalAccum;
    float3 nLerp = w.xxx * deltaN + normalAccum;
    normalAccum  = nLerp * rsqrt(dot(nLerp, nLerp));

    // Mask UV: weighted (yx-swizzled) delta.
    float2 biased = s.maskUV * float2(0.99609375, 1.0) + float2(0.00390625, 0.0);
    float2 deltaM = biased - maskUVAccum.yx;
    maskUVAccum = w.xx * deltaM.yx + maskUVAccum;
}
