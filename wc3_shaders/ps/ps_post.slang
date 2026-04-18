// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026, Fernando Sahmkow
// See LICENSE in the repository root for full terms.
//
// Post-lighting helpers shared by every HD pixel shader entry point:
//   - backface discard (cheap, runs early),
//   - the MRT alpha test used when writing to the deferred G-buffer,
//   - packing the world-space normal for the normal MRT target,
//   - the debug-visualisation switch.
//
// The debug modes are intentionally loose — they exist only to produce
// readable colored imagery for the artist/engineer debug HUD; the
// actual values don't need to match any engine convention beyond
// "distinct per mode".

implementing wc3_shaders;

// Unorm8 pack factors for the normal MRT target. (n * 0.5 + 0.5) puts
// a [-1,1] vector into [0,1], to be written to an R8G8B8A8_UNORM RT.
static const float3 NORMAL_PACK_SCALE = float3(0.5, 0.5, 0.5);
static const float3 NORMAL_PACK_BIAS  = float3(0.5, 0.5, 0.5);

// Scale factor applied to `inverseSoftness` when evaluating the
// deferred MRT alpha test — picks an effective threshold of 0.75 of
// the author-authored softness so cutout silhouettes are slightly
// thicker in the deferred pass (hides 1-pixel aliased seams).
static const float MRT_SOFTNESS_FACTOR = 0.75;

// Debug-visualisation palette. The per-LOD colors cycle R, G, B, Y, M
// and the per-light-count tints use dim ramps for 1/2/3 lights so
// small counts don't blow out the display.
static const float3 DBG_NORMAL_SCALE  = float3(0.5, 0.5, 0.5);
static const float  DBG_LOD_BIAS      = 0.5;
static const float3 DBG_LIGHT1_COLOR  = float3(0.0500000007, 0, 0);
static const float3 DBG_LIGHT2_COLOR  = float3(0, 0.0500000007, 0);
static const float3 DBG_LIGHT3_COLOR  = float3(0, 0, 0.100000001);
static const float3 DBG_LIGHT_DEFAULT = float3(0.25, 0.25, 0.25);

// Drop fragments the VS flagged as "behind the fog-clip plane" (a
// negative fog depth). This runs before lighting for every non-IBL
// non-prepass draw so we don't waste BRDF work on fragments that are
// going to be discarded anyway.
public void backfaceDiscard(float clipDepth)
{
    if (clipDepth < 0)
        discard;
}

// Secondary alpha test applied only when writing to the MRT G-buffer.
// `inverseSoftness` is 1 for hard cut-outs and approaches 0 for soft
// feathered cut-outs; the test turns that into a threshold just below
// 1, so only near-fully-opaque pixels end up in the deferred normal
// buffer.
public void mrtAlphaTest(float albedoAlpha, float inverseSoftness)
{
    float threshold = 1.0 - inverseSoftness;
    float test = albedoAlpha - threshold * MRT_SOFTNESS_FACTOR;
    if (test < 0)
        discard;
}

// Produce the unorm-packed world-space normal for the normal MRT.
public float3 computeMRTNormal(PSInput input, bool isFrontFace)
{
    float3 normalSample = t_normal.Sample(s_normal, input.uv.xy).xyw;
    float3 tsNormal = decodeNormalMap(normalSample);
    float3 worldNormal = tangentToWorld(tsNormal, input.tangentWS.xyz, input.normalWS.xyz, input.tangentWS.w);
    worldNormal = orientNormal(worldNormal, isFrontFace);
    return worldNormal * NORMAL_PACK_SCALE + NORMAL_PACK_BIAS;
}

// Debug visualisation router. `psCB3.debugMode` picks which view to
// emit; mode 0 is the normal-game visualisation path (still applies
// the color matrix), mode 1 shows raw albedo, mode 2 visualises the
// world normal (mixed with its pre-normalised length so artists can
// spot denormalised tangents), mode 3 shows the sampled mip level as a
// stepped palette, and mode 4 shows the active light count.
// Modes 5..8 are reserved in the original engine and output black.
public float4 debugVisualize(float4 color, PSInput input,
    float3 worldNormal, float normalLen)
{
    float lodVal = t_albedo.CalculateLevelOfDetail(s_albedo, input.uv.xy);
    float lodFrac = frac(lodVal);
    lodFrac = lodFrac * DBG_LOD_BIAS + DBG_LOD_BIAS;
    int lodInt = (int)lodVal;

    // Per-LOD color ramp. Anything past LOD 4 collapses to white.
    float3 lodColor;
    switch (lodInt)
    {
        case 0: lodColor = float3(1, 0, 0); break;
        case 1: lodColor = float3(0, 1, 0); break;
        case 2: lodColor = float3(0, 0, 1); break;
        case 3: lodColor = float3(1, 1, 0); break;
        case 4: lodColor = float3(1, 0, 1); break;
        default: lodColor = float3(1, 1, 1); lodFrac = 1; break;
    }

    int dbgMode = asint(psCB3.debugMode);

    if (dbgMode == 0)
    {
        // Normal game output (still needs the post color-matrix).
        return float4(applyColorMatrix(color.xyz), color.w);
    }
    else if (dbgMode == 1)
    {
        // Raw albedo (no lighting).
        return float4(t_albedo.Sample(s_albedo, input.uv.xy).xyz, color.w);
    }
    else if (dbgMode == 2)
    {
        // Normal visualisation mixed with its pre-normalised length;
        // denormalised tangents show up as darker regions.
        float3 nVis = (worldNormal * normalLen + float3(1, 1, 1)) * DBG_NORMAL_SCALE;
        return float4(nVis, color.w);
    }
    else if (dbgMode == 3)
    {
        // Stepped per-LOD heat map, with the fractional portion
        // modulating brightness inside each LOD band.
        return float4(lodColor * lodFrac, color.w);
    }
    else if (dbgMode == 4)
    {
        // Per-light-count tint. 0 = black, 1..3 are dim R/G/B ramps,
        // 4..7 are bright primaries, 8 is white.
        float3 lcColor;
        uint lc = asuint(psCB2.lightCount);
        switch (lc)
        {
            case 0: lcColor = float3(0, 0, 0); break;
            case 1: lcColor = DBG_LIGHT1_COLOR; break;
            case 2: lcColor = DBG_LIGHT2_COLOR; break;
            case 3: lcColor = DBG_LIGHT3_COLOR; break;
            case 4: lcColor = float3(1, 0, 0); break;
            case 5: lcColor = float3(0, 1, 0); break;
            case 6: lcColor = float3(0, 0, 1); break;
            case 7: lcColor = float3(1, 1, 0); break;
            case 8: lcColor = float3(1, 1, 1); break;
            default: lcColor = DBG_LIGHT_DEFAULT; break;
        }
        return float4(lcColor, color.w);
    }
    else
    {
        // Reserved debug slots emit black; everything else falls
        // through to the normal post-processed output.
        if (dbgMode >= 5 && dbgMode <= 8)
            return float4(0, 0, 0, color.w);

        return float4(applyColorMatrix(color.xyz), color.w);
    }
}
