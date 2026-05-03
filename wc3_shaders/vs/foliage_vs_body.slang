// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026, Fernando Sahmkow
// See LICENSE in the repository root for full terms.
//
// Foliage vertex shader body. Two compile-time axes:
//
//   HAS_SHADOWS — RECEIVE_SHADOWS && !SHADOW_PASS in engine terms.
//                 When false the cascade outputs are gated out via the
//                 VSOutput's `Conditional<>` slots.
//   HAS_WIND    — sample the wind-direction texture at three nearby
//                 texels, recover a 2D gradient, and pre-displace +
//                 swing the vertex XY in world space. Z stays fixed.
//                 The displaced XY feeds the clip-space transform,
//                 the world-space position output, and (when shadows
//                 are active) the shadow projection.
//
// Wind algorithm (matches the shipped engine VS bit-for-bit):
//   1. Skip the entire block when `vertColor.w == 0` (vertex isn't
//      wind-affected).
//   2. Strength `s = 0.015 · vertColor.w · min(100, |tangent.w|)`.
//   3. Map vertex world XY into wind-area UV via cb3[0] (origin .xy,
//      size .zw). Sample the wind texture at three taps — center,
//      `-1 texel` in U, `-1 texel` in V — and take the differences as
//      a 2D gradient clamped to [-1, 1].
//   4. Pre-displace world XY by `10 · |tangent.w| · gradient`, then
//      add a `(cos, sin) · s` swing whose phase is
//      `2π · (frac(time) + preDisp)`.
//
// Tangent-handedness stencil:
//   `worldTangent.w = sign(tangent.w)` ∈ {-1, 0, +1}. The PS uses this
//   to flip the bitangent (bitangent = cross(N, T) · sign).

implementing wc3_shaders;

// World 3×3 applied to an object-space direction. Each cb2[4..6].xyz
// is a row of the rotation; the .w lanes are unused. No
// renormalisation here — callers either re-normalise (for normals)
// or rely on the result staying in tangent space (for raw tangents).
float3 foliageTransformDir(float3 d, float4 worldRow0,
                           float4 worldRow1, float4 worldRow2)
{
    float3 r = d.y * worldRow1.xyz;
    r = d.x * worldRow0.xyz + r;
    return d.z * worldRow2.xyz + r;
}

// Object → world point: 3×3 rotation + translation.
float3 foliageTransformPos(float3 p, float4 worldRow0, float4 worldRow1,
                           float4 worldRow2, float4 worldTranslate)
{
    return foliageTransformDir(p, worldRow0, worldRow1, worldRow2)
         + worldTranslate.xyz;
}

// 2π — used to convert the wind phase from [0, 1) into radians.
static const float TWO_PI = 6.28318548;

// Sample the wind texture at three taps and recover a gradient. Inputs
// are the in-area UV (`uv`) and the texel size (`texelU`, `texelV` —
// foliage assumes square texels so both are the same value, but we
// pass both for clarity at the call site). Returns the per-axis
// gradient clamped to [-1, +1].
float2 sampleWindGradient(float2 uv, float texelU)
{
    // Engine convention: V is sampled flipped (1 - uv.y).
    float vFlipped = 1.0 - uv.y;
    float center = t_foliageWind.SampleLevel(s_foliageWind,
                       float2(uv.x, vFlipped), 0).x;

    // -1 texel in V (which corresponds to +1 texel in vFlipped because
    // of the V flip): sample at vFlipped+texelU.
    float vMinus  = t_foliageWind.SampleLevel(s_foliageWind,
                        float2(uv.x - texelU, vFlipped), 0).x;
    float dx      = vMinus - center;

    // -1 texel in U: sample at uv.x - texelU but with the original
    // vFlipped + texelU mapping the engine uses.
    float uMinus  = t_foliageWind.SampleLevel(s_foliageWind,
                        float2(uv.x, 1.0 - (uv.y - texelU)), 0).x;
    float dy      = uMinus - center;

    // Engine clamps the gradient to [-1, 1] in both axes.
    return clamp(float2(dx, dy), float2(-1, -1), float2(1, 1));
}

// Wind-driven XY displacement. Returns the (possibly displaced) world
// XY; Z stays at the input value. When the vertex is wind-inert
// (vertColor.w == 0) the input XY is returned unchanged.
//
// `tangentW` carries the per-vertex wind-strength scalar in the
// tangent's .w lane (also used as the handedness sign downstream —
// the abs() here keeps the strength orientation-agnostic).
float2 applyWindDisplacement(float2 inputXY, float vertWeight,
                             float tangentW)
{
    if (vertWeight <= 0.0)
        return inputXY;

    // Strength = 0.015 · vertWeight · min(100, |tangent.w|).
    float strength = 0.0149999997
                   * vertWeight
                   * min(100.0, abs(tangentW));

    // Wind-area UV via cb3 origin / size.
    float2 areaUV = (inputXY - foliageVsCB3.windOrigin) / foliageVsCB3.windSize;

    // Texel size for the gradient taps.
    uint w, h, mips;
    t_foliageWind.GetDimensions(0, w, h, mips);
    float texel = 1.0 / (float)w;

    // Recover the wind direction gradient via three texture samples.
    float2 gradient = sampleWindGradient(areaUV, texel);

    // Pre-displacement: 10 · |tangent.w| · gradient, applied to the
    // input XY before the cos/sin swing.
    float2 preDisp = inputXY + 10.0 * abs(tangentW) * gradient;

    // Phase = 2π · (frac(time) + preDisp), one component per axis.
    float2 phase = (frac(foliageVsCB2.effectTime) + preDisp) * TWO_PI;

    // Swing on top of the pre-displacement, gated by whether the
    // gradient is non-zero (no wind anywhere → keep preDisp).
    bool gateX = gradient.x == 0.0;
    bool gateY = gradient.y == 0.0;
    bool windActive = !(gateX && gateY);
    float2 swing = float2(cos(phase.x), sin(phase.y)) * strength;
    return windActive ? (preDisp + swing) : preDisp;
}

// Tangent-handedness stencil: sign(tangent.w) ∈ {-1, 0, +1}. HLSL's
// `sign()` lowers to lt+lt+iadd+itof, matching the engine's pattern.
float foliageHandedness(float tangentW)
{
    return sign(tangentW);
}

public FoliageVSOutput<HAS_SHADOWS> foliageVSBody<
    let HAS_SHADOWS : bool,
    let HAS_WIND    : bool>(FoliageVSInput input)
{
    FoliageVSOutput<HAS_SHADOWS> output;

    // 1. Optionally apply wind to the vertex XY. Z is always fixed.
    float2 xy = HAS_WIND
        ? applyWindDisplacement(input.position.xy, input.vertColor.w, input.tangent.w)
        : input.position.xy;
    float3 displaced = float3(xy, input.position.z);

    // 2. World-space position (uses the displaced XY).
    float3 worldP = foliageTransformPos(
        displaced,
        foliageVsCB2.worldRow0, foliageVsCB2.worldRow1,
        foliageVsCB2.worldRow2, foliageVsCB2.worldTranslate);
    output.worldPos = float4(worldP, 0);

    // 3. Clip-space position (uses the displaced XY).
    output.position = mul(foliageVsCB2.worldViewProj, float4(displaced, 1.0));

    // 4. Tint × per-vertex colour (vertColor.w forced to 1 — the .w
    // lane carries the wind-weight, not a real alpha).
    float4 vertCol = float4(input.vertColor.xyz, 1.0);
    output.color = foliageVsCB2.tintColor * vertCol;

    // 5. UV0 transformed by the 2×3 affine in cb2[14..15].xyw, UV1
    // forwarded raw.
    float3 uvH = float3(input.uv0, 1.0);
    output.texCoord = float4(
        dot(foliageVsCB2.texMtx0Row0.xyw, uvH),
        dot(foliageVsCB2.texMtx0Row1.xyw, uvH),
        input.uv1.x, input.uv1.y);

    // 6. World-space normal (uses the original normal — wind sways but
    // doesn't bend the geometry's facing).
    float3 worldN = foliageTransformDir(
        input.normal,
        foliageVsCB2.worldRow0, foliageVsCB2.worldRow1, foliageVsCB2.worldRow2);
    output.worldNormal = worldN * rsqrt(dot(worldN, worldN));

    // 7. World-space tangent (raw, not normalised) + handedness in .w.
    float3 worldT = foliageTransformDir(
        input.tangent.xyz,
        foliageVsCB2.worldRow0, foliageVsCB2.worldRow1, foliageVsCB2.worldRow2);
    output.worldTangent = float4(worldT, foliageHandedness(input.tangent.w));

    // 8. Shadow cascades — the pre-shadow transform uses the displaced
    // XY + original Z, so wind-disturbed pixels self-shadow correctly.
    if (HAS_SHADOWS)
    {
        float4 shadowPos = mul(foliageVsCB2.shadowWorld,
                               float4(displaced, 1.0));
        output.shadowClip0 = mul(vsCB1.cascade0, shadowPos);
        output.shadowClip1 = mul(vsCB1.cascade1, shadowPos);
        output.shadowClip2 = mul(vsCB1.cascade2, shadowPos);
    }

    return output;
}
