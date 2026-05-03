// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026, Fernando Sahmkow
// See LICENSE in the repository root for full terms.
//
// Color-space helpers for the HD path.
//
// Unlike the textbook sRGB transfer, these are the exact cubic / pow-2.4
// approximations the original engine uses; they are kept verbatim so
// the output is bit-identical to the stock shader. Both the gamma curve
// and the color-grading matrix match what the engine ships.

implementing wc3_shaders;

// Cubic approximation of sRGB → linear (degamma).
// Polynomial form avoids `pow` on ancient GPU ALUs — this is Hable's
// quadratic approximation, close enough for surface-shading purposes.
public float3 srgbToLinear(float3 srgb)
{
    return srgb * (srgb * (srgb * 0.305306017 + 0.682171106) + 0.0125228781);
}

// Blizzard's post-grade color-matrix. A very mild contrast-warming
// adjustment that desaturates blue and slightly lifts red — baked into
// every HD pixel shader after lighting, before the fog / debug overlays.
public float3 applyColorMatrix(float3 color)
{
    return float3(
        dot(float3( 1.07000005, -0.0600000173, -0.0100000026), color),
        dot(float3(-0.0300000086, 1.03999996,  -0.0100000026), color),
        dot(float3(-0.0300000086, -0.0600000173, 1.09000003),  color));
}

// Proper (piecewise) linear → sRGB encoding. Used by the SD-on-HD path
// to output sRGB space when the render target is not a _SRGB format.
// Below 0.0031308 the transfer is linear (slope 12.92); above, it uses
// the classic 1.055 · x^(1/2.4) − 0.055 form, implemented as
// exp2(log2(x)·(1/2.4)) to avoid calling `pow` directly.
public float3 linearToSrgb(float3 linear)
{
    float3 curve = exp2(log2(abs(linear)) * 0.416666657);
    curve = curve * 1.05499995 + (-0.0549999997);
    float3 lin = 12.9200001 * linear;
    return select(linear <= float3(0.00313080009), lin, curve);
}

// Inverse of `linearToSrgb`: piecewise sRGB → linear decoding (IEC
// 61966-2-1). Used by the sprite pipeline to decode an sRGB-encoded
// source texture into linear before downstream sampling. Below 0.04045
// the transfer is the linear segment (1/12.92); above, ((c + 0.055) /
// 1.055)^2.4, again expressed as exp2(log2(...)·2.4) so the compiler
// avoids a `pow` instrinsic.
public float3 srgbToLinearIEC(float3 srgb)
{
    float3 curve = exp2(log2(abs((srgb + 0.0549999997) * 0.947867334)) * 2.4000001);
    float3 lin = 0.0773993805 * srgb;
    return select(srgb <= float3(0.0404499993), lin, curve);
}
