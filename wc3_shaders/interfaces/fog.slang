// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026, Fernando Sahmkow
// See LICENSE in the repository root for full terms.
//
// Compile-time fog-mode policy. The shape of the three active modes
// mirrors the fixed-function fog modes exposed by the original engine
// (linear, exponential, exp²), each post-processed through the same
// smoothstep-like remap so the fog fade reads smoother than a linear
// lerp without requiring a fetch into a fog LUT.

implementing wc3_shaders;

public interface IFogMode
{
    // Used by callers to skip fog-dependent work (such as re-applying
    // `inverseSoftness` after fog has already scaled alpha).
    static const bool hasFog;

    // Blend `color` toward `fogColor` as a function of view-space depth.
    // `fogStart`/`fogEnd` parameterise FogLinear; `fogDensity` is used by
    // the exponential variants. `viewZ` is the per-pixel depth the VS
    // stored in worldPos.w.
    static float3 apply(float3 color, float3 fogColor,
                        float viewZ, float fogStart, float fogEnd, float fogDensity);
};

// Fog disabled: passthrough, compiled out of every call site.
public struct FogNone : IFogMode
{
    static const bool hasFog = false;

    static float3 apply(float3 color, float3 fogColor,
                        float viewZ, float fogStart, float fogEnd, float fogDensity)
    {
        return color;
    }
};

// Standard linear fog between fogStart and fogEnd.
// fH(f) = -f² + 2f is a Hermite smoothstep expressed as a FMA: it reaches
// 1 at f=1 and has zero derivative there, so the fog wall fades out
// smoothly instead of producing a visible kink at the far plane.
public struct FogLinear : IFogMode
{
    static const bool hasFog = true;

    static float3 apply(float3 color, float3 fogColor,
                        float viewZ, float fogStart, float fogEnd, float fogDensity)
    {
        float3 delta = color - fogColor;
        float f = saturate((fogEnd - viewZ) / (fogEnd - fogStart));
        float fH = -f * f + f + f;
        return fH * delta + fogColor;
    }
};

// Exponential fog: visibility = exp(-density * z).
// Implemented via `exp2` with the `1/ln(2)` scale (1.44269502) so the
// compiler can fold it to a single `exp2` instruction; the min() clamps
// visibility to 1 when the distance is negative. The same smoothstep
// remap as FogLinear is applied to the visibility factor.
public struct FogExponential : IFogMode
{
    static const bool hasFog = true;

    static float3 apply(float3 color, float3 fogColor,
                        float viewZ, float fogStart, float fogEnd, float fogDensity)
    {
        float3 delta = color - fogColor;
        float e = fogDensity * viewZ;
        e = 1.44269502 * e;
        e = exp2(e);
        float f = min(1.0, 1.0 / e);
        float fH = -f * f + f + f;
        return fH * delta + fogColor;
    }
};

// Exp² fog: visibility = exp(-(density * z)²). Produces a slower onset
// and a sharper drop-off than plain exponential fog; typical choice for
// outdoor distance haze.
public struct FogExp2 : IFogMode
{
    static const bool hasFog = true;

    static float3 apply(float3 color, float3 fogColor,
                        float viewZ, float fogStart, float fogEnd, float fogDensity)
    {
        float3 delta = color - fogColor;
        float dz = fogDensity * fogDensity * viewZ * viewZ;
        dz = 1.44269502 * dz;
        float e = exp2(dz);
        float f = 1.0 / e;
        float fH = -f * f + f + f;
        return fH * delta + fogColor;
    }
};
