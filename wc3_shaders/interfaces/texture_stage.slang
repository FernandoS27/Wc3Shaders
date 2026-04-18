// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026, Fernando Sahmkow
// See LICENSE in the repository root for full terms.
//
// Compile-time fixed-function texture-stage policy used by the SD
// classic pixel shader. Each stage maps to one of the D3D9-era combiner
// ops the original renderer could chain: disabled, modulate, lerp,
// modulate-RGB (preserve alpha), and modulate2x.
// Specialising on these as traits lets the compiler fold the combiner
// chain into straight-line code per material permutation, matching the
// per-material shader variants the engine shipped.

implementing wc3_shaders;

public interface ITextureStage
{
    // When disabled, the stage contributes nothing and chained stages
    // are also short-circuited in the caller.
    static const bool isEnabled;

    // Combine the current accumulator with the newly fetched texel.
    static float4 applyOp(float4 current, float4 texSample);
};

// No-op stage. Used as the second stage for single-texture materials.
public struct StageDisabled : ITextureStage
{
    static const bool isEnabled = false;
    static float4 applyOp(float4 current, float4 texSample) { return current; }
};

// D3DTOP_MODULATE: component-wise multiply including alpha.
public struct StageModulate : ITextureStage
{
    static const bool isEnabled = true;
    static float4 applyOp(float4 current, float4 texSample) { return current * texSample; }
};

// D3DTOP_BLENDTEXTUREALPHA: alpha-weighted lerp between the incoming
// color and the new sample. Alpha is taken from the current accumulator
// (i.e. the stage only affects RGB).
public struct StageLerp : ITextureStage
{
    static const bool isEnabled = true;
    static float4 applyOp(float4 current, float4 texSample)
    {
        float3 rgb = lerp(current.xyz, texSample.xyz, texSample.w);
        return float4(rgb, current.w);
    }
};

// D3DTOP_MODULATE on RGB only; alpha passes through unchanged.
public struct StageModulateRGB : ITextureStage
{
    static const bool isEnabled = true;
    static float4 applyOp(float4 current, float4 texSample)
    {
        return float4(current.xyz * texSample.xyz, current.w);
    }
};

// D3DTOP_MODULATE2X: modulate then double, used by the classic lighting
// path to restore dynamic range after the per-vertex lit color was
// premultiplied by 0.5 in the VS.
public struct StageModulate2X : ITextureStage
{
    static const bool isEnabled = true;
    static float4 applyOp(float4 current, float4 texSample) { return current * texSample * 2.0; }
};
