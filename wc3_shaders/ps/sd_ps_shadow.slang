// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026, Fernando Sahmkow
// See LICENSE in the repository root for full terms.
//
// Shadow cascade sampling for the SD-on-HD path.
//
// Up to three cascades are tested in order of priority (cascade 0 is
// the highest-resolution / closest-to-camera). For the first cascade
// that contains the pixel, a 3×3 PCF test is performed and the
// resulting visibility is returned. If the pixel is outside every
// cascade, the function returns 1 (fully lit) rather than 0 so distant
// fragments aren't spuriously shadowed.

implementing wc3_shaders;

// Is this cascade-space clip coordinate inside the frustum?
// UV is derived from clip XY; depth check uses the NDC [0, 1] convention.
static bool isInShadowFrustum(float3 clipCoord)
{
    float2 uv = clipCoord.xy * float2(0.5, -0.5) + float2(0.5, 0.5);
    float depth = clipCoord.z;
    return uv.x >= 0 && uv.x <= 1 &&
           uv.y >= 0 && uv.y <= 1 &&
           depth >= 0 && depth < 1;
}

// 3×3 percentage-closer filter. Loops over the 9-tap kernel centered on
// the cascade-space UV and averages the comparison results. Using
// `SampleCmpLevelZero` lets the hardware do the depth-compare and
// bilinear filter in one fetch, so the loop actually computes a
// 3×3 ×(hw 2×2) = 6×6 effective footprint.
static float samplePCF2x2(
    Texture2D<float4> shadowTex,
    SamplerComparisonState shadowSampler,
    float3 clipCoord)
{
    float2 uv = clipCoord.xy * float2(0.5, -0.5) + float2(0.5, 0.5);
    float depth = clipCoord.z;

    // Query the shadow map texel size from mip 0 so the same sampler
    // works across cascades of different resolutions.
    uint texW, texH, mips;
    shadowTex.GetDimensions(0, texW, texH, mips);
    float texelSize = 1.0 / (float)texW;

    float sum = 0;
    float count = 0;
    for (int row = -1; row < 2; row++)
    {
        for (int col = -1; col < 2; col++)
        {
            count += 1.0;
            float2 offset = float2(col, row) * texelSize;
            sum += shadowTex.SampleCmpLevelZero(shadowSampler, uv + offset, depth);
        }
    }
    return saturate(sum / count);
}

// Select the first containing cascade and return its PCF visibility.
// `numCascades` lets the runtime switch the number of active cascades
// at runtime (e.g. low-quality setting uses only two).
public float sdSampleShadowCascades(
    float4 shadowClip0, float4 shadowClip1, float4 shadowClip2,
    uint numCascades)
{
    int cascadeIdx = -1;
    for (int i = 0; i < (int)numCascades && i < 3; i++)
    {
        float3 clip;
        if (i == 0) clip = shadowClip0.xyz;
        else if (i == 1) clip = shadowClip1.xyz;
        else clip = shadowClip2.xyz;

        if (isInShadowFrustum(clip))
        {
            cascadeIdx = i;
            break;
        }
    }

    // Outside every cascade: assume no shadow contribution.
    if (cascadeIdx < 0)
        return 1.0;

    // Sample the chosen cascade. The three branches compile into a
    // straightforward data-dependent load on modern GPUs.
    if (cascadeIdx == 0)
        return samplePCF2x2(sd_t_shadow0, sd_s_shadow0, shadowClip0.xyz);
    else if (cascadeIdx == 1)
        return samplePCF2x2(sd_t_shadow1, sd_s_shadow1, shadowClip1.xyz);
    else
        return samplePCF2x2(sd_t_shadow2, sd_s_shadow2, shadowClip2.xyz);
}
