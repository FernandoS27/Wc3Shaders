// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026, Fernando Sahmkow
// See LICENSE in the repository root for full terms.
//
// Terrain cascaded shadow visibility.
//
// The terrain pipeline samples three independent cascaded shadow maps
// at t10/t11/t12 (paired with comparison samplers s10/s11/s12) and runs
// a 9-tap PCF kernel against the chosen cascade. Cascade selection runs
// once per pixel: pick the lowest-numbered cascade whose light-space UV
// + depth all land in [0, 1]. Cascades beyond `cb1[0].x` (the active
// cascade count) are skipped — the engine packs an unused-cascade
// signal into that scalar so a debug build can render with one
// cascade only.
//
// Returns visibility ∈ [0, 1] (1 = fully lit, 0 = fully shadowed).

implementing wc3_shaders;

// Map a cascade clip-space point into [0, 1]² UV + depth, then test
// whether the result lies inside the unit cube. Cascade depth must
// strictly stay below 1 — exactly 1 means "behind the far plane",
// which the comparison sampler would otherwise misinterpret.
bool cascadeContains(float4 clip, out float2 uv, out float depth)
{
    uv = clip.xy * float2(0.5, -0.5) + float2(0.5, 0.5);
    depth = clip.z;
    return uv.x >= 0.0 && uv.x <= 1.0
        && uv.y >= 0.0 && uv.y <= 1.0
        && depth >= 0.0 && depth <  1.0;
}

// 9-tap (3×3) PCF using one cascade's comparison sampler. Texel size
// comes from the cascade map itself — the engine uses square shadow
// maps so a single dimension query is enough.
float pcfCascade(Texture2D<float4> shadowMap,
                 SamplerComparisonState shadowSampler,
                 float2 uv, float depth)
{
    uint w, h, mips;
    shadowMap.GetDimensions(0, w, h, mips);
    float texel = 1.0 / (float)w;

    float taps   = 0.0;
    float sumLit = 0.0;
    for (int i = -1; i < 2; i++)
    {
        for (int j = -1; j < 2; j++)
        {
            float2 ofs = float2((float)i * texel, (float)j * texel);
            taps   = taps + 1.0;
            sumLit = sumLit
                   + shadowMap.SampleCmpLevelZero(shadowSampler, uv + ofs, depth).x;
        }
    }
    return saturate(sumLit / taps);
}

// Compute terrain shadow visibility from the three per-cascade clip
// positions emitted by the VS. Iterates active cascades in order and
// runs PCF against the first one that contains the pixel.
//
// Putting the cascade-count test in the loop condition lets fxc emit a
// single `breakc_nz` for it instead of the if/break/endif triplet a
// `for + break` would produce.
public float computeTerrainShadow(
    float4 clip0, float4 clip1, float4 clip2,
    uint activeCascades)
{
    float4 cascades[3];
    cascades[0] = clip0;
    cascades[1] = clip1;
    cascades[2] = clip2;

    int chosen;
    [loop] for (chosen = 0;
                (uint)chosen < activeCascades;
                chosen++)
    {
        float2 uv;
        float  depth;
        if (cascadeContains(cascades[chosen], uv, depth))
            break;
    }

    float visibility = 1.0;
    if (chosen == 0)
    {
        float2 uv;
        float  depth;
        if (cascadeContains(clip0, uv, depth))
            visibility = pcfCascade(sd_t_shadow0, sd_s_shadow0, uv, depth);
    }
    else if (chosen == 1)
    {
        float2 uv;
        float  depth;
        if (cascadeContains(clip1, uv, depth))
            visibility = pcfCascade(sd_t_shadow1, sd_s_shadow1, uv, depth);
    }
    else if (chosen == 2)
    {
        float2 uv;
        float  depth;
        if (cascadeContains(clip2, uv, depth))
            visibility = pcfCascade(sd_t_shadow2, sd_s_shadow2, uv, depth);
    }
    return visibility;
}
