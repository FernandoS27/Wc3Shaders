// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026, Fernando Sahmkow
// See LICENSE in the repository root for full terms.
//
// Vertex-shader utility functions shared between the HD, SD-on-HD, and
// SD highspec pipelines. These are straight matrix-vector helpers kept
// out of the entry shaders to make the VS bodies read as a linear
// sequence of transform steps.

implementing wc3_shaders;

// Object → world for a point (uses the full 4x4 with w=1).
public float3 transformByWorld(float3 pos, float4x4 world)
{
    return mul(world, float4(pos, 1.0)).xyz;
}

// Object → clip in one multiply (bypasses the intermediate world pos).
public float4 transformClipPos(float3 pos, float4x4 wvp)
{
    return mul(wvp, float4(pos, 1.0));
}

// Object → world for a direction, with renormalization. Used by the HD
// path, which requires a unit-length world normal for BRDF evaluation.
public float3 transformNormal(float3 n, float4x4 world)
{
    float3 result = mul((float3x3)world, n);
    float len2 = dot(result, result);
    return result * rsqrt(len2);
}

// Same as transformNormal but without the rsqrt. The SD-on-HD path
// passes this directly to the PS, which renormalises on demand — and
// the skinning policy has already produced a unit-length vector for
// skinned meshes.
public float3 transformNormalRaw(float3 n, float4x4 world)
{
    return mul((float3x3)world, n);
}

// Apply a 2×3 affine texture-coordinate transform stored as two 4-vec
// rows (column layout matches the legacy D3D `D3DTSS_TEXTURETRANSFORM`
// format where only .xy w are meaningful per row).
public float2 transformUV(float2 uv, float4 mtxRow0, float4 mtxRow1)
{
    float3 uvH = float3(uv, 1.0);
    return float2(
        dot(mtxRow0.xyw, uvH),
        dot(mtxRow1.xyw, uvH));
}

// Apply the UV0 and UV1 transforms iff the vertex format advertises
// each slot. Output is packed float4(uv0, uv1) because the PS expects a
// single TEXCOORD0 register.
public float4 transformUVs<TVF : IVertexFormat>(
    float2 uv0, float2 uv1,
    float4 mtx0Row0, float4 mtx0Row1,
    float4 mtx1Row0, float4 mtx1Row1)
{
    float4 result = float4(0, 0, 0, 0);
    if (TVF.hasUV0)
    {
        result.xy = transformUV(uv0, mtx0Row0, mtx0Row1);
        if (TVF.hasUV1)
            result.zw = transformUV(uv1, mtx1Row0, mtx1Row1);
    }
    return result;
}

// Evaluate the three cascade-space clip positions for PCF shadow
// sampling. Each cascade matrix is the combined cascade-view-projection
// for its split, so the input view-space position is reprojected
// directly into each cascade's light clip space.
public void computeShadowCascades(
    float4 viewPos, ShadowCascades cascades,
    out float4 clip0, out float4 clip1, out float4 clip2)
{
    clip0 = mul(cascades.cascade0, viewPos);
    clip1 = mul(cascades.cascade1, viewPos);
    clip2 = mul(cascades.cascade2, viewPos);
}

// View-space position used by the fog/shadow math. Despite the name,
// `viewProj` in the HD VSPerDraw is actually the view matrix — the
// original engine's constant buffer naming was kept verbatim.
public float4 computeViewPos(float3 pos, float4x4 viewProj)
{
    return mul(viewProj, float4(pos, 1.0));
}

// Biased view-space Z used by the fog apply() calls in the PS.
// `clipHeight` lets the game push fog "up" above water lines and
// `underWater` flips sign underwater so depth increases from the eye
// downward, producing the underwater blue fall-off.
public float computeFogDepth(float viewZ, float clipHeight, float underWater)
{
    return (viewZ - clipHeight) * underWater;
}

// Fast path for computeViewPos().z when only Z is required (non-shadow
// branches). Reading a single projection-matrix row saves 3 multiplies
// and a transpose compared to computing the full view pos.
public float computeScalarViewZ(float3 pos, float4x4 viewProj)
{
    return dot(viewProj[2], float4(pos, 1.0));
}
