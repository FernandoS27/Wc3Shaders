// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026, Fernando Sahmkow
// See LICENSE in the repository root for full terms.
//
// Vertex-shader utility functions shared between the HD, SD-on-HD, and
// SD highspec pipelines. These are straight matrix-vector helpers kept
// out of the entry shaders to make the VS bodies read as a linear
// sequence of transform steps.

implementing wc3_shaders;

// Object → view for a point (uses the full 4x4 with w=1). Despite the
// `worldPos` label on the PS input this produces, the output lives in view
// space — the PS brings it back out into probe / world space via `invView`
// when it needs to. Named after the matrix it applies.
public float3 transformByWorldView(float3 pos, float4x4 worldView)
{
    return mul(worldView, float4(pos, 1.0)).xyz;
}

// Object → world for a point. Produces actual world-space coordinates —
// used for fog altitude (.z vs clipHeight) and as the input to the shadow
// cascade matrices.
public float3 transformByWorld(float3 pos, float4x4 world)
{
    return mul(world, float4(pos, 1.0)).xyz;
}

// Object → clip in one multiply (bypasses the intermediate world pos).
public float4 transformClipPos(float3 pos, float4x4 wvp)
{
    return mul(wvp, float4(pos, 1.0));
}

// Apply the upper 3x3 of a 4x4 transform to a direction, with
// renormalization. Caller passes whichever object-space-rotating matrix
// is relevant — typically the `worldView` matrix on the HD path, whose
// 3x3 takes object directions to view space.
public float3 transformNormal(float3 n, float4x4 m)
{
    float3 result = mul((float3x3)m, n);
    float len2 = dot(result, result);
    return result * rsqrt(len2);
}

// Same as transformNormal but without the rsqrt. The SD-on-HD path
// passes this directly to the PS, which renormalises on demand — and
// the skinning policy has already produced a unit-length vector for
// skinned meshes.
public float3 transformNormalRaw(float3 n, float4x4 m)
{
    return mul((float3x3)m, n);
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

// World-space position used by the fog + shadow-cascade math. Its .z
// carries the world-space altitude the fog apply() compares against
// `clipHeight`; its full xyzw is the input the cascade matrices need.
public float4 computeWorldPos4(float3 pos, float4x4 world)
{
    return mul(world, float4(pos, 1.0));
}

// Biased world-space altitude used by the fog apply() calls in the PS.
// `clipHeight` is the water-line altitude so the game can push fog "up"
// above it, and `underWater` flips sign underwater so depth increases
// from the eye downward, producing the underwater blue fall-off.
public float computeFogDepth(float worldZ, float clipHeight, float underWater)
{
    return (worldZ - clipHeight) * underWater;
}

// Fast path for computeWorldPos4().z when only Z is required (non-shadow
// branches). Reading a single matrix row saves 3 multiplies and a
// transpose compared to computing the full world pos.
public float computeScalarWorldZ(float3 pos, float4x4 world)
{
    return dot(world[2], float4(pos, 1.0));
}
