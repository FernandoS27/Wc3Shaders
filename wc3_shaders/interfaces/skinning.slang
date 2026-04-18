// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026, Fernando Sahmkow
// See LICENSE in the repository root for full terms.
//
// Compile-time skinning policy. `Rigid` passes vertex attributes through
// unchanged; `FourBoneSkinning` / `SDFourBoneSkinning` apply a 4-bone
// linear blend against the bone palette in `BonePalette`.
//
// Each bone is a float4x3 affine transform (transpose of the engine's
// 3x4 row-major upload). The position multiply puts the vector on the
// left — `mul(float4(pos, 1), M)` — and the normal/tangent use the
// top-left 3x3 so the translation column drops out.

implementing wc3_shaders;

public interface ISkinning
{
    static const bool isSkinned;

    static void transform(
        float3 objectPos, float3 objectNormal, float4 objectTangent,
        float4 weights, uint4 boneIndices,
        BonePalette bones,
        out float3 skinnedPos, out float3 skinnedNormal, out float3 skinnedTangent);
};

// Non-skinned geometry: the object-space attributes are forwarded as-is.
public struct Rigid : ISkinning
{
    static const bool isSkinned = false;

    static void transform(
        float3 objectPos, float3 objectNormal, float4 objectTangent,
        float4 weights, uint4 boneIndices,
        BonePalette bones,
        out float3 skinnedPos, out float3 skinnedNormal, out float3 skinnedTangent)
    {
        skinnedPos     = objectPos;
        skinnedNormal  = objectNormal;
        skinnedTangent = objectTangent.xyz;
    }
};

// HD 4-bone linear blend skinning.
// Weighted-sum the four bone matrices into `M`, then apply it to the
// position (affine) and to the rotation part for normal/tangent. The
// .y weight is summed first because the original shader was compiled
// that way, and preserving the accumulation order keeps the numerical
// output bit-identical.
// The `weightSum != 0` guard protects against unloaded or zero-weight
// vertices which would otherwise produce a zero-position spike.
public struct FourBoneSkinning : ISkinning
{
    static const bool isSkinned = true;

    static void transform(
        float3 objectPos, float3 objectNormal, float4 objectTangent,
        float4 weights, uint4 boneIndices,
        BonePalette bones,
        out float3 skinnedPos, out float3 skinnedNormal, out float3 skinnedTangent)
    {

        float weightSum = dot(weights, float4(1, 1, 1, 1));
        if (weightSum != 0.0)
        {
            float4x3 M  = bones.matrices[boneIndices.y] * weights.y;
            M          += bones.matrices[boneIndices.x] * weights.x;
            M          += bones.matrices[boneIndices.z] * weights.z;
            M          += bones.matrices[boneIndices.w] * weights.w;

            float3x3 R = (float3x3)M;

            // Full affine transform for the position (homogeneous w = 1).
            skinnedPos = mul(float4(objectPos, 1.0), M);

            // Normal / tangent use only the rotation portion. The HD path
            // skips the length rescale after skinning because the PS
            // renormalises both vectors when it reconstructs the
            // world-space basis.
            skinnedNormal  = mul(objectNormal,       R);
            skinnedTangent = mul(objectTangent.xyz,  R);
        }
        else
        {
            // Degenerate vertex (no weights): treat as rigid.
            skinnedPos     = objectPos;
            skinnedNormal  = objectNormal;
            skinnedTangent = objectTangent.xyz;
        }
    }
};

// SD-on-HD variant. Identical to FourBoneSkinning except the skinned
// normal is renormalised in the VS — the SD pixel shader paths use the
// raw (unrenormalised) world normal and expect the VS to have already
// produced a unit-length normal at transform time.
public struct SDFourBoneSkinning : ISkinning
{
    static const bool isSkinned = true;

    static void transform(
        float3 objectPos, float3 objectNormal, float4 objectTangent,
        float4 weights, uint4 boneIndices,
        BonePalette bones,
        out float3 skinnedPos, out float3 skinnedNormal, out float3 skinnedTangent)
    {
        float weightSum = dot(weights, float4(1, 1, 1, 1));
        if (weightSum != 0.0)
        {
            float4x3 M  = bones.matrices[boneIndices.y] * weights.y;
            M          += bones.matrices[boneIndices.x] * weights.x;
            M          += bones.matrices[boneIndices.z] * weights.z;
            M          += bones.matrices[boneIndices.w] * weights.w;

            float3x3 R = (float3x3)M;

            skinnedPos     = mul(float4(objectPos, 1.0), M);
            skinnedNormal  = mul(objectNormal,      R);

            // Required by the SD shading path, which does not renormalise
            // in the PS before evaluating NdotL/NdotV.
            float len2 = dot(skinnedNormal, skinnedNormal);
            skinnedNormal = skinnedNormal * rsqrt(len2);

            skinnedTangent = mul(objectTangent.xyz, R);
        }
        else
        {
            skinnedPos     = objectPos;
            skinnedNormal  = objectNormal;
            skinnedTangent = objectTangent.xyz;
        }
    }
};
