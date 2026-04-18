// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026, Fernando Sahmkow
// See LICENSE in the repository root for full terms.
//
// SD-on-HD vertex shader entry point.
//
// Structurally identical to hd_vs.slang, but uses `transformNormalRaw`
// for both the normal and the tangent: the SD path renormalises in
// the pixel shader (or relies on the skinning step having already
// produced unit-length vectors for skinned meshes), so we can skip the
// rsqrt here and save a few cycles per vertex.

implementing wc3_shaders;

[shader("vertex")]
public VSOutput sd_on_hd_vs_main<
    TSkin : ISkinning,
    TVF   : IVertexFormat,
    let HAS_SHADOWS : bool
>(VSInput<TVF> input)
{
    VSOutput output;

    // Skinning — same policy choice as the HD path. `SDFourBoneSkinning`
    // renormalises the skinned normal internally so the raw transform
    // below is safe.
    float3 skinnedPos, skinnedNormal, skinnedTangent;
    TSkin.transform(
        input.position, input.normal, input.tangent,
        input.weights, input.boneIndices, vsCB3,
        skinnedPos, skinnedNormal, skinnedTangent);

    output.worldPos.xyz = transformByWorld(skinnedPos, vsCB2.world);
    output.position = transformClipPos(skinnedPos, vsCB2.worldViewProj);

    // No renormalise — the SD PS does its own normalize() after
    // interpolation shortens the vector across the triangle.
    output.normalWS.xyz = transformNormalRaw(skinnedNormal, vsCB2.world);
    output.normalWS.w = 0;

    if (TVF.hasTangent)
    {
        output.tangentWS.xyz = transformNormalRaw(skinnedTangent, vsCB2.world);
        output.tangentWS.w = input.tangent.w;
    }
    else
    {
        output.tangentWS = float4(0, 0, 0, 0);
    }

    if (HAS_SHADOWS)
    {
        // Cascade clip coords for the SD shadow sampler.
        float4 viewPos = computeViewPos(skinnedPos, vsCB2.viewProj);
        output.worldPos.w = computeFogDepth(viewPos.z, vsCB2.clipHeight, vsCB2.underWater);

        computeShadowCascades(viewPos, vsCB1,
            output.shadowClip0, output.shadowClip1, output.shadowClip2);
    }
    else
    {
        float viewZ = computeScalarViewZ(skinnedPos, vsCB2.viewProj);
        output.worldPos.w = computeFogDepth(viewZ, vsCB2.clipHeight, vsCB2.underWater);

        output.shadowClip0 = float4(0, 0, 0, 0);
        output.shadowClip1 = float4(0, 0, 0, 0);
        output.shadowClip2 = float4(0, 0, 0, 0);
    }

    if (TVF.hasVertexColor)
        output.color = vsCB2.diffuseColor * input.vertColor;
    else
        output.color = vsCB2.diffuseColor;

    output.texCoord = transformUVs<TVF>(
        input.uv0, input.uv1,
        vsCB2.texMtx0Row0, vsCB2.texMtx0Row1,
        vsCB2.texMtx1Row0, vsCB2.texMtx1Row1);

    return output;
}
