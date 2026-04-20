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
public VSOutput<HAS_SHADOWS> sd_on_hd_vs_main<
    TSkin : ISkinning,
    TVF   : IVertexFormat,
    let HAS_SHADOWS : bool
>(VSInput<TVF> input)
{
    VSOutput<HAS_SHADOWS> output;

    // Skinning — same policy choice as the HD path. `SDFourBoneSkinning`
    // renormalises the skinned normal internally so the raw transform
    // below is safe.
    float3 skinnedPos, skinnedNormal, skinnedTangent;
    TSkin.transform(
        input.position, input.normal, input.tangent,
        input.weights, input.boneIndices, vsCB3,
        skinnedPos, skinnedNormal, skinnedTangent);

    output.worldPos.xyz = transformByWorldView(skinnedPos, vsCB2.worldView);
    output.position     = transformClipPos(skinnedPos, vsCB2.worldViewProj);

    // No renormalise — the SD PS does its own normalize() after
    // interpolation shortens the vector across the triangle.
    output.normalWS = transformNormalRaw(skinnedNormal, vsCB2.worldView);

    if (TVF.hasTangent)
    {
        output.tangentWS.xyz = transformNormalRaw(skinnedTangent, vsCB2.worldView);
        output.tangentWS.w = input.tangent.w;
    }
    else
    {
        output.tangentWS = float4(0, 0, 0, 0);
    }

    if (HAS_SHADOWS)
    {
        // Cascade clip coords for the SD shadow sampler — cascade
        // matrices take world-space positions, so we need the `world`
        // product (not the view-space one above). Local temps because
        // `shadowClip*` are Conditional<float4, HAS_SHADOWS> and can't
        // be bound directly to the `out float4` cascade helper.
        float4 worldPos4 = computeWorldPos4(skinnedPos, vsCB2.world);
        output.worldPos.w = computeFogDepth(worldPos4.z, vsCB2.clipHeight, vsCB2.underWater);

        float4 sc0, sc1, sc2;
        computeShadowCascades(worldPos4, vsCB1, sc0, sc1, sc2);
        output.shadowClip0 = sc0;
        output.shadowClip1 = sc1;
        output.shadowClip2 = sc2;
    }
    else
    {
        float worldZ = computeScalarWorldZ(skinnedPos, vsCB2.world);
        output.worldPos.w = computeFogDepth(worldZ, vsCB2.clipHeight, vsCB2.underWater);

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
