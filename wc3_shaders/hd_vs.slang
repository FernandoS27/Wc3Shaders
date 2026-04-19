// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026, Fernando Sahmkow
// See LICENSE in the repository root for full terms.
//
// HD vertex shader entry point.
//
// Specialisation space:
//   TSkin         — ISkinning policy (rigid or four-bone).
//   TVF           — IVertexFormat (which optional attribs are live).
//   HAS_SHADOWS   — whether the three cascade clip coords need computing.
//
// The VS is intentionally thin: all non-trivial work lives in the
// helpers in vs/ and interfaces/, so each permutation is a small linear
// sequence of transforms.

implementing wc3_shaders;

[shader("vertex")]
public VSOutput vs_main<
    TSkin : ISkinning,
    TVF   : IVertexFormat,
    let HAS_SHADOWS : bool
>(VSInput<TVF> input)
{
    VSOutput output;

    // Step 1: skinning. Rigid passes the attributes through; FourBone
    // performs a 4-bone linear blend against the bone palette (vsCB3).
    float3 skinnedPos, skinnedNormal, skinnedTangent;
    TSkin.transform(
        input.position, input.normal, input.tangent,
        input.weights, input.boneIndices, vsCB3,
        skinnedPos, skinnedNormal, skinnedTangent);

    // Step 2: view-space position (emitted as the PS's "worldPos" input
    // — the PS transforms it back through `invView` as needed) + clip
    // position. Clip comes directly from object space via worldViewProj
    // to avoid losing precision to the intermediate pos.
    output.worldPos.xyz = transformByWorldView(skinnedPos, vsCB2.worldView);
    output.position     = transformClipPos(skinnedPos, vsCB2.worldViewProj);

    // Step 3: view-space normal (renormalised — HD PS assumes it's
    // unit-length before any BRDF work). Lives in the same space as
    // `worldPos` above; the PS co-rotates both with `invView`.
    output.normalWS.xyz = transformNormal(skinnedNormal, vsCB2.worldView);
    output.normalWS.w = 0;

    // Tangent only when the vertex format advertises one; `tangent.w`
    // carries the handedness sign for bitangent reconstruction in the
    // pixel shader (see tangentToWorld).
    if (TVF.hasTangent)
    {
        output.tangentWS.xyz = transformNormal(skinnedTangent, vsCB2.worldView);
        output.tangentWS.w = input.tangent.w;
    }
    else
    {
        output.tangentWS = float4(0, 0, 0, 0);
    }

    if (HAS_SHADOWS)
    {
        // When shadows are needed, compute the world-space position
        // once and reuse it for both fog altitude and the three cascade
        // clip-space projections (the cascade matrices take world pos).
        float4 worldPos4 = computeWorldPos4(skinnedPos, vsCB2.world);
        output.worldPos.w = computeFogDepth(worldPos4.z, vsCB2.clipHeight, vsCB2.underWater);

        computeShadowCascades(worldPos4, vsCB1,
            output.shadowClip0, output.shadowClip1, output.shadowClip2);
    }
    else
    {
        // Non-shadow path skips the full world transform and only reads
        // the Z row of `world` — cheaper by 3 multiplies + an adder.
        float worldZ = computeScalarWorldZ(skinnedPos, vsCB2.world);
        output.worldPos.w = computeFogDepth(worldZ, vsCB2.clipHeight, vsCB2.underWater);

        output.shadowClip0 = float4(0, 0, 0, 0);
        output.shadowClip1 = float4(0, 0, 0, 0);
        output.shadowClip2 = float4(0, 0, 0, 0);
    }

    // Step 4: per-vertex tint. `diffuseColor` is the material tint;
    // when the format carries a per-vertex color (e.g. AO from the
    // baker) we modulate the two together.
    if (TVF.hasVertexColor)
        output.color = vsCB2.diffuseColor * input.vertColor;
    else
        output.color = vsCB2.diffuseColor;

    // Step 5: transform the UVs through the two 2x3 UV matrices (used
    // for scrolling / tiling animations). Dead UV slots compile out.
    output.texCoord = transformUVs<TVF>(
        input.uv0, input.uv1,
        vsCB2.texMtx0Row0, vsCB2.texMtx0Row1,
        vsCB2.texMtx1Row0, vsCB2.texMtx1Row1);

    return output;
}
