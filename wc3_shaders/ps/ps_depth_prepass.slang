// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026, Fernando Sahmkow
// See LICENSE in the repository root for full terms.
//
// HD depth-prepass fragment body.
//
// The depth prepass writes only depth; the entry shader (hd_ps.slang)
// short-circuits the color output and returns zero. What this file
// provides is the *alpha test* that the prepass still needs to run so
// cut-out foliage / decals punch the right silhouette into the depth
// buffer.
//
// Two alpha-computation branches are provided for multi-layer and
// standard materials — they must produce the exact same final alpha
// that the lit pass does, otherwise the prepass depth won't line up
// with the color pass (self-shadow / SSAO holes on alpha-tested edges).

implementing wc3_shaders;

// Back-face discard stub used when the alpha-test policy is "off": the
// prepass skips all the alpha math but must still drop back faces that
// the VS marked via negative worldPos.w (see computeFogDepth's
// underWater sign convention).
public void depthPrepassStub(PSInput input)
{
    if (input.worldPos.w < 0)
        discard;
}

// Full alpha-tested prepass. Mirrors the alpha computation from
// ps_standard / ps_multi_layer and runs the configured TAlpha policy
// against it. The goal is to produce an identical discard mask to the
// lit pass without computing color.
public void depthPrepassBody<TAlpha : IAlphaTest, TMat : IMaterialLayer>(
    PSInput input, bool isFrontFace)
{
    // Back-face cull (same as the stub).
    if (input.worldPos.w < 0)
        discard;

    // When the alpha test is disabled we have nothing to compute.
    if (!TAlpha.enabled)
        return;

    // Sample + decode the normal so we can compute NdotV for the
    // fresnel-driven alpha contribution (cloak, hero rim).
    float3 normalSample = t_normal.Sample(s_normal, input.uv.xy).xyw;
    float3 tsNormal = decodeNormalMap(normalSample);

    float3 worldNormal = tangentToWorld(tsNormal, input.tangentWS.xyz, input.normalWS.xyz, input.tangentWS.w);
    worldNormal = orientNormal(worldNormal, isFrontFace);

    float3 viewDir = normalize(-input.worldPos.xyz);
    float NdotV = saturate(dot(viewDir, worldNormal));

    // Classic fresnel-squared rim factor.
    float oneMinusNdotV = 1.0 - NdotV;
    float fresnelSq = oneMinusNdotV * oneMinusNdotV;
    float edgeMask = min(1.0, (1.0 - fresnelSq) * EDGE_DETECT_SCALE);

    // Only the albedo alpha is sampled — color is never materialised.
    float albedoAlpha = t_albedo.Sample(s_albedo, input.uv.xy).w;
    float fresnelAlphaParam = psCB2.fresnelColor.w;
    float vertAlpha = input.vertColor.w;

    float finalAlpha;

    if (TMat.hasMultiLayer)
    {
        // Multi-layer uses the team-color blend weight to scale the
        // cloak-edge ramp, matching the alpha produced by applyCloakML.
        float fresnelContrib = fresnelAlphaParam * fresnelSq;
        float uncloakedAlpha = saturate(vertAlpha * albedoAlpha + fresnelContrib * albedoAlpha);

        float blendWeight = t_orm.Sample(s_orm, input.uv.xy).w;
        float alphaRamp = blendWeight * ML_ALPHA_RAMP_SCALE + ML_ALPHA_RAMP_BIAS;
        float cloakAlpha = edgeMask * alphaRamp + CLOAK_ALPHA_BIAS;
        cloakAlpha = psCB2.cloakAmount * (cloakAlpha - vertAlpha) + vertAlpha;
        float cloakedAlpha = albedoAlpha * cloakAlpha;

        finalAlpha = (psCB2.cloakAmount > 0) ? cloakedAlpha : uncloakedAlpha;
    }
    else
    {
        // Single-layer path mirrors applyCloak's alpha.
        float cloakAlpha = edgeMask * CLOAK_EDGE_SCALE + CLOAK_ALPHA_BIAS;
        cloakAlpha = psCB2.cloakAmount * (cloakAlpha - vertAlpha) + vertAlpha;

        float fresnelContrib = fresnelAlphaParam * fresnelSq;
        float cloakedAlpha = albedoAlpha * cloakAlpha;
        float uncloakedAlpha = saturate(vertAlpha * albedoAlpha + fresnelContrib * albedoAlpha);

        finalAlpha = (psCB2.cloakAmount > 0) ? cloakedAlpha : uncloakedAlpha;
    }

    TAlpha.test(finalAlpha, psCB2.alphaRef);
}
