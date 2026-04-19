// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026, Fernando Sahmkow
// See LICENSE in the repository root for full terms.
//
// Crystal depth-prepass fragment body.
//
// The crystal depth prepass computes the same cut-out alpha the lit pass
// does so the depth buffer agrees with the colour pass on alpha-tested
// edges. It differs from `depthPrepassBody` only in the Fresnel alpha
// parameter: crystal remaps that by the refract mask so transparent
// pixels still produce a visible rim contribution (otherwise the prepass
// discard mask would under-cover the silhouette).
//
// Everything else — the backface discard stub, normal decode, NdotV /
// edge / cloak-alpha math, and the final TAlpha.test — is shared with
// the HD depth prepass. We reuse `depthPrepassStub` directly and only
// re-implement the alpha body here.

implementing wc3_shaders;

// Crystal alpha-tested depth prepass. See `depthPrepassBody` for the
// algorithm; the single crystal-specific delta is that the Fresnel
// alpha parameter is remapped by `refractMask` before being used to
// compute `fresnelContrib`.
public void crystalDepthPrepassBody<TAlpha : IAlphaTest, TMat : IMaterialLayer>(
    PSInput input, bool isFrontFace)
{
    // Back-face cull (same as `depthPrepassStub`).
    if (input.worldPos.w < 0)
        discard;

    if (!TAlpha.enabled)
        return;

    // Normal decode for NdotV.
    float3 normalSample = t_normal.Sample(s_normal, input.uv.xy).xyw;
    float3 tsNormal = decodeNormalMap(normalSample);
    float3 worldNormal = tangentToWorld(tsNormal, input.tangentWS.xyz, input.normalWS.xyz, input.tangentWS.w);
    worldNormal = orientNormal(worldNormal, isFrontFace);

    float3 viewDir = normalize(-input.worldPos.xyz);
    float  NdotV   = saturate(dot(viewDir, worldNormal));

    float oneMinusNdotV = 1.0 - NdotV;
    float fresnelSq     = oneMinusNdotV * oneMinusNdotV;
    float edgeMask      = min(1.0, (1.0 - fresnelSq) * EDGE_DETECT_SCALE);

    // Sample only what we need for the alpha: centre albedo.w and the
    // refractMask. (Crystal never refract-blends the alpha shape.)
    float albedoAlpha = t_albedo.Sample(s_albedo, input.uv.xy).w;
    float refractMask = t_orm.Sample(s_orm, input.uv.xy).x;
    float fresnelAlphaMod = refractModulatedFresnelAlpha(psCB2.fresnelColor.w, refractMask);

    float vertAlpha = input.vertColor.w;
    float finalAlpha;

    if (TMat.hasMultiLayer)
    {
        // Multi-layer uses the team-blend weight to scale the cloak-edge
        // ramp, matching `applyCloakML`.
        float blendWeight = t_orm.Sample(s_orm, input.uv.xy).w;
        float fresnelContrib = fresnelAlphaMod * fresnelSq;
        float uncloakedAlpha = saturate(vertAlpha * albedoAlpha + fresnelContrib * albedoAlpha);

        float alphaRamp = blendWeight * ML_ALPHA_RAMP_SCALE + ML_ALPHA_RAMP_BIAS;
        float cloakAlpha = edgeMask * alphaRamp + CLOAK_ALPHA_BIAS;
        cloakAlpha = psCB2.cloakAmount * (cloakAlpha - vertAlpha) + vertAlpha;
        float cloakedAlpha = albedoAlpha * cloakAlpha;

        finalAlpha = (psCB2.cloakAmount > 0) ? cloakedAlpha : uncloakedAlpha;
    }
    else
    {
        float cloakAlpha = edgeMask * CLOAK_EDGE_SCALE + CLOAK_ALPHA_BIAS;
        cloakAlpha = psCB2.cloakAmount * (cloakAlpha - vertAlpha) + vertAlpha;

        float fresnelContrib = fresnelAlphaMod * fresnelSq;
        float cloakedAlpha   = albedoAlpha * cloakAlpha;
        float uncloakedAlpha = saturate(vertAlpha * albedoAlpha + fresnelContrib * albedoAlpha);

        finalAlpha = (psCB2.cloakAmount > 0) ? cloakedAlpha : uncloakedAlpha;
    }

    TAlpha.test(finalAlpha, psCB2.alphaRef);
}
