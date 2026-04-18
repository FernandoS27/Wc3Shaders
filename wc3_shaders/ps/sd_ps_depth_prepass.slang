// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026, Fernando Sahmkow
// See LICENSE in the repository root for full terms.
//
// SD-on-HD depth prepass.
//
// Simpler than the HD prepass because the SD path has no tangent frame
// and no multi-layer alpha math. We only need to match the alpha that
// the lit SD body computes: vertex alpha × texture alpha, with an
// optional cloak ramp driven by the fresnel edge factor.

implementing wc3_shaders;

// No-op backface discard for cases where alpha test is disabled.
public void sdDepthPrepassStub(PSInput input)
{
    if (input.worldPos.w < 0)
        discard;
}

// Alpha-tested prepass. Produces the same alpha that sdStandardBody /
// sdIBLBody would compute, so the final depth has the correct cut-out
// silhouette for cloaked or cut-out materials.
public void sdDepthPrepassBody<TAlpha : IAlphaTest>(PSInput input, bool isFrontFace)
{
    if (input.worldPos.w < 0)
        discard;

    // No tangent frame: the SD path uses the raw interpolated normal,
    // flipped for back faces. Renormalise because interpolation across
    // the triangle shortens the vector.
    float3 normalWS = normalize(input.normalWS.xyz);
    normalWS = isFrontFace ? normalWS : -normalWS;

    // Fresnel edge factor, same shape as the HD path.
    float3 viewDir = normalize(-input.worldPos.xyz);
    float NdotV = saturate(dot(viewDir, normalWS));

    float oneMinusNdotV = 1.0 - NdotV;
    float edgeMask = min(1.0, (1.0 - oneMinusNdotV * oneMinusNdotV) * EDGE_DETECT_SCALE);
    float cloakAlpha = edgeMask * CLOAK_EDGE_SCALE + CLOAK_ALPHA_BIAS;
    cloakAlpha = sdPsCB2.cloakAmount * (cloakAlpha - input.vertColor.w) + input.vertColor.w;

    float texAlpha = t_albedo.Sample(s_albedo, input.uv.xy).w;

    // Two candidate alphas — the runtime picks based on cloak state.
    float cloakedAlpha = texAlpha * cloakAlpha;
    float normalAlpha = saturate(input.vertColor.w * texAlpha);
    float finalAlpha = (sdPsCB2.cloakAmount > 0) ? cloakedAlpha : normalAlpha;

    TAlpha.test(finalAlpha, sdPsCB2.alphaRef);
}
