// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026, Fernando Sahmkow
// See LICENSE in the repository root for full terms.
//
// SD-on-HD non-IBL pixel shader body.
//
// This is the cheapest SD-on-HD path: the vertex color already carries
// per-vertex lighting, so we just modulate it by the albedo sample
// (after degamma to linear), optionally apply cloak, and return.

implementing wc3_shaders;

public float4 sdStandardBody(PSInput input, bool isFrontFace)
{
    // Needed only for the cloak fresnel factor below.
    float3 normalWS = normalize(input.normalWS.xyz);
    normalWS = isFrontFace ? normalWS : -normalWS;

    float3 viewDir = normalize(-input.worldPos.xyz);
    float NdotV = saturate(dot(viewDir, normalWS));

    // Linearise the per-vertex color; albedo is multiplied in linear
    // space so any downstream color-matrix / sRGB encoding in the
    // entry shader sees the right numbers.
    float3 vcLinear = srgbToLinear(input.vertColor.xyz);

    float4 texColor = t_albedo.Sample(s_albedo, input.uv.xy);

    float3 baseColor = texColor.xyz * vcLinear;

    CloakResult cloak = applyCloak(
        baseColor, texColor.w, texColor.w,
        NdotV, sdPsCB2.cloakAmount, input.vertColor.w);

    float nonCloakAlpha = saturate(input.vertColor.w * texColor.w);
    float4 nonCloakResult = float4(baseColor, nonCloakAlpha);

    bool cloakActive = sdPsCB2.cloakAmount > 0;
    return cloakActive ? cloak.colorAlpha : nonCloakResult;
}
