// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026, Fernando Sahmkow
// See LICENSE in the repository root for full terms.
//
// Non-IBL crystal pixel shader bodies.
//
// These mirror `psStandardBody` / `psMultiLayerBody` but with the crystal
// albedo-refraction path in place of the usual `t_albedo.Sample()` and
// the refract-modulated Fresnel alpha in place of the authored one. The
// BRDF math itself is trivial for the non-IBL crystal paths — the
// per-vertex color already carries lighting and the team blend /
// fresnel overlay / cloak composite the final colour.
//
// Both variants provide a `HAS_DEBUG_VIS` specialisation that honours
// the two cb3 override bits: bit 0 replaces the albedo with a constant
// colour and bit 1 replaces the scalar refractMask with an authored one
// (typical use: sweeping the slider in an artist tool to isolate one
// half of the blend).

implementing wc3_shaders;

// Single-layer (non-ML, non-IBL) crystal body.
public float4 crystalStandardBody<let HAS_DEBUG_VIS : bool>(PSInput input, bool isFrontFace)
{
    // Crystal control-map sample: only `.x` (refractMask) is needed in
    // the non-ML path. Fetched first so DEBUG_VIS can override it
    // before the refraction blend consumes it below.
    float refractMask = t_orm.Sample(s_orm, input.uv.xy).x;
    if (HAS_DEBUG_VIS && (asuint(psCB3.enabledShaders) & 2) != 0)
        refractMask = psCB3.overrideOrm.x;

    // Refracted albedo — `.w` comes from the centre sample only, so the
    // downstream cloak / Fresnel alpha behaviour is unchanged by the
    // refraction (refraction only affects colour, never opacity shape).
    float4 albedo = sampleCrystalAlbedo(input.uv.xy, input.normalWS.xyz, refractMask);
    if (HAS_DEBUG_VIS && (asuint(psCB3.enabledShaders) & 1) != 0)
        albedo.xyz = psCB3.overrideAlbedo;

    float3 emissive = t_emissive.Sample(s_emissive, input.uv.xy).xyz;

    // World-space normal.
    float3 normalSample = t_normal.Sample(s_normal, input.uv.xy).xyw;
    float3 tsNormal = decodeNormalMap(normalSample);
    float3 worldNormal = tangentToWorld(tsNormal, input.tangentWS.xyz, input.normalWS.xyz, input.tangentWS.w);
    worldNormal = orientNormal(worldNormal, isFrontFace);

    // View direction + NdotV.
    float3 viewDir = normalize(-input.worldPos.xyz);
    float  NdotV   = saturate(dot(viewDir, worldNormal));

    // Lit colour: (emissive·gain + albedo) * linearVert.
    float3 linearVert = srgbToLinear(input.vertColor.xyz);
    float3 litColor   = (emissive * psCB2.emissiveGain + albedo.xyz) * linearVert;

    // Fresnel overlay with the refract-modulated alpha so fully
    // refracted pixels still receive a visible hero rim.
    float fresnelAlphaMod = refractModulatedFresnelAlpha(psCB2.fresnelColor.w, refractMask);
    FresnelOverlayResult fr = applyFresnelOverlay(
        litColor, albedo.w, NdotV,
        psCB2.fresnelColor.xyz, fresnelAlphaMod, psCB2.fresnelTeamColor,
        input.vertColor.w);

    if (psCB2.cloakAmount > 0)
    {
        CloakResult cr = applyCloak(
            fr.color, fr.alpha, albedo.w,
            NdotV, psCB2.cloakAmount, input.vertColor.w);
        return cr.colorAlpha;
    }
    return float4(fr.color, fr.alpha);
}

// Multi-layer (non-IBL) crystal body. Same shape as `crystalStandardBody`
// but runs the team-colour blend on the refracted albedo and uses the
// multi-layer fresnel + cloak overlays.
public float4 crystalMultiLayerBody<let HAS_DEBUG_VIS : bool>(PSInput input, bool isFrontFace)
{
    // Crystal control map (.x=refract, .w=blendWeight); the two other
    // channels (.y=roughness, .z=metalness) are only consumed in the
    // IBL path.
    float2 ctrl = t_orm.Sample(s_orm, input.uv.xy).xw;
    float refractMask = ctrl.x;
    float blendWeight = ctrl.y;
    if (HAS_DEBUG_VIS && (asuint(psCB3.enabledShaders) & 2) != 0)
        refractMask = psCB3.overrideOrm.x;

    float4 albedo = sampleCrystalAlbedo(input.uv.xy, input.normalWS.xyz, refractMask);
    if (HAS_DEBUG_VIS && (asuint(psCB3.enabledShaders) & 1) != 0)
        albedo.xyz = psCB3.overrideAlbedo;

    float3 teamAlbedo = t_teamColor.Sample(s_teamColor, input.uv.xy).xyz;
    float3 emissive   = t_emissive.Sample(s_emissive, input.uv.xy).xyz;

    // Team-colour blend on the refracted albedo + emissive.
    MultiLayerResult ml = multiLayerBlend(albedo.xyz, teamAlbedo, blendWeight, emissive);

    // World-space normal.
    float3 normalSample = t_normal.Sample(s_normal, input.uv.xy).xyw;
    float3 tsNormal = decodeNormalMap(normalSample);
    float3 worldNormal = tangentToWorld(tsNormal, input.tangentWS.xyz, input.normalWS.xyz, input.tangentWS.w);
    worldNormal = orientNormal(worldNormal, isFrontFace);

    float3 viewDir = normalize(-input.worldPos.xyz);
    float  NdotV   = saturate(dot(viewDir, worldNormal));

    float3 linearVert = srgbToLinear(input.vertColor.xyz);
    float3 litColor   = (ml.emissive * psCB2.emissiveGain + ml.albedo) * linearVert;

    // ML fresnel overlay needs the team-albedo luma peak so its intensity
    // ramp matches the diffuse layer's brightness.
    float teamLuma = max(max(teamAlbedo.x, teamAlbedo.y), teamAlbedo.z);
    float fresnelAlphaMod = refractModulatedFresnelAlpha(psCB2.fresnelColor.w, refractMask);
    FresnelOverlayResult fr = applyFresnelOverlayML(
        litColor, albedo.w, NdotV,
        psCB2.fresnelColor.xyz, fresnelAlphaMod, psCB2.fresnelTeamColor,
        input.vertColor.w,
        teamAlbedo, teamLuma);

    if (psCB2.cloakAmount > 0)
    {
        CloakResult cr = applyCloakML(
            fr.color, fr.alpha, albedo.w,
            NdotV, psCB2.cloakAmount, input.vertColor.w,
            blendWeight);
        return cr.colorAlpha;
    }
    return float4(fr.color, fr.alpha);
}
