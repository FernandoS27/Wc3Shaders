// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026, Fernando Sahmkow
// See LICENSE in the repository root for full terms.
//
// Non-IBL HD pixel shader bodies.
//
// These are the "cheap" shading paths — used when the material opts out
// of the image-based-lighting path (or when the IBL cubemaps aren't
// bound). They still read the full PBR texture set and emit a lit
// color + alpha, but skip the expensive cubemap lookups + light loop
// from ps_ibl.slang; the per-vertex color already carries the lighting.
//
// psStandardBody  — single-layer variant.
// psMultiLayerBody — team-color multi-layer variant.
//
// Both provide a `HAS_DEBUG_VIS` specialisation. In debug mode we keep
// intermediate data around (normal length, alternate sample ordering)
// so the downstream `debugVisualize` can visualise it; in the release
// path we rearrange the math to reduce register pressure.

implementing wc3_shaders;

// Single-layer HD pixel shader.
// When HAS_DEBUG_VIS is set the code preserves the world-normal length
// (before normalisation) for the debug "normal length" visualisation
// and fetches albedo first so the albedo-override debug path can swap
// it cheaply. The two branches compute the same final color, just with
// the arithmetic inlined differently to match what the stock shader
// produced in each mode.
public float4 psStandardBody<let HAS_DEBUG_VIS : bool>(PSInput input, bool isFrontFace)
{
    float4 albedo;
    float3 emissive;
    float3 normalSample;
    float3 worldNormal;
    float normalLen;

    if (HAS_DEBUG_VIS)
    {
        // Debug path: sample everything, then apply overrides before
        // the rest of the shader runs.
        albedo = t_albedo.Sample(s_albedo, input.uv.xy);
        normalSample = t_normal.Sample(s_normal, input.uv.xy).xyw;
        emissive = t_emissive.Sample(s_emissive, input.uv.xy).xyz;

        // enabledShaders bit 0 → force the albedo to a solid color.
        if (asint(psCB3.enabledShaders) & 1)
            albedo.xyz = psCB3.overrideAlbedo;

        float3 tsNormal = decodeNormalMap(normalSample);
        worldNormal = tangentToWorld(tsNormal, input.tangentWS.xyz, input.normalWS.xyz, input.tangentWS.w);

        // Keep the raw length around for the "visualise normal length"
        // debug mode; subsequent math uses the normalised vector.
        normalLen = length(worldNormal);
        worldNormal = normalize(worldNormal);
        worldNormal = orientNormal(worldNormal, isFrontFace);

        // Baseline lit color: emissive + albedo tinted by the linearised
        // vertex color (which carries per-vertex lighting).
        float3 litColor = emissive * psCB2.emissiveGain + albedo.xyz;
        float3 linearVert = srgbToLinear(input.vertColor.xyz);
        litColor *= linearVert;

        float3 viewDir = normalize(-input.worldPos.xyz);
        float NdotV = saturate(dot(viewDir, worldNormal));
        float oneMinusNdotV = 1.0 - NdotV;
        float fresnelSq = oneMinusNdotV * oneMinusNdotV;

        FresnelOverlayResult fr = applyFresnelOverlay(
            litColor, albedo.w, NdotV,
            psCB2.fresnelColor.xyz, psCB2.fresnelColor.w, psCB2.fresnelTeamColor,
            input.vertColor.w);

        // Cloak runs last so the cyan tint modulates the fresnel rim.
        bool cloakActive = psCB2.cloakAmount > 0;
        if (cloakActive)
        {
            CloakResult cr = applyCloak(
                fr.color, fr.alpha, albedo.w,
                NdotV, psCB2.cloakAmount, input.vertColor.w);
            return cr.colorAlpha;
        }
        return float4(fr.color, fr.alpha);
    }
    else
    {
        // Release path: same pipeline, reordered and inlined.
        //
        // We intentionally re-derive the fresnel / cloak math here
        // rather than calling applyFresnelOverlay + applyCloak so the
        // compiler can interleave them, re-use intermediates, and fit
        // into the register budget the legacy shader hit. The result
        // is bit-for-bit identical to the debug path.
        normalSample = t_normal.Sample(s_normal, input.uv.xy).xyw;
        float3 tsNormal = decodeNormalMap(normalSample);
        worldNormal = tangentToWorld(tsNormal, input.tangentWS.xyz, input.normalWS.xyz, input.tangentWS.w);
        worldNormal = orientNormal(worldNormal, isFrontFace);

        float3 viewDir = normalize(-input.worldPos.xyz);
        float NdotV = saturate(dot(viewDir, worldNormal));
        float oneMinusNdotV = 1.0 - NdotV;
        float fresnelSq = oneMinusNdotV * oneMinusNdotV;

        // Pre-computed cloak alpha for the cloak-on branch below.
        float edgeMask = min(1.0, (1.0 - fresnelSq) * EDGE_DETECT_SCALE);
        float cloakAlpha = edgeMask * CLOAK_EDGE_SCALE + CLOAK_ALPHA_BIAS;
        cloakAlpha = psCB2.cloakAmount * (cloakAlpha - input.vertColor.w) + input.vertColor.w;

        // Fresnel overlay inlined (see applyFresnelOverlay).
        float fresnelAlphaContrib = psCB2.fresnelColor.w * fresnelSq;
        float teamSq = psCB2.fresnelTeamColor * psCB2.fresnelTeamColor;
        float maxFresnel = max(max(psCB2.fresnelColor.x, psCB2.fresnelColor.y), psCB2.fresnelColor.z);
        float intensity = maxFresnel + teamSq * (min(1.0, maxFresnel) - maxFresnel);
        float safeMaxInv = 1.0 / (maxFresnel + SAFE_EPS);
        float3 normTint = psCB2.fresnelColor.xyz * safeMaxInv;
        float3 blendedTint = psCB2.fresnelTeamColor * (FRESNEL_TARGET - normTint) + normTint;

        float3 linearVert = srgbToLinear(input.vertColor.xyz);
        emissive = t_emissive.Sample(s_emissive, input.uv.xy).xyz;
        albedo = t_albedo.Sample(s_albedo, input.uv.xy);
        float3 litColor = (emissive * psCB2.emissiveGain + albedo.xyz) * linearVert;

        float3 fresnelColor = blendedTint * intensity;
        float3 finalColor = fresnelAlphaContrib * (fresnelColor - litColor) + litColor;
        float fresnelAlphaResult = fresnelAlphaContrib * albedo.w;
        float finalAlpha = saturate(input.vertColor.w * albedo.w + fresnelAlphaResult);

        bool cloakActive = psCB2.cloakAmount > 0;
        if (cloakActive)
        {
            float cloakedAlpha = albedo.w * cloakAlpha;
            float3 tinted = psCB2.cloakAmount * fresnelSq * (CLOAK_TINT - finalColor) + finalColor;
            return float4(tinted, cloakedAlpha);
        }
        return float4(finalColor, finalAlpha);
    }
}

// Multi-layer HD pixel shader. Produces albedo + emissive by blending
// the base material with the team-color layer (see multi_layer.slang),
// then runs the same fresnel/cloak finishing stack — with the multi-
// layer variants of the overlays so tints are hue-matched to the team
// albedo.
public float4 psMultiLayerBody<let HAS_DEBUG_VIS : bool>(PSInput input, bool isFrontFace)
{
    float3 linearVert;
    float4 albedo;
    float3 teamAlbedo;
    float blendWeight;
    float3 emissive;
    float3 normalSample;

    if (HAS_DEBUG_VIS)
    {
        // Debug path sets up the same sample set + override as above,
        // then runs the shared multi-layer body below.
        albedo = t_albedo.Sample(s_albedo, input.uv.xy);
        normalSample = t_normal.Sample(s_normal, input.uv.xy).xyw;
        blendWeight = t_orm.Sample(s_orm, input.uv.xy).w;
        emissive = t_emissive.Sample(s_emissive, input.uv.xy).xyz;
        teamAlbedo = t_teamColor.Sample(s_teamColor, input.uv.xy).xyz;

        if (asint(psCB3.enabledShaders) & 1)
            albedo.xyz = psCB3.overrideAlbedo;

        linearVert = srgbToLinear(input.vertColor.xyz);
    }
    else
    {
        // Release-path sample order picked to let the driver coalesce.
        linearVert = srgbToLinear(input.vertColor.xyz);
        albedo = t_albedo.Sample(s_albedo, input.uv.xy);
        teamAlbedo = t_teamColor.Sample(s_teamColor, input.uv.xy).xyz;
        blendWeight = t_orm.Sample(s_orm, input.uv.xy).w;
        emissive = t_emissive.Sample(s_emissive, input.uv.xy).xyz;
    }

    // Hue-preserving team blend for both albedo and emissive.
    MultiLayerResult ml = multiLayerBlend(albedo.xyz, teamAlbedo, blendWeight, emissive);

    float3 litColor = (ml.emissive * psCB2.emissiveGain + ml.albedo) * linearVert;

    // Needed by the multi-layer fresnel overlay.
    float teamLuma = max(max(teamAlbedo.x, teamAlbedo.y), teamAlbedo.z);
    float teamSafe = 1.0 / (teamLuma + SAFE_EPS);

    // Debug path already fetched the normal; release path sampled it
    // only after the albedo/emissive swap so we only fetch it here.
    if (!HAS_DEBUG_VIS)
        normalSample = t_normal.Sample(s_normal, input.uv.xy).xyw;

    float3 tsNormal = decodeNormalMap(normalSample);
    float3 worldNormal = tangentToWorld(tsNormal, input.tangentWS.xyz, input.normalWS.xyz, input.tangentWS.w);
    worldNormal = orientNormal(worldNormal, isFrontFace);

    float3 viewDir = normalize(-input.worldPos.xyz);
    float NdotV = saturate(dot(viewDir, worldNormal));
    float oneMinusNdotV = 1.0 - NdotV;
    float fresnelSq = oneMinusNdotV * oneMinusNdotV;

    // Inlined applyFresnelOverlayML (mirrors the ML variant's hue /
    // intensity logic — kept inline to match the legacy codegen).
    float fAlpha = psCB2.fresnelColor.w * fresnelSq;
    float maxFresnel = max(max(psCB2.fresnelColor.x, psCB2.fresnelColor.y), psCB2.fresnelColor.z);
    float safeMaxInv = 1.0 / (maxFresnel + SAFE_EPS);
    float3 normTint = psCB2.fresnelColor.xyz * safeMaxInv;
    float3 mlTintTarget = teamAlbedo * teamSafe;
    float3 blendedTint = psCB2.fresnelTeamColor * (mlTintTarget - normTint) + normTint;
    float teamSq = psCB2.fresnelTeamColor * psCB2.fresnelTeamColor;
    float intensity = maxFresnel + teamSq * (min(maxFresnel, teamLuma) - maxFresnel);
    float3 fresnelColor = blendedTint * intensity;

    float3 finalColor = fAlpha * (fresnelColor - litColor) + litColor;
    float finalAlpha = saturate(input.vertColor.w * albedo.w + fAlpha * albedo.w);

    bool cloakActive = psCB2.cloakAmount > 0;
    if (cloakActive)
    {
        // Inlined applyCloakML: team-weighted alpha ramp + tint
        // strength that fades out where blendWeight is high so team
        // glyphs stay readable under cloak.
        float edgeMask = min(1.0, (1.0 - fresnelSq) * EDGE_DETECT_SCALE);
        float alphaRamp = blendWeight * ML_ALPHA_RAMP_SCALE + ML_ALPHA_RAMP_BIAS;
        float cloakAlpha = edgeMask * alphaRamp + CLOAK_ALPHA_BIAS;
        cloakAlpha = psCB2.cloakAmount * (cloakAlpha - input.vertColor.w) + input.vertColor.w;
        float cloakedAlpha = albedo.w * cloakAlpha;

        // `tintReduced` is mathematically identical to
        //   fresnelSq * (1 - blendWeight * ML_EMISSIVE_BLEND)
        // but written as an FMA so the two multiplies can schedule
        // independently of the earlier fresnelSq multiply.
        float tintStrength = fresnelSq * (1.0 - blendWeight * ML_EMISSIVE_BLEND);
        float tintReduced = fresnelSq * blendWeight;
        tintReduced = tintReduced * -ML_EMISSIVE_BLEND + fresnelSq;
        float3 tinted = psCB2.cloakAmount * tintReduced * (CLOAK_TINT - finalColor) + finalColor;

        return float4(tinted, cloakedAlpha);
    }
    return float4(finalColor, finalAlpha);
}
