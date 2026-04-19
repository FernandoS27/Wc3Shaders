// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026, Fernando Sahmkow
// See LICENSE in the repository root for full terms.
//
// HD image-based-lighting + analytical-light pixel shader.
//
// This is the heaviest path in the HD pipeline. It:
//   1. Samples albedo / normal / ORM / emissive / team-albedo.
//   2. Runs the multi-layer blend when the material opts in.
//   3. Builds the world-space normal (with optional specular-AA).
//   4. For the first light, if it is directional, evaluates a full
//      Cook-Torrance lobe together with the IBL probe blend and the
//      "Blizzard ambient" mix — this is the key path that makes
//      dynamic lighting match the pre-baked ambient probes smoothly.
//   5. Runs a fixed-size light loop for the remaining analytic lights
//      (Lambert diffuse + Cook-Torrance specular).
//   6. Applies fresnel-overlay + cloak with the right (standard or
//      multi-layer) variants and returns the final lit color + alpha.
//
// The IBL probe blend uses two cubemap-array bindings (`from`, `to`) so
// the runtime can smoothly transition environments; `envTransitionT` is
// the lerp factor and `envFromMipEnd` / `envToMipEnd` pick the
// roughness-mapped mip for each probe independently (allowing the two
// probes to have different resolutions / mip counts).

implementing wc3_shaders;

// Split factors for the Blizzard ambient mix. The analytical-ambient
// contribution is decomposed into two pieces — an "upper" path that
// blends between the probe and the light's ambient color and a "lower"
// path that is scaled down by the NdotL coverage so unlit sides darken
// gracefully as the diffuse light fades out.
static const float IBL_SPLIT_UPPER = 0.675000012;
static const float IBL_SPLIT_LOWER = 0.324999988;

public struct IBLSampleResult
{
    float3 diffuseIrradiance;
    float3 specularRadiance;
    float2 brdfLUT;
    float  isActive;   // 0 when no probes are bound, 1 otherwise.
};

// Sample both the "from" and "to" IBL probes, blend them, and fetch the
// split-sum BRDF LUT. When either probe is absent (`envFromMipEnd *
// envToMipEnd == 0`) the result is cleared and `isActive = 0`, letting
// the caller skip the IBL contribution altogether.
//
// The `xzy` swizzles on the world-space vectors + the (1,1,-1) flip in
// the cubemap coord convert from the engine's Z-up world to the
// cubemap's Y-up, right-handed sampling convention.
public IBLSampleResult sampleIBL(
    float3 worldPos, float3 worldNormal, float roughnessAA,
    float4x4 invView, float envFromMipEnd, float envToMipEnd, float envTransitionT)
{
    IBLSampleResult result;

    // Fast-out when no probes are bound.
    float iblTest = envFromMipEnd * envToMipEnd;
    if (iblTest != 0.0)
    {
        // Transform position and normal into probe space. The probes are
        // authored in view space so we rotate from world back into view
        // using the transposed view matrix (invView).
        float3 viewPosDir  = normalize(mul((float3x3)invView, worldPos).xzy);
        float3 viewNormal  = normalize(mul((float3x3)invView, worldNormal).xzy);

        // Reflection vector for the specular lobe sample.
        float VdotN = dot(viewPosDir, viewNormal);
        float3 reflDir = viewPosDir - viewNormal * (VdotN + VdotN);
        reflDir = normalize(reflDir);

        // Diffuse irradiance sampled at mip 0 (slice 0).
        float4 diffCoord = float4(viewNormal * float3(1, 1, -1), 0);
        float3 diffFrom = t_iblFrom.Sample(s_iblFrom, diffCoord).xyz;
        float3 diffTo   = t_iblTo.Sample(s_iblTo, diffCoord).xyz;
        result.diffuseIrradiance = envTransitionT * (diffTo - diffFrom) + diffFrom;

        // Specular radiance at the roughness-mapped mip of each probe
        // (slice 1 holds the pre-convolved specular levels).
        float2 mipLevels = float2(envFromMipEnd, envToMipEnd) * roughnessAA;
        float4 specCoord = float4(reflDir * float3(1, 1, -1), 1);
        float3 specFrom = t_iblFrom.SampleLevel(s_iblFrom, specCoord, mipLevels.x).xyz;
        float3 specTo   = t_iblTo.SampleLevel(s_iblTo, specCoord, mipLevels.y).xyz;
        result.specularRadiance = envTransitionT * (specTo - specFrom) + specFrom;

        // Split-sum BRDF LUT: x is the Fresnel scale, y is the bias.
        // We pass the unclamped NdotV so glancing pixels are allowed to
        // darken past zero, which is what the LUT was baked against.
        float NdotV_unclamped = dot(worldNormal, normalize(-worldPos));
        float2 lutCoord = float2(NdotV_unclamped, roughnessAA);
        result.brdfLUT = t_brdfLut.Sample(s_brdfLut, lutCoord).xy;
        result.isActive = 1.0;
    }
    else
    {
        result.diffuseIrradiance = float3(0, 0, 0);
        result.specularRadiance = float3(0, 0, 0);
        result.brdfLUT = float2(0, 0);
        result.isActive = 0.0;
    }
    return result;
}

// The "Blizzard ambient" mix.
//
// Given the first analytical light's ambient term and the IBL diffuse
// irradiance, this produces a blend that:
//   - uses the analytical ambient when no IBL probe is active,
//   - matches the probe hue when it is,
//   - fades smoothly between the two depending on the relative
//     intensity of the probe vs. the analytical ambient (maxAmbient),
//   - and additionally darkens the "shadow side" (low NdotL) using
//     the `lower` split so unlit areas don't look flat.
public float3 blizzardAmbientBlend(
    float3 analyticalAmbient, float3 diffuseIBL,
    float NdotL, float maxAmbient, float iblActive, float ambientWeight)
{
    float safeDenom = 1.0 / (maxAmbient + SAFE_EPS);
    float totalWeight = maxAmbient + iblActive;
    float ambientIBLWeight = totalWeight * iblActive;
    ambientIBLWeight = ambientWeight * ambientIBLWeight;

    // Hue-separate both terms so the mix preserves saturation.
    float maxDiffIBL = max(max(diffuseIBL.x, diffuseIBL.y), diffuseIBL.z);
    float3 normIBL = diffuseIBL / (maxDiffIBL + SAFE_EPS);

    float3 normAmbient = analyticalAmbient * safeDenom;
    float3 delta = normAmbient - normIBL;

    // "Upper" term — dominant when the surface is lit (high NdotL).
    float3 upper = delta * IBL_SPLIT_UPPER + normIBL;
    upper *= maxDiffIBL;
    upper = upper * totalWeight - analyticalAmbient;
    upper = iblActive * upper + analyticalAmbient;

    // "Lower" term — dominant on the shadow side (low NdotL).
    float3 lower = delta * IBL_SPLIT_LOWER + normIBL;
    lower *= maxDiffIBL;

    float3 lowerScaled = lower * ambientIBLWeight;
    float3 result = NdotL * (upper - lowerScaled) + lowerScaled;
    return result;
}

// Analytical light loop. Runs Cook-Torrance + Lambert per light and
// accumulates into `accum`. The first directional light, if any, is
// handled separately by the caller (it also runs the Blizzard ambient
// blend); `startLight` is the loop index to resume from. All the
// roughness-dependent BRDF constants (a², a⁴, k) are pre-hoisted
// outside the loop since they don't change across lights.
public float3 lightLoop(
    float3 worldPos, float3 worldNormal, float3 viewDir, float viewDirLen,
    float3 F0, float3 oneMinusF0, float3 lambertianDiffuse,
    float roughnessAA, float NdotV_precomp,
    uint startLight)
{
    // Hoisted constants — used to reconstruct D (GGX) and V (Schlick-GGX).
    float a2 = roughnessAA * roughnessAA;
    float a4 = a2 * a2;
    float a4m1 = a4 - 1.0;

    float a1 = 1.0 + roughnessAA;
    float a12 = a1 * a1;
    float k = a12 * SCHLICK_K_FACTOR;
    float oneMinusK = 1.0 - k;
    float gvDenom = NdotV_precomp * oneMinusK + k;

    float3 accum = float3(0, 0, 0);
    uint lightIdx = startLight;

    while (true)
    {
        if (lightIdx >= asuint(psCB2.lightCount))
            break;

        ShaderLight light = psCB2.lights[lightIdx];
        float3 lightDir;
        float attenuation;

        if (light.type > 0)
        {
            // Point light: direction toward the light + inverse-square
            // attenuation (with the tiny distance bias from POINT_EPS).
            float3 lightVec = light.position - worldPos;
            float distSq = dot(lightVec, lightVec);
            float dist = sqrt(distSq);
            attenuation = pointLightAttenuation(distSq, dist);
            lightDir = lightVec / dist;
        }
        else
        {
            // Directional light: position is already the unit direction.
            lightDir = light.position;
            attenuation = 1.0;
        }

        // Evaluate the Cook-Torrance components inline so the compiler
        // can share intermediates (H, VdotH, NdotH) across D/F/V.
        float3 H = normalize(viewDir + lightDir);
        float VdotH = dot(H, viewDir);
        float NdotH = dot(worldNormal, H);
        float NdotL = dot(worldNormal, lightDir);

        // Schlick Fresnel (F = F0 + (1-F0)(1-VdotH)^5).
        float t = 1.0 - VdotH;
        float t2 = t * t;
        float t5 = t2 * t2 * t;
        float3 F = oneMinusF0 * t5 + F0;

        // GGX normal distribution.
        float nh2 = NdotH * NdotH;
        float denom = nh2 * a4m1 + 1.0;
        float D = a4 / (PI * denom * denom);

        // Schlick-GGX visibility (half of the classic G/(4·NdotL·NdotV)).
        float glDenom = NdotL * oneMinusK + k;
        float V = saturate(1.0 / (4.0 * glDenom * gvDenom));

        float clampedNdotL = max(0.0, NdotL);
        float contribution = attenuation * clampedNdotL;

        // Direct-light contribution: (diffuse + specular) * light color.
        float3 specular = D * F * V;
        float3 brdf = specular + lambertianDiffuse;
        accum += light.diffuse * brdf * contribution;

        // Per-light ambient (attenuated, independent of NdotL) —
        // matches the analytical-ambient term the engine uploads.
        accum += light.ambient * lambertianDiffuse * attenuation;

        lightIdx++;
    }
    return accum;
}

// HD IBL + analytical pixel shader body. See file banner for the
// overall flow; the comments below highlight the non-obvious steps.
public float4 psIBLBody<
    TMat : IMaterialLayer,
    let HAS_EXTRA_VERTS : bool,
    let HAS_DEBUG_VIS : bool>(
    PSInput input, bool isFrontFace)
{
    // Shadow cascade attenuation — sampled up front so both the main
    // directional light and the HLSL scheduler see the same register
    // layout as the retail shader (shadow visibility lands in r2.z
    // before the normal-map decode). Skipped when the VS didn't pass
    // shadowClip coords (HAS_EXTRA_VERTS=false).
    float shadowAtten = 1.0;
    if (HAS_EXTRA_VERTS)
    {
        shadowAtten = sdSampleShadowCascades(
            input.shadowClip0, input.shadowClip1, input.shadowClip2,
            asuint(sdPsCB1.numCascades));
    }

    // Base material sampling. Multi-layer materials also need the blend
    // weight from t2.w, so we take the .yzw swizzle up front and let
    // the non-ML path ignore the .w lane — this matches the single
    // sample op the retail shader emits.
    float4 albedo = t_albedo.Sample(s_albedo, input.uv.xy);
    float3 normalSample = t_normal.Sample(s_normal, input.uv.xy).xyw;
    float3 ormFull = t_orm.Sample(s_orm, input.uv.xy).yzw;
    float2 orm = ormFull.xy;
    float3 emissive = t_emissive.Sample(s_emissive, input.uv.xy).xyz;

    // Debug overrides — applied right after the texture fetches so the
    // override values feed every downstream consumer (ML blend, normal
    // decode, Cook-Torrance). Matches the retail DXBC which emits the
    // and+movc pair immediately after the t0/t1/t2/t3 samples.
    //   bit 0 — replace albedo.rgb with overrideAlbedo.
    //   bit 1 — replace (roughness, metalness) with overrideOrm.yz,
    //           mirroring the cb3[2].yzyy movc in the original DXBC.
    //           The ML blend weight (ormFull.z) is intentionally left
    //           alone so debug doesn't reshape the team-color layer.
    if (HAS_DEBUG_VIS)
    {
        uint enabled = asuint(psCB3.enabledShaders);
        if (enabled & 1)
            albedo.xyz = psCB3.overrideAlbedo;
        if (enabled & 2)
            orm = psCB3.overrideOrm.yz;
    }

    float3 teamAlbedo = float3(0, 0, 0);
    float blendWeight = 0.0;
    float3 teamHueNorm = float3(0, 0, 0);
    float teamLuma = 0.0;

    if (TMat.hasMultiLayer)
    {
        blendWeight = ormFull.z;
        teamAlbedo = t_teamColor.Sample(s_teamColor, input.uv.xy).xyz;
        teamLuma = max(max(teamAlbedo.x, teamAlbedo.y), teamAlbedo.z);
        teamHueNorm = teamAlbedo / (teamLuma + SAFE_EPS);
    }

    float roughness = orm.x;
    float metalness = orm.y;

    // Apply the multi-layer blend once so downstream math sees a single
    // effective albedo / emissive.
    if (TMat.hasMultiLayer)
    {
        MultiLayerResult ml = multiLayerBlend(albedo.xyz, teamAlbedo, blendWeight, emissive);
        albedo.xyz = ml.albedo;
        emissive = ml.emissive;
    }

    // Build the world-space normal and reconstruct a roughness mip that
    // accounts for sub-pixel normal variance (specular-AA) when the
    // `useNdf` flag is set on this draw.
    float3 tsNormal = decodeNormalMap(normalSample);
    float3 worldNormal = tangentToWorld(tsNormal, input.tangentWS.xyz, input.normalWS.xyz, input.tangentWS.w);
    worldNormal = orientNormal(worldNormal, isFrontFace);

    float roughnessMip = roughnessToMip(roughness);
    float roughnessAA = psCB2.useNdf != 0 ? specularAA(roughnessMip, worldNormal) : roughnessMip;

    // View direction. We reuse `viewDirLen` (= 1/|worldPos|) in later
    // dot products to avoid recomputing the normalisation.
    float viewDirLen = rsqrt(dot(-input.worldPos.xyz, -input.worldPos.xyz));
    float3 viewDir = -input.worldPos.xyz * viewDirLen;

    // Metalness splits the albedo into diffuse (non-metal) and F0
    // (metal) components; the Lambertian diffuse scales by 1/π so the
    // final PI-scale at the end undoes it for the direct lights.
    float3 F0 = metalness * albedo.xyz;
    float3 diffuseAlbedo = (1.0 - metalness) * albedo.xyz;
    float3 lambertianDiffuse = INV_PI * diffuseAlbedo;

    float3 accum = float3(0, 0, 0);
    uint firstLightProcessed = 0;

    // If light 0 is directional, handle it together with the IBL blend
    // so the probe contribution and the main analytical light share
    // their ambient mix. Subsequent lights run through the generic
    // light loop.
    bool hasFirstLight = asuint(psCB2.lightCount) > 0;
    bool firstIsDirectional = hasFirstLight && (psCB2.lights[0].type == 0.0);

    if (firstIsDirectional)
    {
        float3 lightDir = psCB2.lights[0].position;

        // Cook-Torrance for the main directional light.
        float3 H = normalize(viewDir + lightDir);
        float VdotH = dot(H, viewDir);
        float NdotH = dot(worldNormal, H);
        float NdotL = dot(worldNormal, lightDir);
        float NdotV = dot(worldNormal, viewDir);

        float3 F = schlickFresnel(F0, VdotH);
        float D = ggxNDF(NdotH, roughnessAA);
        float V = schlickGGXVisibility(NdotL, NdotV, roughnessAA);
        float3 spec = D * F * V;
        float clampedNdotL = max(0.0, NdotL);

        // Shadow cascades attenuate only the main directional light's
        // direct contribution (diffuse ambient modulation + direct
        // specular). Indirect IBL terms stay unshadowed, matching the
        // retail DXBC's `mul r2.x, r2.z, r2.x` fold that re-scales the
        // clamped NdotL in-place before the ambient / specular combines.
        float litNdotL = HAS_EXTRA_VERTS ? (clampedNdotL * shadowAtten) : clampedNdotL;

        // Sample both IBL probes for this pixel.
        IBLSampleResult ibl = sampleIBL(
            input.worldPos.xyz, worldNormal, roughnessAA,
            psCB2.invView,
            psCB2.envFromMipEnd, psCB2.envToMipEnd, psCB2.envTransitionT);

        float maxDiffuse = max(max(psCB2.lights[0].diffuse.x, psCB2.lights[0].diffuse.y), psCB2.lights[0].diffuse.z);
        float ambientWeight = psCB2.lights[0]._pad2.x; // Per-light ambient weight.

        // Diffuse ambient: analytical+IBL mix, modulated by shadow-
        // attenuated NdotL so the lit / shadow sides of the Blizzard
        // ambient split respond to occlusion.
        float3 ambientResult = blizzardAmbientBlend(
            psCB2.lights[0].diffuse, ibl.diffuseIrradiance,
            litNdotL, maxDiffuse, ibl.isActive, ambientWeight);
        ambientResult += psCB2.lights[0].ambient;

        // Direct + indirect specular combine. The direct spec is scaled
        // by the combined weight so it matches the ambient strength,
        // and the indirect term reconstructs a reflection from the
        // split-sum LUT (F0 * scale + bias) then modulates by the IBL
        // specular radiance.
        float totalWeight = maxDiffuse + ibl.isActive;
        float3 scaledSpec = spec * totalWeight * litNdotL;

        float3 indirectSpec = F0 * ibl.brdfLUT.x + ibl.brdfLUT.y;
        float ambientIBLWeight = totalWeight * ibl.isActive * ambientWeight;
        indirectSpec *= ambientIBLWeight;
        indirectSpec += scaledSpec * psCB2.lights[0].diffuse;
        indirectSpec *= ibl.specularRadiance;

        accum = ambientResult * lambertianDiffuse + indirectSpec;
        firstLightProcessed = 1; // Skip light 0 in the loop below.
    }

    // Analytical loop for remaining lights.
    float NdotV_precomp = dot(worldNormal, viewDir);
    float3 oneMinusF0 = 1.0 - F0;

    float3 loopResult = lightLoop(
        input.worldPos.xyz, worldNormal, viewDir, viewDirLen,
        F0, oneMinusF0, lambertianDiffuse,
        roughnessAA, NdotV_precomp,
        firstLightProcessed);
    accum += loopResult;

    // Multiply by π (un-normalising the Lambertian divide), add
    // emissive, and modulate by the per-vertex color (which carries AO
    // and hero tints).
    float3 finalColor = PI * accum;
    finalColor += emissive * psCB2.emissiveGain;

    float3 linearVert = srgbToLinear(input.vertColor.xyz);
    finalColor *= linearVert;

    // Finishing stack: fresnel overlay → cloak. Variant selection
    // matches the material's multi-layer flag.
    float NdotV_sat = saturate(NdotV_precomp);
    float oneMinusNdotV = 1.0 - NdotV_sat;
    float fresnelSq = oneMinusNdotV * oneMinusNdotV;
    float fAlpha = psCB2.fresnelColor.w * fresnelSq;

    float3 fresnelResult;
    float finalAlpha;

    if (TMat.hasMultiLayer)
    {
        FresnelOverlayResult fr = applyFresnelOverlayML(
            finalColor, albedo.w, NdotV_sat,
            psCB2.fresnelColor.xyz, psCB2.fresnelColor.w, psCB2.fresnelTeamColor,
            input.vertColor.w,
            teamAlbedo, teamLuma);
        fresnelResult = fr.color;
        finalAlpha = fr.alpha;
    }
    else
    {
        FresnelOverlayResult fr = applyFresnelOverlay(
            finalColor, albedo.w, NdotV_sat,
            psCB2.fresnelColor.xyz, psCB2.fresnelColor.w, psCB2.fresnelTeamColor,
            input.vertColor.w);
        fresnelResult = fr.color;
        finalAlpha = fr.alpha;
    }

    bool cloakActive = psCB2.cloakAmount > 0;
    if (cloakActive)
    {
        if (TMat.hasMultiLayer)
        {
            CloakResult cr = applyCloakML(
                fresnelResult, finalAlpha, albedo.w,
                NdotV_sat, psCB2.cloakAmount, input.vertColor.w,
                blendWeight);
            return cr.colorAlpha;
        }
        else
        {
            CloakResult cr = applyCloak(
                fresnelResult, finalAlpha, albedo.w,
                NdotV_sat, psCB2.cloakAmount, input.vertColor.w);
            return cr.colorAlpha;
        }
    }

    return float4(fresnelResult, finalAlpha);
}
