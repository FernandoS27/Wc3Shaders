// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026, Fernando Sahmkow
// See LICENSE in the repository root for full terms.
//
// Terrain analytical-light evaluation.
//
// Two distinct paths:
//
//   evaluateDominantLight() — runs once for light index 0 when it is a
//     directional light, and combines a Cook-Torrance lobe with the
//     dual-cubemap IBL split-sum + the asymmetric "Blizzard ambient"
//     mix that the HD path also uses (see ps_ibl.slang's
//     blizzardAmbientBlend). When `HAS_SHADOWS` is set the dominant
//     light's clamped NdotL is multiplied by the cascade visibility
//     before any of the ambient / spec combines.
//
//   evaluateAdditionalLights() — runs the remaining lights through a
//     plain Cook-Torrance loop (no IBL, no shadows). Mirrors the
//     hd_ps light-loop contract: typed `uint` counter so fxc skips
//     the float↔uint round-trip, and inlined `(idx<<2)+N` indexing
//     so the address arithmetic folds into the cb access.

implementing wc3_shaders;

// Re-orient a world-space vector into the engine's IBL-cubemap basis
// (world to view rotation as an `.xzy` swizzle of the rows in cb2[7..9]).
// The PS uses this for the dominant light's reflection sampling.
float3 worldToProbeBasis(float3 v)
{
    float3 r = v.y * psCB2.view[1].xzy;
    r = v.x * psCB2.view[0].xzy + r;
    r = v.z * psCB2.view[2].xzy + r;
    return r * rsqrt(dot(r, r));
}

// Split-sum-style IBL block: dual cubemap diffuse + specular sample
// blended by `psCB2.envTransitionT` and the BRDF LUT lookup at NdotV.
// Returns env diffuse, env specular, BRDF LUT, and an `iblActive`
// boolean (1 when both cubemaps are bound, 0 otherwise).
struct TerrainIBLSample
{
    float3 envDiffuse;
    float3 envSpec;
    float2 brdfLut;
    float  isActive;
};

TerrainIBLSample sampleTerrainIBL(
    float3 viewPos, float3 worldNormal, float roughness, float NdotV)
{
    TerrainIBLSample r;

    float gate = psCB2.envFromMipEnd * psCB2.envToMipEnd;
    if (gate != 0.0)
    {
        // World-space view direction rotated into the probe basis.
        float3 viewDirProbe = worldToProbeBasis(viewPos);
        float3 normalProbe  = worldToProbeBasis(worldNormal);

        // Reflection of viewDir about the probe-space normal.
        float vdn = dot(viewDirProbe, normalProbe);
        float3 refl = -2.0 * vdn * normalProbe + viewDirProbe;
        refl = refl * rsqrt(dot(refl, refl));

        // The (1, 1, -1) flip + slice index converts world-space samples
        // into the engine's Y-up, right-handed cubemap convention. The
        // diffuse half is at slice 0 and is sampled at full resolution;
        // specular sits at slice 1 with a roughness-mapped mip.
        float4 diffCoord = float4(normalProbe * float3(1, 1, -1), 0);
        float3 dFrom = t_iblFrom.Sample(s_iblFrom, diffCoord).xyz;
        float3 dTo   = t_iblTo.Sample(s_iblTo, diffCoord).xyz;
        r.envDiffuse = psCB2.envTransitionT * (dTo - dFrom) + dFrom;

        float2 mipLevels = float2(psCB2.envFromMipEnd, psCB2.envToMipEnd) * roughness;
        float4 specCoord = float4(refl * float3(1, 1, -1), 1);
        float3 sFrom = t_iblFrom.SampleLevel(s_iblFrom, specCoord, mipLevels.x).xyz;
        float3 sTo   = t_iblTo.SampleLevel(s_iblTo, specCoord, mipLevels.y).xyz;
        r.envSpec = psCB2.envTransitionT * (sTo - sFrom) + sFrom;

        r.brdfLut  = t_brdfLut.Sample(s_brdfLut, float2(NdotV, roughness)).xy;
        r.isActive = 1.0;
    }
    else
    {
        r.envDiffuse = float3(0, 0, 0);
        r.envSpec    = float3(0, 0, 0);
        r.brdfLut    = float2(0, 0);
        r.isActive   = 0.0;
    }
    return r;
}

// Result of the dominant-light pass.
public struct TerrainDominantLight
{
    public float3 lit;        // ambient + IBL + direct-spec blend
    public float3 domAmbient; // ambient × colorClamp (saved for debug viz)
    public float3 domSpec;    // direct + indirect spec (saved for debug viz)
    public uint   firstLight; // 1 if dominant was processed, else 0
};

// Apply the dominant-light + IBL combine. Inputs:
//   viewPos      — interpolated view-space pixel position (PS .xyz of TEXCOORD1)
//   worldNormal  — TBN-transformed splat normal in world space
//   viewDir      — normalize(-viewPos)
//   diffuseScaled— layer-weighted (1-α)·albedo·(1/π) diffuse tint base
//   F0           — α·albedo (Schlick base / metallic-equivalent for terrain)
//   roughness    — current ndfAlpha (smoothed roughness, .y of `r0`)
//   NdotV        — dot(worldNormal, viewDir)
//   shadowVis    — cascade visibility (1.0 when HAS_SHADOWS=false)
public TerrainDominantLight evaluateDominantLight(
    float3 viewPos, float3 worldNormal, float3 viewDir,
    float3 diffuseScaled, float3 F0, float roughness,
    float NdotV, float shadowVis)
{
    TerrainDominantLight result;
    result.domAmbient = float3(0, 0, 0);
    result.domSpec    = float3(0, 0, 0);

    // The dominant light is light 0 if it's directional (.w == 0).
    bool hasLight0 = asuint(psCB2.lightCount) > 0;
    bool light0IsDir = hasLight0 && (psCB2.lights[0].type == 0.0);

    if (!light0IsDir)
    {
        result.lit = float3(0, 0, 0);
        result.firstLight = 0;
        return result;
    }

    float3 lightDir = psCB2.lights[0].position;

    // Half-vector + dot products. NdotV is precomputed by caller.
    float3 H = viewDir + lightDir;
    H = H * rsqrt(dot(H, H));
    float VdotH = dot(H, viewDir);
    float NdotH = dot(worldNormal, H);
    float NdotL = dot(worldNormal, lightDir);

    // Schlick Fresnel (F = F0 + (1 - F0)·t⁵)
    float t  = 1.0 - VdotH;
    float t2 = t * t;
    float t5 = t2 * t2 * t;
    float3 F = (1.0 - F0) * t5 + F0;

    // GGX NDF using α² as the smoothed roughness.
    float a2 = roughness * roughness;
    float a4 = a2 * a2;
    float nh2 = NdotH * NdotH;
    float denom = nh2 * (a4 - 1.0) + 1.0;
    denom = denom * denom;
    float D = a4 / (PI * denom);

    // Schlick-GGX visibility.
    float a1  = 1.0 + roughness;
    float a12 = a1 * a1;
    float k         = a12 * SCHLICK_K_FACTOR;
    float oneMinusK = -a12 * SCHLICK_K_FACTOR + 1.0;
    float gl = NdotL * oneMinusK + k;
    float gv = NdotV * oneMinusK + k;
    float V  = saturate(1.0 / (4.0 * gl * gv));

    float3 spec = F * D * V;
    float clampedNdotL = max(0.0, NdotL);
    clampedNdotL = clampedNdotL * shadowVis;

    // IBL split-sum sample (when probes are bound).
    TerrainIBLSample ibl = sampleTerrainIBL(viewPos, worldNormal, roughness, NdotV);

    // Asymmetric ambient blend (Blizzard split: 0.675 upper / 0.325 lower).
    float3 lightDiffuse = psCB2.lights[0].diffuse;
    float3 lightAmbient = psCB2.lights[0].ambient;

    float maxLightDiff = max(lightDiffuse.y, lightDiffuse.z);
    maxLightDiff = max(lightDiffuse.x, maxLightDiff);
    float invLightDiffuse = 1.0 / (SAFE_EPS + maxLightDiff);
    float totalWeight = maxLightDiff + ibl.isActive;
    float ambientIBLWeight = totalWeight * ibl.isActive;
    ambientIBLWeight = psCB2.lights[0]._pad2.x * ambientIBLWeight;

    float maxIBLDiff = max(ibl.envDiffuse.y, ibl.envDiffuse.z);
    maxIBLDiff = max(ibl.envDiffuse.x, maxIBLDiff);
    float invIBLDiff = 1.0 / (SAFE_EPS + maxIBLDiff);
    float3 normIBL = ibl.envDiffuse * invIBLDiff;

    float3 hueDiff = lightDiffuse * invLightDiffuse - normIBL;
    float3 upper = hueDiff * 0.675000012 + normIBL;
    upper = upper * maxIBLDiff;
    upper = upper * totalWeight - lightDiffuse;
    upper = ibl.isActive * upper + lightDiffuse;
    float3 lower = hueDiff * 0.324999988 + normIBL;
    lower = lower * maxIBLDiff;
    float3 lowerScaled  = lower * ambientIBLWeight;
    float3 ambientCombo = -lower * ambientIBLWeight + upper;
    ambientCombo = clampedNdotL * ambientCombo + lowerScaled;
    ambientCombo = lightAmbient + ambientCombo;

    // Direct + indirect specular combine. Direct scales by the combined
    // weight so it matches the ambient strength; indirect reconstructs
    // a reflection from the split-sum LUT (F0 · scale + bias) modulated
    // by the IBL specular radiance.
    float3 directSpec = spec;                             // F · D · V
    directSpec = directSpec * totalWeight;
    directSpec = directSpec * clampedNdotL;
    float3 brdfTerm = F0 * ibl.brdfLut.x + ibl.brdfLut.yyy;
    brdfTerm = brdfTerm * ambientIBLWeight;
    brdfTerm = directSpec * lightDiffuse + brdfTerm;
    float3 specEnv = brdfTerm * ibl.envSpec;

    // Save the dominant-light component pieces — debug viz uses them
    // even on the !TINT path (the entry shader just throws them out).
    result.domAmbient = ambientCombo * diffuseScaled;
    result.domSpec    = specEnv;

    result.lit = result.domAmbient + result.domSpec;
    result.firstLight = 1;
    return result;
}

// Iterative loop over the remaining analytic lights. Mirrors HD's
// `lightLoop` (in ps_ibl.slang) — same Cook-Torrance form, no shadow
// masking, point-light attenuation via inverse-square + tiny radial bias.
public float3 evaluateAdditionalLights(
    float3 viewPos, float3 worldNormal, float3 viewDir,
    float3 dominantLit,
    float3 diffuseScaled, float3 F0, float roughness,
    float NdotV, uint startLight)
{
    // Pre-hoist GGX / Schlick-GGX constants — they don't depend on the
    // per-light parameters.
    float a2 = roughness * roughness;
    float a4 = a2 * a2;
    float a4m1 = a4 - 1.0;

    float a1  = 1.0 + roughness;
    float a12 = a1 * a1;
    float k         = a12 * SCHLICK_K_FACTOR;
    float oneMinusK = -a12 * SCHLICK_K_FACTOR + 1.0;
    float gvDenom = NdotV * oneMinusK + k;

    // (1 - F0) precompute for the in-loop Schlick fresnel.
    float3 oneMinusF0 = -F0 * 1.0 + 1.0;

    float3 accum = dominantLit;
    [loop] for (uint lightIdx = startLight;
                lightIdx < asuint(psCB2.lightCount);
                lightIdx++)
    {
        ShaderLight light = psCB2.lights[lightIdx];

        float3 lightDir;
        float  attenuation;
        if (light.type > 0.0)
        {
            float3 lightVec = light.position - viewPos;
            float  distSq = dot(lightVec, lightVec);
            float  dist = sqrt(distSq);
            attenuation = 1.0 / (distSq + POINT_EPS * dist);
            lightDir    = lightVec / dist;
        }
        else
        {
            lightDir    = light.position;
            attenuation = 1.0;
        }

        float3 H = viewDir + lightDir;
        H = H * rsqrt(dot(H, H));
        float VdotH = dot(H, viewDir);
        float NdotH = dot(worldNormal, H);
        float NdotL = dot(worldNormal, lightDir);

        float t  = 1.0 - VdotH;
        float t2 = t * t;
        float t5 = t2 * t2 * t;
        float3 F = oneMinusF0 * t5 + F0;

        float nh2 = NdotH * NdotH;
        float dn  = nh2 * a4m1 + 1.0;
        float D   = a4 / (PI * dn * dn);

        float gl = NdotL * oneMinusK + k;
        float V  = saturate(1.0 / (4.0 * gl * gvDenom));

        float clampedNdotL = max(0.0, NdotL);
        float contribution = attenuation * clampedNdotL;

        float3 spec  = F * D * V;
        float3 brdf  = spec + diffuseScaled;
        accum += light.diffuse * brdf * contribution;
        accum += light.ambient * diffuseScaled * attenuation;
    }
    return accum;
}
