// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026, Fernando Sahmkow
// See LICENSE in the repository root for full terms.
//
// Water pixel shader body + water-specific helpers.
//
// The water pass is structurally independent of the HD mesh path: it
// samples the previous-frame scene color and depth to produce both
// refracted (under-water) and reflected (sky / SSR) contributions, then
// composites them using a Fresnel-like mix.
//
// Helpers below are intentionally file-local (non-`public`) — they are
// water-specific and don't apply to other rendering paths:
//   • `compositeWaterNormal` — 4-sample animated normal-map blend
//   • `snellRefractTangent` — Snell's law → screen-space refraction 2D
//   • `waterPointAttenuation` — 1 / (1 + ε·d²) (water's own falloff)
//   • `ssrMarch` — 32-step screen-space reflection ray march
//
// Where the HD path already has a suitable helper (`decodeNormalMap`,
// `srgbToLinear`, `blizzardAmbientBlend`, the GGX/Schlick BRDF terms,
// the `IFogMode` policies, etc.) this file calls into it directly.

implementing wc3_shaders;

// ============================================================
// Named constants (water-specific; shared constants live in brdf.slang)
// ============================================================

// Normal-map animation: two octaves, bidirectional scroll.
static const float NORMAL_SCROLL_SMALL  = 0.00350000034;
static const float NORMAL_SCROLL_LARGE  = 0.0375000015;
static const float NORMAL_SCALE_SMALL   = 0.0140000004;
static const float NORMAL_SCALE_LARGE   = 0.150000006;

// Near-shore flattening — `flattenT = min(1, vertα² + 0.2)`.
static const float NORMAL_FLATTEN_BIAS  = 0.200000003;

// Snell's law for water (η = n_air / n_water ≈ 1 / 1.33).
static const float WATER_N_RATIO        = 0.751879692;
static const float WATER_N_RATIO_SQ     = 0.565323055;

// Screen-space refraction scale (tuned for 1080p frame) + shore ramps.
static const float2 REFRACT_OFFSET         = float2(0.00999999978, -0.00200000009);
static const float  REFRACT_DEPTH_SCALE_X  = 0.0142857144;   // ≈ 1/70
static const float  REFRACT_DEPTH_SCALE_Y  = 0.550000012;

// Screen-space reflection march parameters.
static const int    SSR_MAX_STEPS       = 32;
static const float  SSR_STEP_SIZE       = 16.0;
static const float  SSR_HIT_THICKNESS   = 50.0;
static const float3 SSR_MISS_COLOR      = float3(1, 0, 0);
static const float3 SSR_INITIAL_COLOR   = float3(0, 1, 0);

// PBR constants unique to the water body.
static const float  NDF_FALLBACK_ALPHA  = 0.167500004;
static const float  NDF_VARIANCE_BIAS   = 0.0280562509;  // ≈ NDF_FALLBACK_ALPHA²
static const float  WATER_F0_SCALE      = 0.27852115;

// SSR-hit blend — lerp(envRefl, ssrColor·dilute, mix).
static const float  SSR_HIT_DILUTE      = 0.875;
static const float  SSR_HIT_MIX         = 0.5;

// Gentle quadratic depth-remap: y = A·x² + (1−A)·x, y(0)=0, y(1)=1.
static const float  DEPTH_REMAP_CURVE_A = 0.25;

// Fresnel reflection-mix bias used at the final composite.
static const float  NDV_ALPHA_MIX_BIAS  = -0.100000001;

// Water-specific point-light attenuation: 1 / (1 + ε · d²).
// Softer than the HD inverse-square; gives the characteristic
// wide, shallow pool of light the engine's water shading expects.
static const float WATER_ATTEN_EPS     = 0.000500000024;

// ============================================================
// Water-specific helpers
// ============================================================

// 4-sample animated composite. Two octaves × bidirectional scroll, then
// average. The large octave contributes only XY so coarse waves never
// tilt the composite off the water plane (the Z reconstruction happens
// only for the small-octave samples, then the XY of the large ones is
// added on top afterward).
float3 compositeWaterNormal(float2 uv, float time)
{
    float4 scroll = frac(time * float4(
        NORMAL_SCROLL_SMALL, NORMAL_SCROLL_SMALL,
        NORMAL_SCROLL_LARGE, NORMAL_SCROLL_LARGE));

    float4 uvFwd = uv.xyxy * float4(
        NORMAL_SCALE_SMALL, NORMAL_SCALE_SMALL,
        NORMAL_SCALE_LARGE, NORMAL_SCALE_LARGE) + scroll;

    float4 uvShift = float4(-0.5, 0.5, -0.5, 0.5) + uv.xyxy;
    float4 uvBwd   = uvShift * float4(
        NORMAL_SCALE_SMALL, NORMAL_SCALE_SMALL,
        NORMAL_SCALE_LARGE, NORMAL_SCALE_LARGE) - scroll;

    float3 nSmallFwd = decodeNormalMap(t_waterNormal.Sample(s_waterNormal, uvFwd.xy).xyw);
    float3 nSmallBwd = decodeNormalMap(t_waterNormal.Sample(s_waterNormal, uvBwd.xy).xyw);
    float3 nLargeFwd = decodeNormalMap(t_waterNormal.Sample(s_waterNormal, uvFwd.zw).xyw);
    float3 nLargeBwd = decodeNormalMap(t_waterNormal.Sample(s_waterNormal, uvBwd.zw).xyw);

    float3 tsN;
    tsN    = 0.5 * (nSmallFwd + nSmallBwd);
    tsN.xy = tsN.xy + 0.5 * (nLargeFwd.xy + nLargeBwd.xy);
    return tsN;
}

// Flatten the composited TS normal toward (0,0,1) based on vertex alpha.
// Returns the normalised world-space normal.
float3 waterTsNormalToWorld(float3 tsNormalRaw, float vertAlpha,
                            float3 tangentWS, float3 normalWS, float tangentSign)
{
    // Center on (0,0,1), scale by flattenT, re-bias by (0,0,1).
    float  invLen  = rsqrt(dot(tsNormalRaw, tsNormalRaw));
    float  flatten = min(1.0, vertAlpha * vertAlpha + NORMAL_FLATTEN_BIAS);
    float3 centred = tsNormalRaw * invLen + float3(0, 0, -1);
    float3 tsN     = normalize(flatten * centred + float3(0, 0, 1));

    // TBN: tangent = upAxis, bitangent = cross(geomN, up) · up.w.
    return tangentToWorld(tsN, tangentWS, normalWS, tangentSign);
}

// Snell's law at a water / air interface. Returns the 2D tangent of the
// refracted direction (in world XY), zeroed when total internal
// reflection would occur. Biased by (worldPos.xy · invViewDist) to
// approximate the screen-space offset for the refraction sample.
//
//   cos²θt = 1 − η²·(1 − cos²θi)
//   tangent2D = η·V.xy − (η·NdotV + cosθt) · N.xy   (only when cos²θt ≥ 0)
float2 snellRefractTangent(float3 viewDirWS, float3 worldN,
                           float3 worldPos, float invViewDist)
{
    float NdotV      = dot(viewDirWS, worldN);
    float cosTheta2  = 1 - WATER_N_RATIO_SQ * (1 - NdotV * NdotV);
    float cosTheta   = sqrt(cosTheta2);
    float refrBias   = NdotV * WATER_N_RATIO + cosTheta;
    bool  tirPass    = (cosTheta2 >= 0);

    float2 tangent = viewDirWS.xy * WATER_N_RATIO - refrBias * worldN.xy;
    tangent = tirPass ? tangent : 0;
    tangent = tangent + worldPos.xy * invViewDist;
    return tangent;
}

// Water's soft point-light falloff: 1 / (1 + ε · d²).
// Intentionally NOT `pointLightAttenuation` (HD) — that helper uses a
// physical 1/(d² + ε·d) which produces much harsher close-range pools.
float waterPointAttenuation(float distSq)
{
    return 1.0 / (1.0 + WATER_ATTEN_EPS * distSq);
}

// ============================================================
// Screen-space reflection march.
// Walks along the reflected view ray in 16-unit world-space steps,
// re-projecting each sample back into screen space and testing it
// against the scene depth buffer. Terminates on hit (|Δz| < 50) or
// when the ray exits the viewport. Returns:
//   - color: the resolved reflection color (scene-sample on hit,
//            red on miss, green if the loop trivially exited).
//   - hit:   true iff a depth hit was recorded (used by the caller
//            to decide whether to mix SSR in with the env sky).
// ============================================================
struct SSRResult
{
    float3 color;
    bool   hit;
};

SSRResult ssrMarch(float3 worldPos, float3 reflectDirWS,
                   float4x4 worldToClip, float4 viewport)
{
    SSRResult result;
    result.color = SSR_INITIAL_COLOR;
    result.hit   = false;
    int escapeOrHit = 0;   // goes −1 on either termination

    // Runtime loop — no unroll. The `Sample` inside the loop would
    // force a gradient intrinsic otherwise; SampleLevel with explicit
    // mip 0 avoids it (and is correct for a depth buffer anyway).
    [loop] for (int step = 1; step < SSR_MAX_STEPS; ++step)
    {
        if (escapeOrHit != 0)
            continue;

        float3 samplePos = step * SSR_STEP_SIZE * reflectDirWS + worldPos;

        // Project to clip space — only .xyw are needed (Z is unused for
        // a depth-buffer lookup once we have the screen-space UV).
        float4 clip    = mul(worldToClip, float4(samplePos, 1));
        float3 clipXYW = clip.xyw;

        // NDC → UV (with the Y flip that matches the viewport origin).
        float2 ndc      = clipXYW.xy / clipXYW.z;
        float2 ndcUV    = ndc * 0.5 + 0.5;
        float2 sampleUV = viewport.xy * float2(ndcUV.x, 1 - ndcUV.y) + viewport.zw;

        float probedDepth = t_waterDepth.SampleLevel(s_waterDepth, sampleUV, 0).x;
        float zDelta      = clipXYW.z - probedDepth;
        bool  withinSlab  = (zDelta < SSR_HIT_THICKNESS);
        bool  inFront     = (zDelta > 0);

        if (inFront && withinSlab)
        {
            result.color = t_waterScene.SampleLevel(s_waterScene, sampleUV, 0).xyz;
            result.hit   = true;
            escapeOrHit  = -1;
        }
        else
        {
            // Escape once the ray leaves the viewport.
            float2 clamped = saturate(sampleUV);
            bool   oob     = any(sampleUV != clamped);
            if (oob)
            {
                result.color = SSR_MISS_COLOR;
                escapeOrHit  = -1;
            }
        }
    }
    return result;
}

// Rotate a world-space direction into the cubemap's view-space
// convention (with the engine's YZ swap + Z flip applied later by
// callers). `invView` — despite the name — is the world→view matrix the
// engine uploads; we only need its 3×3 rotation.
float3 rotateToProbeView(float3 v)
{
    return mul((float3x3)waterPsCB2.invView, v).xzy;
}

// ============================================================
// Base environment reflection (mip 0 dual-cubemap blend).
// Used as the "sky" color for the water reflection, irrespective of
// whether the analytic / IBL path contributes any light.
// ============================================================
float3 sampleWaterEnvRefl(float3 reflectDirProbeSpace)
{
    float4 coord = float4(reflectDirProbeSpace * float3(1, 1, -1), 1);
    float3 fromEnv = t_iblFrom.Sample(s_iblFrom, coord).xyz;
    float3 toEnv   = t_iblTo.Sample(s_iblTo,   coord).xyz;
    return waterPsCB2.envTransitionT * (toEnv - fromEnv) + fromEnv;
}

// ============================================================
// IBL sample block used inside the dominant-light branch.
// Same probe + BRDF-LUT triple as sampleIBL() for the HD path, but
// implemented inline so the dominant-light setup can share the
// already-computed probe-space directions + normal.
// ============================================================
struct WaterIBLSample
{
    float3 diffuseIrradiance;
    float3 specularRadiance;
    float2 brdfLUT;
    float  isActive;
};

WaterIBLSample sampleWaterIBL(
    float3 probeNormal, float3 probeReflect,
    float NdotV, float roughnessAA)
{
    WaterIBLSample s;
    float iblTest = waterPsCB2.envFromMipEnd * waterPsCB2.envToMipEnd;
    if (iblTest != 0)
    {
        // Diffuse irradiance: sample at the mirrored normal direction.
        float4 diffCoord = float4(probeNormal * float3(1, 1, -1), 0);
        float3 diffFrom  = t_iblFrom.Sample(s_iblFrom, diffCoord).xyz;
        float3 diffTo    = t_iblTo.Sample(s_iblTo,   diffCoord).xyz;
        s.diffuseIrradiance = waterPsCB2.envTransitionT * (diffTo - diffFrom) + diffFrom;

        // Specular radiance: roughness-biased mip at the reflect dir.
        float2 mipLevels  = float2(waterPsCB2.envFromMipEnd, waterPsCB2.envToMipEnd)
                          * roughnessAA;
        float4 specCoord  = float4(probeReflect * float3(1, 1, -1), 1);
        float3 specFrom   = t_iblFrom.SampleLevel(s_iblFrom, specCoord, mipLevels.x).xyz;
        float3 specTo     = t_iblTo.SampleLevel(s_iblTo,    specCoord, mipLevels.y).xyz;
        s.specularRadiance = waterPsCB2.envTransitionT * (specTo - specFrom) + specFrom;

        s.brdfLUT  = t_brdfLut.Sample(s_brdfLut, float2(NdotV, roughnessAA)).xy;
        s.isActive = 1.0;
    }
    else
    {
        s.diffuseIrradiance = 0;
        s.specularRadiance  = 0;
        s.brdfLUT           = 0;
        s.isActive          = 0;
    }
    return s;
}

// ============================================================
// Analytic light loop (water flavour).
// Same Cook-Torrance + per-light ambient structure as the HD path, but
// uses `waterPointAttenuation` and adds per-light-ambient that way too.
// Starts at `startLight`; the caller handles light 0 separately when
// it's directional.
// ============================================================
float3 waterLightLoop(
    float3 worldPos, float3 worldN, float3 viewDir, float invViewDist,
    float3 f0Schlick, float3 oneMinusF0, float3 f0Water,
    float roughnessAA, float NdotV,
    uint startLight)
{
    // Hoisted per-pixel constants (same as HD's lightLoop).
    float a2   = roughnessAA * roughnessAA;
    float a4   = a2 * a2;
    float a4m1 = a4 - 1.0;

    float a1        = 1.0 + roughnessAA;
    float a12       = a1 * a1;
    float k         = a12 * SCHLICK_K_FACTOR;
    float oneMinusK = 1.0 - k;
    float gvDenom   = NdotV * oneMinusK + k;

    float3 accum    = float3(0, 0, 0);
    uint   lightIdx = startLight;

    while (true)
    {
        if (lightIdx >= asuint(waterPsCB2.lightCount))
            break;

        ShaderLight light = waterPsCB2.lights[lightIdx];
        float3 Ldir;
        float  atten;

        if (light.type > 0)
        {
            // Point light: direction + water's own soft falloff.
            float3 toLight = light.position - worldPos;
            float  distSq  = dot(toLight, toLight);
            float  dist    = sqrt(distSq);
            atten = waterPointAttenuation(distSq);
            Ldir  = toLight / dist;
        }
        else
        {
            Ldir  = light.position;
            atten = 1.0;
        }

        // Cook-Torrance (inlined so intermediates can be shared).
        float3 H     = normalize(-worldPos * invViewDist + Ldir);
        float  VdotH = dot(H, viewDir);
        float  NdotH = dot(worldN, H);
        float  NdotL = dot(worldN, Ldir);

        // Schlick Fresnel.
        float  t   = 1.0 - VdotH;
        float  t2  = t * t;
        float  t5  = t2 * t2 * t;
        float3 F   = oneMinusF0 * t5 + f0Schlick;

        // GGX NDF.
        float  nh2 = NdotH * NdotH;
        float  den = nh2 * a4m1 + 1.0;
        float  D   = a4 / (PI * den * den);

        // Schlick-GGX visibility (reusing the pre-computed gvDenom).
        float  glDenom = NdotL * oneMinusK + k;
        float  V       = saturate(1.0 / (4.0 * glDenom * gvDenom));

        float  NdotLsat = max(0.0, NdotL);
        float  gate     = atten * NdotLsat;

        // Diffuse + specular modulated by the light's diffuse colour.
        float3 brdf = F * D * V + f0Water;
        accum += light.diffuse * brdf * gate;
        accum += light.ambient * f0Water * atten;

        ++lightIdx;
    }
    return accum;
}

// ============================================================
// Fog helper. Wraps `TFog.apply` so the water body can apply fog to
// three independent color layers with a single call — and exposes a
// `hasFog` predicate so the caller can compile out the pre-fog layer
// composition on the no-fog permutation.
// ============================================================
float3 applyWaterFog<TFog : IFogMode>(float3 color, float viewZ)
{
    return TFog.apply(color, waterPsCB2.fogColor.xyz, viewZ,
                      waterPsCB2.fogStart, waterPsCB2.fogEnd,
                      waterPsCB2.fogDensity);
}

// ============================================================
// Main water body — returns the final float4 SV_TARGET0 color.
// ============================================================
public float4 waterPSBody<TFog : IFogMode>(WaterPSInput input)
{
    // --- Screen-space UV for scene / depth samples ---
    uint2 targetSize;  uint mipCount;
    t_waterScene.GetDimensions(0, targetSize.x, targetSize.y, mipCount);
    float2 screenUV = input.svPosition.xy / (float2)(int2)targetSize;

    // --- Composited, flattened, world-space water surface normal ---
    float3 tsNormalRaw = compositeWaterNormal(input.uv, waterPsCB2.effectTime);
    float3 worldN = waterTsNormalToWorld(
        tsNormalRaw, input.vertColor.w,
        input.upAxisWS.xyz, input.geomNormalWS, input.upAxisWS.w);

    // --- View direction + depth-weighted Snell refraction ---
    float  invViewDist = rsqrt(dot(-input.worldPos, -input.worldPos));
    float3 viewDir     = -input.worldPos * invViewDist;

    float2 refractTangent = snellRefractTangent(
        viewDir, worldN, input.worldPos, invViewDist);

    // Compare depth at the centred vs pre-offset UV to derive a
    // refraction-softness weight + a shore-mix scalar used later.
    float  centerDepth = t_waterDepth.Sample(s_waterDepth, screenUV).x;
    float2 probeUV     = refractTangent * REFRACT_OFFSET + screenUV;
    float  offsetDepth = t_waterDepth.Sample(s_waterDepth, probeUV).x;
    float  depthDelta  = max(offsetDepth, centerDepth) - input.worldPos.z;
    float  refractW    = saturate(depthDelta * REFRACT_DEPTH_SCALE_X);
    float  shoreDepthScalar = depthDelta * REFRACT_DEPTH_SCALE_Y;

    float2 refractedUV    = refractTangent * REFRACT_OFFSET * refractW + screenUV;
    float3 refractedScene = t_waterScene.Sample(s_waterScene, refractedUV).xyz;

    // --- Reflected view direction (world space) ---
    float3 reflectDirWS = normalize(reflect(-viewDir, worldN));

    // --- SSR march + probe-space env reflection ---
    SSRResult ssr = ssrMarch(input.worldPos, reflectDirWS,
                             waterPsCB2.worldToClip, waterPsCB2.viewport);

    float3 probePosDir = normalize(rotateToProbeView(input.worldPos));
    float3 probeNormal = normalize(rotateToProbeView(worldN));
    float3 probeReflect = normalize(
        probePosDir - probeNormal * (2 * dot(probePosDir, probeNormal)));
    float3 envReflColor = sampleWaterEnvRefl(probeReflect);

    // --- Emissive / lightmap probe (single-value average) ---
    float3 emissiveProbe = t_waterProbe.SampleLevel(s_waterProbe, float2(0, 0), 16).xyz;

    // --- Roughness (specular-AA) + F0 base terms ---
    float ndfAlpha = waterPsCB2.useNdf != 0
        ? specularAA(sqrt(NDF_VARIANCE_BIAS), worldN)  // sqrt(bias) → base α
        : NDF_FALLBACK_ALPHA;
    // Note: `specularAA` adds σ² to roughness², so seeding with sqrt
    // (NDF_VARIANCE_BIAS) = fallback α reproduces the original formula
    // exactly (α = sqrt(fallback² + σ²), clamped to 1 by the helper).

    float3 f0Schlick  = SCHLICK_K_FACTOR  * emissiveProbe;  // Fresnel additive base
    float3 f0Water    = WATER_F0_SCALE    * emissiveProbe;  // final BRDF multiplier
    float3 oneMinusF0 = 1.0 - SCHLICK_K_FACTOR * emissiveProbe;

    // --- Dominant-light Cook-Torrance + IBL split-sum (only if light 0
    //     exists and is directional — matching HD's pattern) ---
    float  NdotV_raw        = dot(viewDir, worldN);
    float3 dominantLitColor = float3(0, 0, 0);
    uint   firstLightIdx    = 0;

    bool hasLights     = asuint(waterPsCB2.lightCount) > 0;
    bool dominantIsDir = waterPsCB2.lights[0].type == 0.0;
    if (hasLights && dominantIsDir)
    {
        ShaderLight light0 = waterPsCB2.lights[0];

        float3 H     = normalize(-input.worldPos * invViewDist + light0.position);
        float  VdotH = dot(H, viewDir);
        float  NdotH = dot(worldN, H);
        float  NdotL = dot(worldN, light0.position);

        float3 F = schlickFresnel(f0Schlick, VdotH);
        float  D = ggxNDF(NdotH, ndfAlpha);
        float  V = schlickGGXVisibility(NdotL, NdotV_raw, ndfAlpha);
        float3 spec     = F * D * V;
        float  NdotLsat = max(0.0, NdotL);

        WaterIBLSample ibl = sampleWaterIBL(
            probeNormal, probeReflect, NdotV_raw, ndfAlpha);

        float maxDiffuse    = max(max(light0.diffuse.x, light0.diffuse.y), light0.diffuse.z);
        float ambientWeight = light0._pad2.x;

        // Blizzard ambient blend (+ per-light ambient term).
        float3 ambient = blizzardAmbientBlend(
            light0.diffuse, ibl.diffuseIrradiance,
            NdotLsat, maxDiffuse, ibl.isActive, ambientWeight);
        ambient += light0.ambient;

        // Direct + indirect specular composition.
        float totalWeight      = maxDiffuse + ibl.isActive;
        float3 scaledSpec      = spec * totalWeight * NdotLsat;
        float3 indirectSpec    = f0Schlick * ibl.brdfLUT.x + ibl.brdfLUT.y;
        float  ambientIBLWeight = totalWeight * ibl.isActive * ambientWeight;
        indirectSpec          *= ambientIBLWeight;
        indirectSpec          += scaledSpec * light0.diffuse;
        indirectSpec          *= ibl.specularRadiance;

        dominantLitColor = ambient * f0Water + indirectSpec;
        firstLightIdx    = 1;
    }

    // --- Analytic light loop for everything after light 0 ---
    float3 lightSum = dominantLitColor + waterLightLoop(
        input.worldPos, worldN, viewDir, invViewDist,
        f0Schlick, oneMinusF0, f0Water,
        ndfAlpha, NdotV_raw,
        firstLightIdx);

    // Un-divide by π (the per-light Lambertian already baked in 1/π).
    float3 lightSumScaled = PI * lightSum;
    if (waterPsCB2.lightCount == 0)
        lightSumScaled = float3(1, 1, 1);

    // --- Vertex-colour sRGB decode + three tint-clamp terms ---
    float3 vertColorLinear = srgbToLinear(input.vertColor.xyz);
    float3 colorClampVL    = min(vertColorLinear, lightSumScaled);
    float3 sceneWithEmiss  = emissiveProbe * refractedScene;
    float3 ambientClamped  = min(min(waterPsCB2.ambientClamp, emissiveProbe),
                                  vertColorLinear);

    // Depth-based shore ramps.
    float depthMixRefr = saturate(shoreDepthScalar / waterPsCB2.refractSoftness);
    float depthMixLit  = saturate(shoreDepthScalar / waterPsCB2.litSoftness);
    float depthRemap   = depthMixLit * (DEPTH_REMAP_CURVE_A * depthMixLit
                                         + (1 - DEPTH_REMAP_CURVE_A));

    // Fresnel-like alpha mix: lerp(1, saturate(NdotV − 0.1), vertα).
    // At grazing + deep water (vertα ≈ 1) → reflection dominates.
    float fresnelAlphaT = saturate(NdotV_raw + NDV_ALPHA_MIX_BIAS);
    float alphaMixT     = input.vertColor.w * (fresnelAlphaT - 1) + 1;

    // --- Compose the three "through-water" colour layers, then
    //     mix with the reflection layer.
    //     With fog: each layer is lerped toward fogColor first.
    //     Without fog: the lerps collapse into direct products. ---
    float3 envReflTint;
    float3 sceneEmissTint;
    float3 litTint;
    if (TFog.hasFog)
    {
        envReflTint    = applyWaterFog<TFog>(envReflColor   * colorClampVL,     input.worldPos.z);
        sceneEmissTint = applyWaterFog<TFog>(sceneWithEmiss * colorClampVL,     input.worldPos.z);
        litTint        = applyWaterFog<TFog>(ambientClamped * lightSumScaled,   input.worldPos.z);
    }
    else
    {
        envReflTint    = envReflColor   * colorClampVL;
        sceneEmissTint = sceneWithEmiss * colorClampVL;
        litTint        = ambientClamped * lightSumScaled;
    }

    // Reflection layer: lerp(env, ssr·dilute, 0.5) when SSR hit, else env.
    float3 ssrMixed = (ssr.color * SSR_HIT_DILUTE - envReflTint) * SSR_HIT_MIX
                    + envReflTint;
    float3 reflectionLay = ssr.hit ? ssrMixed : envReflTint;

    // Underwater composition: refracted → +emissive → +lit.
    float3 layer0 = refractedScene;
    float3 layer1 = layer0 + depthMixRefr * (sceneEmissTint - layer0);
    float3 layer2 = layer1 + depthRemap   * (litTint        - layer1);

    // Final mix between reflection and underwater-lit.
    float3 finalColor = reflectionLay + alphaMixT * (layer2 - reflectionLay);
    return float4(finalColor, 1);
}
