// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026, Fernando Sahmkow
// See LICENSE in the repository root for full terms.
//
// PopcornFX pixel-shader body + popcorn-specific helpers.
//
// The body is split into two top-level entry helpers:
//
//   • `popcornMotionPassBody`  — encodes 2D screen-space motion
//                                vectors into RT0.xy. Used by the
//                                MOTION_PASS specialization; depth-
//                                discards behind the opaque scene
//                                and (with VC) tints the motion by
//                                vertex-colour .xy + forwards .w
//                                into the output alpha.
//   • `popcornColorPassBody`   — full lit / diffuse / fog / alpha-LUT
//                                / soft-particles surface shading
//                                returned as an `(rgba, gbufferAlpha)`
//                                pair so the entry can route the
//                                gbuffer-alpha to RT2 separately.
//
// Helpers that aren't reused outside the popcorn pipeline are kept
// file-local (non-`public`). Helpers that map cleanly onto an
// existing wc3_shaders building block (`decodeNormalMap`,
// `tangentToWorld`, `orientNormal`, `ggxNDF`, `schlickGGXVisibility`,
// `specularAA`, the `IFogMode` policies) are called through directly
// rather than re-implemented.

implementing wc3_shaders;

// ============================================================
// Named constants
// ============================================================

// Motion-vector encoding offsets. The engine packs each motion
// component as `(motion + 1.00391996) / 2`, so unpacking is
// `motion = sample · 2 − 1.00391996` (note the asymmetric
// −1.00391996 / +0.99608, matching an 8-bit signed roundtrip).
static const float MOTION_UNPACK_BIAS  = -1.00391996;

// Motion-vector magnitude scale: the source LUT stores motion in
// "world units per second at one metre depth", and the runtime
// rescales by sqrt(1000 / depth) so the projected screen-space
// magnitude stays roughly constant.
static const float MOTION_DEPTH_NUMER  = 1000.0;

// Default per-particle deferred material payload — neutral diffuse
// tint + 0.5 roughness/metallic-like values matching the shipped
// engine's deferred resolve assumptions.
static const float3 GBUFFER_DEFAULT_MAT = float3(1.0, 0.5, 0.5);

// "Tonal mix" weight used to bias the IBL diffuse irradiance toward
// the light0 base material colour — chosen to roughly match the
// retail engine's environment-tinting behaviour.
static const float IBL_TONAL_MIX_T      = 0.325;

// ============================================================
// Per-mode UV resolution
// ============================================================

// Resolves the (single) sample UV for the per-mode diffuse / motion
// fetch, plus the second atlas UV when applicable. The two outputs
// pack a struct so a single helper covers all three live modes
// (basic / billboard / atlas) and the entry doesn't have to know the
// per-mode UV decoding rules.
public struct PopcornUV
{
    public float2 sampleUV;     // primary sample point
    public float2 sampleUV2;    // secondary (ATLAS only — second frame)
};

// Each particle is rendered as two triangles sharing the same atlas
// cell, with the quad UV (0,0)/(0,1)/(1,0)/(1,1) coming from the VS's
// SV_VertexID derivation. The two triangles map mirrored portions of
// the atlas cell, so we pick the half by `uv.x + uv.y < 1` and
// remap into the per-cell scale/bias from `slot1` (TEXCOORD1).
float2 popcornBillboardUV(float2 quadUV, float4 uvScaleBias, float4 uvAxisScales)
{
    float2 mirrored = (1.0 - quadUV) / uvAxisScales.zw;
    mirrored        = 1.0 - mirrored;
    float2 direct   = quadUV / uvAxisScales.xy;
    bool   firstTri = (quadUV.x + quadUV.y) < 1.0;
    float2 sampleUV = firstTri ? direct : mirrored;
    return sampleUV * uvScaleBias.xy + uvScaleBias.zw;
}

public PopcornUV popcornResolveUV<TMode : IPopcornMode>(PopcornPSInput input)
{
    PopcornUV uv;
    uv.sampleUV  = float2(0, 0);
    uv.sampleUV2 = float2(0, 0);
    if (TMode.isBillboard)
    {
        uv.sampleUV = popcornBillboardUV(input.uv.xy, input.slot1, input.uvAxis);
    }
    else if (TMode.isAtlas)
    {
        uv.sampleUV  = input.uv.xy;
        uv.sampleUV2 = input.uv.zw;
    }
    else if (TMode.hasUV)
    {
        // Basic UV mode: straight pass-through of TEXCOORD0.xy.
        uv.sampleUV = input.uv.xy;
    }
    return uv;
}

// ============================================================
// Albedo + diffuse texture
// ============================================================

// Sample the diffuse texture at the resolved UV, lerping between two
// atlas frames when in ATLAS mode. Returns white when there's no UV
// stream (the popcorn pipeline encodes "no diffuse" by simply not
// binding a UV).
public float4 popcornSampleDiffuse<TMode : IPopcornMode>(PopcornUV uv)
{
    if (TMode.isAtlas)
    {
        // Atlas mode: lerp between current and next frame by the
        // life-ratio fractional part stored at TEXCOORD1.x by the VS.
        // We re-fetch the blend factor here so the entry doesn't have
        // to thread it through; the same lookup is dirt-cheap.
        float4 a = t_popcornDiffuse.Sample(s_popcornDiffuse, uv.sampleUV);
        float4 b = t_popcornDiffuse.Sample(s_popcornDiffuse, uv.sampleUV2);
        return a;  // overwritten by caller; see popcornColorPassBody
        // (atlas blend lives there because it needs the slot1 input).
    }
    if (TMode.hasUV)
        return t_popcornDiffuse.Sample(s_popcornDiffuse, uv.sampleUV);
    return float4(1, 1, 1, 1);
}

// ============================================================
// Soft-particles depth fade
// ============================================================

// Reproject worldPos through `worldToClip`, normalise to NDC, sample
// the scene depth at the matching screen-space UV, and return
// `(sceneZ - worldPos.z)` — the caller folds this into a saturated
// fade scalar, or in MOTION_PASS uses it both for the discard test
// and as the multiplier for the motion magnitude.
public float popcornSceneDepthDelta(float3 worldPos)
{
    float3 clip = worldPos.x * popcornPsCB2.worldToClip[0].xyw
                + worldPos.y * popcornPsCB2.worldToClip[1].xyw
                + worldPos.z * popcornPsCB2.worldToClip[2].xyw
                +              popcornPsCB2.worldToClip[3].xyw;
    float2 ndc      = clip.xy / clip.z;
    ndc             = ndc * 0.5 + 0.5;
    float2 depthUV  = float2(ndc.x, 1.0 - ndc.y);
    depthUV         = popcornPsCB2.depthUVRemap.xy * depthUV
                    + popcornPsCB2.depthUVRemap.zw;
    float  sceneZ   = t_popcornDepth.Sample(s_popcornDepth, depthUV).x;
    return sceneZ - worldPos.z;
}

// ============================================================
// LIT path
// ============================================================

// Per-particle world-space basis transform. The IBL block rotates
// both the world position (used as a "view direction proxy" through
// the per-particle frame) and the world-space normal into the
// cubemap-local frame stored at cb2[7..9]. The `.xzy` swizzle on
// each row reproduces the engine's frame layout (rows are stored
// XZY rather than XYZ).
float3 popcornRotateToParticleFrame(float3 v)
{
    return v.x * popcornPsCB2.particleBasis[0].xzy
         + v.y * popcornPsCB2.particleBasis[1].xzy
         + v.z * popcornPsCB2.particleBasis[2].xzy;
}

// Reflect a normalised "view-from-particle" direction through the
// particle-local normal — used to look up the pre-filtered specular
// cubemap.
float3 popcornReflectAcross(float3 viewVec, float3 normalVec)
{
    float vDotN = dot(viewVec, normalVec);
    return normalize(viewVec - 2.0 * vDotN * normalVec);
}

// Karis' split-sum environment-BRDF integration result, expressed
// per-channel because F0 is per-channel (F0 = metallic · albedo).
float3 popcornSplitSumBRDF(float3 F0, float2 brdfLut)
{
    return F0 * brdfLut.x + brdfLut.y;
}

// Pre-filtered IBL contributions sampled from the two cubemaps and
// blended by `iblBlend` (cb2[19].z). The two cubes use independent
// roughness-to-mip multipliers (cb2[19].xy) so the engine can swap
// between probes with different mip pyramids.
struct PopcornIBLSample
{
    float3 diffuseCube;     // mip-0 cubemap blend (used as IBL diffuse irradiance)
    float3 specCube;        // pre-filtered cubemap blend (specular radiance)
    float2 brdfLUT;         // (NdotV, roughness) → split-sum LUT
};

PopcornIBLSample popcornSampleIBL(float3 worldNormal, float3 worldPos,
                                  float NdotV, float roughness)
{
    PopcornIBLSample out_;

    float3 viewRot = normalize(popcornRotateToParticleFrame(worldPos));
    float3 nRot    = normalize(popcornRotateToParticleFrame(worldNormal));
    float3 reflRot = popcornReflectAcross(viewRot, nRot);

    // Diffuse irradiance from mip 0. The Z negation matches the
    // engine's cubemap orientation (left-handed sample direction).
    float4 diffDir = float4(nRot.x, nRot.y, -nRot.z, 0);
    float3 d13 = t_popcornCubeFrom.Sample(s_popcornCubeFrom, diffDir).xyz;
    float3 d14 = t_popcornCubeTo  .Sample(s_popcornCubeTo,   diffDir).xyz;
    out_.diffuseCube = lerp(d13, d14, popcornPsCB2.iblBlend);

    // Pre-filtered specular cubemap; each cube has its own mip range
    // so the call uses SampleLevel with per-cube roughness scales.
    float4 specDir = float4(reflRot.x, reflRot.y, -reflRot.z, 1);
    float  mipFrom = popcornPsCB2.iblMipFrom * roughness;
    float  mipTo   = popcornPsCB2.iblMipTo   * roughness;
    float3 s13 = t_popcornCubeFrom.SampleLevel(s_popcornCubeFrom, specDir, mipFrom).xyz;
    float3 s14 = t_popcornCubeTo  .SampleLevel(s_popcornCubeTo,   specDir, mipTo  ).xyz;
    out_.specCube = lerp(s13, s14, popcornPsCB2.iblBlend);

    // Karis split-sum BRDF LUT, indexed by (NdotV, roughness).
    out_.brdfLUT = t_popcornBrdfLut.Sample(s_popcornBrdfLut,
                                            float2(NdotV, roughness)).xy;
    return out_;
}

// IBL "tonal mix" — biases the per-pixel diffuse cubemap colour
// toward the light-0 base material colour by a fixed
// `IBL_TONAL_MIX_T`, then composes the diffuse and specular
// contributions into a single linear-light tint. When `iblOn` is
// false the iblScale collapses to zero so only the ambient flows
// through (matches the original engine's behaviour where a
// useIbl=false branch still emits an ambient-only contribution).
float3 popcornComposeIBLTint(
    PopcornIBLSample ibl, float3 F0, float3 diffTerm, float iblOn)
{
    float3 baseMat = popcornPsCB2.lights[0].diffuse;     // cb2[22].xyz
    float3 ambient = popcornPsCB2.lights[0].ambient;     // cb2[21].xyz
    float  matMax    = max(baseMat.x, max(baseMat.y, baseMat.z));
    float  invMatMax = 1.0 / (SAFE_EPS + matMax);
    float  iblWeight = matMax + iblOn;
    float  iblScale  = iblWeight * iblOn * popcornPsCB2.lights[0]._pad2.x;  // cb2[24].x

    float3 diffCube  = ibl.diffuseCube;
    float  diffMax   = max(diffCube.x, max(diffCube.y, diffCube.z));
    float  invDiffMax = 1.0 / (SAFE_EPS + diffMax);
    diffCube        *= invDiffMax;
    float3 envTint   = baseMat * invMatMax - diffCube;
    diffCube         = envTint * IBL_TONAL_MIX_T + diffCube;
    diffCube         = diffCube * iblWeight;
    diffCube         = diffCube * iblScale + ambient;

    float3 F            = popcornSplitSumBRDF(F0, ibl.brdfLUT);
    float3 specCubeTerm = (F * iblScale) * ibl.specCube;
    return diffCube * diffTerm + specCubeTerm;
}

// Per-light Cook-Torrance + ambient contribution. The popcorn engine
// uses a split-colour model where `ambient` (cb2[i*4+21]) is an
// always-on diffuse tint (no NoL gate) and `diffuse` (cb2[i*4+22])
// is the directional / spec colour gated by NoL.
struct PopcornLightContrib
{
    float3 spec;       // per-pixel light_spec · (D·F·vis + diffTerm) · NoL · atten
    float3 diff;       // per-pixel light_amb · diffTerm · atten
};

PopcornLightContrib popcornEvalLight(
    ShaderLight light, float3 worldPos, float3 worldN, float3 viewDir,
    float3 F0, float3 oneMinusF0, float3 diffTerm, float roughness, float NdotV)
{
    float3 lightVec;
    float  lightAtten;
    bool   isPunctual = (light.type > 0.0);
    if (isPunctual)
    {
        lightVec = light.position - worldPos;
        float d2 = dot(lightVec, lightVec);
        float d  = sqrt(d2);
        lightAtten = pointLightAttenuation(d2, d);
        lightVec  /= d;
    }
    else
    {
        lightVec   = light.position;
        lightAtten = 1.0;
    }

    float3 halfVec = normalize(lightVec + viewDir);
    float  vDotH   = dot(viewDir, halfVec);
    float  nDotH   = dot(worldN, halfVec);
    float  nDotL   = dot(worldN, lightVec);

    float3 F        = schlickFresnel(F0, vDotH);
    float  D        = ggxNDF(nDotH, roughness);
    float  vis      = schlickGGXVisibility(nDotL, NdotV, roughness);
    float  energy   = max(0.0, nDotL) * lightAtten;

    PopcornLightContrib c;
    c.spec = (D * F * vis + diffTerm) * light.diffuse * energy;
    c.diff = light.ambient * diffTerm * lightAtten;
    return c;
}

// Whole-LIT-pipeline body — normal map decode, tangent-space basis
// reconstruction, specular-AA roughness, IBL block, analytic-light
// loop. Returns the linear-RGB lit colour (the entry pairs it with
// the pre-LIT alpha to form the final output).
public float3 popcornLitBody(PopcornPSInput input, PopcornUV uv,
                             bool isFrontFace, float3 albedoRGB)
{
    // Tangent-space normal from the packed map. With no UV stream
    // the maps are still bound and sampled at (0, 0) — effectively
    // a one-texel "constant material" mode.
    float2 litMapUV     = uv.sampleUV;
    float3 normalSample = t_popcornNormal   .Sample(s_popcornNormal,    litMapUV).xyw;
    float2 smoothMet    = t_popcornSmoothMet.Sample(s_popcornSmoothMet, litMapUV).yz;

    float3 tsNormal  = decodeNormalMap(normalSample);
    float3 worldN    = tangentToWorld(tsNormal, input.tangentWS.xyz,
                                      input.normalWS, input.tangentWS.w);
    worldN           = orientNormal(worldN, isFrontFace);

    // Specular-AA: widen the BRDF on aliased high-frequency normals.
    // cb2[20].w toggles between the smoothed value and the raw map
    // sample (the engine flips this off on hardware where the
    // ddx/ddy variance estimate is too noisy).
    float roughnessSmoothed = specularAA(smoothMet.x, worldN);
    float roughness         = popcornPsCB2.useSmoothedNormal != 0.0
                                ? roughnessSmoothed : smoothMet.x;

    // Key BRDF terms.
    float3 viewDir   = normalize(-input.worldPos);
    float  metallic  = smoothMet.y;
    float3 F0        = metallic * albedoRGB;
    float3 oneMinusF0 = 1.0 - F0;
    float3 diffTerm  = (1.0 - metallic) * albedoRGB * INV_PI;
    float  NdotV     = dot(worldN, viewDir);

    // IBL block — gated on (lightCount > 0) AND light0 is directional
    // (light0 is the special "directional sun + IBL driver" slot).
    bool  lightsOn = (asuint(popcornPsCB2.lightCount) > 0u)
                   && (popcornPsCB2.lights[0].type == 0.0);
    float3 iblTinted = float3(0, 0, 0);
    if (lightsOn)
    {
        bool useIbl = (popcornPsCB2.iblMipFrom * popcornPsCB2.iblMipTo) != 0.0;
        PopcornIBLSample ibl;
        float iblOn;
        if (useIbl)
        {
            ibl   = popcornSampleIBL(worldN, input.worldPos, NdotV, roughness);
            iblOn = 1.0;
        }
        else
        {
            ibl.diffuseCube = float3(0, 0, 0);
            ibl.specCube    = float3(0, 0, 0);
            ibl.brdfLUT     = float2(0, 0);
            iblOn           = 0.0;
        }
        iblTinted = popcornComposeIBLTint(ibl, F0, diffTerm, iblOn);
    }

    // Analytic-light loop. When lightsOn is true, light 0 has been
    // consumed by the IBL block above so the loop starts at index 1.
    // Without IBL, the loop processes every light from 0.
    float3 lightAccum = iblTinted;
    uint   lightCount = asuint(popcornPsCB2.lightCount);
    uint   startLight = lightsOn ? 1u : 0u;
    [loop]
    for (uint i = startLight; i < lightCount; ++i)
    {
        PopcornLightContrib c = popcornEvalLight(
            popcornPsCB2.lights[i], input.worldPos, worldN, viewDir,
            F0, oneMinusF0, diffTerm, roughness, NdotV);
        lightAccum += c.spec + c.diff;
    }

    // Final scale: π undoes the `INV_PI` baked into the diffuse
    // term so per-light contributions land at the right magnitude.
    return PI * lightAccum;
}

// ============================================================
// Color pass body
// ============================================================

// Composite return value so the entry can route the gbuffer alpha
// (= post-VC, post-LUT, pre-SP-fade alpha) to RT2 separately from
// the SP-faded RT0 alpha.
public struct PopcornColorOut
{
    public float4 color;            // RT0 — final RGBA
    public float  gbufferAlpha;     // RT2.w — pre-SP / pre-fog alpha
};

public PopcornColorOut popcornColorPassBody<
    TMode  : IPopcornMode,
    TFog   : IFogMode,
    let HAS_VC        : bool,
    let HAS_ALPHA_LUT : bool,
    let HAS_SP        : bool,
    let HAS_LIT       : bool>(PopcornPSInput input, bool isFrontFace)
{
    // ---- Resolve UV + sample diffuse ----
    PopcornUV uv = popcornResolveUV<TMode>(input);

    float4 albedo;
    if (TMode.isAtlas)
    {
        // Atlas: lerp between current and next frame by the life-
        // ratio fractional part the VS stored at slot1.x.
        float4 a = t_popcornDiffuse.Sample(s_popcornDiffuse, uv.sampleUV);
        float4 b = t_popcornDiffuse.Sample(s_popcornDiffuse, uv.sampleUV2);
        albedo = lerp(a, b, input.slot1.x);
    }
    else if (TMode.hasUV)
    {
        albedo = t_popcornDiffuse.Sample(s_popcornDiffuse, uv.sampleUV);
    }
    else
    {
        albedo = float4(1, 1, 1, 1);
    }

    // ---- Vertex colour modulation ----
    if (HAS_VC)
        albedo *= input.vertColor;

    // ---- Alpha LUT remap ----
    // The LUT input is the post-VC alpha; the result replaces alpha
    // and feeds every downstream consumer (LIT alpha source, SP
    // fade, gbuffer write, fog).
    if (HAS_ALPHA_LUT)
    {
        albedo.w = t_popcornAlphaLut.Sample(
            s_popcornAlphaLut, float2(albedo.w, input.random.x)).x;
    }

    // ---- Lighting ----
    float3 colorRGB;
    if (HAS_LIT)
        colorRGB = popcornLitBody(input, uv, isFrontFace, albedo.xyz);
    else
        colorRGB = albedo.xyz;

    float colorAlpha = albedo.w;

    // ---- Capture pre-SP alpha for the gbuffer write ----
    float gbufferAlpha = colorAlpha;

    // ---- Soft-particles fade (multiplies BOTH RGB and alpha,
    //      matching the engine's premultiplied-alpha pattern). ----
    if (HAS_SP)
    {
        float fade = saturate(popcornPsCB2.depthFadeScale
                              * popcornSceneDepthDelta(input.worldPos));
        colorRGB   *= fade;
        colorAlpha *= fade;
    }

    // ---- Distance fog (RGB only — alpha stays untouched) ----
    if (TFog.hasFog)
    {
        colorRGB = TFog.apply(colorRGB, popcornPsCB2.fogColor.xyz,
                              input.worldPos.z,
                              popcornPsCB2.fogStart, popcornPsCB2.fogEnd,
                              popcornPsCB2.fogDensity);
    }

    PopcornColorOut out_;
    out_.color        = float4(colorRGB, colorAlpha);
    out_.gbufferAlpha = gbufferAlpha;
    return out_;
}

// ============================================================
// Motion-vector pass body
// ============================================================

// In MOTION_PASS the same UV resolution drives the t0 sample, but
// the .xy lanes of the texel are interpreted as a packed motion
// vector (2 × 8-bit unsigned, biased by ~−1.0039). The ALPHA_LUT,
// FOG, GBUFFER and LIT bits are dead in this pass — the engine
// keeps them on only so the input signature stays aligned across
// pass switches. SP and VC are the only flags with a functional
// effect here.
public float4 popcornMotionPassBody<
    TMode  : IPopcornMode,
    let HAS_VC : bool,
    let HAS_SP : bool>(PopcornPSInput input)
{
    PopcornUV uv = popcornResolveUV<TMode>(input);

    // Depth re-projection + scene-depth fetch. Pre-compute the soft
    // fade scalar BEFORE the discard so SP perms can still scale by
    // it (the discard saturates depthDelta < 0 to 0 anyway).
    float depthDelta = popcornSceneDepthDelta(input.worldPos);
    float softFade   = HAS_SP
                     ? saturate(popcornPsCB2.depthFadeScale * depthDelta)
                     : 1.0;
    if (depthDelta < 0.0) discard;

    // Motion source. Atlas mode lerps two t0 samples; basic /
    // billboard just sample the resolved UV. Only .xy is used.
    float2 motSrc;
    if (TMode.isAtlas)
    {
        float2 a = t_popcornDiffuse.Sample(s_popcornDiffuse, uv.sampleUV ).xy;
        float2 b = t_popcornDiffuse.Sample(s_popcornDiffuse, uv.sampleUV2).xy;
        motSrc = lerp(a, b, input.slot1.x);
    }
    else
    {
        motSrc = t_popcornDiffuse.Sample(s_popcornDiffuse, uv.sampleUV).xy;
    }

    // Unpack [0, 1] → ~[−1, +1], scale by sqrt(1000 / depth) so
    // screen-space magnitude stays roughly stable as particles
    // recede from the camera.
    float2 motion = motSrc * 2.0 + MOTION_UNPACK_BIAS;
    motion       *= sqrt(MOTION_DEPTH_NUMER / input.worldPos.z);

    // VERTEX_COLOR is the only "color" feature that meaningfully
    // affects MOTION_PASS — it tints the motion vectors by vc.xy
    // and forwards vc.w into the output alpha. Without VC the
    // alpha is the constant 1.
    float motionAlpha = 1.0;
    if (HAS_VC)
    {
        motion      *= input.vertColor.xy;
        motionAlpha  = input.vertColor.w;
    }

    if (HAS_SP) motion *= softFade;

    // Re-bias to [0, 1] for the RT format. .z is forced to zero
    // (the engine reserves it as a "no special encoding" flag).
    float4 outColor;
    outColor.xy = motion * 0.5 + 0.5;
    outColor.z  = 0.0;
    outColor.w  = motionAlpha;
    return outColor;
}
