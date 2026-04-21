// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026, Fernando Sahmkow
// See LICENSE in the repository root for full terms.
//
// Toon (Breath-of-the-Wild style) pixel shader bodies. Parallels the
// three HD bodies in ps_ibl.slang / ps_standard.slang so the toon entry
// point can dispatch with the same (HAS_IBL, TMat.hasMultiLayer) logic:
//
//   psToonIBLBody          — full lit path with shadow cascades + IBL.
//   psToonMultiLayerBody   — cheap non-IBL path + team-color blend.
//   psToonStandardBody     — cheapest non-IBL path.
//
// The shading model differs from HD in three places:
//
//   1. Diffuse is stepped into two hard bands by `toonDiffuseBand` —
//      shadow side is darker flat color, lit side is the full light
//      contribution. Shadow cascades attenuate NdotL before the step so
//      cast shadows produce the same clean silhouette.
//   2. Specular is a Blinn-Phong highlight run through `step`, giving a
//      flat disc instead of a GGX gradient.
//   3. A `toonRim` fresnel term is added at silhouettes to pop the shape
//      against the background (BOTW-style edge glow).
//
// Everything downstream of lighting (fresnel overlay, cloak, emissive,
// per-vertex tint, color matrix, fog, MRT packing, debug visualisation)
// is shared with the HD path unchanged — these effects are orthogonal to
// the shading model and the engine routes them through the same state.

implementing custom_shaders;

// =====================================================================
// Toon lighting tunables
// =====================================================================

// Three-band diffuse thresholds applied to NdotL * shadowAtten. Below
// the shadow threshold the pixel sits in the core shadow band; above
// the lit threshold it is fully lit; between the two it occupies a mid
// band. Three bands give the stepped "stylised" look BOTW uses instead
// of the two-band look which reads as low-contrast PBR.
static const float TOON_SHADOW_THRESHOLD = 0.25;
static const float TOON_LIT_THRESHOLD    = 0.6;

// Brightness of each band. Shadow band keeps enough albedo for the
// surface to still read; mid band is between so the step is clearly
// visible without looking like a flat fill. Shadow sits lower than
// before to widen the band spread and push readable contrast.
static const float TOON_LIT_INTENSITY    = 1.0;
static const float TOON_MID_INTENSITY    = 0.65;
static const float TOON_SHADOW_INTENSITY = 0.3;

// Classic stylised warm/cool split: lit side picks up a subtle warm
// tint, shadow side a cool one. Mid sits neutral so the transition
// reads as a shift in temperature, not a colour jump.
static const float3 TOON_LIT_TINT    = float3(1.08, 1.02, 0.92);
static const float3 TOON_MID_TINT    = float3(1.0,  1.0,  1.0);
static const float3 TOON_SHADOW_TINT = float3(0.85, 0.92, 1.10);

// Specular highlight cutoff applied to NdotH^power. The smoothstep
// window widens the edge by a pixel or two — still flat, but avoids
// the sub-pixel shimmer pure `step` produces on curved surfaces.
static const float TOON_SPEC_THRESHOLD_LOW  = 0.55;
static const float TOON_SPEC_THRESHOLD_HIGH = 0.65;
static const float TOON_SPEC_POWER_MIN   = 16.0;   // at roughness = 1
static const float TOON_SPEC_POWER_MAX   = 256.0;  // at roughness = 0

// Rim-light shape: lower power widens the rim, higher strength brightens
// it. These are tuned to clearly read on silhouettes without bleeding
// into front-facing geometry.
static const float TOON_RIM_POWER        = 2.0;
static const float TOON_RIM_STRENGTH     = 0.5;

// Screen-space pseudo-outline tunables. A true inverted-hull outline
// needs a second draw pass, which we can't issue without changing the
// HD pipeline layout we patch into. The mask is split into two roles:
//
//   - silhouette (1 - NdotV) : produces ONLY the outer border. The
//       threshold sits right up against grazing so a curved cape or
//       cloak doesn't turn into a dark gradient on its far side.
//   - cavity (ORM.R)         : handles all inner inking — armor seams,
//       joint folds, anything artists baked into the occlusion map.
//
// The old screen-space crease and depth-edge terms have been removed:
// they caught some inner creases correctly but also false-fired on
// low-tessellation cloth (triangle-seam banding on capes) and on any
// surface curving steeply toward grazing. Cavity via the AO channel
// replaces them with a cleaner, artist-controlled inner-line signal.
static const float3 TOON_OUTLINE_COLOR       = float3(0.04, 0.04, 0.06);
static const float  TOON_OUTLINE_SIL_START   = 0.92;
static const float  TOON_OUTLINE_SIL_END     = 0.995;
// Cavity inking uses the tangent-space normal map's Z component as the
// signal: on a flat pixel the decoded normal is (~0, ~0, 1), so
// tsNormal.z sits near 1. Wherever the artist baked pressed-in detail
// (cracks, seams, joint recesses) the normal tilts and tsNormal.z
// drops below 1. This is purely "does the texture encode a cavity
// here" and is independent of ORM.R, which in WC3 HD is used as a
// generic darkening mask (uniform low values on hair / fur / dark
// shields) rather than a cavity signal — thresholding on ORM.R made
// whole geosets ink over at close range.
//
// Mipmap blending naturally fades the signal at distance (downsampled
// normals average toward flat), so no explicit distance fade is
// needed.
static const float  TOON_OUTLINE_CAVITY_HIGH     = 0.85;
static const float  TOON_OUTLINE_CAVITY_LOW      = 0.55;
static const float  TOON_OUTLINE_CAVITY_STRENGTH = 1.0;

// Three-band diffuse step. `shadowAtten` (0..1) comes from the cascade
// PCF and pushes lit pixels toward the shadow band when a cascade
// occludes them.
public float toonDiffuseBand(float NdotL, float shadowAtten)
{
    float attenuated = NdotL * shadowAtten;
    float midMask = step(TOON_SHADOW_THRESHOLD, attenuated);
    float litMask = step(TOON_LIT_THRESHOLD, attenuated);
    float band = TOON_SHADOW_INTENSITY;
    band = lerp(band, TOON_MID_INTENSITY, midMask);
    band = lerp(band, TOON_LIT_INTENSITY, litMask);
    return band;
}

// Warm/cool temperature shift matching the three diffuse bands. Applied
// to diffuse light contributions only — the vertex-lit quantisation in
// the non-IBL bodies uses the scalar band directly to avoid double-
// tinting vertex colour that already encodes the level's key tint.
public float3 toonBandTint(float NdotL, float shadowAtten)
{
    float attenuated = NdotL * shadowAtten;
    float midMask = step(TOON_SHADOW_THRESHOLD, attenuated);
    float litMask = step(TOON_LIT_THRESHOLD, attenuated);
    float3 tint = TOON_SHADOW_TINT;
    tint = lerp(tint, TOON_MID_TINT, midMask);
    tint = lerp(tint, TOON_LIT_TINT, litMask);
    return tint;
}

// Blinn-Phong style hard highlight. Intentionally skips GGX — we want a
// flat disc, not a gradient. Roughness picks the specular exponent so
// smooth materials still get tight highlights and rough materials get
// broader ones.
public float toonSpecular(float NdotH, float roughness)
{
    float power = lerp(TOON_SPEC_POWER_MAX, TOON_SPEC_POWER_MIN, roughness);
    float highlight = pow(max(NdotH, 0.0), power);
    return smoothstep(TOON_SPEC_THRESHOLD_LOW, TOON_SPEC_THRESHOLD_HIGH, highlight);
}

// Silhouette rim. Pure fresnel with a tighter power than the overlay.
public float toonRimFactor(float NdotV)
{
    float oneMinusNdotV = 1.0 - saturate(NdotV);
    return pow(oneMinusNdotV, TOON_RIM_POWER) * TOON_RIM_STRENGTH;
}

// Outline mask built from two signals: a tight silhouette for the outer
// border and a cavity term for inner inking. `NdotV_sat` is the
// saturated dot against the view direction, `tsNormalZ` is the Z
// component of the decoded tangent-space normal (pass 1.0 when
// unavailable — the cavity term then evaluates to zero).
public float toonOutlineMask(float NdotV_sat, float tsNormalZ)
{
    // Silhouette: fires only right at the border. Threshold is tight so
    // curved surfaces away from the camera aren't darkened on their
    // face — only the actual grazing edge pixels qualify.
    float silhouette = smoothstep(
        TOON_OUTLINE_SIL_START, TOON_OUTLINE_SIL_END, 1.0 - NdotV_sat);

    // Cavity: wherever the normal map's tangent-space Z drops below
    // the threshold, the surface has an authored indentation — fold,
    // seam, crack. Flat surfaces (tsNormal.z ~= 1) stay unink'd.
    float cavity = smoothstep(
        TOON_OUTLINE_CAVITY_HIGH, TOON_OUTLINE_CAVITY_LOW, tsNormalZ)
        * TOON_OUTLINE_CAVITY_STRENGTH;

    return saturate(max(silhouette, cavity));
}

// =====================================================================
// Full lit toon body (IBL / directional / shadow cascades).
// =====================================================================
// Mirrors the structure of psIBLBody — same sample set, same multi-
// layer + debug-override handling, same downstream fresnel/cloak — but
// replaces the Cook-Torrance + IBL specular + Blizzard ambient combine
// with the 2-band toon model.
public float4 psToonIBLBody<
    TMat : IMaterialLayer,
    let HAS_EXTRA_VERTS : bool,
    let HAS_DEBUG_VIS   : bool>(
    PSInput input, bool isFrontFace)
{
    // Shadow cascades (when the VS provided the three clip-space
    // outputs). Same call as the HD IBL body uses.
    float shadowAtten = 1.0;
    if (HAS_EXTRA_VERTS)
    {
        shadowAtten = sdSampleShadowCascades(
            input.shadowClip0, input.shadowClip1, input.shadowClip2,
            asuint(sdPsCB1.numCascades));
    }

    // Material sample set — matches psIBLBody's .xyz/.yzw swizzle pattern
    // so ML materials pick up the blend-weight in ormFull.z without a
    // second sample.
    float4 albedo = t_albedo.Sample(s_albedo, input.uv.xy);
    float3 normalSample = t_normal.Sample(s_normal, input.uv.xy).xyw;
    float4 ormSample = t_orm.Sample(s_orm, input.uv.xy);
    // HD ignores the occlusion channel (.x); the toon path reads it for
    // its cavity outline term.
    float ao = ormSample.x;
    float3 ormFull = ormSample.yzw;
    float2 orm = ormFull.xy;
    float3 emissive = t_emissive.Sample(s_emissive, input.uv.xy).xyz;

    // Debug overrides — same bit layout as HD so the debug HUD works
    // identically on toon materials.
    if (HAS_DEBUG_VIS)
    {
        uint enabled = asuint(psCB3.enabledShaders);
        if ((enabled & 1u) != 0)
            albedo.xyz = psCB3.overrideAlbedo;
        if ((enabled & 2u) != 0)
            orm = psCB3.overrideOrm.yz;
    }

    float3 teamAlbedo = float3(0, 0, 0);
    float blendWeight = 0.0;
    float teamLuma = 0.0;

    if (TMat.hasMultiLayer)
    {
        blendWeight = ormFull.z;
        teamAlbedo = t_teamColor.Sample(s_teamColor, input.uv.xy).xyz;
        teamLuma = max(max(teamAlbedo.x, teamAlbedo.y), teamAlbedo.z);
    }

    float roughness = orm.x;
    float metalness = orm.y;

    if (TMat.hasMultiLayer)
    {
        MultiLayerResult ml = multiLayerBlend(albedo.xyz, teamAlbedo, blendWeight, emissive);
        albedo.xyz = ml.albedo;
        emissive = ml.emissive;
    }

    // View-space shading normal. Toon doesn't run specular-AA — the
    // step-based highlight already hides sub-pixel variance.
    float3 tsNormal = decodeNormalMap(normalSample);
    float3 viewNormal = tangentToWorld(tsNormal, input.tangentWS.xyz, input.normalWS.xyz, input.tangentWS.w);
    viewNormal = orientNormal(viewNormal, isFrontFace);

    float3 viewDir = normalize(-input.worldPos.xyz);
    float NdotV = dot(viewNormal, viewDir);
    float NdotV_sat = saturate(NdotV);

    // Toon is not PBR: the full PBR metal/dielectric split kills the
    // albedo on metals, which — combined with IBL specular being skipped
    // — made reflective surfaces read as pitch black. BOTW-style cel
    // shading treats albedo as the surface color regardless of metalness,
    // and uses a dielectric-default F0 lerped toward albedo so the hard
    // highlight still picks up gold/silver tints.
    float3 F0 = lerp(float3(0.04, 0.04, 0.04), albedo.xyz, metalness);
    float3 diffuseAlbedo = albedo.xyz;

    float3 accumDiffuse = float3(0, 0, 0);
    float3 accumSpec    = float3(0, 0, 0);
    uint firstLightProcessed = 0;

    bool hasFirstLight = asuint(psCB2.lightCount) > 0;
    bool firstIsDirectional = hasFirstLight && (psCB2.lights[0].type == 0.0);

    // Main directional light + IBL: hue-tint the shadow side with the
    // IBL diffuse so environment probes still influence the toon ambient.
    if (firstIsDirectional)
    {
        float3 lightDir = psCB2.lights[0].position;

        float3 H = normalize(viewDir + lightDir);
        float NdotH = dot(viewNormal, H);
        float NdotL = dot(viewNormal, lightDir);
        float VdotH = dot(H, viewDir);

        // Sample the two IBL probes for the shadow-side ambient tint.
        // The specular cubemap lookup is ignored — toon reflects only as
        // the hard highlight from the analytical light.
        IBLSampleResult ibl = sampleIBL(
            input.worldPos.xyz, viewNormal, roughness,
            psCB2.view,
            psCB2.envFromMipEnd, psCB2.envToMipEnd, psCB2.envTransitionT);

        float ambientWeight = psCB2.lights[0]._pad2.x;

        // Banded diffuse contribution; the spec mask gates on the top
        // lit threshold so highlights only appear in the fully-lit band.
        float diffuseBand = toonDiffuseBand(NdotL, shadowAtten);
        float litMask = step(TOON_LIT_THRESHOLD, NdotL * shadowAtten);

        float3 ambientTint = psCB2.lights[0].ambient
                           + ibl.isActive * ibl.diffuseIrradiance * ambientWeight;
        float3 bandTint    = toonBandTint(NdotL, shadowAtten);
        float3 litTint     = psCB2.lights[0].diffuse * diffuseBand * bandTint;

        accumDiffuse += ambientTint + litTint;

        // Hard specular, only fires where the surface is lit.
        float3 F = schlickFresnel(F0, VdotH);
        float specMask = toonSpecular(NdotH, roughness) * litMask;
        accumSpec += specMask * F * psCB2.lights[0].diffuse;

        // Metals also reflect the environment outside the hard highlight
        // disc — without this they'd go flat/black in shadow. Scaled by
        // metalness so dielectrics stay purely diffuse. The split-sum
        // weighting (F0 * brdfLUT.x + brdfLUT.y) also forces slangc to
        // emit a dcl_resource for the BRDF LUT so our IBL perms bind
        // the same resource set as HD — the engine rejects a shader
        // whose declared resources don't match the pipeline template.
        float3 iblSpecWeight = F0 * ibl.brdfLUT.x + ibl.brdfLUT.y;
        accumSpec += metalness * ibl.isActive * ibl.specularRadiance * iblSpecWeight;

        firstLightProcessed = 1;
    }

    // Remaining analytical lights: same 2-band model, no shadow
    // attenuation (cascades only cover the main directional).
    uint lightIdx = firstLightProcessed;
    uint lightCount = asuint(psCB2.lightCount);
    while (lightIdx < lightCount)
    {
        ShaderLight light = psCB2.lights[lightIdx];
        float3 lightDir;
        float attenuation;

        if (light.type > 0)
        {
            float3 lightVec = light.position - input.worldPos.xyz;
            float distSq = dot(lightVec, lightVec);
            float dist = sqrt(distSq);
            attenuation = pointLightAttenuation(distSq, dist);
            lightDir = lightVec / dist;
        }
        else
        {
            lightDir = light.position;
            attenuation = 1.0;
        }

        float3 H = normalize(viewDir + lightDir);
        float NdotH = dot(viewNormal, H);
        float NdotL = dot(viewNormal, lightDir);
        float VdotH = dot(H, viewDir);

        float diffuseBand = toonDiffuseBand(NdotL, 1.0);
        float litMask = step(TOON_LIT_THRESHOLD, NdotL);
        float3 bandTint = toonBandTint(NdotL, 1.0);

        accumDiffuse += light.ambient * attenuation
                      + light.diffuse * diffuseBand * attenuation * bandTint;

        float3 F = schlickFresnel(F0, VdotH);
        float specMask = toonSpecular(NdotH, roughness) * litMask;
        accumSpec += specMask * F * light.diffuse * attenuation;

        lightIdx++;
    }

    // Combine diffuse × albedo, add specular + rim.
    float3 finalColor = accumDiffuse * diffuseAlbedo + accumSpec;
    finalColor += toonRimFactor(NdotV_sat) * albedo.xyz;
    finalColor += emissive * psCB2.emissiveGain;

    // Per-vertex tint / AO — sRGB-linearised as in the HD path.
    float3 linearVert = srgbToLinear(input.vertColor.xyz);
    finalColor *= linearVert;

    // Screen-space outline — applied after the lit colour is assembled
    // so the outline colour is constant instead of being modulated by
    // vertex tint. Stays under the fresnel overlay so selection rings
    // still read on top of outlined silhouettes.
    float outlineMask = toonOutlineMask(NdotV_sat, tsNormal.z);
    finalColor = lerp(finalColor, TOON_OUTLINE_COLOR, outlineMask);

    // Finishing stack: fresnel overlay → cloak. Signatures match what
    // psIBLBody uses so selection rings / stealth look the same on toon
    // materials.
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

// =====================================================================
// Cheap single-layer toon body (no IBL, no analytical lights).
// =====================================================================
// The engine drives the non-IBL path with pre-baked vertex lighting, so
// there is no NdotL to step against. We instead quantise the per-vertex
// lighting's luma into the two toon bands, which reproduces the BOTW
// "flat shadow" look on assets that don't opt into the full lit path.
public float4 psToonStandardBody<let HAS_DEBUG_VIS : bool>(PSInput input, bool isFrontFace)
{
    float4 albedo = t_albedo.Sample(s_albedo, input.uv.xy);
    float3 normalSample = t_normal.Sample(s_normal, input.uv.xy).xyw;
    float3 emissive = t_emissive.Sample(s_emissive, input.uv.xy).xyz;

    if (HAS_DEBUG_VIS && (asint(psCB3.enabledShaders) & 1))
        albedo.xyz = psCB3.overrideAlbedo;

    float3 tsNormal = decodeNormalMap(normalSample);
    float3 worldNormal = tangentToWorld(tsNormal, input.tangentWS.xyz, input.normalWS.xyz, input.tangentWS.w);
    worldNormal = orientNormal(worldNormal, isFrontFace);

    float3 viewDir = normalize(-input.worldPos.xyz);
    float NdotV = saturate(dot(viewDir, worldNormal));

    // Quantise the vertex-lit color into 2 bands using its peak channel
    // as the lighting intensity; preserves hue so colored key lights
    // still tint the toon output correctly.
    float3 linearVert = srgbToLinear(input.vertColor.xyz);
    float vertLum = max(max(linearVert.x, linearVert.y), linearVert.z);
    float bandScale = toonDiffuseBand(vertLum, 1.0);
    float3 bandedVert = linearVert * (bandScale / max(vertLum, SAFE_EPS));

    float3 litColor = (emissive * psCB2.emissiveGain + albedo.xyz) * bandedVert;
    litColor += toonRimFactor(NdotV) * albedo.xyz;

    float outlineMask = toonOutlineMask(NdotV, tsNormal.z);
    litColor = lerp(litColor, TOON_OUTLINE_COLOR, outlineMask);

    FresnelOverlayResult fr = applyFresnelOverlay(
        litColor, albedo.w, NdotV,
        psCB2.fresnelColor.xyz, psCB2.fresnelColor.w, psCB2.fresnelTeamColor,
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

// =====================================================================
// Cheap multi-layer toon body.
// =====================================================================
// Same idea as psToonStandardBody but runs the team-color multiLayerBlend
// first so team-tinted heroes still read correctly under toon shading.
public float4 psToonMultiLayerBody<let HAS_DEBUG_VIS : bool>(PSInput input, bool isFrontFace)
{
    float4 albedo = t_albedo.Sample(s_albedo, input.uv.xy);
    float3 teamAlbedo = t_teamColor.Sample(s_teamColor, input.uv.xy).xyz;
    float4 ormSample = t_orm.Sample(s_orm, input.uv.xy);
    float ao = ormSample.x;
    float blendWeight = ormSample.w;
    float3 emissive = t_emissive.Sample(s_emissive, input.uv.xy).xyz;
    float3 normalSample = t_normal.Sample(s_normal, input.uv.xy).xyw;

    if (HAS_DEBUG_VIS && (asint(psCB3.enabledShaders) & 1))
        albedo.xyz = psCB3.overrideAlbedo;

    MultiLayerResult ml = multiLayerBlend(albedo.xyz, teamAlbedo, blendWeight, emissive);
    float teamLuma = max(max(teamAlbedo.x, teamAlbedo.y), teamAlbedo.z);

    float3 tsNormal = decodeNormalMap(normalSample);
    float3 worldNormal = tangentToWorld(tsNormal, input.tangentWS.xyz, input.normalWS.xyz, input.tangentWS.w);
    worldNormal = orientNormal(worldNormal, isFrontFace);

    float3 viewDir = normalize(-input.worldPos.xyz);
    float NdotV = saturate(dot(viewDir, worldNormal));

    float3 linearVert = srgbToLinear(input.vertColor.xyz);
    float vertLum = max(max(linearVert.x, linearVert.y), linearVert.z);
    float bandScale = toonDiffuseBand(vertLum, 1.0);
    float3 bandedVert = linearVert * (bandScale / max(vertLum, SAFE_EPS));

    float3 litColor = (ml.emissive * psCB2.emissiveGain + ml.albedo) * bandedVert;
    litColor += toonRimFactor(NdotV) * ml.albedo;

    float outlineMask = toonOutlineMask(NdotV, tsNormal.z);
    litColor = lerp(litColor, TOON_OUTLINE_COLOR, outlineMask);

    FresnelOverlayResult fr = applyFresnelOverlayML(
        litColor, albedo.w, NdotV,
        psCB2.fresnelColor.xyz, psCB2.fresnelColor.w, psCB2.fresnelTeamColor,
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
