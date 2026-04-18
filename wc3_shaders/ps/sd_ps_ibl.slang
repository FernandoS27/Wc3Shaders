// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026, Fernando Sahmkow
// See LICENSE in the repository root for full terms.
//
// SD-on-HD IBL pixel shader body.
//
// This is the remastered SD lighting path: it keeps the SD data layout
// (vertex color as pre-lit color, no per-pixel normal map, diffuse-only
// albedo) but upgrades the shading with:
//   - simplified specular using a fresnel × cheap-geometric visibility,
//   - two-probe IBL blend (same pattern as the HD path),
//   - optional 3-cascade PCF shadows,
//   - analytical loop for the remaining lights.
// The arithmetic is reorganised to match the SD pixel shader that
// ships with Reforged — the sequence of FMAs below preserves the
// legacy register scheduling and numerical output.

implementing wc3_shaders;

// Splits used to mix the probe diffuse irradiance with the primary
// analytical light's ambient contribution. Same meaning as the
// constants in ps_ibl.slang.
static const float IBL_SPLIT_UPPER = 0.675000012;
static const float IBL_SPLIT_LOWER = 0.324999988;
// (1, 1, -1) flip converts between the engine's world Z-up convention
// and the cubemap's Y-up right-handed sampling convention.
static const float3 IBL_COORD_FLIP = float3(1, 1, -1);

// Cheap specular approximation used by the SD-on-HD path.
// F ≈ (1 - VdotH)^5 (no F0 term — specular color is folded into the
// diffuse luma at the call site), G ≈ 1 / (4 · (0.5·NdotL+0.5) ·
// (0.5·NdotV+0.5)). Divided by π so the PI-scale at the end of the
// pixel shader undoes it.
static float sdSimplifiedSpec(float VdotH, float NdotL, float NdotV)
{
    float t = 1.0 - VdotH;
    float t2 = t * t;
    float t4 = t2 * t2;
    float fresnel = t * t4;

    // Biased (0.5·NdotL + 0.5) form matches the "wrap" factor the
    // retail SD shader used for its geometric term — softer than the
    // HD Schlick-GGX but close enough for the detail-free SD albedo.
    float gNdotL = NdotL * 0.5 + 0.5;
    float gNdotV = NdotV * 0.5 + 0.5;
    float geom = saturate(1.0 / (4.0 * gNdotL * gNdotV));

    return fresnel * geom * INV_PI;
}

struct IBLSample
{
    float3 diffuseIBL;
    float3 specIBL;
    float  brdfY;   // Y (bias) component of the split-sum LUT.
};

// Two-probe IBL sample. Short-circuits to zeros when no probe is
// bound (mip_end product == 0). The matrix multiply is hand-unrolled
// because `invView` is uploaded as three float4 rows rather than a
// full float4x4 (see SDPSPerDraw).
static IBLSample sampleIBL(
    float3 worldPos, float3 normalWS, float NdotV,
    float4 invViewRow0, float4 invViewRow1, float4 invViewRow2,
    float envFromMip, float envToMip, float envTransitionT)
{
    IBLSample result;
    result.diffuseIBL = float3(0, 0, 0);
    result.specIBL = float3(0, 0, 0);
    result.brdfY = 0;

    if (envFromMip * envToMip == 0.0)
        return result;

    // Build the view-space position direction, in cubemap xzy ordering.
    float3 viewPosDir;
    viewPosDir  = invViewRow1.xzy * worldPos.y;
    viewPosDir += invViewRow0.xzy * worldPos.x;
    viewPosDir += invViewRow2.xzy * worldPos.z;
    viewPosDir = normalize(viewPosDir);

    // Same for the normal.
    float3 viewNormal;
    viewNormal  = invViewRow1.xzy * normalWS.y;
    viewNormal += invViewRow0.xzy * normalWS.x;
    viewNormal += invViewRow2.xzy * normalWS.z;
    viewNormal = normalize(viewNormal);

    // Reflection direction for the specular probe fetch.
    float nDotV = dot(viewPosDir, viewNormal);
    float3 reflDir = viewPosDir - 2.0 * nDotV * viewNormal;
    reflDir = normalize(reflDir);

    // Diffuse (slice 0) — no mip selection, always sample the top.
    float4 diffCoord = float4(IBL_COORD_FLIP * viewNormal, 0);
    float3 fromDiff = t_iblFrom.Sample(s_iblFrom, diffCoord).xyz;
    float3 toDiff = t_iblTo.Sample(s_iblTo, diffCoord).xyz;
    result.diffuseIBL = lerp(fromDiff, toDiff, envTransitionT);

    // Specular (slice 1) at the runtime-supplied mip.
    float4 specCoord = float4(IBL_COORD_FLIP * reflDir, 1);
    float3 fromSpec = t_iblFrom.SampleLevel(s_iblFrom, specCoord, envFromMip).xyz;
    float3 toSpec = t_iblTo.SampleLevel(s_iblTo, specCoord, envToMip).xyz;
    result.specIBL = lerp(fromSpec, toSpec, envTransitionT);

    // SD only uses the Y (bias) channel; the X (scale) channel is
    // irrelevant because the SD path has no explicit F0.
    result.brdfY = t_brdfLut.Sample(s_brdfLut, float2(NdotV, 1)).y;

    return result;
}

public float4 sdIBLBody<let HAS_EXTRA_VERTS : bool, let HAS_DEBUG_VIS : bool>(
    PSInput input, bool isFrontFace)
{
    // Diffuse albedo + vertex-color modulation happen at the end; for
    // now we only need the texture sample.
    float4 texColor = t_albedo.Sample(s_albedo, input.uv.xy);

    // Interpolated world-space normal, flipped for back faces.
    float3 normalWS = normalize(input.normalWS.xyz);
    normalWS = isFrontFace ? normalWS : -normalWS;

    float3 viewDir = normalize(-input.worldPos.xyz);
    float NdotV = dot(normalWS, viewDir);

    // Lambertian albedo (pre-divided by π; the final PI multiply at
    // the end cancels it for the direct light terms).
    float3 diffAlbedo = texColor.xyz * INV_PI;

    // Shadow attenuation comes from the 3-cascade PCF helper when the
    // VS provided the cascade-space clip coords. The generic SD path
    // passes this flag false and skips shadow work entirely.
    float shadowAtten = 1.0;
    if (HAS_EXTRA_VERTS)
    {
        shadowAtten = sdSampleShadowCascades(
            input.shadowClip0, input.shadowClip1, input.shadowClip2,
            asuint(sdPsCB1.numCascades));
    }

    // `lightCountSlot.z` stores the uint-reinterpreted light count —
    // the SD CB uses a float slot here.
    uint lightCount = asuint(sdPsCB2.lightCountSlot.z);

    float3 diffAccum = float3(0, 0, 0);
    float3 ambAccum = float3(0, 0, 0);

    int loopStart = 0;
    if (lightCount > 0 && sdPsCB2.lights[0].type == 0.0)
    {
        // Special-case the first directional light to combine it with
        // the IBL blend, matching the HD path's scheme.
        float3 L = sdPsCB2.lights[0].position;

        float3 H = normalize(viewDir + L);
        float VdotH = dot(H, viewDir);
        float NdotL = dot(normalWS, L);
        float specTerm = sdSimplifiedSpec(VdotH, NdotL, NdotV);
        float clampedNdotL = max(0, NdotL);

        IBLSample ibl = sampleIBL(
            input.worldPos.xyz, normalWS, NdotV,
            sdPsCB2.invViewRow0, sdPsCB2.invViewRow1, sdPsCB2.invViewRow2,
            sdPsCB2.envFromMipEnd, sdPsCB2.envToMipEnd, sdPsCB2.envTransitionT);

        // Apply shadow only to the direct (NdotL) term, not to ambient.
        float litNdotL = HAS_EXTRA_VERTS ? (clampedNdotL * shadowAtten) : clampedNdotL;

        float3 lightDiffuse = sdPsCB2.lights[0].diffuse;
        float maxAmb = max(max(lightDiffuse.x, lightDiffuse.y), lightDiffuse.z);
        float invMaxAmb = 1.0 / (maxAmb + SAFE_EPS);

        // Ambient + IBL combining weight and its scaled-LUT product —
        // used by both the ambient blend and the specular indirect
        // term below, so we hoist the computation.
        float ambPlusLut = maxAmb + ibl.brdfY;
        float lutContrib = ambPlusLut * ibl.brdfY;
        float scaledLut = sdPsCB2.lights[0]._pad2.x * lutContrib;

        // Hue-separate the IBL diffuse for the ambient mix.
        float maxIBL = max(max(ibl.diffuseIBL.x, ibl.diffuseIBL.y), ibl.diffuseIBL.z);
        float invMaxIBL = 1.0 / (maxIBL + SAFE_EPS);
        float3 iblNorm = ibl.diffuseIBL * invMaxIBL;

        // Upper / lower ambient split — same construction as the HD
        // blizzardAmbientBlend but inlined because the SD path reuses
        // intermediates in the specular combine below.
        float3 ambDelta = lightDiffuse * invMaxAmb - iblNorm;
        float3 upper = ambDelta * IBL_SPLIT_UPPER + iblNorm;
        upper = upper * maxIBL;
        upper = upper * ambPlusLut - lightDiffuse;
        upper = ibl.brdfY * upper + lightDiffuse;

        float3 lower = ambDelta * IBL_SPLIT_LOWER + iblNorm;
        float3 lowerScaled = lower * maxIBL;
        float3 litLower = lowerScaled * scaledLut;
        float3 unlitLower = -lowerScaled * scaledLut + upper;
        // Blend between upper (lit) and litLower (shadow-side ambient)
        // by litNdotL so the shadow side retains the IBL contribution.
        unlitLower = litNdotL * unlitLower + litLower;
        ambAccum = sdPsCB2.lights[0].ambient + unlitLower;

        // Specular combine: direct lobe + indirect from the IBL
        // specular probe weighted by the LUT.
        float combinedSpec = specTerm * ambPlusLut;
        combinedSpec = combinedSpec * litNdotL;
        float specIBL = scaledLut * ibl.brdfY;
        diffAccum = combinedSpec * lightDiffuse + specIBL;
        diffAccum = diffAccum * ibl.specIBL;
        diffAccum = ambAccum * diffAlbedo + diffAccum;

        loopStart = 1; // Skip light 0 in the generic loop.
    }

    // Generic analytical loop for the remaining lights (up to 8).
    for (uint i = (uint)loopStart; i < lightCount && i < 8; i++)
    {
        float3 lightPos = sdPsCB2.lights[i].position;
        float lightType = sdPsCB2.lights[i].type;

        float3 L;
        float atten;
        if (lightType > 0)
        {
            // Point light: inverse-square attenuation.
            float3 delta = lightPos - input.worldPos.xyz;
            float distSq = dot(delta, delta);
            float dist = sqrt(distSq);
            atten = pointLightAttenuation(distSq, dist);
            L = delta / dist;
        }
        else
        {
            // Directional light (`position` is already a unit vector).
            L = lightPos;
            atten = 1.0;
        }

        float3 H = normalize(viewDir + L);
        float VdotH = dot(H, viewDir);
        float NdotL = dot(normalWS, L);
        float specTerm = sdSimplifiedSpec(VdotH, NdotL, NdotV);
        float clampedNdotL = max(0, NdotL) * atten;

        // Per-light specular + diffuse, weighted by the light color.
        float3 specAndDiff = specTerm * INV_PI + diffAlbedo;
        specAndDiff = sdPsCB2.lights[i].diffuse * specAndDiff;
        diffAccum += specAndDiff * clampedNdotL;

        ambAccum += sdPsCB2.lights[i].ambient * atten;
    }

    // Convert back to physical units (undo the INV_PI scale) and
    // modulate by the linearised vertex color (carries AO / tints).
    float3 litColor = PI * diffAccum;

    float3 vcLinear = srgbToLinear(input.vertColor.xyz);
    float3 finalColor = vcLinear * litColor;

    // Cloak is the only overlay the SD-on-HD path supports; the
    // fresnel overlay is HD-only. Alpha collapses to vertex × texture.
    float satNdotV = saturate(NdotV);
    CloakResult cloak = applyCloak(
        finalColor, texColor.w, texColor.w,
        satNdotV, sdPsCB2.cloakAmount, input.vertColor.w);

    float nonCloakAlpha = saturate(input.vertColor.w * texColor.w);
    float4 nonCloakResult = float4(finalColor, nonCloakAlpha);

    bool cloakActive = sdPsCB2.cloakAmount > 0;
    return cloakActive ? cloak.colorAlpha : nonCloakResult;
}
