// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026, Fernando Sahmkow
// See LICENSE in the repository root for full terms.
//
// DeferredLight family (deferredlight.fx) — VS + PS.
//
// The deferred point/spot light pass: rasterise the light volume, reconstruct the
// world position from the depth prepass, shade it with one light, and modulate by
// the shadow map.  `DeferredLight_BuildSection` @0x102b480f0 registers five own
// axes and links [Common, **Lighting**]:
//
//     b_iLightDebugMode      @104 w2  (dead — its arm compares against a mode the
//                                      retail set never selects)
//     b_iLightType           @106 w2  LIGHTTYPE_POINT 1 / LIGHTTYPE_SPOT 2
//     b_useDeffSpecularPower @108 w1
//     b_useVSBackBufferUV    @109 w1
//     b_useViewport          @110 w1  (pinned 1 across the whole retail set)
//
// Its b_iUseSpecular / b_iAffectedByAO come from the linked Lighting sub-section,
// and its shadow axes (b_useShadows, b_iUseSoftShadows, b_iSoftShadowTaps,
// b_useTransparentShadows) from Common — which is why the PS body can call the
// Model family's already-ported `shadowIntensityUV` unchanged.
//
// NOTE deferredlight.fx also references `b_iStereoscopicCorrection`, which NO
// section registers.  Like `b_useVertexLighting` it is dead source reading a
// constant 0 — and here the block it guards is commented out anyway.
//
// Unlike the other M4 families this one uses the SHARED interpolant transport
// (`InitShader(vertOut)` / `WRITE_INTERPOLANT_*`), so it keeps the default
// `inject_preamble: true` and appears in sc2_shaders_cfg._SHARED_TRANSPORT.  Its
// live set is tiny — {ScreenPos} or {BackBufferUV, ScreenPos} — because the only
// varying it can carry is the back-buffer UV; HPosAsUV is written by the VS but
// never read by the PS (VPOS_SEMANTIC is 1 on ps_5_0), so it is never live.
//
// Retail set: 8 VS perms, 82 PS perms.

implementing sc2_shaders;
#include "types/perm_cb.slangh"

#ifndef b_iLightDebugMode
#define b_iLightDebugMode 0
#endif
#ifndef b_iLightType
#define b_iLightType 0
#endif
#ifndef b_useDeffSpecularPower
#define b_useDeffSpecularPower 0
#endif
#ifndef b_useVSBackBufferUV
#define b_useVSBackBufferUV 0
#endif
#ifndef b_useViewport
#define b_useViewport 0
#endif
#ifndef b_iUseSpecular
#define b_iUseSpecular 0
#endif
#ifndef b_iAffectedByAO
#define b_iAffectedByAO 0
#endif
#ifndef b_iDebugMode
#define b_iDebugMode 0
#endif

#ifndef b_iUse8BitHDR
#define b_iUse8BitHDR 0
#endif

// lighttypes.fx
#define LIGHTTYPE_DIRECTIONAL 0
#define LIGHTTYPE_POINT       1
#define LIGHTTYPE_SPOT        2

// deferredlight.fx
#define LIGHT_DEBUG_NONE         0
#define LIGHT_DEBUG_COLORS       1
#define LIGHT_DEBUG_SINGLE_LIGHT 2

// psdebugmodes.fx (only the modes this root tests).
#define DL_NO_DEBUG                0
#define DL_DIFFUSE_ONLY            2
#define DL_SPECULAR_ONLY           3
#define DL_LIGHTING_ONLY           4
#define DL_DIFFUSE_LIGHTING_ONLY   5
#define DL_SPECULAR_LIGHTING_ONLY  6
#define DL_SHADOWS_ONLY            13
#define DL_LIGHTING_OVERLAP        16
#define DL_OVERDRAW                19

public struct DeferredLightVSInput
{
    float4 vPosition : POSITION;
};

// The engine-generated `VertexTransport` for this family: HPos plus, when the VS
// computes the back-buffer UV itself, the one live varying.  Field set and
// semantics come from the same interp_defines that drive gen_preamble, so the two
// legs' signatures match slot for slot.
public struct DeferredLightTransport
{
    float4 HPos : SV_Position;
#if SC2_HAS_BackBufferUV
    float2 m_BackBufferUV : SC2_SEM_BackBufferUV;
#endif
};

public struct DeferredLightPSInput
{
    float4 HPos : SV_Position;
#if SC2_HAS_BackBufferUV
    float2 m_BackBufferUV : SC2_SEM_BackBufferUV;
#endif

    float2 BackBufferUV()
    {
#if SC2_HAS_BackBufferUV
        return m_BackBufferUV;
#else
        return float2(0, 0);
#endif
    }
};

// vsutils.fx : GetBackBufferUV — NDC to [0..1], then into the viewport rect.
float2 vsGetBackBufferUV(float4 vHPos)
{
    float3 vPos = vHPos.xyz / vHPos.w;
    float2 vResult = vPos.xy * float2(0.5f, -0.5f) + 0.5f;
#if b_useViewport
    return vsGlobals.vViewportOrigin + vResult * vsGlobals.vViewportScale;
#else
    return vResult;
#endif
}

[shader("vertex")]
public DeferredLightTransport deferredlight_vs_main(DeferredLightVSInput vertIn)
{
    DeferredLightTransport vertOut;
    vertOut.HPos = mul(vertIn.vPosition, vsGlobals.mWVPTransform);
#if SC2_HAS_BackBufferUV
    vertOut.m_BackBufferUV = vsGetBackBufferUV(vertOut.HPos);
#endif
    // WRITE_INTERPOLANT_HPosAsUV is a no-op: the PS reads SV_Position instead
    // (VPOS_SEMANTIC), so HPosAsUV is never in the live set.
    return vertOut;
}

[shader("pixel")]
public float4 deferredlight_ps_main(DeferredLightPSInput vertOut) : SV_Target0
{
    float2 vOffsetUV;
    float2 vUV;
    float4 vViewPos;

#if b_useVSBackBufferUV
    vUV = vertOut.BackBufferUV();
    // Offset it so it samples exact pixels.
    vOffsetUV = vUV + psGlobals.vRecipRenderTargetSizeOffset.zw;
    vViewPos.xz = vUV.xy * psGlobals.vUVToViewPos.xy + psGlobals.vUVToViewPos.zw;
#else
    // VPOS_SEMANTIC is 1 on ps_5_0, so the optimised SV_Position path is the only
    // live arm; the psutils.fx HPosAsUV fallback is compiled out.
    vOffsetUV = vertOut.HPos.xy * psGlobals.vRecipRenderTargetSizeOffset.xy
              + psGlobals.vRecipRenderTargetSizeOffset.zw;
    vViewPos.xz = vertOut.HPos.xy * psGlobals.vUVToViewPos.xy + psGlobals.vUVToViewPos.zw;
#endif

    // pscommon.fx : SampleNormalDepth (ps/ps_common.slang).
    float4 vNormalDepth = psSampleNormalDepth(p_sNormalDepthMap_t, vOffsetUV);

    float4 cDiffuse = p_sDiffuseMap_t.Sample(p_sSamp, vOffsetUV);
    float4 cSpecular;
    // Same SRV either way -- only which channel supplies .a differs.
    float4 cSpecularSample = p_sSpecularMap_t.Sample(p_sSamp, vOffsetUV);
    cSpecular.rgb = cSpecularSample.rgb;
    cSpecular.a = SC2_AXIS(b_useDeffSpecularPower) ? cSpecularSample.a
                                                  : psGlobals.vSpecColorSpecularity.a;
    [branch] if (SC2_AXIS(b_iUse8BitHDR))
        cSpecular.a /= psGlobals.vHDRScaling.y;

    // PIXEL_NORMAL / PIXEL_DEPTH are the .rgb / .a of vNormalDepth.
    vNormalDepth.rgb = normalize(vNormalDepth.rgb);

    vViewPos.yw = float2(vNormalDepth.a, 1.0f);
    vViewPos.xz = vViewPos.xz * (vNormalDepth.a * psGlobals.mPTransform[1][3]
                                + psGlobals.mPTransform[3][3]);

    // b_iStereoscopicCorrection's block is commented out upstream.

    // Back to world space.
    float3 vPos = mul(psGlobals.mInvViewTransform, vViewPos).xyz;

    float3 lightDiffSpecIntensity = float3(0, 0, 0);
    const int iLightType = SC2_AXIS(b_iLightType);
    [branch] if (iLightType == LIGHTTYPE_POINT)
        lightDiffSpecIntensity = computePointLight(
            vNormalDepth.rgb, vPos, psGlobals.vPositionRecipRadius.xyz,
            psGlobals.vPositionRecipRadius.w, psGlobals.cColorAttenMultiplier.w,
            cSpecular.w);
    else [branch] if (iLightType == LIGHTTYPE_SPOT)
        lightDiffSpecIntensity = computeSpotLight(
            vNormalDepth.rgb, vPos, psGlobals.vPositionRecipRadius.xyz,
            psGlobals.vDirection, psGlobals.vPositionRecipRadius.w,
            psGlobals.cColorAttenMultiplier.w,
            psGlobals.vFalloffBiasScaleAndHotSpotMultiplier.x,
            psGlobals.vFalloffBiasScaleAndHotSpotMultiplier.y,
            psGlobals.vFalloffBiasScaleAndHotSpotMultiplier.z,
            cSpecular.w);

    float4 vShadowMapUV = mul(float4(vPos, 1), psGlobals.mShadowTransform);

    // Spot-light shadows have a fixed viewpoint, so the shadow-map UV doubles as the
    // soft-shadow noise UV.
    float4 cShadowColor = shadowIntensityUV(
        vShadowMapUV, vShadowMapUV.xy * psGlobals.fShadowMapSize / 32.0f);

#if b_iAffectedByAO
    cShadowColor *= p_sSSAO_t.Sample(p_sSamp, vOffsetUV).a;
#endif

#if (b_iDebugMode == DL_LIGHTING_ONLY)
    cDiffuse = 1.0f;
    cSpecular = 1.0f;
#elif (b_iDebugMode == DL_DIFFUSE_LIGHTING_ONLY)
    cDiffuse = 1.0f;
    cSpecular = 0.0f;
#elif (b_iDebugMode == DL_SPECULAR_LIGHTING_ONLY)
    cDiffuse = 0.0f;
    cSpecular = 1.0f;
#elif (b_iDebugMode == DL_DIFFUSE_ONLY)
    cSpecular = 0.0f;
#elif (b_iDebugMode == DL_SPECULAR_ONLY)
    cDiffuse = 0.0f;
#elif (b_iDebugMode == DL_SHADOWS_ONLY)
    return float4(cShadowColor.rgb, 1.0f);
#elif (b_iDebugMode == DL_LIGHTING_OVERLAP)
    return 1.0f / 8.0f;
#elif (b_iDebugMode == DL_OVERDRAW)
    return 0;
#elif (b_iDebugMode != DL_NO_DEBUG)
    return 0.0f;
#endif

#if (b_iLightDebugMode == LIGHT_DEBUG_COLORS)
    // The D3D arm (the GL variant additionally tests the colour products, which are
    // commented out here).  b_iLightDebugMode is 0 across the whole retail set.
    if (SC2_AXIS(b_iUseSpecular) && lightDiffSpecIntensity.y >= 0.02f)
    {
        if (cShadowColor.a > 0.0f)
            return float4(0.0f, 0.0f, 1.0f, 1.0f);      // Blue = specular highlight
        else
            return float4(0.5f, 0.0f, 0.0f, 1.0f);      // Dark red = shadowed specular
    }
    else if (lightDiffSpecIntensity.x <= 0.02f)
        return float4(1.0f, 0.0f, 0.0f, 1.0f);          // Red = <= 1% contribution
    else if (cShadowColor.a == 0.0f)
        return float4(0.5f, 0.0f, 0.0f, 1.0f);          // Dark red = shadowed lit pixel
    else if (cShadowColor.a < 1.0f)
        return float4(0.0f, 0.5f, 0.0f, 1.0f);          // Dark green = partially lit
    else
        return float4(0.0f, 1.0f, 0.0f, 1.0f);          // Green = lit pixel
#endif

    float4 result;
    result.rgb = cDiffuse.rgb * psGlobals.cColorAttenMultiplier.xyz
               * lightDiffSpecIntensity.x * cShadowColor.rgb;
    [branch] if (SC2_AXIS(b_iUseSpecular))
        result.rgb += cSpecular.rgb * psGlobals.vSpecColorSpecularity.rgb
                    * lightDiffSpecIntensity.y * cShadowColor.rgb;
    result.a = 1.0f;
    [branch] if (SC2_AXIS(b_iUse8BitHDR))
        result.rgb *= psGlobals.vHDRScaling.x;

    return result;
}
