// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026, Fernando Sahmkow
// See LICENSE in the repository root for full terms.
//
// Model-family fog-of-war (psfow.fx : ApplyFogOfWar), transcribed faithfully.
//
// The FOW UV arrives as INTERPOLANT_FOWUV.xy through Ndc2Tex.  Two live-forms
// exist in the reference (sc2_interp.gen_preamble): the packed FOWUV varying
// (this increment), or an in-PS recompute mul(WorldPos, p_mFOWMatrix) when the
// varying was dropped but WorldPos survives.  This increment covers the packed
// form (SC2_HAS_FOWUV); the WorldPos-recompute form lands with the WorldPos
// interpolant in a later increment.
//
// Every b_* here is a permutation define substituted as an integer literal, so
// the `#if` ladders below dead-code-eliminate to exactly the branch fxc keeps —
// which is what makes the reference's referenced-constant set match ours.

implementing sc2_shaders;
#include "types/perm_cb.slangh"

// psfow.fx FOW-mode enum.
#define FOWMODE_STORM_SMOOTH    0
#define FOWMODE_STORM           1
#define FOWMODE_STAR2           2
#define FOWMODE_RAW             3

// ---- axis defaults (0 = engine default) ------------------------------------
#ifndef b_FOWEnabled
#define b_FOWEnabled 0
#endif
#ifndef b_SampleFOW
#define b_SampleFOW 0
#endif
#ifndef b_FOWMode
#define b_FOWMode 0
#endif
#ifndef b_FastFOW
#define b_FastFOW 0
#endif
#ifndef b_useBlendAdd
#define b_useBlendAdd 0
#endif
#ifndef b_FOWAdditiveScale
#define b_FOWAdditiveScale 0
#endif

// common.fx : Ndc2Tex — NDC (-1..1) to texture (0..1) with y flipped.  The
// flipV arg is unused in the .fx body (it always flips), so it is dropped here.
float2 ndc2Tex(float2 vNDC)
{
    return vNDC * float2(0.5f, -0.5f) + 0.5f;
}

// psfow.fx : ApplyFogOfWar(float2 FOWUV, cResult, inout cDeferredDiffuse).
// cDeferredDiffuse is threaded so the MRT-diffuse target stays correct once MRT
// lands; on the non-MRT standard path the caller passes a stub and ignores it.
float4 applyFogOfWar(float2 FOWUV, float4 cResult, inout float3 cDeferredDiffuse)
{
    [branch] if (!SC2_AXIS(b_FOWEnabled))
        return cResult;

    float4 cFOWFactor;
  #if b_SampleFOW
    #if (b_FOWMode == FOWMODE_STORM) || (b_FOWMode == FOWMODE_STORM_SMOOTH)
      #if !b_FastFOW
        cFOWFactor.rgb = saturate(2.0f * p_sFOWSampler_t.Sample(p_sSamp, FOWUV).rgb);
        cFOWFactor.a = dot(cFOWFactor.rgb, psGlobals.vFOWLevels.rgb);
      #else
        cFOWFactor.a = p_sFOWSampler_t.Sample(p_sSamp, FOWUV).a;
      #endif
        cFOWFactor.rgb = psGlobals.cFOWBlackMaskColor;
    #elif (b_FOWMode == FOWMODE_RAW)
        cFOWFactor = p_sFOWSampler_t.Sample(p_sSamp, FOWUV);
        return float4(cFOWFactor.rgb, cResult.a);
    #endif
  #else
    cFOWFactor = psGlobals.cMaterialFOWTint;
  #endif

    float4 cDiffuseFOWFactor = cFOWFactor;

    [branch] if (SC2_AXIS(b_useBlendAdd))
    {
        float blendFactor = cResult.a;
        cDeferredDiffuse = lerp(cDeferredDiffuse.rgb,
            lerp(cDeferredDiffuse.rgb, 1.0f, blendFactor) * cFOWFactor.rgb, cFOWFactor.a);
        return float4(lerp(cResult.rgb,
            lerp(cResult.rgb, 1.0f, blendFactor) * cFOWFactor.rgb, cFOWFactor.a), cResult.a);
    }
    [branch] if (SC2_AXIS(b_FOWAdditiveScale))
    {
        cDeferredDiffuse = lerp(cDeferredDiffuse.rgb,
            cDeferredDiffuse.rgb * cDiffuseFOWFactor.rgb, cDiffuseFOWFactor.a);
        return float4(lerp(cResult.rgb,
            cResult.rgb * cFOWFactor.rgb, cFOWFactor.a), cResult.a);
    }
    cDeferredDiffuse = lerp(cDeferredDiffuse.rgb, cDiffuseFOWFactor.rgb, cDiffuseFOWFactor.a);
    return float4(lerp(cResult.rgb, cFOWFactor.rgb, cFOWFactor.a), cResult.a);
}
