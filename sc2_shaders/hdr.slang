// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026, Fernando Sahmkow
// See LICENSE in the repository root for full terms.
//
// HDR family (hdr.fx + pstonemap.fx) — VS + PS.
//
// The tonemapping resolve: sample the HDR scene target, optionally add bloom,
// run the luminance through one of five tone curves, optionally push the result
// through a 3D colour-grading LUT, and stash an FXAA luma in alpha.
//
// `HDR_BuildSection` @0x102b52c40 registers TEN own axes and links NO shared
// sub-sections — the ten match hdr.fx's ten `b_*` references exactly:
//
//     b_iHDRPass              @104 w3   (only MODE_TONEMAP=3 ships)
//     b_useBloom              @107 w1
//     b_useWhiteShift         @108 w1
//     b_addMotionBlur         @109 w1   (dead — its block is commented out)
//     b_useColorMapping       @110 w1
//     b_iColorMapSize         @111 w5   (16 whenever colour mapping is on)
//     b_iUse8BitHDRPostProcess@116 w1
//     b_iScaleBloom           @117 w1
//     b_iAALumaMode           @118 w3
//     b_iOperator             @121 w3
//
// Own IO (`VertexOutput`), so `inject_preamble: false`.  hdr.fx does still call
// the engine's zero-argument `InitShader()`; the reference leg gets the empty
// definition from `sc2_interp.gen_initshader_stub`, and the slang port simply
// has nothing to initialise.
//
// Retail set: 1 VS perm, 320 PS perms (2^5 booleans x 5 operators x 2 luma modes).

implementing sc2_shaders;

// SC2_AXIS comes from here -- one macro for the whole module now that the payload
// layout is global; the per-axis SC2_PERMDATA_ flag does the selecting.
#include "types/perm_cb.slangh"

#ifndef b_iHDRPass
#define b_iHDRPass 0
#endif
#ifndef b_useBloom
#define b_useBloom 0
#endif
#ifndef b_useWhiteShift
#define b_useWhiteShift 0
#endif
#ifndef b_addMotionBlur
#define b_addMotionBlur 0
#endif
#ifndef b_useColorMapping
#define b_useColorMapping 0
#endif
#ifndef b_iColorMapSize
#define b_iColorMapSize 0
#endif
#ifndef b_iUse8BitHDRPostProcess
#define b_iUse8BitHDRPostProcess 0
#endif
#ifndef b_iScaleBloom
#define b_iScaleBloom 0
#endif
#ifndef b_iAALumaMode
#define b_iAALumaMode 0
#endif
#ifndef b_iOperator
#define b_iOperator 0
#endif

// hdr.fx FXAA luma modes.
#define HDR_LUMA_OFF        0
#define HDR_LUMA_OPTIMAL    1
#define HDR_LUMA_EQUAL      2
#define HDR_LUMA_IGNOREBLUE 3

// hdr.fx pass modes (only MODE_TONEMAP survives — the three downsample arms are
// commented out in the shipped source).
#define HDR_MODE_DOWNSAMPLE_LUM_LOG 0
#define HDR_MODE_DOWNSCALE_LUM      1
#define HDR_MODE_DOWNSAMPLE_LUM_EXP 2
#define HDR_MODE_TONEMAP            3

// pstonemap.fx tone curves.
#define REINHARD_OPERATOR      0
#define EXPONENTIONAL_OPERATOR 1
#define LOGARITHMIC_OPERATOR   2
#define LINEAR_OPERATOR        3
#define FILMIC_OPERATOR        4

static const float3 c_vLuminanceWeights = float3(0.35f, 0.5f, 0.125f);
static const float  c_fHDREpsilon = 0.0001f;

public struct HDRVSInput
{
    float3 vPosition : POSITION;
    float4 cColor    : COLOR0;
    float2 vUV       : TEXCOORD0;
};

// hdr.fx : VertexOutput.
public struct HDRTransport
{
    float4 vPosition : SV_Position;
    float2 vUV0      : TEXCOORD0;
};

public struct HDRPSInput
{
    float4 vPosition : SV_Position;
    float2 vUV0      : TEXCOORD0;
};

//--------------------------------------------------------------------------------
// pstonemap.fx : Tonemap.  Scale and gain are dead in the shipped source.
float hdrTonemap(float fSrcL)
{
    float fTonemappedL = 0.f;
    // Five operators over the same scalar: no texture, no output, no discard and no
    // loop differs between them, so the operator is a constant-buffer field and this
    // ladder is a scalar branch.  It alone is a 5x multiplier on the retail grid.
    const int sc2Operator = SC2_AXIS(b_iOperator);
    [branch] if (sc2Operator == REINHARD_OPERATOR)
    {
        fTonemappedL = pow(fSrcL, psGlobals.fContrast) * psGlobals.fExposure;
        float fTonemappedLScaled =
            fTonemappedL * (1 + (fTonemappedL / (psGlobals.fWhitePoint * psGlobals.fWhitePoint)));
        fTonemappedL = fTonemappedLScaled / (1 + fTonemappedL);
    }
    else [branch] if (sc2Operator == EXPONENTIONAL_OPERATOR)
    {
        fTonemappedL = 1.0f - exp(-psGlobals.fExposure * pow(fSrcL, psGlobals.fContrast));
    }
    else [branch] if (sc2Operator == LOGARITHMIC_OPERATOR)
    {
        fTonemappedL = log(pow(fSrcL, psGlobals.fContrast) + 1) / log(psGlobals.fExposure + 1);
    }
    else [branch] if (sc2Operator == LINEAR_OPERATOR)
    {
        fTonemappedL = pow(fSrcL, psGlobals.fContrast);
    }
    else [branch] if (sc2Operator == FILMIC_OPERATOR)
    {
        // fSrcL = pow(fSrcL, p_fContrast) is commented out upstream (always 1.0f).
        fTonemappedL = pow(fSrcL, 2.2f) * psGlobals.fExposure;
        float x = fTonemappedL;
        fTonemappedL = (x * (12.3f * x + .5f)) / (x * (12.3f * x + 4.0f) + 0.02f);
    }
    return fTonemappedL;
}

[shader("vertex")]
public HDRTransport hdr_vs_main(HDRVSInput input)
{
    HDRTransport output;
    output.vPosition = float4(input.vPosition, 1.0f);
    output.vUV0 = input.vUV;
    return output;
}

[shader("pixel")]
public float4 hdr_ps_main(HDRPSInput input) : SV_Target0
{
    // The three luminance-downsample passes are commented out in hdr.fx, so the
    // function body is a single `if (b_iHDRPass == MODE_TONEMAP)` with NO else —
    // every other pass value falls off the end returning an undefined value.
    // Retail only ever ships MODE_TONEMAP.
#if (b_iHDRPass == HDR_MODE_TONEMAP)
    float3 cColor = p_sSrcMap_t.Sample(p_sSamp, input.vUV0.xy).rgb;
    [branch] if (SC2_AXIS(b_iUse8BitHDRPostProcess))
        cColor.rgb *= psGlobals.fHDRScaling;
  #if b_useBloom
    {
        float3 bloom = p_sBloomMap_t.Sample(p_sSamp, input.vUV0.xy).rgb;
        // Exaggerate bloom to compensate for capped HDR.
        [branch] if (SC2_AXIS(b_iScaleBloom))
            bloom *= 2.0f;
        cColor += bloom;
    }
  #endif

    float fSrcL = dot(c_vLuminanceWeights, cColor);

    // White shift.
    [branch] if (SC2_AXIS(b_useWhiteShift) && (SC2_AXIS(b_iOperator) != REINHARD_OPERATOR))
    {
        float fRatio = saturate((fSrcL - 1.5f) / fSrcL);
        cColor = lerp(cColor, float3(fSrcL, fSrcL, fSrcL), fRatio);
    }

    float fToneMappedL = hdrTonemap(fSrcL);

    // Saturation removed upstream (`pow(cColor / fSrcL, p_fSaturation)`).
    cColor = fToneMappedL * cColor / fSrcL;

  #if b_useColorMapping
    cColor.rgb += 0.5f / b_iColorMapSize;
    cColor.rgb = p_sColorMap_t.Sample(p_sSamp, saturate(cColor.rgb)).rgb;
  #endif

    float fAlpha = 1.0;
    const int sc2AALuma = SC2_AXIS(b_iAALumaMode);
    [branch] if (sc2AALuma != HDR_LUMA_OFF)
    {
    // Store luma in alpha — see the FXAA description in psantialiasing.fx.
    //
    // NOTE the dangling `else` upstream: the source reads
    //     if (mode == LUMA_EQUAL)      fAlpha = (r+g+b)/3;
    //     if (mode == LUMA_IGNOREBLUE) fAlpha = r*.337 + g*.663;
    //     else                         fAlpha = dot(cColor, luma709);
    // so the else binds to the SECOND if, not the first — LUMA_EQUAL computes the
    // average and then immediately overwrites it with the dot.  Transcribed as
    // written; only LUMA_OPTIMAL ships anyway.
        [branch] if (sc2AALuma == HDR_LUMA_EQUAL)
            fAlpha = (cColor.r + cColor.g + cColor.b) / 3.0;
        [branch] if (sc2AALuma == HDR_LUMA_IGNOREBLUE)
            fAlpha = cColor.r * 0.337 + cColor.g * 0.663;
        else
            fAlpha = dot(cColor, float3(0.299, 0.587, 0.114));
    }

    return float4(cColor, fAlpha);
#else
    return float4(0, 0, 0, 0);
#endif
}
