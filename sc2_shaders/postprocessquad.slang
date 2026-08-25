// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026, Fernando Sahmkow
// See LICENSE in the repository root for full terms.
//
// PostProcessQuad family (postprocessquad.fx + vspostprocess.fx + pspostprocess.fx
// + psdof.fx + psssao.fx + psantialiasing.fx + pshalo.fx) — VS + PS.
//
// The full-screen post-process quad: fifteen independent modes behind one axis,
// wrapped in a shared output-conditioning tail (channel filters, scale/add, clamp,
// post-mask, alpha test).  `PostProcessQuad_BuildSection` @0x102b63770 registers
// THIRTY-SIX own axes spanning 79 bits — the widest own-axis block in the game —
// then links [Common] at bit 184.
//
// Modes 7 (ConvertToGradient) and 8 (DepthEncode) never ship, but every mode is
// transcribed; they cost nothing once b_iMode selects an arm.
//
// Like DeferredLight this is an own ROOT on the SHARED interpolant transport: it
// calls InitShader(vertOut) and carries UV, the per-tap GaussianBlurSample array,
// DownscaleUV0/1 and ScreenPos.  So `inject_preamble` stays true and the family is
// in sc2_shaders_cfg._SHARED_TRANSPORT.  It is the ONLY root that makes
// GaussianBlurSample live, which is why sc2_interp.gen_preamble had never had to
// emit that array before.
//
// The many never-selected axes (b_iDOFDebug, b_iSSAOOutput, b_iSSAOEncodedDepth,
// b_iSSAOOptimizedSampleDelta, b_iSSAOBlockerLookup) are all 0 across the retail
// set; their arms are transcribed anyway and fold out.
//
// Retail set: 23 VS perms, 1031 PS perms.

implementing sc2_shaders;
#include "types/perm_cb.slangh"

// ---- own axes ---------------------------------------------------------------
#ifndef b_iMode
#define b_iMode 0
#endif
#ifndef b_iScaleOutput
#define b_iScaleOutput 0
#endif
#ifndef b_iAddToOutput
#define b_iAddToOutput 0
#endif
#ifndef b_iPostMaskModulate
#define b_iPostMaskModulate 0
#endif
#ifndef b_iTakeMin
#define b_iTakeMin 0
#endif
#ifndef b_iClampNegativeOutput
#define b_iClampNegativeOutput 0
#endif
#ifndef b_iOutputChannelFilter
#define b_iOutputChannelFilter 0
#endif
#ifndef b_iInputChannelFilter
#define b_iInputChannelFilter 0
#endif
#ifndef b_iTapCount
#define b_iTapCount 0
#endif
#ifndef b_iBlurAlphaOnly
#define b_iBlurAlphaOnly 0
#endif
#ifndef b_iBlurAllChannels
#define b_iBlurAllChannels 0
#endif
#ifndef b_iInteriorBlur
#define b_iInteriorBlur 0
#endif
#ifndef b_iLayeredBlur
#define b_iLayeredBlur 0
#endif
#ifndef b_iWeightedBlur
#define b_iWeightedBlur 0
#endif
#ifndef b_iUseSeparateBlurMap
#define b_iUseSeparateBlurMap 0
#endif
#ifndef b_iDOFDotBlend
#define b_iDOFDotBlend 0
#endif
#ifndef b_iCorrectDOFForDepth
#define b_iCorrectDOFForDepth 0
#endif
#ifndef b_iConstrainToViewport
#define b_iConstrainToViewport 0
#endif
#ifndef b_iDepthScaledBlur
#define b_iDepthScaledBlur 0
#endif
#ifndef b_iDOFDebug
#define b_iDOFDebug 0
#endif
#ifndef b_iHaloBlurMode
#define b_iHaloBlurMode 0
#endif
#ifndef b_iSSAOEncodedDepth
#define b_iSSAOEncodedDepth 0
#endif
#ifndef b_iSSAOOptimizedSampleDelta
#define b_iSSAOOptimizedSampleDelta 0
#endif
#ifndef b_iSSAOBlockerLookup
#define b_iSSAOBlockerLookup 0
#endif
#ifndef b_iSSAOOutput
#define b_iSSAOOutput 0
#endif
#ifndef b_iUseSeparateDetailSSAO
#define b_iUseSeparateDetailSSAO 0
#endif
#ifndef b_iParticleScreenTexture
#define b_iParticleScreenTexture 0
#endif
#ifndef b_useTransform
#define b_useTransform 0
#endif
#ifndef b_iAAMode
#define b_iAAMode 0
#endif
#ifndef b_PostProcessFastFOW
#define b_PostProcessFastFOW 0
#endif
#ifndef b_iHaloRastermode
#define b_iHaloRastermode 0
#endif
#ifndef b_iSampleCount
#define b_iSampleCount 0
#endif
#ifndef b_iSampleInterpolantCount
#define b_iSampleInterpolantCount 0
#endif
#ifndef b_iSampleCountPS
#define b_iSampleCountPS 0
#endif
#ifndef b_iShowSingleTap
#define b_iShowSingleTap 0
#endif
#ifndef b_iSSAOSampleCount
#define b_iSSAOSampleCount 0
#endif
// From the linked Common sub-section.  (Each `__include`d file has its own
// preprocessor state, so the depth-encoding axis needs its default here too.)
#ifndef b_iAlphaTest
#define b_iAlphaTest 0
#endif
#ifndef b_iInEncoding
#define b_iInEncoding 0
#endif

// pspostprocess.fx modes.
#define MODE_Copy                0
#define MODE_SetValue            1
#define MODE_GaussianBlur        2
#define MODE_Downscale           3
#define MODE_COC                 4
#define MODE_DOF                 5
#define MODE_SSAO                6
#define MODE_ConvertToGradient   7
#define MODE_DepthEncode         8
#define MODE_ScreenSpaceParticle 9
#define MODE_AntiAliasing        10
#define MODE_FOWSmoothing        11
#define MODE_HaloCopy            12
#define MODE_HaloPrepare         13
#define MODE_HaloBlur            14

#define CHANNEL_ALL   0
#define CHANNEL_RED   1
#define CHANNEL_GREEN 2
#define CHANNEL_BLUE  3
#define CHANNEL_ALPHA 4

// psdof.fx debug modes.
#define DOFDEBUG_NODOF           1
#define DOFDEBUG_SMALL           2
#define DOFDEBUG_MEDIUM          3
#define DOFDEBUG_LARGE           4
#define DOFDEBUG_COC             5
#define DOFDEBUG_UNBLURREDCOC    6
#define DOFDEBUG_BLURREDCOC      7
#define DOFDEBUG_DOWNSCALEDDEPTH 8
#define DOFDEBUG_WEIGHTS         9
#define DOFDEBUG_ALPHA           10

// psssao.fx outputs.
#define SSAO_OUTPUT_MIN     0
#define SSAO_OUTPUT_STATIC  1
#define SSAO_OUTPUT_DYNAMIC 2

// pspostprocess.fx halo modes.
#define HALOBLUR_GAUSSIAN  0
#define HALOBLUR_LINEAR    1
#define HALOBLUR_QUADRATIC 2
#define HALOBLUR_SHARP     3

#define HALO_ADD   0
#define HALO_MUL   1
#define HALO_ALPHA 2

#define HALO_RADIUS_MULTIPLIER 32.0

// psantialiasing.fx FXAA modes + the QUALITY-3 step table.
#define AAMODE_OFF           0
#define AAMODE_FXAA3         1
#define AAMODE_SUBPIXEL      2
#define AAMODE_EDGEDIRECTION 3

#define FXAA_SAMPLES 8
#define FXAA_STEP0 1.0
#define FXAA_STEP1 1.5
#define FXAA_STEP2 2.0
#define FXAA_STEP3 2.0
#define FXAA_STEP4 2.0
#define FXAA_STEP5 2.0
#define FXAA_STEP6 4.0
#define FXAA_STEP7 8.0

// pscommon.fx (redeclared — see ps/ps_common.slang on why the enum cannot be
// inherited through __include).
#define PPQ_ENCODING_FLOAT 0

// ---- IO ---------------------------------------------------------------------
// postprocessquad.fx : Input.
public struct PostProcessQuadVSInput
{
    float4 vPosition : POSITION;
    float4 cColor    : COLOR0;
    float2 vUV0      : TEXCOORD0;
};

// The engine-generated `VertexTransport`: HPos, the sorted live scalars
// (DownscaleUV0/1), then the UV array, then the GaussianBlurSample array — exactly
// sc2_interp.gen_preamble's field order.
public struct PostProcessQuadTransport
{
    float4 HPos : SV_Position;
#if SC2_HAS_DownscaleUV0
    float4 m_DownscaleUV0 : SC2_SEM_DownscaleUV0;
#endif
#if SC2_HAS_DownscaleUV1
    float4 m_DownscaleUV1 : SC2_SEM_DownscaleUV1;
#endif
#if SC2_HAS_UV
    float4 m_UV[SC2_UV_COUNT] : SC2_SEM_UV;
#endif
#if SC2_HAS_GaussianBlurSample
    float4 m_GaussianBlurSample[SC2_GBS_COUNT] : SC2_SEM_GaussianBlurSample;
#endif
};

public struct PostProcessQuadPSInput
{
    float4 HPos : SV_Position;
#if SC2_HAS_DownscaleUV0
    float4 m_DownscaleUV0 : SC2_SEM_DownscaleUV0;
#endif
#if SC2_HAS_DownscaleUV1
    float4 m_DownscaleUV1 : SC2_SEM_DownscaleUV1;
#endif
#if SC2_HAS_UV
    float4 m_UV[SC2_UV_COUNT] : SC2_SEM_UV;
#endif
#if SC2_HAS_GaussianBlurSample
    float4 m_GaussianBlurSample[SC2_GBS_COUNT] : SC2_SEM_GaussianBlurSample;
#endif

    float4 UV(int i)
    {
#if SC2_HAS_UV
        return m_UV[i];
#else
        return float4(0, 0, 0, 0);
#endif
    }
    float4 GaussianBlurSample(int i)
    {
#if SC2_HAS_GaussianBlurSample
        return m_GaussianBlurSample[i];
#else
        return float4(0, 0, 0, 0);
#endif
    }
    float4 DownscaleUV0()
    {
#if SC2_HAS_DownscaleUV0
        return m_DownscaleUV0;
#else
        return float4(0, 0, 0, 0);
#endif
    }
    float4 DownscaleUV1()
    {
#if SC2_HAS_DownscaleUV1
        return m_DownscaleUV1;
#else
        return float4(0, 0, 0, 0);
#endif
    }
};

// =============================================================================
// Vertex shader (vspostprocess.fx + postprocessquad.fx)
// =============================================================================

// postprocessquad.fx : EmitPostProcessQuadHPos.
float4 ppqEmitHPos(PostProcessQuadVSInput vertIn)
{
#if b_useTransform
    return mul(vertIn.vPosition, vsGlobals.mTransform);
#else
    return vertIn.vPosition;
#endif
}

// postprocessquad.fx : EmitPostProcessQuadUV — the emitter index is ignored.
float4 ppqEmitUV(PostProcessQuadVSInput vertIn, int iIndex)
{
    return float4(vertIn.vUV0, 0, 1);
}

// vspostprocess.fx : EmitGaussianBlurSample — two taps per interpolant.
float4 ppqEmitGaussianBlurSample(PostProcessQuadVSInput vertIn, int iIndex)
{
    float4 vSample = vertIn.vUV0.xyxy;
    vSample.xy += vsGlobals.vBlurOffsets[iIndex * 2 + 1];
    vSample.zw += vsGlobals.vBlurOffsets[iIndex * 2 + 2];
    return vSample;
}

// vspostprocess.fx : EmitDownscaleUV0/1 — the 4x4 bilinear downscale taps, placed
// exactly halfway between neighbouring texel centres.
float4 ppqEmitDownscaleUV0(PostProcessQuadVSInput vertIn)
{
    float2 vRecipSrcSize = 1.0f / vsGlobals.vSrcSize;
    return vertIn.vUV0.xyxy + (-0.5 + float4(-1, -1, 1, -1)) * vRecipSrcSize.xyxy;
}

float4 ppqEmitDownscaleUV1(PostProcessQuadVSInput vertIn)
{
    float2 vRecipSrcSize = 1.0f / vsGlobals.vSrcSize;
    return vertIn.vUV0.xyxy + (-0.5 + float4(-1, 1, 1, 1)) * vRecipSrcSize.xyxy;
}

[shader("vertex")]
public PostProcessQuadTransport postprocessquad_vs_main(PostProcessQuadVSInput vertIn)
{
    PostProcessQuadTransport vertOut;
    vertOut.HPos = ppqEmitHPos(vertIn);

    // postprocessquad.fx hard-defines b_iUVEmitterCount to 1 for the vertex stage
    // (":HACK:"), so this loop is a single emitter.
#if SC2_HAS_UV
    [unroll]
    for (int i = 0; i < 1; i++)
        vertOut.m_UV[i] = ppqEmitUV(vertIn, i);
#endif

#if (b_iSampleCountPS == 0 && SC2_HAS_GaussianBlurSample)
    [unroll]
    for (int j = 0; j < b_iSampleInterpolantCount; j++)
        vertOut.m_GaussianBlurSample[j] = ppqEmitGaussianBlurSample(vertIn, j);
#endif

#if SC2_HAS_DownscaleUV0
    vertOut.m_DownscaleUV0 = ppqEmitDownscaleUV0(vertIn);
#endif
#if SC2_HAS_DownscaleUV1
    vertOut.m_DownscaleUV1 = ppqEmitDownscaleUV1(vertIn);
#endif

    return vertOut;
}

// =============================================================================
// Pixel shader
// =============================================================================

// pspostprocess.fx : PostProcessCopy.
float4 ppqCopy(PostProcessQuadPSInput vertOut)
{
    return p_sSrcMap_t.Sample(p_sSamp, vertOut.UV(0).xy);
}

// pspostprocess.fx : PostProcessSetValue.
float4 ppqSetValue(PostProcessQuadPSInput vertOut)
{
    return psGlobals.vValue;
}

// pspostprocess.fx : PostProcessDownscale — 4x4 bilinear downscale.
float4 ppqDownscale(PostProcessQuadPSInput vertOut)
{
    return (p_sSrcMap_t.Sample(p_sSamp, vertOut.DownscaleUV0().xy)
          + p_sSrcMap_t.Sample(p_sSamp, vertOut.DownscaleUV0().zw)
          + p_sSrcMap_t.Sample(p_sSamp, vertOut.DownscaleUV1().xy)
          + p_sSrcMap_t.Sample(p_sSamp, vertOut.DownscaleUV1().zw))
         * float4(0.25f, 0.25f, 0.25f, 0.25f);
}

// pspostprocess.fx : PostProcessConvertToGradient (never shipped).
float4 ppqConvertToGradient(PostProcessQuadPSInput vertOut)
{
    return float4(lerp(psGlobals.cGradientStartColor, psGlobals.cGradientEndColor,
                       saturate(p_sSrcMap_t.Sample(p_sSamp, vertOut.UV(0).xy).a
                                / psGlobals.fGradientScale)), 1.0f);
}

// pspostprocess.fx : PostProcessDepthEncode (never shipped).
float4 ppqDepthEncode(PostProcessQuadPSInput vertOut)
{
    float4 vResult;
    float fDepth = saturate(p_sSrcMap_t.Sample(p_sSamp, vertOut.UV(0).xy).a / 10.0f);
    vResult.rg = p_sDepthEncodeRG_t.Sample(p_sSamp, float2(fDepth, fDepth * 32.0f)).rg;
    vResult.ba = p_sDepthEncodeB_t.Sample(p_sSamp, float2(fDepth * 2048.0f, fDepth * 2048.0f)).bb;
    return vResult;
}

// p_fBlurWeights read through a function, NOT indexed at the call site.
//
// slangc's WGSL backend segfaults (access violation, no diagnostic) whenever an
// element of a SCALAR array in a constant buffer is passed straight into a call —
// `float p_fBlurWeights[N]` trips it, `float2 p_vBlurOffsets[N]` right next to it
// does not.  Reproduced on slangc 2026.1 / 2026.5.2 / 2026.8.1; D3D, SPIR-V and
// Metal all emit it fine.  Doing arithmetic on the value first does not help (the
// fold puts the raw load back); taking the INDEX here and returning the load's
// RESULT does.  Keep this a plain function — `[ForceInline]` reintroduces the
// crash — and it costs nothing on the D3D path: fxc inlines it straight back, so
// the DXBC the validation harness checks stays byte-identical.
float ppqBlurWeight(int i)
{
    return psGlobals.fBlurWeights[i];
}

// pspostprocess.fx : GaussSample — one weighted tap, with the optional viewport
// clamp / interior / weighted / layered rejections.
float ppqGaussSample(float2 vTapUV, float fWeight, inout float4 vResult, float4 vNormalDepth)
{
    [branch] if (SC2_AXIS(b_iConstrainToViewport))
    {
        vTapUV = max(vTapUV, psGlobals.vPostProcessViewport.xy);
        vTapUV = min(vTapUV, psGlobals.vPostProcessViewport.zw);
    }

    float4 cValue;
#if b_iUseSeparateBlurMap
    cValue = p_sSeparateBlurMap_t.Sample(p_sSamp, vTapUV);
#else
    cValue = p_sSrcMap_t.Sample(p_sSamp, vTapUV);
#endif

#if b_iInteriorBlur
    {
        float4 vSampleNormalDepth = psSampleNormalDepth(p_sNormalDepthMap_t, vTapUV);
        if (dot(vSampleNormalDepth.rgb, vNormalDepth.rgb) < 0.9f
            || abs(vSampleNormalDepth.a - vNormalDepth.a)
               > ((b_iInEncoding == PPQ_ENCODING_FLOAT) ? 0.01f : 0.1f))
            fWeight = 0.0f;
    }
#endif
#if b_iWeightedBlur
    fWeight *= p_sWeightMap_t.Sample(p_sSamp, vTapUV).a;
#endif
#if b_iLayeredBlur
    {
        float vDepth = psSampleNormalDepth(p_sNormalDepthMap_t, vTapUV).a;
        if (vNormalDepth.a < vDepth)
            fWeight = 0.0f;
    }
#endif

    vResult += cValue * fWeight;
    return fWeight;
}

// pspostprocess.fx : PostProcessGaussianBlur.  The whole body is gated on the mode
// upstream too ("INTERPOLANT_GaussianBlurSample may not be present").
float4 ppqGaussianBlur(PostProcessQuadPSInput vertOut)
{
#if (b_iMode == MODE_GaussianBlur)
    float2 vCenterTap = vertOut.UV(0).xy;
    float4 cValue     = p_sSrcMap_t.Sample(p_sSamp, vCenterTap.xy);
    float4 cResult    = cValue * psGlobals.fBlurWeights[0];
    float fTotalWeight = psGlobals.fBlurWeights[0];

    float4 vNormalDepth = float4(0, 0, 0, 0);
  #if b_iInteriorBlur
    vNormalDepth = psSampleNormalDepth(p_sNormalDepthMap_t, vCenterTap.xy);
  #else
    vNormalDepth.a = psSampleNormalDepth(p_sNormalDepthMap_t, vCenterTap.xy).a;
    #if (b_iLayeredBlur && (b_iInEncoding != PPQ_ENCODING_FLOAT))
    vNormalDepth.a += 0.01f;    // Slack for layered blur.
    #endif
  #endif

    float depthScale = 1.f;
  #if b_iDepthScaledBlur
    {
        float depth = p_sWeightMap_t.Sample(p_sSamp, vCenterTap.xy).r;
        float2 s = vCenterTap.xy * 2.f - 1.f;

        // Back-transform to recover the original depth from the plane.
        float x0 = s.x * psGlobals.vBlurDepthCameraConstants.x;
        float z0 = s.y * psGlobals.vBlurDepthCameraConstants.y;
        float w = x0 + z0 + psGlobals.vBlurDepthCameraConstants.z - depth;

        if (w <= 0.f)
        {
            depthScale = 1.f;   // Depth is infinite.
        }
        else
        {
            // Use depth to scale the size of the blur circle.
            float invW = 1.f / w;
            float z0b = depth * w;
            float blurRadius = z0b * psGlobals.vBlurDepthRadiusConstants.z;  // view-space units

            // Convert the blur radius to post-projection space units.
            depthScale = saturate(abs(blurRadius
                * (psGlobals.vBlurOffsets[1].x == 0 ? psGlobals.vBlurDepthRadiusConstants.y
                                                    : psGlobals.vBlurDepthRadiusConstants.x)
                * invW));
        }
    }
  #endif

  #if b_iSampleCountPS
    // Sample offsets stored in uniforms.  NOT unrolled: the .fx's [unroll] is
    // `#if __bsl__` only, and forcing it changes fxc's mad-accumulation order.
    for (int i = 1; i <= b_iSampleCount; i++)
    {
        if (b_iShowSingleTap >= b_iSampleCount * 2 || (b_iShowSingleTap % b_iSampleCount) == i - 1)
        {
      #if b_iDepthScaledBlur
            fTotalWeight += ppqGaussSample(vCenterTap + depthScale * psGlobals.vBlurOffsets[i],
                                           ppqBlurWeight(i), cResult, vNormalDepth);
      #else
            fTotalWeight += ppqGaussSample(vCenterTap + psGlobals.vBlurOffsets[i],
                                           ppqBlurWeight(i), cResult, vNormalDepth);
      #endif
        }
    }
  #else
    // Sample positions stored in interpolants (same: no forced unroll).
    for (int i = 0; i < b_iSampleInterpolantCount; i++)
    {
        if (b_iShowSingleTap >= b_iSampleCount * 2 || (b_iShowSingleTap % b_iSampleCount) == i * 2)
            fTotalWeight += ppqGaussSample(vertOut.GaussianBlurSample(i).xy,
                                           ppqBlurWeight(i * 2 + 1), cResult, vNormalDepth);

        if (b_iShowSingleTap >= b_iSampleCount * 2 || (b_iShowSingleTap % b_iSampleCount) == i * 2 + 1)
            fTotalWeight += ppqGaussSample(vertOut.GaussianBlurSample(i).zw,
                                           ppqBlurWeight(i * 2 + 2), cResult, vNormalDepth);
    }
  #endif

  #if (b_iInteriorBlur || b_iWeightedBlur || b_iLayeredBlur)
    cResult *= 1.0f / fTotalWeight;
  #endif

    [branch] if (!SC2_AXIS(b_iBlurAllChannels))
    {
        [branch] if (SC2_AXIS(b_iBlurAlphaOnly))
            cResult.rgb = 0;        // Discard colour.
        else
            cResult.a = cValue.a;   // Use the original alpha.
    }

  #if b_iDepthScaledBlur
    cResult.rgb = lerp(cValue.rgb, cResult.rgb, cValue.a);
    cResult.a = cValue.a;
  #endif

    return cResult;
#else
    return float4(0, 0, 0, 0);
#endif
}

// pspostprocess.fx : PostProcessScreenSpaceParticle.
float4 ppqScreenSpaceParticle(PostProcessQuadPSInput vertOut)
{
    float4 vResult = p_sSrcMap_t.Sample(p_sSamp, vertOut.UV(0).xy);

    // Only standard screen-space particles; the frame-buffer read is commented out.
    float fogDensity = 0;

    float4 vParticleTexture = float4(0, 0, 0, 0);
#if b_iParticleScreenTexture
    {
        float2 vUv = mul(psGlobals.mParticleTextureTransform,
                         float4(vertOut.UV(0).x, vertOut.UV(0).y, 0.0f, 1.0f));
        vParticleTexture = p_sParticleTexture_t.Sample(p_sSamp, vUv.xy);
    }
#endif

    vResult.xyz = vResult.xyz * psGlobals.vParticleAmbientLight.xyz + vResult.xyz;

#if b_iParticleScreenTexture
    vResult.xyz = lerp(vParticleTexture.xyz, psGlobals.vParticleFogColor.xyz, fogDensity);
#endif

    vResult.a = step(psGlobals.fParticleThreshold, vResult.a);
    return vResult;
}

// pspostprocess.fx : PostProcessFOWSmoothing — a 5-tap cross, centre double-weighted.
float4 ppqFOWSmoothing(PostProcessQuadPSInput vertOut)
{
    float4 vResult;
    vResult  = 2 * p_sSrcMap_t.Sample(p_sSamp, vertOut.UV(0).xy);
    vResult += p_sSrcMap_t.Sample(p_sSamp, vertOut.UV(0).xy
                   + float2(0, -psGlobals.fFOWSmoothing / psGlobals.vSrcSize.y));
    vResult += p_sSrcMap_t.Sample(p_sSamp, vertOut.UV(0).xy
                   + float2(0,  psGlobals.fFOWSmoothing / psGlobals.vSrcSize.y));
    vResult += p_sSrcMap_t.Sample(p_sSamp, vertOut.UV(0).xy
                   + float2(-psGlobals.fFOWSmoothing / psGlobals.vSrcSize.x, 0));
    vResult += p_sSrcMap_t.Sample(p_sSamp, vertOut.UV(0).xy
                   + float2( psGlobals.fFOWSmoothing / psGlobals.vSrcSize.x, 0));
    vResult /= 6.0f;
#if b_PostProcessFastFOW
    vResult.a = dot(saturate(2.0f * vResult.rgb), float3(0.375f, 0.3125f, 0.3125f));
#endif
    return vResult;
}

// pshalo.fx : GetHaloColor.
float4 ppqGetHaloColor(float2 vUV, float4 cColor, float fStrobeOffset,
                       float fStrobeSizeRcp, float fStrobeFalloff)
{
    float4 cResult = cColor;
    float Dist = (abs(vUV.y - fStrobeOffset) * fStrobeSizeRcp);
    cResult.rgb *= saturate(max(1.0 - pow(Dist, fStrobeFalloff), 0.0) + cColor.a);
    cResult.a = 1.0;
    return cResult;
}

float4 ppqHaloCopy(PostProcessQuadPSInput vertOut)
{
    float2 vUV = vertOut.UV(0).xy;
    float4 cResult = p_sSrcMap_t.Sample(p_sSamp, vUV);
    // The GL arm uses a 0.01 threshold; D3D tests against 0.
    return ppqGetHaloColor(vUV, psGlobals.cHaloColor, psGlobals.vStrobeParams.x,
                           psGlobals.vStrobeParams.y, psGlobals.vStrobeParams.z)
         * ((cResult.a > 0) ? 1.0 : 0.0);
}

// pspostprocess.fx : HaloSample.
float4 ppqHaloSample(float2 vUV)
{
    [branch] if (SC2_AXIS(b_iConstrainToViewport))
    {
        vUV = max(vUV, psGlobals.vPostProcessViewport.xy);
        vUV = min(vUV, psGlobals.vPostProcessViewport.zw);
    }
    return p_sSrcMap_t.Sample(p_sSamp, vUV);
}

void ppqHaloPrepareSingle(float4 cInput, float fOffset, inout float4 cResult)
{
    [branch] if (SC2_AXIS(b_iBlurAllChannels))
    {
        float testValue = (1.0 - cInput.a) + fOffset;
        if (testValue < cResult.a)
        {
            cResult.a = testValue;
            cResult.rgb = cInput.rgb;
        }
    }
    else
    {
        float testValue = (1.0 - cInput.r) + fOffset;
        cResult.r = min(cResult.r, testValue);
    }
}

float4 ppqHaloPrepare(PostProcessQuadPSInput vertOut)
{
    float2 vUV = vertOut.UV(0).xy;
    float4 cResult = p_sSrcMap_t.Sample(p_sSamp, vUV);

    [branch] if (SC2_AXIS(b_iBlurAllChannels))
        cResult.a = (1.0 - cResult.a);
    else
        cResult.r = (1.0 - cResult.r);

    float texOffset = psGlobals.vRecipSrcSizeOffset.y;
    float radius = 1.0 / HALO_RADIUS_MULTIPLIER;
    for (int i = 1; i < b_iTapCount; i++)
    {
        float4 inputSample = ppqHaloSample(vUV + float2(0, texOffset));
        ppqHaloPrepareSingle(inputSample, radius, cResult);

        inputSample = ppqHaloSample(vUV - float2(0, texOffset));
        ppqHaloPrepareSingle(inputSample, radius, cResult);

        texOffset += psGlobals.vRecipSrcSizeOffset.y;
        radius += 1.0 / HALO_RADIUS_MULTIPLIER;
    }

    return cResult;
}

void ppqHaloBlurSingle(float4 cInput, float fDimSq, inout float fMinSq, inout float4 cResult)
{
    float testSq, test;

    test = SC2_AXIS(b_iBlurAllChannels) ? cInput.a : cInput.r;
    test *= HALO_RADIUS_MULTIPLIER;
    testSq = test * test + fDimSq;

    if (testSq < fMinSq)
    {
        fMinSq = testSq;
        [branch] if (SC2_AXIS(b_iBlurAllChannels))
            cResult.rgb = cInput.rgb;
        else
            cResult.r = cInput.r;
    }
}

float4 ppqHaloBlur(PostProcessQuadPSInput vertOut)
{
    float2 vUV = vertOut.UV(0).xy;
    float4 cResult = p_sSrcMap_t.Sample(p_sSamp, vUV);
    float minSq = SC2_AXIS(b_iBlurAllChannels) ? cResult.a : cResult.r;
    minSq = minSq * minSq * HALO_RADIUS_MULTIPLIER * HALO_RADIUS_MULTIPLIER;

    // Inside the blurred object: write nothing.
    sc2_clip(minSq - 0.001);

    float offset = psGlobals.vRecipSrcSizeOffset.x;
    for (int i = 1; i < b_iTapCount; i++)
    {
        float valSq = i * i;

        float4 testValue = ppqHaloSample(vUV + float2(offset, 0));
        ppqHaloBlurSingle(testValue, valSq, minSq, cResult);

        testValue = ppqHaloSample(vUV - float2(offset, 0));
        ppqHaloBlurSingle(testValue, valSq, minSq, cResult);

        offset += psGlobals.vRecipSrcSizeOffset.x;
    }

    // Outside the max radius: write nothing.
    sc2_clip(b_iTapCount * b_iTapCount - minSq);

    // Halo mode determines the strength curve.
    float strength = 0.f;
    const int iHaloBlurMode = SC2_AXIS(b_iHaloBlurMode);
    [branch] if (iHaloBlurMode == HALOBLUR_GAUSSIAN)
    {
        float weight = exp(-minSq * psGlobals.fHaloRadiusRcp);
        strength = max(weight, 0.0);
    }
    else [branch] if (iHaloBlurMode == HALOBLUR_LINEAR)
        strength = max(1.0 - (sqrt(minSq) * psGlobals.fHaloRadiusRcp), 0.0);
    else [branch] if (iHaloBlurMode == HALOBLUR_QUADRATIC)
        strength = max(1.0 - (minSq * psGlobals.fHaloRadiusRcp), 0.0);
    else [branch] if (iHaloBlurMode == HALOBLUR_SHARP)
        strength = max(1.0 - (minSq * minSq * minSq * minSq * psGlobals.fHaloRadiusRcp), 0.0);

    [branch] if (SC2_AXIS(b_iBlurAllChannels))
        cResult.a = strength;
    else
        cResult = float4(strength, strength, strength, strength);
    return cResult;
}

// ---- psdof.fx ---------------------------------------------------------------
float ppqComputeCOC(float depth)
{
    return max(0.0f, max((depth - psGlobals.fFocusDistance) * psGlobals.vCoCWindow.x
                             + psGlobals.vCoCWindow.y,
                         (psGlobals.fFocusDistance - depth) * psGlobals.vCoCWindow.z
                             + psGlobals.vCoCWindow.w));
}

float4 ppqTex2DOffset(Texture2D<float4> t, float2 vUV, float2 vOffset, float2 vImageSize)
{
    float2 vUVFinal = vUV + vOffset * float2(1.0f / vImageSize.x, 1.0f / vImageSize.y);
    [branch] if (SC2_AXIS(b_iConstrainToViewport))
        vUVFinal = clamp(vUVFinal, psGlobals.vPostProcessViewport.xy,
                         psGlobals.vPostProcessViewport.zw);
    return t.Sample(p_sSamp, vUVFinal);
}

// A 16-tap blur: 4 bilinear taps, each covering 4 texels.
float4 ppqGetSmallBlurSample(Texture2D<float4> t, float2 vUV, float2 vImageSize)
{
    float4 cSum;
    const float weight = 4.0 / 16.0f;
    cSum = 0;
    cSum += weight * ppqTex2DOffset(t, vUV, float2( 0.5f, -1.5f), vImageSize);
    cSum += weight * ppqTex2DOffset(t, vUV, float2(-1.5f, -0.5f), vImageSize);
    cSum += weight * ppqTex2DOffset(t, vUV, float2(-0.5f,  1.5f), vImageSize);
    cSum += weight * ppqTex2DOffset(t, vUV, float2( 1.5f,  0.5f), vImageSize);
    return cSum;
}

float4 ppqCOC(PostProcessQuadPSInput vertOut)
{
    float4 vNormalDepth = psSampleNormalDepth(p_sNormalDepthMap_t, vertOut.UV(0).xy);
    return ppqComputeCOC(vNormalDepth.a);
}

float4 ppqDOF(PostProcessQuadPSInput vertOut)
{
    float2 vUV = vertOut.UV(0).xy;

    float4 vNormalDepth = psSampleNormalDepth(p_sNormalDepthMap_t, vUV);
    float4 cLarge       = p_sLargeBlurMap_t.Sample(p_sSamp, vUV);   // 1/4 size, blurred
    float fUnblurredCOC = ppqComputeCOC(vNormalDepth.a);
    float vDownscaledDepth = 0;
    float fCoC;
#if b_iCorrectDOFForDepth
    vDownscaledDepth = psSampleNormalDepth(p_sDownscaledDepth_t, vUV).a
                     + psGlobals.fCorrectDepthDOFBias;
    if (vDownscaledDepth > vNormalDepth.a)
        fCoC = fUnblurredCOC;   // Sharp object behind the downscaled depth: stay sharp.
    else
        fCoC = 2.0f * max(cLarge.a, fUnblurredCOC) - fUnblurredCOC;
#else
    fCoC = fUnblurredCOC;
#endif

    float d0 = psGlobals.vBlurRamp.x;
    float d1 = psGlobals.vBlurRamp.y;
    float d2 = psGlobals.vBlurRamp.z;
    fCoC = saturate(fCoC);

#if (b_iDOFDebug == DOFDEBUG_SMALL)
    fCoC = d0;
#elif (b_iDOFDebug == DOFDEBUG_MEDIUM)
    fCoC = d0 + d1;
#elif (b_iDOFDebug == DOFDEBUG_LARGE)
    fCoC = d0 + d1 + d2;
#endif

    float3 cSharp = p_sSrcMap_t.Sample(p_sSamp, vUV).rgb;
    float3 cSmall = ppqGetSmallBlurSample(p_sSrcMap_t, vUV, psGlobals.vSrcSize).rgb;
    float3 cMed   = p_sMediumBlurMap_t.Sample(p_sSamp, vUV).rgb;   // full size, blurred

    float3 cColor;
    float fAlpha = 1;
    float4 weights;
    [branch] if (SC2_AXIS(b_iDOFDotBlend))
    {
        // Cross-blend weights for each sample:
        //   0     -> d0        : sharp -> small blur
        //   d0    -> d0+d1     : small -> medium
        //   d0+d1 -> d0+d1+d2  : medium -> large
        float4 vLerpScale  = float4(-1 / d0, -1 / d1, -1 / d2, 1 / d2);
        float4 vLerpOffset = float4(1, (1 - d2) / d1, 1 / d2, (d2 - 1) / d2);
        weights    = saturate(fCoC * vLerpScale + vLerpOffset);
        weights.yz = min(weights.yz, 1 - weights.xy);

        cColor = weights.x * cSharp + weights.y * cSmall + weights.z * cMed
               + weights.w * cLarge.rgb;
        fAlpha = 1;
    }
    else
    {
        float3 cColorA;
        float3 cColorB;
        float t;
        weights = 0;
        if (fCoC <= d0)
        {
            t = fCoC / d0;
            cColorA = cSharp;
            cColorB = cSmall;
            weights = lerp(float4(0, 0, 0, 0), float4(0, 1, 0, 0), t);
        }
        else if (fCoC <= (d1 + d0))
        {
            t = (fCoC - d0) / d1;
            cColorA = cSmall;
            cColorB = cMed;
            weights = lerp(float4(0, 1, 0, 0), float4(0, 0, 1, 0), t);
        }
        else
        {
            t = (fCoC - d0 - d1) / d2;
            cColorA = cMed;
            cColorB = cLarge.rgb;
            weights = lerp(float4(0, 0, 1, 0), float4(0, 0, 0, 1), t);
        }

        cColor = lerp(cColorA, cColorB, t).rgb;
        fAlpha = 1;
    }

#if ((b_iDOFDebug > DOFDEBUG_LARGE) && (b_iDOFDebug != DOFDEBUG_ALPHA))
    fAlpha = 1;
#endif
#if (b_iDOFDebug == DOFDEBUG_NODOF)
    fAlpha = 0;
#elif (b_iDOFDebug == DOFDEBUG_COC)
    cColor = saturate(fCoC);
#elif (b_iDOFDebug == DOFDEBUG_UNBLURREDCOC)
    cColor = saturate(fUnblurredCOC);
#elif (b_iDOFDebug == DOFDEBUG_BLURREDCOC)
    cColor = saturate(p_sLargeBlurMap_t.Sample(p_sSamp, vUV).a);
#elif (b_iDOFDebug == DOFDEBUG_DOWNSCALEDDEPTH)
    cColor = vDownscaledDepth;
#elif (b_iDOFDebug == DOFDEBUG_WEIGHTS)
    cColor = weights.yzw;
#elif (b_iDOFDebug == DOFDEBUG_ALPHA)
    cColor = fAlpha;
    fAlpha = 1;
#endif
    return float4(cColor, fAlpha);
}

// ---- psssao.fx --------------------------------------------------------------
// NOTE: named ssaoReflect, not `reflect` — the .fx shadows the HLSL intrinsic with
// a DIFFERENT formula (it normalises, and does not negate the way `reflect` does).
float3 ssaoReflect(float3 vSample, float3 vNormal)
{
    return normalize(vSample - 2.0f * dot(vSample, vNormal) * vNormal);
}

bool ppqTestOcclusion(float3 vViewPos, float3 vSamplePointDelta, float3 vNormal,
                      out float fBlock, float fOcclusionRadius,
                      Texture2D<float4> tBlockerFunc, float2 fSampleRecip,
                      float fFullOcclusionThreshold, float fNoOcclusionThreshold,
                      float2 SSAOCenterUV, float SSAOLength)
{
    if (dot(vSamplePointDelta, vNormal.xzy) < 0)
        vSamplePointDelta = -vSamplePointDelta;

    float3 vSamplePoint = vViewPos + fOcclusionRadius * vSamplePointDelta;
    float2 vSamplePointUV;
#if b_iSSAOOptimizedSampleDelta
    // Sacrifices some contrast.  SSAO term A = reciprocal of HALF camera size times
    // (0.5, -0.5); term B = (1,-1) * (0.5,-0.5) = (0.5, 0.5).
    vSamplePointUV = (psGlobals.vSSAOTermA / vSamplePoint.z) * vSamplePoint.xy
                   + float2(0.5f, 0.5f);
#else
    vSamplePointUV = vSamplePoint.xy / vSamplePoint.z;
    vSamplePointUV = vSamplePointUV / psGlobals.vCameraSize / 0.5f;
    vSamplePointUV = vSamplePointUV + float2(1.0f, -1.0f);
    vSamplePointUV = vSamplePointUV / float2(2.0f, -2.0f);
#endif

    if (vSamplePointUV.x < 0.0f || vSamplePointUV.y < 0.0f ||
        vSamplePointUV.x > 1.0f || vSamplePointUV.y > 1.0f)
    {
        fBlock = 0.0f;
        return true;
    }

    SSAOLength += length(vSamplePointUV - SSAOCenterUV);

    float fSampleDepth;
#if b_iSSAOEncodedDepth
    fSampleDepth = decodeDepth(p_sEncodedDepth_t.Sample(p_sSamp, vSamplePointUV));
#else
    fSampleDepth = p_sEncodedDepth_t.Sample(p_sSamp, vSamplePointUV).r;
#endif

    float fDistance = vSamplePoint.z - fSampleDepth;

#if b_iSSAOBlockerLookup
    // Clipped so there is no occlusion when the sampled depth is behind the sample.
    fBlock = tBlockerFunc.Sample(p_sSamp, fDistance).r;
#else
    if (fDistance > 0.01f)
    {
        // Past this distance there is no occlusion.
        float fNoOcclusionRange = fNoOcclusionThreshold - fFullOcclusionThreshold;
        if (fDistance < fFullOcclusionThreshold)
            fBlock = 1.0f;
        else
            fBlock = max(1.0f - ((fDistance - fFullOcclusionThreshold) / fNoOcclusionRange), 0.0f);
    }
    else fBlock = 0.0f;
#endif

    return true;
}

float4 ppqSSAO(PostProcessQuadPSInput vertOut)
{
    float2 SSAOCenterUV;
    float SSAOLength;

    // VPOS_SEMANTIC is 1 on ps_5_0, so the screen position comes from SV_Position.
    float2 vViewSpaceUV = vertOut.HPos.xy * psGlobals.vRecipSrcSizeOffset.xy;
    float2 vScreenUV    = vViewSpaceUV + psGlobals.vRecipSrcSizeOffset.zw;
    float4 vNormalDepth = psSampleNormalDepth(p_sNormalDepthMap_t, vScreenUV);

    SSAOLength = 0.0f;

    float3 vNormal = normalize(mul((float3x3)psGlobals.mViewTransform, vNormalDepth.rgb));

    vViewSpaceUV = vViewSpaceUV * float2(2.0f, -2.0f);   // 0..1 -> 0..2
    vViewSpaceUV = vViewSpaceUV + float2(-1.0f, 1.0f);   // 0..2 -> -1..1
    vViewSpaceUV = vViewSpaceUV * psGlobals.vCameraSize * 0.5f;

    SSAOCenterUV = vScreenUV;

    float3 vViewPos;
#if b_iSSAOEncodedDepth
    vViewPos = float3(vViewSpaceUV.x, vViewSpaceUV.y, 1.0f)
             * decodeDepth(p_sEncodedDepth_t.Sample(p_sSamp, vScreenUV));
#else
    vViewPos = float3(vViewSpaceUV.x, vViewSpaceUV.y, 1.0f)
             * p_sEncodedDepth_t.Sample(p_sSamp, vScreenUV).r;
#endif

    // Reused for every sample instead of a perspective divide; the error is small.
    float2 fSampleRecip = psGlobals.vSSAOTermA / vViewPos.z;

    float fAdjustedOcclusionRadius = psGlobals.fOcclusionRadius * 0.15f;
    float fAdjustedDetailOcclusionRadius = psGlobals.fDetailOcclusionRadius * 0.15f;
    float fScreenOcclusionRadius = fAdjustedOcclusionRadius / vViewPos.z;
    if (fScreenOcclusionRadius > psGlobals.fSSAOLimiter)
    {
        fAdjustedOcclusionRadius = fAdjustedOcclusionRadius * psGlobals.fSSAOLimiter
                                 / fScreenOcclusionRadius;
    }

    float3 vRandomNormal = normalize(
        p_sSSAONoise_t.Sample(p_sSamp, vScreenUV * psGlobals.vSrcSize / 256.0f).xyz
        * 2.0f - 1.0f);

#if b_iUseSeparateDetailSSAO
    int iSampleCount = b_iSSAOSampleCount * 3 / 4;
#else
    int iSampleCount = b_iSSAOSampleCount;
#endif

    float fAccumBlock = 0.0f;
    int iSamples = 0;
    for (int i = 0; i < iSampleCount; i++)
    {
        float3 vSamplePointDelta = ssaoReflect(psGlobals.vSSAOSamplePoints[i], vRandomNormal);
        float fBlock;
        if (ppqTestOcclusion(vViewPos, vSamplePointDelta, vNormal, fBlock,
                             fAdjustedOcclusionRadius, p_sBlockerFunc_t, fSampleRecip,
                             psGlobals.fFullOcclusionThreshold,
                             psGlobals.fNoOcclusionThreshold, SSAOCenterUV, SSAOLength))
        {
            fAccumBlock += fBlock;
            iSamples++;
        }
    }
    fAccumBlock /= iSamples;
    float fOcclusion = 1.0f - fAccumBlock;
    fOcclusion = pow(fOcclusion, psGlobals.fOcclusionPower);

    float fResult;
#if b_iUseSeparateDetailSSAO
    {
        fAccumBlock = 0.0f;
        iSamples = 0;
        for (int j = b_iSSAOSampleCount * 3 / 4; j < b_iSSAOSampleCount; j++)
        {
            float3 vSamplePointDelta = ssaoReflect(psGlobals.vSSAOSamplePoints[j], vRandomNormal);
            float fBlock;
            if (ppqTestOcclusion(vViewPos, vSamplePointDelta, vNormal, fBlock,
                                 fAdjustedDetailOcclusionRadius, p_sDetailBlockerFunc_t,
                                 fSampleRecip, psGlobals.fDetailFullOcclusionThreshold,
                                 psGlobals.fDetailNoOcclusionThreshold,
                                 SSAOCenterUV, SSAOLength))
            {
                fAccumBlock += fBlock;
                iSamples++;
            }
        }
        fAccumBlock /= iSamples;
        float fOcclusion2 = 1.0f - fAccumBlock;
        fOcclusion2 = pow(fOcclusion2, psGlobals.fDetailOcclusionPower);

        SSAOLength /= b_iSSAOSampleCount;

        fResult = min(fOcclusion, fOcclusion2);
    }
#else
    fResult = fOcclusion;
#endif

    // fStaticAO is sampled but never read upstream (the b_iSSAOOutput arm only ever
    // overwrites fResult with 1), so it DCEs away on both legs.
#if (b_iSSAOOutput == SSAO_OUTPUT_STATIC)
    fResult = 1.0f;
#endif

    return fResult;
}

// ---- psantialiasing.fx (FXAA 3.10-ish, Timothy Lottes) ----------------------
float ppqGetLuma(float4 cIn) { return cIn.a; }

float4 ppqSampleOff(float2 p, float2 o, float2 r)
{
    return p_sSrcMap_t.SampleLevel(p_sSamp, p + (o * r), 0);
}

float4 ppqSampleTop(float2 p)
{
    return p_sSrcMap_t.SampleLevel(p_sSamp, p, 0);
}

float4 ppqFxaaPixelShader(float2 pos, float4 posPos, float2 vRcpFrame, float4 rcpFrameOpt)
{
    float2 vPos;
    vPos.x = pos.x;
    vPos.y = pos.y;
    float4 cCenter = ppqSampleTop(vPos);
    float fLumaCenter = ppqGetLuma(cCenter);
    float fLumaDown  = ppqGetLuma(ppqSampleOff(vPos, float2( 0,  1), vRcpFrame.xy));
    float fLumaRight = ppqGetLuma(ppqSampleOff(vPos, float2( 1,  0), vRcpFrame.xy));
    float fLumaUp    = ppqGetLuma(ppqSampleOff(vPos, float2( 0, -1), vRcpFrame.xy));
    float fLumaLeft  = ppqGetLuma(ppqSampleOff(vPos, float2(-1,  0), vRcpFrame.xy));

    // Luma range.
    float fLumaMaxDownCenter = max(fLumaDown, fLumaCenter);
    float fLumaMinDownCenter = min(fLumaDown, fLumaCenter);
    float fLumaMaxRightDownCenter = max(fLumaRight, fLumaMaxDownCenter);
    float fLumaMinRightDownCenter = min(fLumaRight, fLumaMinDownCenter);
    float fLumaMaxUpLeft = max(fLumaUp, fLumaLeft);
    float fLumaMinUpLeft = min(fLumaUp, fLumaLeft);
    float fLumaRangeMax = max(fLumaMaxUpLeft, fLumaMaxRightDownCenter);
    float fLumaRangeMin = min(fLumaMinUpLeft, fLumaMinRightDownCenter);
    float fLumaRangeMaxScaled = fLumaRangeMax * psGlobals.vAAEdgeThreshold.x;
    float fLumaRange = fLumaRangeMax - fLumaRangeMin;
    float fLumaRangeMaxClamped = max(psGlobals.vAAEdgeThreshold.y, fLumaRangeMaxScaled);

    // Too low a luma range: the pixel does not suffer from aliasing.
    if (fLumaRange < fLumaRangeMaxClamped)
    {
        if ((SC2_AXIS(b_iAAMode) == AAMODE_SUBPIXEL) || (SC2_AXIS(b_iAAMode) == AAMODE_EDGEDIRECTION))
            return float4(0, 0, 0, 0);
        else
            return cCenter;
    }

    // Direction and subpixel aliasing.
    float fSubpixel;
    bool bHorizontal;
    {
        float fLumaUpLeft    = ppqGetLuma(ppqSampleOff(vPos, float2(-1, -1), vRcpFrame.xy));
        float fLumaDownRight = ppqGetLuma(ppqSampleOff(vPos, float2( 1,  1), vRcpFrame.xy));
        float fLumaUpRight   = ppqGetLuma(ppqSampleOff(vPos, float2( 1, -1), vRcpFrame.xy));
        float fLumaDownLeft  = ppqGetLuma(ppqSampleOff(vPos, float2(-1,  1), vRcpFrame.xy));

        float fLumaUpDown = fLumaUp + fLumaDown;
        float fLumaLeftRight = fLumaLeft + fLumaRight;
        float fEdgeCenterHorz = (-2.0 * fLumaCenter) + fLumaUpDown;
        float fEdgeCenterVert = (-2.0 * fLumaCenter) + fLumaLeftRight;

        float fLumaRightCorners = fLumaUpRight + fLumaDownRight;
        float fLumaUpCorners = fLumaUpLeft + fLumaUpRight;
        float fEdgeRight = (-2.0 * fLumaRight) + fLumaRightCorners;
        float fEdgeUp = (-2.0 * fLumaUp) + fLumaUpCorners;

        float fLumaLeftCorners = fLumaUpLeft + fLumaDownLeft;
        float fLumaDownCorners = fLumaDownLeft + fLumaDownRight;
        float fEdgeLeft = (-2.0 * fLumaLeft) + fLumaLeftCorners;
        float fEdgeDown = (-2.0 * fLumaDown) + fLumaDownCorners;

        float fEdgeHorzCombined = abs(fEdgeCenterHorz) * 2.0 + abs(fEdgeLeft) + abs(fEdgeRight);
        float fEdgeVertCombined = abs(fEdgeCenterVert) * 2.0 + abs(fEdgeUp) + abs(fEdgeDown);

        bHorizontal = fEdgeHorzCombined >= fEdgeVertCombined;

        if (SC2_AXIS(b_iAAMode) == AAMODE_EDGEDIRECTION)
            return float4(0, (bHorizontal) ? 1.0 : 0, (bHorizontal) ? 0 : 1.0, 0);

        // Subpixel aliasing.
        float fLumaRangeRcp = 1.0 / fLumaRange;
        float fLumaUpDownLeftRight = fLumaUpDown + fLumaLeftRight;
        float fLumaAllCorners = fLumaLeftCorners + fLumaRightCorners;

        // Overall luminance of the 9x9 block, and how much the centre differs.
        float fLumaFull = fLumaUpDownLeftRight * 2.0 + fLumaAllCorners;
        float fLumaCenterDiff = (fLumaFull * (1.0 / 12.0)) - fLumaCenter;
        fLumaCenterDiff -= min(fLumaCenterDiff, 0.0) * psGlobals.vAASubPixel.y;
        float fLumaCenterDiffNormalized = saturate(abs(fLumaCenterDiff) * fLumaRangeRcp);

        // S-curve.
        float fSubpixelLinear = ((-2.0) * fLumaCenterDiffNormalized) + 3.0;
        float fSubpixelQuadratic = fLumaCenterDiffNormalized * fLumaCenterDiffNormalized;
        float fSubpixelProduct = fSubpixelLinear * fSubpixelQuadratic;
        float fSubpixelFinal = fSubpixelProduct * fSubpixelProduct;

        fSubpixel = fSubpixelFinal * psGlobals.vAASubPixel.x;

        if (SC2_AXIS(b_iAAMode) == AAMODE_SUBPIXEL)
            return float4(1.0, fSubpixel, 0.0, 0.0);
    }

    // Which side of the pixel to search.
    float fLengthSign;
    float fGradient;
    float fLumaTarget;
    {
        float fLumaFront = (bHorizontal) ? fLumaUp : fLumaLeft;
        float fLumaBack = (bHorizontal) ? fLumaDown : fLumaRight;
        fLengthSign = (bHorizontal) ? vRcpFrame.y : vRcpFrame.x;

        float fGradientFront = fLumaFront - fLumaCenter;
        float fGradientBack = fLumaBack - fLumaCenter;
        float fLumaFrontCenter = fLumaFront + fLumaCenter;
        float fLumaBackCenter = fLumaBack + fLumaCenter;

        float fGradientFrontAbs = abs(fGradientFront);
        float fGradientBackAbs = abs(fGradientBack);
        if (fGradientFrontAbs >= fGradientBackAbs)
        {
            fGradient = fGradientFrontAbs;
            fLengthSign = -fLengthSign;
            fLumaTarget = fLumaFrontCenter;
        }
        else
        {
            fGradient = fGradientBackAbs;
            fLumaTarget = fLumaBackCenter;
        }
    }

    // Search parameters.
    float2 vPosBase;
    vPosBase.x = vPos.x;
    vPosBase.y = vPos.y;
    float2 vStep;
    vStep.x = (!bHorizontal) ? 0.0 : vRcpFrame.x;
    vStep.y = ( bHorizontal) ? 0.0 : vRcpFrame.y;
    if (!bHorizontal) vPosBase.x += fLengthSign * 0.5;
    if ( bHorizontal) vPosBase.y += fLengthSign * 0.5;

    float2 vPosFront = vPosBase - vStep;
    float2 vPosBack = vPosBase + vStep;
    float fLumaEndFront = ppqGetLuma(ppqSampleTop(vPosFront));
    float fLumaEndBack = ppqGetLuma(ppqSampleTop(vPosBack));

    float fGradientScaled = fGradient * 1.0 / 4.0;
    float fLumaDiff = fLumaCenter - fLumaTarget * 0.5;
    bool bLumaCenterIsLess = fLumaDiff < 0.0;

    // Search for the ends of the line (manually unrolled upstream).
    fLumaEndFront -= fLumaTarget * 0.5;
    fLumaEndBack -= fLumaTarget * 0.5;
    bool bDoneFront = abs(fLumaEndFront) >= fGradientScaled;
    bool bDoneBack = abs(fLumaEndBack) >= fGradientScaled;
    if (!bDoneFront) vPosFront -= vStep * FXAA_STEP0;
    bool bNotDone = (!bDoneFront) || (!bDoneBack);
    if (!bDoneBack) vPosBack += vStep * FXAA_STEP0;

    if (bNotDone) {
        if (!bDoneFront) fLumaEndFront = ppqGetLuma(ppqSampleTop(vPosFront.xy));
        if (!bDoneBack) fLumaEndBack = ppqGetLuma(ppqSampleTop(vPosBack.xy));
        if (!bDoneFront) fLumaEndFront = fLumaEndFront - fLumaTarget * 0.5;
        if (!bDoneBack) fLumaEndBack = fLumaEndBack - fLumaTarget * 0.5;
        bDoneFront = abs(fLumaEndFront) >= fGradientScaled;
        bDoneBack = abs(fLumaEndBack) >= fGradientScaled;
        if (!bDoneFront) vPosFront -= vStep * FXAA_STEP1;
        bNotDone = (!bDoneFront) || (!bDoneBack);
        if (!bDoneBack) vPosBack += vStep * FXAA_STEP1;

        if (bNotDone) {
            if (!bDoneFront) fLumaEndFront = ppqGetLuma(ppqSampleTop(vPosFront.xy));
            if (!bDoneBack) fLumaEndBack = ppqGetLuma(ppqSampleTop(vPosBack.xy));
            if (!bDoneFront) fLumaEndFront = fLumaEndFront - fLumaTarget * 0.5;
            if (!bDoneBack) fLumaEndBack = fLumaEndBack - fLumaTarget * 0.5;
            bDoneFront = abs(fLumaEndFront) >= fGradientScaled;
            bDoneBack = abs(fLumaEndBack) >= fGradientScaled;
            if (!bDoneFront) vPosFront -= vStep * FXAA_STEP2;
            bNotDone = (!bDoneFront) || (!bDoneBack);
            if (!bDoneBack) vPosBack += vStep * FXAA_STEP2;

            if (bNotDone) {
                if (!bDoneFront) fLumaEndFront = ppqGetLuma(ppqSampleTop(vPosFront.xy));
                if (!bDoneBack) fLumaEndBack = ppqGetLuma(ppqSampleTop(vPosBack.xy));
                if (!bDoneFront) fLumaEndFront = fLumaEndFront - fLumaTarget * 0.5;
                if (!bDoneBack) fLumaEndBack = fLumaEndBack - fLumaTarget * 0.5;
                bDoneFront = abs(fLumaEndFront) >= fGradientScaled;
                bDoneBack = abs(fLumaEndBack) >= fGradientScaled;
                if (!bDoneFront) vPosFront -= vStep * FXAA_STEP3;
                bNotDone = (!bDoneFront) || (!bDoneBack);
                if (!bDoneBack) vPosBack += vStep * FXAA_STEP3;

                if (bNotDone) {
                    if (!bDoneFront) fLumaEndFront = ppqGetLuma(ppqSampleTop(vPosFront.xy));
                    if (!bDoneBack) fLumaEndBack = ppqGetLuma(ppqSampleTop(vPosBack.xy));
                    if (!bDoneFront) fLumaEndFront = fLumaEndFront - fLumaTarget * 0.5;
                    if (!bDoneBack) fLumaEndBack = fLumaEndBack - fLumaTarget * 0.5;
                    bDoneFront = abs(fLumaEndFront) >= fGradientScaled;
                    bDoneBack = abs(fLumaEndBack) >= fGradientScaled;
                    // NOTE: from here on the original steps only .x, even on the
                    // vertical-edge path — transcribed as written.
                    if (!bDoneFront) vPosFront.x -= (vStep * FXAA_STEP4).x;
                    bNotDone = (!bDoneFront) || (!bDoneBack);
                    if (!bDoneBack) vPosBack.x += (vStep * FXAA_STEP4).x;

                    if (bNotDone && (FXAA_SAMPLES > 5)) {
                        if (!bDoneFront) fLumaEndFront = ppqGetLuma(ppqSampleTop(vPosFront.xy));
                        if (!bDoneBack) fLumaEndBack = ppqGetLuma(ppqSampleTop(vPosBack.xy));
                        if (!bDoneFront) fLumaEndFront = fLumaEndFront - fLumaTarget * 0.5;
                        if (!bDoneBack) fLumaEndBack = fLumaEndBack - fLumaTarget * 0.5;
                        bDoneFront = abs(fLumaEndFront) >= fGradientScaled;
                        bDoneBack = abs(fLumaEndBack) >= fGradientScaled;
                        if (!bDoneFront) vPosFront.x -= (vStep * FXAA_STEP5).x;
                        bNotDone = (!bDoneFront) || (!bDoneBack);
                        if (!bDoneBack) vPosBack.x += (vStep * FXAA_STEP5).x;

                        if (bNotDone && (FXAA_SAMPLES > 6)) {
                            if (!bDoneFront) fLumaEndFront = ppqGetLuma(ppqSampleTop(vPosFront.xy));
                            if (!bDoneBack) fLumaEndBack = ppqGetLuma(ppqSampleTop(vPosBack.xy));
                            if (!bDoneFront) fLumaEndFront = fLumaEndFront - fLumaTarget * 0.5;
                            if (!bDoneBack) fLumaEndBack = fLumaEndBack - fLumaTarget * 0.5;
                            bDoneFront = abs(fLumaEndFront) >= fGradientScaled;
                            bDoneBack = abs(fLumaEndBack) >= fGradientScaled;
                            if (!bDoneFront) vPosFront.x -= (vStep * FXAA_STEP6).x;
                            bNotDone = (!bDoneFront) || (!bDoneBack);
                            if (!bDoneBack) vPosBack.x += (vStep * FXAA_STEP6).x;

                            if (bNotDone && (FXAA_SAMPLES > 7)) {
                                if (!bDoneFront) fLumaEndFront = ppqGetLuma(ppqSampleTop(vPosFront.xy));
                                if (!bDoneBack) fLumaEndBack = ppqGetLuma(ppqSampleTop(vPosBack.xy));
                                if (!bDoneFront) fLumaEndFront = fLumaEndFront - fLumaTarget * 0.5;
                                if (!bDoneBack) fLumaEndBack = fLumaEndBack - fLumaTarget * 0.5;
                                bDoneFront = abs(fLumaEndFront) >= fGradientScaled;
                                bDoneBack = abs(fLumaEndBack) >= fGradientScaled;
                                if (!bDoneFront) vPosFront.x -= (vStep * FXAA_STEP7).x;
                                if (!bDoneBack) vPosBack.x += (vStep * FXAA_STEP7).x;
                            }
                        }
                    }
                }
            }
        }
    }

    // Distance and final gradient.
    float fDistanceFront = (bHorizontal) ? vPos.x - vPosFront.x : vPos.y - vPosFront.y;
    float fDistanceBack = (bHorizontal) ? vPosBack.x - vPos.x : vPosBack.y - vPos.y;

    bool bFrontDiffers = (fLumaEndFront < 0.0) != bLumaCenterIsLess;
    float fSpanLength = (fDistanceFront + fDistanceBack);
    bool bBackDiffers = (fLumaEndBack < 0.0) != bLumaCenterIsLess;
    float fSpanLengthRcp = 1.0 / fSpanLength;

    // Are we actually on the right side of the line?
    bool bSpanDiff;
    float fDistance;
    if (fDistanceFront < fDistanceBack)
    {
        bSpanDiff = bFrontDiffers;
        fDistance = fDistanceFront;
    }
    else
    {
        bSpanDiff = bBackDiffers;
        fDistance = fDistanceBack;
    }
    float fPixelOffset = (fDistance * (-fSpanLengthRcp)) + 0.5;
    float fPixelOffsetFinal = bSpanDiff ? fPixelOffset : 0.0;

    // Include the subpixel blend, then sample the screen texture.
    fPixelOffsetFinal = max(fPixelOffsetFinal, fSubpixel);
    if (!bHorizontal) vPos.x += fPixelOffsetFinal * fLengthSign;
    if ( bHorizontal) vPos.y += fPixelOffsetFinal * fLengthSign;
    return ppqSampleTop(vPos);
}

float4 ppqAntiAliasing(PostProcessQuadPSInput vertOut)
{
    float4 vRcpOpt = float4(psGlobals.vRecipSrcSizeOffset.xy, psGlobals.vRecipSrcSizeOffset.xy);
    vRcpOpt.xy *= 2.0;
    vRcpOpt.zw *= 0.5;
    float4 vPosCorners;
    vPosCorners.xy = vertOut.UV(0).xy - vRcpOpt.zw;
    vPosCorners.zw = vertOut.UV(0).xy + vRcpOpt.zw;
    return ppqFxaaPixelShader(vertOut.UV(0).xy, vPosCorners,
                              psGlobals.vRecipSrcSizeOffset.xy, vRcpOpt);
}

// =============================================================================
// pspostprocess.fx : PostProcess — mode dispatch + the shared output tail.
// =============================================================================
[shader("pixel")]
public float4 postprocessquad_ps_main(PostProcessQuadPSInput vertOut) : SV_Target0
{
    float4 cResult = float4(0, 0, 0, 0);

#if (b_iMode == MODE_Copy)
    cResult = ppqCopy(vertOut);
#elif (b_iMode == MODE_SetValue)
    cResult = ppqSetValue(vertOut);
#elif (b_iMode == MODE_GaussianBlur)
    cResult = ppqGaussianBlur(vertOut);
#elif (b_iMode == MODE_Downscale)
    cResult = ppqDownscale(vertOut);
#elif (b_iMode == MODE_COC)
    cResult = ppqCOC(vertOut);
#elif (b_iMode == MODE_DOF)
    cResult = ppqDOF(vertOut);
#elif (b_iMode == MODE_SSAO)
    cResult = ppqSSAO(vertOut);
#elif (b_iMode == MODE_ConvertToGradient)
    cResult = ppqConvertToGradient(vertOut);
#elif (b_iMode == MODE_DepthEncode)
    cResult = ppqDepthEncode(vertOut);
#elif (b_iMode == MODE_ScreenSpaceParticle)
    cResult = ppqScreenSpaceParticle(vertOut);
#elif (b_iMode == MODE_AntiAliasing)
    cResult = ppqAntiAliasing(vertOut);
#elif (b_iMode == MODE_FOWSmoothing)
    cResult = ppqFOWSmoothing(vertOut);
#elif (b_iMode == MODE_HaloCopy)
    cResult = ppqHaloCopy(vertOut);
#elif (b_iMode == MODE_HaloPrepare)
    cResult = ppqHaloPrepare(vertOut);
#elif (b_iMode == MODE_HaloBlur)
    cResult = ppqHaloBlur(vertOut);
#endif

#if b_iTakeMin
    {
        float4 cOperand = p_sOperandMap_t.Sample(p_sSamp, vertOut.UV(0).xy);
        cResult = min(cOperand, cResult);
    }
#endif

    const int iInputChannelFilter  = SC2_AXIS(b_iInputChannelFilter);
    const int iOutputChannelFilter = SC2_AXIS(b_iOutputChannelFilter);
    const int iHaloRastermode      = SC2_AXIS(b_iHaloRastermode);

    [branch] if (iInputChannelFilter == CHANNEL_RED)
        cResult = cResult.r;
    else [branch] if (iInputChannelFilter == CHANNEL_GREEN)
        cResult = cResult.g;
    else [branch] if (iInputChannelFilter == CHANNEL_BLUE)
        cResult = cResult.b;
    else [branch] if (iInputChannelFilter == CHANNEL_ALPHA)
        cResult = cResult.a;

    [branch] if (iHaloRastermode == HALO_ALPHA && iInputChannelFilter != CHANNEL_ALL)
        cResult.rgb = float3(1.0, 1.0, 1.0);

    [branch] if (SC2_AXIS(b_iScaleOutput))
        cResult *= psGlobals.vScale;
    [branch] if (SC2_AXIS(b_iAddToOutput))
        cResult += psGlobals.fAdd;

    [branch] if (iOutputChannelFilter == CHANNEL_RED)
        cResult = cResult.r;
    else [branch] if (iOutputChannelFilter == CHANNEL_GREEN)
        cResult = cResult.g;
    else [branch] if (iOutputChannelFilter == CHANNEL_BLUE)
        cResult = cResult.b;
    else [branch] if (iOutputChannelFilter == CHANNEL_ALPHA)
        cResult = cResult.a;

    [branch] if (SC2_AXIS(b_iClampNegativeOutput))
        cResult = max(float4(0.0f, 0.0f, 0.0f, 0.0f), cResult);

    [branch] if (SC2_AXIS(b_iPostMaskModulate))
        cResult *= psGlobals.vPostMaskColor;

    [branch] if (iHaloRastermode == HALO_MUL)
        cResult = lerp(float4(1.0, 1.0, 1.0, 1.0), cResult, cResult.a);

#if b_iAlphaTest
    sc2_clip(cResult.a - psGlobals.fAlphaThreshold);
#endif

    return cResult;
}
