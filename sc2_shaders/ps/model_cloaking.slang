// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026, Fernando Sahmkow
// See LICENSE in the repository root for full terms.
//
// Model-family SHADINGMODE_CLOAKING arm (pscloaking.fx : Cloaking) — the cloak/blit
// effect: blit the off-screen buffer (pass 0), a transparent cloak (pass 1), or the
// opaque cloak that composites the background with an envio/alpha-mask (pass 2).
// ComputeEnvMappings is dead (no reachable cloaking layer has an envio UV mapping;
// the CloakingEnvio layer is always constant-colour), so it is omitted.

implementing sc2_shaders;

// SC2_LAYER_<Prop>: layer properties are also read OUTSIDE the setup
// functions, and every such site has to go through the same accessor.
#include "ps/model_layers_axes.slangh"

// A `#if` tolerates an undefined axis; ordinary CODE does not, and the
// convention only passes -D for NONZERO axes.  Converting these reads from
// `#if` to a runtime `if` therefore needs their file-local defaults, or the
// permutations where they are zero stop compiling.
// (docs: the sc2_shaders file-scope rule.)
#ifndef b_iCloakingEnvioOp
#define b_iCloakingEnvioOp 0
#endif

#define CLOAKINGMODE_BLITOFFSCREEN    0
#define CLOAKINGMODE_CLOAKTRANSPARENT 1
#define CLOAKINGMODE_CLOAKOPAQUE      2

#define LAYEROP_ADD  2
#define LAYEROP_LERP 3

#ifndef b_iCloakingPass
#define b_iCloakingPass 0
#endif
#ifndef b_iCloakingNormalLayerEnable
#define b_iCloakingNormalLayerEnable 0
#endif
#ifndef b_iCloakingAlphaMaskOriginalLayerEnable
#define b_iCloakingAlphaMaskOriginalLayerEnable 0
#endif
#ifndef b_iCloakingEnvioLayerEnable
#define b_iCloakingEnvioLayerEnable 0
#endif
#ifndef b_iCloakingEnvioMaskLayerEnable
#define b_iCloakingEnvioMaskLayerEnable 0
#endif
#ifndef b_iCloakingAlphaTestLayerEnable
#define b_iCloakingAlphaTestLayerEnable 0
#endif
#ifndef b_iCloakingEnvioOp
#define b_iCloakingEnvioOp 0
#endif

// pscloaking.fx : Cloaking.
float4 cloaking(ModelPSInput vertOut, inout DeferredOutputs def)
{
#if (b_iCloakingPass == CLOAKINGMODE_BLITOFFSCREEN)
    float2 vUV = getBackBufferUV(vertOut);
    float4 rvalue = p_sCloakingSrc_t.Sample(p_sSamp, vUV);
    def.cDiffuse = rvalue.rgb;
    return rvalue;

#elif (b_iCloakingPass == CLOAKINGMODE_CLOAKTRANSPARENT)
    float2 vUV = getBackBufferUV(vertOut);
    float4 objectColor = p_sCloakingSrc_t.Sample(p_sSamp, vUV);

    float3 vDeferredNormal = normalize(vertOut.Normal().xyz);
    def.vNormal = vDeferredNormal;

    float4 rvalue = float4(psGlobals.fCloakingAlpha * objectColor.rgb, 1.0f - psGlobals.fCloakingAlpha);

  #if b_iCloakingAlphaTestLayerEnable
    {
        TriPlanarUVs tp = initTriPlanarUVs(vertOut);
        SLayerConfig cfg = setupCloakingAlphaTestLayer();
        float alphaTestValue = getLayerColor2D(vertOut, cfg, p_sCloakingAlphaTestSampler_t,
                                               p_sSamp, 1.0f, vDeferredNormal, tp).a;
        if (alphaTestValue >= psGlobals.fAlphaThreshold)
        {
            rvalue.rgb = objectColor.rgb;
            rvalue.a = 0.0f;
        }
    }
  #endif
    return rvalue;

#elif (b_iCloakingPass == CLOAKINGMODE_CLOAKOPAQUE)
    float4 rvalue = float4(0, 0, 0, 0);
    TriPlanarUVs tp = initTriPlanarUVs(vertOut);

    float3 vDeferredNormal;
    SLayerConfig normCfg = setupCloakingNormalLayer();
  #if b_iCloakingNormalLayerEnable
    vDeferredNormal = decodeNormalInternal(vertOut, normCfg,
                                           p_sCloakingNormalSampler_t, p_sSamp);
  #else
    vDeferredNormal = normalize(vertOut.Normal().xyz);
  #endif
    def.vNormal = vDeferredNormal;

    {
        SLayerConfig cfg = setupCloakingAlphaMaskLayer();
        rvalue.a = psGlobals.fCloakingAlpha
                 * saturate(getLayerColor2D(vertOut, cfg, p_sCloakingAlphaMaskSampler_t,
                                            p_sSamp, 1.0f, vDeferredNormal, tp).a);
    }

  #if b_iCloakingAlphaMaskOriginalLayerEnable
    {
        SLayerConfig cfg = setupCloakingAlphaMaskOriginalLayer();
        float alphaTestValue = getLayerColor2D(vertOut, cfg, p_sCloakingAlphaMaskOriginalSampler_t,
                                               p_sSamp, 1.0f, vDeferredNormal, tp).a;
        if (alphaTestValue < psGlobals.fOGThreshold)
        {
            rvalue.rgb = float3(0, 0, 0);
            rvalue.a = 1.0f;
            return rvalue;
        }
        rvalue.a *= alphaTestValue;
    }
  #endif

    float2 vUV = getBackBufferUV(vertOut);
    float4 backgroundColor = p_sCloakingSrc_t.Sample(p_sSamp, vUV);
    rvalue.rgb = backgroundColor.rgb;

  #if b_iCloakingEnvioLayerEnable
    {
        SLayerConfig envCfg = setupCloakingEnvioLayer();
        float4 envio = getLayerColor2D(vertOut, envCfg, p_sCloakingEnvioSampler_t,
                                       p_sSamp, 1.0f, vDeferredNormal, tp);
    #if b_iCloakingEnvioMaskLayerEnable
        {
            SLayerConfig maskCfg = setupCloakingEnvioMaskLayer();
            envio.a = getLayerColor2D(vertOut, maskCfg, p_sCloakingEnvioMaskSampler_t,
                                      p_sSamp, 1.0f, vDeferredNormal, tp).a * envio.a;
        }
    #endif

        // Simulate the reversed src/dst blend in the shader.
        const int sc2CloakOp = SC2_LAYER_Op(b_iCloakingEnvioOp);
        [branch] if (sc2CloakOp == LAYEROP_ADD)
        {
            rvalue.rgb = (1.0f - rvalue.a) * rvalue.rgb + rvalue.a * envio.rgb * envio.a;
        }
        else [branch] if (sc2CloakOp == LAYEROP_LERP)
        {
            rvalue.rgb = (1.0f - rvalue.a) * rvalue.rgb + rvalue.a * envio.rgb * envio.a;
            rvalue.a = rvalue.a - rvalue.a * envio.a;
        }
        else
        {
            rvalue.rgb *= 1.0f - rvalue.a;
        }
    }
  #else
    rvalue.rgb *= 1.0f - rvalue.a;
  #endif

  #if b_iCloakingAlphaTestLayerEnable
    {
        SLayerConfig cfg = setupCloakingAlphaTestLayer();
        float alphaTestValue = getLayerColor2D(vertOut, cfg, p_sCloakingAlphaTestSampler_t,
                                               p_sSamp, 1.0f, vDeferredNormal, tp).a;
        if (alphaTestValue >= psGlobals.fAlphaThreshold)
        {
            rvalue.rgb = float3(0, 0, 0);
            rvalue.a = 1.0f;
        }
    }
  #endif
    return rvalue;
#else
    return float4(0.0f, 1.0f, 0.0f, 0.0f);
#endif
}
