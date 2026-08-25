// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026, Fernando Sahmkow
// See LICENSE in the repository root for full terms.
//
// Model-family SHADINGMODE_DISPLACEMENT arm (psdisplacement.fx : Displace) — a
// screen-space refraction/heat-haze: offset the back-buffer UV by the view-space
// normal scaled by a strength layer, with an optional depth-fix that cancels the
// displacement when an occluder is in front.  Debug-mode early returns are dead (no
// reachable perm sets b_iDebugMode) and ComputeEnvMappings is dead (EyeToVertex never
// live; the strength layer samples with allowEnvio=false), so both are omitted.

implementing sc2_shaders;
#include "types/perm_cb.slangh"

#define PASS_SINGLEPASS_DISPLACE 0
#define PASS_COPYBACK            1

#ifndef b_iDisplacementPass
#define b_iDisplacementPass 0
#endif
#ifndef b_useNormalMapping
#define b_useNormalMapping 0
#endif
#ifndef b_iDisplacementUseConstantColor
#define b_iDisplacementUseConstantColor 0
#endif
#ifndef b_iDisplacementUVEmitter
#define b_iDisplacementUVEmitter 0
#endif
#ifndef b_useVertexAlphaStrength
#define b_useVertexAlphaStrength 0
#endif
#ifndef b_distortionDepthFix
#define b_distortionDepthFix 0
#endif

// psdisplacement.fx : Displace.
float4 displace(ModelPSInput vertOut)
{
#if (b_iDisplacementPass == PASS_SINGLEPASS_DISPLACE)
    float3 vDeferredNormal = float3(0, 0, 0);
    SLayerConfig dispCfg = setupDisplacementLayer();

    // Sample the perturbation normal (tangent-space) and rotate to world.
    float fStub = 0.0f;
    float3 vNormal;
    float3 vTextureNormal;
  #if b_useNormalMapping
    #if b_iDisplacementUseConstantColor
    vTextureNormal = dispCfg.p_cConstantColor.xyz;
    #else
    {
        float4 vUV = getUVEmitter(vertOut, b_iDisplacementUVEmitter,
                                  dispCfg.p_vMultiplyAddAlphaTrans.w, dispCfg.p_mUVTransform);
        vTextureNormal = decodeTextureNormal(p_sDisplacementSampler_t, p_sSamp, vUV, fStub);
    }
    #endif
    vNormal = tangentToWorld(vTextureNormal, vertOut.Normal().xyz,
                             vertOut.Tangent().xyz, vertOut.Binormal());
  #else
    vTextureNormal = vertOut.Normal().xyz;
    vNormal = vertOut.Normal().xyz;
  #endif
    vDeferredNormal = vertOut.Normal().xyz;     // stored for fresnel; unused without MRT

    // Sample the strength layer (.a).  ComputeEnvMappings is dead here.
    float fStrengthSample;
    {
        TriPlanarUVs tp = initTriPlanarUVs(vertOut);
        SLayerConfig strCfg = setupDisplacementStrengthLayer();
        fStrengthSample = getLayerColor2D(vertOut, strCfg, p_sDisplacementStrengthSampler_t,
                                          p_sSamp, 1.0f, vDeferredNormal, tp).a;
    }
    [branch] if (SC2_AXIS(b_useVertexAlphaStrength))
        fStrengthSample *= saturate(vertOut.VertexColor().w);

    // Adjust by depth: project the normal to view space and divide by view depth.
    vNormal = mul((float3x3)psGlobals.mViewTransform, vNormal);
    vNormal = vNormal / vertOut.ViewPos().y;

    // Calculate the displaced UV.
    float2 vUV = getBackBufferUV(vertOut);
    float2 vDisplacedUV = vUV + vNormal.xz * psGlobals.vStrength2D * fStrengthSample
                              * psGlobals.vViewportScale;

    // Depth-aware fix (PIXEL_SHADER_VERSION >= PS_30 is always true): if the sampled
    // occluder is in front of this fragment, cancel the displacement.
    // The depth map is already bound on this path (the displacement pass samples it
    // unconditionally above), so this is a value axis, not a texture-set one.
    [branch] if (SC2_AXIS(b_distortionDepthFix))
    {
        float fSampledDepth = decodeDepth(p_sNormalDepthMap_t.Sample(p_sSamp, vDisplacedUV));
        if (fSampledDepth < vertOut.ViewPos().y)
            vDisplacedUV = vUV;
    }

    return p_sDisplacementSrc_t.Sample(p_sSamp, vDisplacedUV);

#elif (b_iDisplacementPass == PASS_COPYBACK)
    float2 vUV = getBackBufferUV(vertOut);
    return p_sResult_t.Sample(p_sSamp, vUV);
#else
    return float4(0, 0, 0, 0);
#endif
}
