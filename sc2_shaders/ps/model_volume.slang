// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026, Fernando Sahmkow
// See LICENSE in the repository root for full terms.
//
// Model-family SHADINGMODE_VOLUME arm (psvolume.fx : Volume).  Every reachable perm
// is VOLUME_TYPE_UNIFORM (b_iVolumeType == 0), so VolumeNoisy / WalkVolume (the
// 40-step ray march) and the noise textures are dead — only VolumeUniform is
// transcribed.  ComputeEnvMappings is likewise dead (EyeToVertex never live, the
// VolumeColor layer is always constant-colour), so it is omitted.

implementing sc2_shaders;
#include "types/perm_cb.slangh"

#define VOLUME_TYPE_UNIFORM 0

#define PASS_CLEAR                 0
#define PASS_ADD_VOLUME_DEPTH      1
#define PASS_SUBTRACT_VOLUME_DEPTH 2
#define PASS_ADD_SCENE_DEPTH       3
#define PASS_SUBTRACT_SCENE_DEPTH  4
#define PASS_VOLUME_FOG            5

#define LINEAR_FOG      0
#define EXPONENTIAL_FOG 1

#ifndef b_iVolumeType
#define b_iVolumeType 0
#endif
#ifndef b_iVolumePass
#define b_iVolumePass 0
#endif
#ifndef b_iVolumeFalloffType
#define b_iVolumeFalloffType 0
#endif
#ifndef b_iVolumeColorLayerEnable
#define b_iVolumeColorLayerEnable 0
#endif
#ifndef b_iVolumeMod
#define b_iVolumeMod 0
#endif

// psvolume.fx : VolumeUniform.
float4 volumeUniform(ModelPSInput vertOut)
{
#if (b_iVolumePass == PASS_CLEAR)
    return float4(0, 0, 0, 0);
#elif (b_iVolumePass == PASS_ADD_VOLUME_DEPTH)
    float fDepth = vertOut.ViewPos().y;
    return float4(fDepth, fDepth, fDepth, fDepth);
#elif (b_iVolumePass == PASS_SUBTRACT_VOLUME_DEPTH)
    float fDepth = -vertOut.ViewPos().y;
    return float4(fDepth, fDepth, fDepth, fDepth);
#elif (b_iVolumePass == PASS_ADD_SCENE_DEPTH)
    // SampleNormalDepth then PIXEL_DEPTH = .a = DecodeDepth(sample); the decoded
    // normal (.rgb) is unused, so only the depth channel is recovered.
    float2 vUV = getBackBufferUV(vertOut);
    float fDepth = decodeDepth(p_sNormalDepthMap_t.Sample(p_sSamp, vUV));
    return float4(fDepth, fDepth, fDepth, fDepth);
#elif (b_iVolumePass == PASS_SUBTRACT_SCENE_DEPTH)
    float2 vUV = getBackBufferUV(vertOut);
    float fDepth = -decodeDepth(p_sNormalDepthMap_t.Sample(p_sSamp, vUV));
    return float4(fDepth, fDepth, fDepth, fDepth);
#elif (b_iVolumePass == PASS_VOLUME_FOG)
    float2 vUV = getBackBufferUV(vertOut);
    float fThickness = p_sThickness_t.Sample(p_sSamp, vUV).x * psGlobals.fDensity;
    fThickness = max(0.0f, fThickness);

    float fogDensity;
    [branch] if (SC2_AXIS(b_iVolumeFalloffType) == EXPONENTIAL_FOG)
        fogDensity = 1.0f - saturate(exp(-fThickness));
    else
        fogDensity = saturate(fThickness);

    float3 cColor = float3(1, 1, 1);
  #if b_iVolumeColorLayerEnable
    {
        TriPlanarUVs tp = initTriPlanarUVs(vertOut);   // VolumeColor is constant / non-triplanar → DCE'd
        SLayerConfig cfg = setupVolumeColorLayer();
        cColor = getLayerColor2D(vertOut, cfg, p_sVolumeColorSampler_t, p_sSamp,
                                 1.0f, float3(0, 0, 0), tp).rgb;
    }
  #endif
    [branch] if (SC2_AXIS(b_iVolumeMod))
        cColor *= fogDensity;
    return float4(cColor, fogDensity);
#else
    return float4(0, 0, 0, 0);
#endif
}

// psvolume.fx : Volume — b_iVolumeType is always UNIFORM on every retail perm; the
// noisy arm has no reachable permutation, so it is left as an explicit fall-through.
float4 volume(ModelPSInput vertOut)
{
#if (b_iVolumeType == VOLUME_TYPE_UNIFORM)
    return volumeUniform(vertOut);
#else
    return float4(0, 0, 0, 0);   // VolumeNoisy: no reachable perm (documented gap)
#endif
}
