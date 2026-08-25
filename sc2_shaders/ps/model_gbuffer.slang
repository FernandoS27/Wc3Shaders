// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026, Fernando Sahmkow
// See LICENSE in the repository root for full terms.
//
// Model-family deferred g-buffer encoding (pscommon.fx : OutputDepth /
// EncodeDepthNormal) + psutils.fx : OutputDepth.
//
// M1 INCREMENT 4 SCOPE: the depth/normal MRT target encoding.  b_iOutEncoding is a
// compile-time axis (FLOAT / RGBA / RGBA16), so the runtime `if (b_iOutEncoding ==
// …)` ladder in the .fx folds to a single branch — reproduced here as `#if`
// branches (the same technique used for the FOW-mode ladder), which keeps the
// candidate's surviving arm identical to fxc's.  The diffuse/specular targets and
// the target-layout selection live in DefaultPixelMain (model_ps.slang).

implementing sc2_shaders;
#include "types/perm_cb.slangh"

// pscommon.fx depth/normal encoding selector.
#define DEPTH_NORMAL_ENCODING_FLOAT   0
#define DEPTH_NORMAL_ENCODING_RGBA    1
#define DEPTH_NORMAL_ENCODING_RGBA16  2

// Axis defaults (the engine passes these only when nonzero).
#ifndef b_iOutEncoding
#define b_iOutEncoding 0
#endif
#ifndef b_iNormalizeDeffNormal
#define b_iNormalizeDeffNormal 0
#endif

// psutils.fx : OutputDepth — the view-space Y is the deferred depth.
float outputDepth(ModelPSInput vertOut)
{
    return vertOut.ViewPos().y;
}

// pscommon.fx : EncodeDepthNormal — pack the world-space normal + depth into the
// vNormalDepth target per the compile-time encoding.  Each encoding writes the
// normal only when b_emitMRTNormal and the depth only when b_emitMRTDepth (else the
// unused half gets the .fx's sentinel: 0 for normal, 1 for depth's .a).
float4 encodeDepthNormal(float3 vNormal, float fDepth)
{
#if b_iNormalizeDeffNormal
    vNormal = normalize(vNormal);
#endif

    // b_emitMRT* stay compile-time -- they ARE the output signature (K2).  Only the
    // ENCODING moves, so each arm keeps its own MRT guards.
    float4 vNormalDepth = float4(0, 0, 0, 0);
    const int iOutEncoding = SC2_AXIS(b_iOutEncoding);

    [branch] if (iOutEncoding == DEPTH_NORMAL_ENCODING_FLOAT)
    {
    #if (b_emitMRTNormal == 1)
        vNormalDepth.rgb = vNormal;
    #else
        vNormalDepth.rgb = float3(0, 0, 0);
    #endif
    #if (b_emitMRTDepth == 1)
        vNormalDepth.a = fDepth;
    #else
        vNormalDepth.a = 1.0f;
    #endif
    }
    else [branch] if (iOutEncoding == DEPTH_NORMAL_ENCODING_RGBA)
    {
    #if (b_emitMRTNormal == 1)
        float3 vNormalVS = mul((float3x3)psGlobals.mViewTransform, vNormal);
        vNormalDepth.rg = (vNormalVS.xz * 0.5f) + 0.5f;
    #else
        vNormalDepth.rg = float2(0, 0);
    #endif
    #if (b_emitMRTDepth == 1)
        float fEnc = saturate(1.0f - 1.0f / (1.0f + fDepth / psGlobals.fMaxDepthFactor));
        vNormalDepth.ba = frac(float2(1.0f, 255.0f) * fEnc);
        vNormalDepth.b -= vNormalDepth.a / 255.0f;
    #else
        vNormalDepth.ba = float2(0, 0);
    #endif
    }
    else [branch] if (iOutEncoding == DEPTH_NORMAL_ENCODING_RGBA16)
    {
    #if (b_emitMRTNormal == 1)
        vNormalDepth.rgb = vNormal * 0.5f + 0.5f;
    #else
        vNormalDepth.rgb = float3(0, 0, 0);
    #endif
    #if (b_emitMRTDepth == 1)
        vNormalDepth.a = saturate(1.0f - 1.0f / (1.0f + fDepth / psGlobals.fMaxDepthFactor));
    #else
        vNormalDepth.a = 1.0f;
    #endif
    }
    return vNormalDepth;
}
