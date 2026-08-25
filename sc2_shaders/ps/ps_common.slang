// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026, Fernando Sahmkow
// See LICENSE in the repository root for full terms.
//
// pscommon.fx : the depth-prepass sample decode, shared by the screen-space roots
// (deferredlight.fx, postprocessquad.fx).
//
// `decodeDepth` — the .a half — already lives in ps/model_main_shading.slang, which
// is where the Model root needs it; this file adds the .rgb half and the combined
// `SampleNormalDepth`.
//
// The encoding enum is REDECLARED here on purpose.  Slang's `__include` merges
// declarations into the module but gives every file its own preprocessor state, so
// another file's `#define DEPTH_NORMAL_ENCODING_RGBA 1` is not visible.  Omitting it
// leaves the symbol at 0 and turns `#if (b_iInEncoding == RGBA)` into `0 == 0`, i.e.
// the FLOAT permutations silently decode as RGBA.

implementing sc2_shaders;
#include "types/perm_cb.slangh"

#ifndef b_iInEncoding
#define b_iInEncoding 0
#endif
#define DEPTH_NORMAL_ENCODING_FLOAT   0
#define DEPTH_NORMAL_ENCODING_RGBA    1
#define DEPTH_NORMAL_ENCODING_RGBA16  2

// pscommon.fx : DecodeNormal(vNormalDepth, false).
float3 psDecodeNormal(float4 vNormalDepth)
{
    const int iInEncoding = SC2_AXIS(b_iInEncoding);
    [branch] if (iInEncoding == DEPTH_NORMAL_ENCODING_RGBA)
    {
        float3 vNormal;
        vNormal.xz = 2.0f * vNormalDepth.rg - 1.0f;
        vNormal.y = -sqrt(max(0, 1 - dot(vNormal.xz, vNormal.xz)));
        vNormal = mul((float3x3)psGlobals.mInvViewTransform, vNormal);
        vNormal = normalize(vNormal);
        return vNormal.rgb;
    }
    [branch] if (iInEncoding == DEPTH_NORMAL_ENCODING_RGBA16)
        return 2.0f * vNormalDepth.rgb - 1.0f;
    // DEPTH_NORMAL_ENCODING_FLOAT (and the trailing `else -> 0`, folded here)
    return vNormalDepth.rgb;
}

// pscommon.fx : SampleNormalDepth — .a is the decoded depth, .rgb the decoded normal.
float4 psSampleNormalDepth(Texture2D<float4> tNormalTexture, float2 vUV)
{
    float4 cSample = tNormalTexture.Sample(p_sSamp, vUV);
    float4 vNormalDepth;
    vNormalDepth.a = decodeDepth(cSample);
    vNormalDepth.rgb = psDecodeNormal(cSample);
    return vNormalDepth;
}
