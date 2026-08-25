// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026, Fernando Sahmkow
// See LICENSE in the repository root for full terms.
//
// Model-family fog (psfog.fx : ApplyFog), transcribed faithfully.
//
// Two static branches:
//   b_UseLightingRegions — blend four per-region fog colours by `mask`.
//   b_fogScaleByAlpha    — additive-effect fog (bend toward fog, then to 0);
//                          else the simple colour-replace lerp.
// The lighting-regions / scale-by-alpha arms reference constants not yet bound
// (p_lightSetups[]) — they compile only when their define is set, which the
// standard-mode minimal permutations never do; they'll bind their constants
// when those features are transcribed.

implementing sc2_shaders;
#include "types/perm_cb.slangh"

// Axes read as CODE need a file-local default: only NONZERO axes are passed
// as -D, so a zero one would otherwise be an undefined identifier.
#ifndef b_fogScaleByAlpha
#define b_fogScaleByAlpha 0
#endif
#ifndef b_useBlendAdd
#define b_useBlendAdd 0
#endif

float4 applyFog(ModelPSInput vertOut, float4 cColor, float fAlpha, float4 mask)
{
    float3 cFogColor = float3(0, 0, 0);

#if b_UseLightingRegions
    cFogColor = psGlobals.lightSetups[0].cFogColor * mask.x
              + psGlobals.lightSetups[1].cFogColor * mask.y
              + psGlobals.lightSetups[2].cFogColor * mask.z
              + psGlobals.lightSetups[3].cFogColor * mask.w;
#else
    cFogColor = psGlobals.cFogColor.xyz;
#endif

    [branch] if (SC2_AXIS(b_fogScaleByAlpha))
    {
        // Additive effects: bend lighting toward fog colour, then toward 0 to fade.
        float fogAtten = saturate(vertOut.FogColor().w);
        float fS = saturate(dot(cColor.rgb, cColor.rgb) * cColor.a);
        cColor.rgb = lerp(cColor.xyz, cFogColor.rgb + cColor.rgb, fogAtten * fS);
        [branch] if (SC2_AXIS(b_useBlendAdd))
            cColor.rgb = lerp(cColor.rgb, cFogColor.rgb * cColor.a, fogAtten * fogAtten);
        else
            cColor.rgb = lerp(cColor.rgb, float3(0, 0, 0), fogAtten * fogAtten);
    }
    else
    {
        // Simple case: colour-replace lerp.  1-minus is free (the saturate forced a
        // mov_sat), and lerping "to" cColor saves a mov.
        float fogAtten = saturate(1 - vertOut.FogColor().w);
        cColor.rgb = lerp(cFogColor.rgb, cColor.rgb, fogAtten);
    }

    return cColor;
}
