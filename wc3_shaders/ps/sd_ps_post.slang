// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026, Fernando Sahmkow
// See LICENSE in the repository root for full terms.
//
// SD-on-HD post helpers: MRT-normal packing and a debug visualiser
// parallel to the HD one in ps_post.slang. No tangent frame is needed,
// so the MRT normal is just the interpolated world normal; and there's
// no "normal length" visualisation because there's nothing to decode.

implementing wc3_shaders;

// Pack the interpolated world-space normal into the unorm MRT target.
public float3 sdComputeMRTNormal(PSInput input, bool isFrontFace)
{
    float3 normalWS = normalize(input.normalWS.xyz);
    normalWS = isFrontFace ? normalWS : -normalWS;
    return normalWS * NORMAL_PACK_SCALE + NORMAL_PACK_BIAS;
}

// Debug visualiser matching ps_post.slang's `debugVisualize`, minus the
// HD-specific normal-length mode. Mode 0 passes the final color
// through unchanged (the SD entry shader has already applied sRGB
// encoding when needed); modes 1..4 mirror the HD variants.
public float4 sdDebugVisualize(float4 result, PSInput input, bool isFrontFace)
{
    float3 normalWS = normalize(input.normalWS.xyz);
    normalWS = isFrontFace ? normalWS : -normalWS;

    float lodVal = t_albedo.CalculateLevelOfDetail(s_albedo, input.uv.xy);
    float lodFrac = frac(lodVal) * DBG_LOD_BIAS + DBG_LOD_BIAS;
    int lodInt = (int)lodVal;

    float3 lodColor;
    switch (lodInt)
    {
        case 0: lodColor = float3(1, 0, 0); break;
        case 1: lodColor = float3(0, 1, 0); break;
        case 2: lodColor = float3(0, 0, 1); break;
        case 3: lodColor = float3(1, 1, 0); break;
        case 4: lodColor = float3(1, 0, 1); break;
        default: lodColor = float3(1, 1, 1); lodFrac = 1; break;
    }

    int dbgMode = asint(psCB3.debugMode);

    if (dbgMode == 0)
    {
        return result;
    }
    else if (dbgMode == 1)
    {
        // Raw albedo.
        return float4(t_albedo.Sample(s_albedo, input.uv.xy).xyz, result.w);
    }
    else if (dbgMode == 2)
    {
        // World normal visualisation — simpler than the HD path since
        // we have no tangent frame / normal-length to mix in.
        return float4((float3(1, 1, 1) + normalWS) * DBG_NORMAL_SCALE, result.w);
    }
    else if (dbgMode == 3)
    {
        return float4(lodColor * lodFrac, result.w);
    }
    else if (dbgMode == 4)
    {
        float3 lcColor;
        uint lc = asuint(sdPsCB2.lightCountSlot.z);
        switch (lc)
        {
            case 0: lcColor = float3(0, 0, 0); break;
            case 1: lcColor = DBG_LIGHT1_COLOR; break;
            case 2: lcColor = DBG_LIGHT2_COLOR; break;
            case 3: lcColor = DBG_LIGHT3_COLOR; break;
            case 4: lcColor = float3(1, 0, 0); break;
            case 5: lcColor = float3(0, 1, 0); break;
            case 6: lcColor = float3(0, 0, 1); break;
            case 7: lcColor = float3(1, 1, 0); break;
            case 8: lcColor = float3(1, 1, 1); break;
            default: lcColor = DBG_LIGHT_DEFAULT; break;
        }
        return float4(lcColor, result.w);
    }
    else
    {
        if (dbgMode >= 5 && dbgMode <= 8)
            return float4(0, 0, 0, result.w);
        return result;
    }
}
