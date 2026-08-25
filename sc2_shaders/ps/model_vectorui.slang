// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026, Fernando Sahmkow
// See LICENSE in the repository root for full terms.
//
// Model-family SHADINGMODE_VECTORUI arm (psvectorui.fx : VectorUI) — screen-space
// splat primitives (circle / crescent) accumulated into the output colour.  Pure
// interpolant math: no textures, no material constants.  The splat-attenuation tail
// is 1 on every reachable perm (b_splatAttenuationEnabled is never set), so it DCEs.

implementing sc2_shaders;
#include "types/perm_cb.slangh"

#define VECTORUI_INVALID  0
#define VECTORUI_CIRCLE   1
#define VECTORUI_CRESCENT 2
#define c_f2PI 6.2831853071795864f

#ifndef b_renderBlack
#define b_renderBlack 0
#endif
#ifndef b_additiveSplat
#define b_additiveSplat 0
#endif
#ifndef b_iDrawType
#define b_iDrawType 0
#endif
#ifndef b_dashedCircleSplats
#define b_dashedCircleSplats 0
#endif
#ifndef b_lowEndShaders
#define b_lowEndShaders 0
#endif
#ifndef b_splatAttenuationEnabled
#define b_splatAttenuationEnabled 0
#endif

// common.fx : SplatAttenuationVectorUI — 1 unless splat attenuation is enabled.
float splatAttenuationVectorUI(float3 worldPos, float4 vPlane, float scalar, float minHeight)
{
    [branch] if (SC2_AXIS(b_splatAttenuationEnabled))
    {
        float fDist = dot(float4(worldPos, 1.0f), vPlane) * scalar;
        return 1.0f - saturate(abs(fDist));
    }
    return 1.0f;
}

// psvectorui.fx : ApplyCircleInfluence.
void applyCircleInfluence(float2 vPosition, float2 vCircleCenter, float4 vCircleRadiiParams,
                          float4 vCircleDashParams, float4 cLitColor, inout float4 cAccumulatedColor)
{
    float2 vDist = vPosition - vCircleCenter;
    float fDist = length(vDist);

    float finalFactor;
    {
        float innerFactor = fDist - (vCircleRadiiParams.x - vCircleRadiiParams.z);
        innerFactor = saturate(innerFactor * vCircleRadiiParams.w);
        float outerFactor = fDist - vCircleRadiiParams.y;
        outerFactor = 1.0f - saturate(outerFactor * vCircleRadiiParams.w);
        finalFactor = innerFactor * outerFactor;
    }

    [branch] if (SC2_AXIS(b_dashedCircleSplats))
    {
        float2 vDir = normalize(vDist);
        float2 xAxis = vCircleDashParams.zw;
        float2 yAxis = float2(xAxis.y, -xAxis.x);
        float2 vRotatedDir = (xAxis * vDir.x) + (yAxis * vDir.y);

        float fDashAngle = vCircleDashParams.x;
        float fPercentSolid = vCircleDashParams.y;
        float fAvgRadius = (vCircleRadiiParams.x + vCircleRadiiParams.y) * 0.5f;
        float fInvAvgRadius = 1.0f / fAvgRadius;

        float fSegmentLength = fDashAngle * fAvgRadius;
        float fRange = vCircleRadiiParams.z * fInvAvgRadius;
        float fOuterAngle = (fSegmentLength * fPercentSolid * fInvAvgRadius) - fRange;

        float fAngle = acos(vRotatedDir.x);
        fAngle = (vRotatedDir.y > 0) ? fAngle : (c_f2PI) - fAngle;

        float fPartial = fmod(fAngle, fDashAngle);
        fPartial = max(fPartial, fDashAngle - fPartial);
        float fDashedFactor = saturate((fPartial - fOuterAngle) / fRange);

        finalFactor = min(fDashedFactor, finalFactor);
    }

    float4 color = cLitColor * finalFactor;
    [branch] if (SC2_AXIS(b_additiveSplat))
    {
        cAccumulatedColor += color;
    }
    else
    {
        cAccumulatedColor.xyz = lerp(cAccumulatedColor.xyz, color.xyz, color.w);
        cAccumulatedColor.w *= (1.0f - color.w);
    }
}

// psvectorui.fx : ApplyCrescentInfulence.
void applyCrescentInfluence(float2 vPosition, float2 vCircleCenter, float4 vCircleRadiiParams,
                            float3 vCameraToVert, float3 vInsideCircleRatios, float4 cLitColor,
                            inout float4 cAccumulatedColor)
{
    float2 vDist = vPosition - vCircleCenter;
    float fDist = length(vDist);

    float innerFactor = fDist - (vCircleRadiiParams.x - vCircleRadiiParams.z);
    innerFactor = saturate(innerFactor * vCircleRadiiParams.w);
    float outerFactor = fDist - vCircleRadiiParams.y;
    outerFactor = saturate(outerFactor * vCircleRadiiParams.w);
    outerFactor *= outerFactor;
    outerFactor = 1.0f - outerFactor;

    float finalFactor = innerFactor * outerFactor;

    [branch] if (SC2_AXIS(b_lowEndShaders))
    {
        cAccumulatedColor = cLitColor * float4(1.0f, 1.0f, 1.0f, finalFactor * 0.5f);
        cAccumulatedColor.w = 1.0f - cAccumulatedColor.w;
        return;
    }

    float4 innerColor = float4(cLitColor.x, cLitColor.y, cLitColor.z, cLitColor.w * 0.5f);
    float4 edgeColor = float4(cLitColor.x, cLitColor.y, cLitColor.z, 1.0f - cLitColor.w);

    cAccumulatedColor = innerColor * float4(1.0f, 1.0f, 1.0f, outerFactor);
    cAccumulatedColor.w = 1.0f - cAccumulatedColor.w;
    cAccumulatedColor = lerp(cAccumulatedColor, edgeColor, finalFactor);

    float2 vOffset = normalize(vCameraToVert.xy) * vInsideCircleRatios.x * vCircleRadiiParams.y;
    float2 vInnerCircleCenter = vCircleCenter.xy + vOffset;
    vDist = vPosition - vInnerCircleCenter;
    fDist = length(vDist);

    float innerCircleInsideBound = vCircleRadiiParams.y * vInsideCircleRatios.y;
    float innerCircleFalloffDistance = 1.0f / (vCircleRadiiParams.y * vInsideCircleRatios.z);
    float innerCircleFactor = 1.0f - saturate((fDist - innerCircleInsideBound) * innerCircleFalloffDistance);
    cAccumulatedColor.w = lerp(cAccumulatedColor.w, 1.0f, innerCircleFactor);
}

// psvectorui.fx : VectorUI.
float4 vectorUI(ModelPSInput vertOut)
{
#if b_renderBlack
    return float4(0.0f, 0.0f, 0.0f, 1.0f);
#else
    float4 accumulatedColor;
    [branch] if (SC2_AXIS(b_additiveSplat))
        accumulatedColor = float4(1.0f, 0.0f, 0.0f, 0.0f);
    else
        accumulatedColor = float4(0.0f, 0.0f, 0.0f, 1.0f);

    // PIXEL_SHADER_VERSION >= PS_20 is always true for the ps_5_0 target.
  #if (b_iDrawType == VECTORUI_CIRCLE)
    // VectorUI0 = camera pos; VectorUI1 = (inner,outer,falloff,1/falloff);
    // VectorUI2 = (segCount, percentSolid, rotationAxis.xy).
    applyCircleInfluence(vertOut.WorldPos().xy, vertOut.VectorUI0().xy, vertOut.VectorUI1(),
                         vertOut.VectorUI2(), vertOut.VertexColor(), accumulatedColor);
  #elif (b_iDrawType == VECTORUI_CRESCENT)
    // VectorUI2 = (inner offset, inner boundary, inner boundary falloff ratios).
    applyCrescentInfluence(vertOut.WorldPos().xy, vertOut.VectorUI0().xy, vertOut.VectorUI1(),
                           vertOut.EyeToVertex(), vertOut.VectorUI2().xyz, vertOut.VertexColor(),
                           accumulatedColor);
  #else
    return float4(1.0f, 0.0f, 0.0f, 1.0f);
  #endif

    [branch] if (SC2_AXIS(b_additiveSplat))
        accumulatedColor = saturate(accumulatedColor);
    else
        accumulatedColor = saturate(float4(accumulatedColor.xyz, 1.0f - accumulatedColor.w));

    // Attenuation factor (SplatAttenuationVectorUI — 1 on every reachable perm).
    accumulatedColor.w *= splatAttenuationVectorUI(vertOut.WorldPos().xyz, vertOut.Vector4(),
                                                   vertOut.VectorUI0().w, vertOut.WorldPos().w);
    return accumulatedColor;
#endif
}
