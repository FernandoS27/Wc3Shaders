// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026, Fernando Sahmkow
// See LICENSE in the repository root for full terms.
//
// Water pixel shader entry point.
//
// Specialisation space:
//   TFog — fog-mode policy (the only axis the water pass varies).
//
// The four water permutations map to:
//   FogNone        — no fog
//   FogLinear      — linear fog between fogStart and fogEnd
//   FogExponential — exp(−density · z)
//   FogExp2        — exp(−density² · z²)
//
// All the heavy lifting lives in `waterPSBody` (ps/water_ps_body.slang);
// this file just binds the policy and returns the result.

implementing wc3_shaders;

[shader("fragment")]
public float4 water_ps_main<TFog : IFogMode>(WaterPSInput input)
    : SV_Target0
{
    return waterPSBody<TFog>(input);
}
