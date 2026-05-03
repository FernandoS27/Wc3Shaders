// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026, Fernando Sahmkow
// See LICENSE in the repository root for full terms.
//
// Distortion pixel shader entry point.
//
// Single-permutation full-screen pass. Samples a 2-channel offset map
// (t1) and uses it to drive a chromatic-aberration blur of a scene
// texture (t0): each output channel is sampled at a slightly different
// displacement, which reads as colour fringing along the offset
// direction.
//
// The offset is decoded from the [0, 1] map sample to [-1, 1], then
// scaled by per-channel factors and added to the base UV to produce
// four sample UVs. The four samples are combined per channel as:
//
//   R  =  s1.r · 0.5  +  s2.r · 0.5
//   G  =  s2.g · 0.5  +  s3.g · 0.5
//   B  =  s4.b
//   A  =  s2.a · 0.5  +  s3.a · 0.5
//
// where s_i is `t0.Sample(uv + offset · k_i)` with the per-channel
// scale factors:
//
//   k1 = 0.0100  (R weight)
//   k2 = 0.0125  (R+G+A weight)
//   k3 = 0.0150  (G+A weight)
//   k4 = 0.0175  (B weight)
//
// The progressively larger displacements per channel produce the
// classic chromatic-aberration "channel fringe" effect — no two output
// channels see the same scene pixel.
//
// Reuses `SpritePSInput` (SV_Position + TEXCOORD0) since both shaders
// take the same PS input layout.

implementing wc3_shaders;

[shader("fragment")]
public float4 distortion_ps_main(SpritePSInput input) : SV_Target0
{
    // 1. Sample the offset map and decode to signed [-1, 1].
    float2 offsetRG = t_distortionMap.Sample(s_distortionMap, input.uv).xy;
    float2 offset   = offsetRG * 2.0 - 1.0;

    // 2. Build the four sample UVs at scales 0.01 / 0.0125 / 0.015 / 0.0175.
    float2 uv1 = input.uv + offset * 0.00999999978;  // R-feeding short-distance
    float2 uv2 = input.uv + offset * 0.0125000002;   // R+G+A medium-distance
    float2 uv3 = input.uv + offset * 0.0149999997;   // G+A long-distance
    float2 uv4 = input.uv + offset * 0.0175000001;   // B-only longest-distance

    // 3. Sample the scene at each UV.
    float4 s1 = t_distortionScene.Sample(s_distortionScene, uv1);
    float4 s2 = t_distortionScene.Sample(s_distortionScene, uv2);
    float4 s3 = t_distortionScene.Sample(s_distortionScene, uv3);
    float4 s4 = t_distortionScene.Sample(s_distortionScene, uv4);

    // 4. Combine per channel. The scaling vectors mirror the original
    //    DXBC's `mad`s exactly so the compiler keeps the dependency
    //    chain identical to the shipped shader.
    float4 result = s2 * float4(0.5, 0.5, 0.0, 0.5);
    result        = s1 * float4(0.5, 0.0, 0.0, 0.0) + result;
    result        = s3 * float4(0.0, 0.5, 0.0, 0.5) + result;
    result        = s4 * float4(0.0, 0.0, 1.0, 0.0) + result;
    return result;
}
