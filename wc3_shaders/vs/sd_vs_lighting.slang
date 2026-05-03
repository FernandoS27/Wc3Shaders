// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026, Fernando Sahmkow
// See LICENSE in the repository root for full terms.
//
// Per-vertex Blinn-Phong-ish lighting used by the SD highspec pipeline.
//
// The SD classic PS does no per-pixel shading — it modulates the
// interpolated vertex color with texture samples and that's it. So
// lighting has to be fully evaluated here, at the vertex, across the
// (up to 8) active lights.

implementing wc3_shaders;

// SD attenuation matches the retail SD highspec VS exactly: 1/(1 + k*d²),
// i.e. quadratic falloff on squared distance — not linear-in-distance.
// The 0.000500000024 coefficient is the engine's `attenuation2` value;
// taking dSq directly avoids the sqrt the linear form would have needed.
public float sdAttenuation(float distSq)
{
    return 1.0 / (1.0 + 0.000500000024 * distSq);
}

// Accumulate Lambertian diffuse and ambient contributions from up to
// NUM_LIGHTS (a compile-time count so the loop unrolls into a straight
// sequence and there are no dynamic-indexed constant buffer loads).
// Returns the final per-vertex lit color, clamped to [0, 1] so over-
// saturated sums don't punch past the legacy output gamut.
public float3 sdComputePerVertexLighting<let NUM_LIGHTS : int>(
    float3 worldNormal, float3 worldPos, float4 material, ShaderLight lights[8])
{
    float3 diffAccum = float3(0, 0, 0);
    float3 ambAccum  = float3(0, 0, 0);

    [ForceUnroll]
    for (int i = 0; i < NUM_LIGHTS; i++)
    {
        float3 lightVec;
        float  atten;

        if (lights[i].type > 0.0)
        {
            // Point light: unit direction toward the light + quadratic
            // falloff on squared distance (see sdAttenuation).
            float3 delta = lights[i].position - worldPos;
            float  dSq = dot(delta, delta);
            float  d = sqrt(dSq);
            lightVec = delta / d;
            atten    = sdAttenuation(dSq);
        }
        else
        {
            // Directional light: `position` is already a unit direction
            // and attenuation is a no-op.
            lightVec = lights[i].position;
            atten    = 1.0;
        }

        // Lambertian NdotL; ambient accumulates unconditionally so
        // unlit sides still pick up the light's ambient contribution.
        float NdotL = saturate(dot(worldNormal, lightVec));
        diffAccum += lights[i].diffuse * NdotL * atten;
        ambAccum  += lights[i].ambient * atten;
    }

    return saturate(diffAccum * material.xyz + ambAccum * material.xyz);
}
