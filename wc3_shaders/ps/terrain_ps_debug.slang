// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026, Fernando Sahmkow
// See LICENSE in the repository root for full terms.
//
// Terrain TINT_OVERRIDE debug visualisation router.
//
// When TINT_OVERRIDE is enabled the engine bypasses the post-grading
// 3×3 colour matrix and routes one of nine "dev mode" outputs through
// SV_Target0 instead. The mode index lives in `psCB3.debugMode` (= the
// .y lane of the same b3 slot the override bitmask hides in .x).
//
// The mode set is:
//
//   0 — pre-color-grade lit color × (1 - AO)
//   1 — undefined (engine left this path uninitialised — preserved
//       as an empty branch so the dxbc movc tree matches)
//   2 — encoded world normal           ((n + 1) · 0.5)
//   3 — current-mip indicator          (per-LOD palette × frac fade)
//   4 — light-count tile id            (per-bucket palette)
//   5 — π · domSpec                    (specular component)
//   6 — π · domAmbient                 (ambient component)
//   7 — π · (domAmbient + domSpec)     (sum of both)
//   8 — π · totalLight                 (final lit, no AO modulation)
//
// Modes 5..8 lean on the dominant-light intermediates the lighting body
// stashes in `TerrainDominantLight.{domAmbient, domSpec}`.

implementing wc3_shaders;

// LOD-band palette + half-frac brightness fade used by mode 3.
struct LodViz { float3 color; float fade; };

LodViz pickLodColor(float lodRaw)
{
    LodViz r;
    r.fade = frac(lodRaw) * 0.5 + 0.5;

    int lodInt = (int)lodRaw;
    switch (lodInt)
    {
        case 0:  r.color = float3(1, 0, 0); break;
        case 1:  r.color = float3(0, 1, 0); break;
        case 2:  r.color = float3(0, 0, 1); break;
        case 3:  r.color = float3(1, 1, 0); break;
        case 4:  r.color = float3(1, 0, 1); break;
        default: r.color = float3(1, 1, 1); r.fade = 1.0; break;
    }
    return r;
}

// Light-count bucket palette used by mode 4.
float3 lightCountColor(int bucket)
{
    switch (bucket)
    {
        case 0:  return float3(0, 0, 0);
        case 1:  return float3(0.05, 0, 0);
        case 2:  return float3(0, 0.05, 0);
        case 3:  return float3(0, 0, 0.1);
        case 4:  return float3(1, 0, 0);
        case 5:  return float3(0, 1, 0);
        case 6:  return float3(0, 0, 1);
        case 7:  return float3(1, 1, 0);
        case 8:  return float3(1, 1, 1);
        default: return float3(0.25, 0.25, 0.25);
    }
}

// Top-level debug router. `preGradeLit` is the pre-color-matrix lit
// colour (= mode 0), `domAmbient`/`domSpec` are the saved dominant-
// light pieces from the lighting pass, `worldNormal` is the already-
// normalised TBN result, and `lodRaw` is the active mip level the
// caller queried from whichever material texture it samples first
// (terrain → splat-color, foliage → albedo). Pulling the LOD query
// up to the caller keeps this helper texture-agnostic so both
// pipelines can share it without dragging unused texture bindings
// into each other's permutations.
public float3 terrainDebugVisualize(
    float3 preGradeLit, float3 totalLight,
    float3 worldNormal, float lodRaw,
    float3 domAmbient, float3 domSpec)
{
    uint mode = asuint(psCB3.debugMode);
    if (mode == 0)
        return preGradeLit;

    // Mode 1 — engine left this path uninitialised; preserve the
    // engine's quirk by returning an undefined value (the original
    // shader leaves r2.xyz untouched, so we just return zero — the
    // upper layers' movc tree will route this branch only when
    // explicitly mode 1, which the engine never asks for).
    if (mode == 1)
        return float3(0, 0, 0);

    if (mode == 2)
    {
        // Encoded world normal: (n + 1) · 0.5.
        return worldNormal * 0.5 + 0.5;
    }
    if (mode == 3)
    {
        LodViz l = pickLodColor(lodRaw);
        return l.color * l.fade;
    }
    if (mode == 4)
    {
        return lightCountColor((int)asint(psCB2.lightCount));
    }

    // Modes 5..8 — visualise the lit-pipeline subterms scaled by π.
    float3 modeSpec = PI.xxx * domSpec;
    float3 modeAmb  = PI.xxx * domAmbient;
    float3 modeSum  = (domAmbient + domSpec) * PI.xxx;
    int4 modeMask = asint(psCB3.debugMode.xxxx) - int4(5, 6, 7, 8);
    bool4 hit = modeMask == int4(0, 0, 0, 0);

    float3 picked = hit.w ? totalLight : preGradeLit;   // mode 8 → π · total
    picked = hit.z ? modeSum : picked;                   // mode 7 → π · (amb+spec)
    picked = hit.y ? modeAmb : picked;                   // mode 6 → π · ambient
    picked = hit.x ? modeSpec : picked;                  // mode 5 → π · spec
    return picked;
}
