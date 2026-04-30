// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026, Fernando Sahmkow
// See LICENSE in the repository root for full terms.
//
// PopcornFX vertex shader entry point.
//
// Specialisation space (recovered from the original BLS bundle —
// 72 perms = 4 effective rendering modes × 8 inner-bit combinations,
// with 8 dead-axis duplicates from the engine compiling redundant
// uv_variant mirrors):
//
//   TMode         — IPopcornMode (NoUV, BasicUV, Billboard, Atlas).
//   HAS_RANDOM    — particle has a per-vertex random scalar (ATTR6).
//   HAS_VC        — particle has a per-vertex RGBA color (ATTR2).
//   HAS_NT        — particle has a per-vertex normal + tangent
//                   (ATTR1 + ATTR7) — required when the matching PS
//                   runs the lit BRDF path.
//
// The VS body is a small linear stack: pivot-blend the per-vertex
// position toward the particle origin, scale uniformly, transform to
// clip + world, then forward the active mode's per-vertex auxiliaries
// (UV variants, axis vectors, random, normal/tangent).

implementing wc3_shaders;

// PopcornFX particles author the per-vertex position as the corner of
// the quad (or the per-vertex offset for mesh particles), with the
// per-particle "pivot" living in ATTR8.xyz and a per-vertex blend
// weight in ATTR8.w. The final local-space position is a lerp from
// pivot to corner driven by the weight, then uniformly scaled by
// `popcornScale` to undo any scale baked into the mesh data.
//
//     localPos = lerp(pivot.xyz, position, pivot.w) * popcornScale
public float3 popcornBlendPivot(float3 position, float4 pivot, float scale)
{
    float3 localPos = lerp(pivot.xyz, position, pivot.www);
    return localPos * scale;
}

// Quad UV from a SV_VertexID in [0, 4).
//
//   id & 3 == 0  → (0, 0)
//   id & 3 == 1  → (0, 1)
//   id & 3 == 2  → (1, 0)
//   id & 3 == 3  → (1, 1)
//
// Matches the original engine's triangle-strip quad pattern.
public float2 popcornQuadUV(uint vertexID)
{
    uint idx = vertexID & 3u;
    return float2(float(idx >> 1), float(idx & 1u));
}

[shader("vertex")]
public PopcornVSOutput<TMode, HAS_RANDOM, HAS_VC, HAS_NT> popcorn_vs_main<
    TMode : IPopcornMode,
    let HAS_RANDOM : bool,
    let HAS_VC     : bool,
    let HAS_NT     : bool
>(PopcornVSInput input)
{
    PopcornVSOutput<TMode, HAS_RANDOM, HAS_VC, HAS_NT> output;

    // Step 1: pivot-blend + uniform scale gives us the local-space
    // position that everything downstream operates on.
    float3 localPos = popcornBlendPivot(
        input.position, input.pivot, popcornVsCB2.popcornScale);

    // Step 2: clip-space position (single-multiply object → clip
    // bypasses the world round-trip).
    output.position = transformClipPos(localPos, popcornVsCB2.worldViewProj);

    // Step 3: vertex color — straight pass-through, no remap.
    if (HAS_VC)
        output.color = input.vertColor;

    // Step 4: per-mode UV / auxiliary outputs.
    //
    // TEXCOORD0 (`output.uv`) and TEXCOORD1 (`output.slot1Out`) are
    // always packed as float4 when active — the PS picks .xy / .xyzw
    // / .x lanes per its mode (see popcorn_ps.slang). The unused
    // lanes are written here as zeros so fxc doesn't warn.
    if (TMode.hasUV)
    {
        if (TMode.isBillboard)
        {
            output.uv       = float4(popcornQuadUV(input.vertexID), 0, 0);
            output.axis0Out = input.modeSlot4;     // axis0 (.xyzw)
            output.axis1Out = input.modeSlot5;     // axis1 (.xyzw)
        }
        else if (TMode.isAtlas)
        {
            // .xy = current frame, .zw = next frame. The PS lerps
            // between two t0 samples by `lifeOut.x` (the life-ratio
            // fractional part written below).
            output.uv = float4(input.uv0, input.modeSlot4.xy);
        }
        else
        {
            // BasicUV — straight pass-through; .zw left as zeros.
            output.uv = float4(input.uv0, 0, 0);
        }
    }

    // Step 5: per-particle random scalar pass-through (only meaningful
    // when there's a UV stream — see PopcornVSOutput for the rationale).
    // Emitted as a float4 with .yzw zeroed because Slang's Conditional
    // doesn't lower to a scalar varying cleanly through fxc; the PS
    // reads only .x.
    if (HAS_RANDOM && TMode.hasUV)
        output.randomOut = float4(input.random, 0, 0, 0);

    // Step 6: world-space normal + tangent. The popcorn world matrix
    // is assumed to be a similarity transform (no shear), so we skip
    // the renormalize step the HD pipeline runs. tangent.w carries the
    // bitangent handedness sign passed straight through. Local temps
    // because `tangentWS` is `Conditional<float4, HAS_NT>` and only
    // accepts whole-value assignment.
    if (HAS_NT)
    {
        float3 worldNormal  = transformNormalRaw(input.normal,       popcornVsCB2.world);
        float3 worldTangent = transformNormalRaw(input.tangent.xyz,  popcornVsCB2.world);
        output.normalWS  = worldNormal;
        output.tangentWS = float4(worldTangent, input.tangent.w);
    }

    // Step 7: world-space position, always emitted (the PS uses it
    // for soft-particles depth reprojection, fog, IBL, etc.).
    output.worldPos = transformByWorld(localPos, popcornVsCB2.world);

    // Step 8: atlas blend factor (.x of TEXCOORD1; the PS reads only
    // .x and ignores the upper lanes). Emitted at the end so the
    // `frac` instruction lands after the worldPos transform — matches
    // the original engine's instruction ordering.
    if (TMode.isAtlas)
        output.lifeOut = float4(frac(input.modeSlot5.x), 0, 0, 0);

    return output;
}
