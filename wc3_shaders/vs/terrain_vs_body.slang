// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026, Fernando Sahmkow
// See LICENSE in the repository root for full terms.
//
// Terrain vertex shader body. Two compile-time axes:
//
//   HAS_VERTEX_COLOR — `true` reads ATTR2 (per-vertex RGBA) into the
//                       PS COLOR0 slot; `false` substitutes the
//                       constant tint from `terrainVsCB2.tintColor`.
//   HAS_SHADOWS      — emit the three cascade-clip outputs. The engine
//                       packs `RECEIVE_SHADOWS && !SHADOW_PASS` into
//                       this single axis — a draw rendering its own
//                       shadow caster pass never emits cascade UVs
//                       even when the receive flag is also set, which
//                       lets the engine reuse the no-shadow PS perms
//                       for the caster pass.
//
// Notable terrain-specific details:
//
//   * Degenerate-vertex kill: a vertex whose ATTR6 (`boneIDs`) is the
//     four-component sentinel `(0xFFFF,…)` is replaced with the
//     clip-space point `(0, 0, -2, 1)` — well behind the near plane,
//     so the entire triangle gets clipped out by the rasterizer. The
//     engine uses this to drop unused vertices from a shared VB
//     without rebuilding the buffer.
//
//   * Normal Z-remap: keeps `n.xy` as-is and remaps `n.z` through
//     `z' = saturate(z) · (z⁸ - z) + z` before normalising. The curve
//     flattens the normal toward the surface plane on shallow slopes
//     (z near 1 → still ~1) and pushes steep z's toward zero — a
//     terrain-specific bias that softens cliff/wall normals so the
//     shading doesn't reveal the underlying tile geometry.

implementing wc3_shaders;

// Re-evaluate `n.z` through the terrain-flatten curve, then transform
// the (xy, z') normal by the world 3×3 rotation rows. Returns a
// world-space, unit-length normal.
float3 terrainTransformNormal(float3 nObject, float4 worldRow0,
                              float4 worldRow1, float4 worldRow2)
{
    // z⁸ - z, expressed as ((z²)²)² - z so fxc keeps the chain in three
    // multiplies + one mad.
    float z2 = nObject.z * nObject.z;
    float z4 = z2 * z2;
    float z8MinusZ = z4 * z4 - nObject.z;

    // saturate(z) · (z⁸ - z) + z. saturate kills the negative tail so
    // the curve only kicks in on the upward-facing hemisphere.
    float zCurved = saturate(nObject.z) * z8MinusZ + nObject.z;

    float3 nCurved = float3(nObject.xy, zCurved);
    nCurved = nCurved * rsqrt(dot(nCurved, nCurved));

    // World 3×3: each cb2[4..6].xyz gives one rotation row. Express it
    // as a column-by-column accumulate so fxc emits the same mul/mad
    // chain the original engine shader did.
    float3 result = nCurved.y * worldRow1.xyz;
    result = nCurved.x * worldRow0.xyz + result;
    return nCurved.z * worldRow2.xyz + result;
}

// World 3×3 applied to an object-space direction. `worldRow0/1/2.xyz`
// are the rows of the rotation; their .w lanes are unused. No
// renormalisation — the tangent's handedness is later tracked
// through the .w bit, so the magnitude is irrelevant.
float3 terrainTransformDirection(float3 dObject, float4 worldRow0,
                                 float4 worldRow1, float4 worldRow2)
{
    float3 result = dObject.y * worldRow1.xyz;
    result = dObject.x * worldRow0.xyz + result;
    return dObject.z * worldRow2.xyz + result;
}

// True iff every component of `boneIDs` is the engine's degenerate-
// vertex sentinel.
bool isDegenerateVertex(uint4 boneIDs)
{
    bool4 m = (int4)boneIDs == int4(0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF);
    return m.x && m.y && m.z && m.w;
}

// Terrain VS body. Returns the fully populated VS output payload; the
// entry shader thin-wrappers this and forwards it.
public TerrainVSOutput<HAS_SHADOWS> terrainVSBody<
    let HAS_VERTEX_COLOR : bool,
    let HAS_SHADOWS      : bool>(TerrainVSInput input)
{
    TerrainVSOutput<HAS_SHADOWS> output;

    // 1. Splat layer indices pass through to the PS unchanged.
    output.boneIDs = input.boneIDs;

    // 2. Clip-space position via worldViewProj. The degenerate-vertex
    // sentinel selects the off-screen clip-out coordinate.
    float4 clipPos = mul(terrainVsCB2.worldViewProj, float4(input.position, 1.0));
    output.position = isDegenerateVertex(input.boneIDs)
        ? float4(0, 0, -2, 1)
        : clipPos;

    // 3. COLOR0 — vertex stream or constant tint.
    output.color = HAS_VERTEX_COLOR ? input.vertColor
                                    : terrainVsCB2.tintColor;

    // 4. UV0 / UV1 packed into a single float4 slot.
    output.texCoord = float4(input.uv0, input.uv1);

    // 5. World-space position. The translation lives in cb2[7].xyz; the
    // .w lane carries 0 (a dead slot the PS reserves for fog data on the
    // HD path — terrain doesn't use it but the layout stays consistent).
    float3 worldP = terrainTransformDirection(
        input.position,
        terrainVsCB2.worldRow0, terrainVsCB2.worldRow1, terrainVsCB2.worldRow2);
    output.worldPos = float4(worldP + terrainVsCB2.worldTranslate.xyz, 0);

    // 6. World-space remapped normal.
    output.worldNormal = terrainTransformNormal(
        input.normal,
        terrainVsCB2.worldRow0, terrainVsCB2.worldRow1, terrainVsCB2.worldRow2);

    // 7. World-space tangent. Tangent .w (handedness) passes through
    // unchanged so the PS can reconstruct the bitangent sign later.
    float3 tangentWS = terrainTransformDirection(
        input.tangent.xyz,
        terrainVsCB2.worldRow0, terrainVsCB2.worldRow1, terrainVsCB2.worldRow2);
    output.worldTangent = float4(tangentWS, input.tangent.w);

    // 8. Cascade UVs (only when receiving shadows, never on a caster
    // pass). The pre-shadow world-space position uses the full 4×4 in
    // cb2[0..3] — distinct from the world matrix in cb2[4..7], because
    // the engine bakes its own near-shadow bias / clip-plane fix into
    // the shadow path's transform.
    if (HAS_SHADOWS)
    {
        float4 shadowPos = mul(terrainVsCB2.shadowWorld,
                               float4(input.position, 1.0));
        output.shadowClip0 = mul(vsCB1.cascade0, shadowPos);
        output.shadowClip1 = mul(vsCB1.cascade1, shadowPos);
        output.shadowClip2 = mul(vsCB1.cascade2, shadowPos);
    }

    return output;
}
