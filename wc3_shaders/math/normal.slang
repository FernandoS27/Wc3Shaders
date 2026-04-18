// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026, Fernando Sahmkow
// See LICENSE in the repository root for full terms.
//
// Normal-map decoding and tangent-frame transforms used by the HD path.
//
// The engine ships normal maps in a DXT5-style two-channel layout
// (X in .R, Y in .G) and stores a high-precision X factor in the alpha
// channel. Z is reconstructed from XY + this extra precision channel.

implementing wc3_shaders;

// Decode a two-channel (+ alpha-biased X) normal-map sample back into a
// unit-length tangent-space normal.
//   nx = 2 · sample.x · sample.w − 1
//   ny = −(2 · sample.y − 1)
//   nz = sqrt(1 − nx² − ny²)
// The sign flip on Y matches the engine's convention (texture Y-down →
// world Y-up). The max() guards against slightly out-of-range XY pairs
// producing a negative nz² under the sqrt.
public float3 decodeNormalMap(float3 texXYW)
{
    float nx = 2.0 * texXYW.x * texXYW.z - 1.0;
    float ny = -(texXYW.y * 2.0 - 1.0);
    float nz2 = max(0.0, 1.0 - nx * nx - ny * ny);
    float nz = sqrt(nz2);
    return normalize(float3(nx, ny, nz));
}

// Tangent-space to world-space conversion.
// Bitangent = cross(normal, tangent) · tangentSign — expressed as a
// hand-unrolled cross product to let the compiler share the yzx/zxy
// swizzles with any downstream math. tangentSign (tangentWS.w) comes
// from the VS and encodes handedness to keep the basis consistent
// across mirrored UV islands.
public float3 tangentToWorld(float3 tsNormal, float3 tangentWS, float3 normalWS, float tangentSign)
{
    float3 bt = tangentWS.yzx * normalWS.zxy - normalWS.yzx * tangentWS.zxy;
    bt *= tangentSign;
    float3 wn = tsNormal.x * tangentWS + tsNormal.y * bt + tsNormal.z * normalWS;
    return normalize(wn);
}

// Flip the world-space normal for back-facing fragments. The HD pipeline
// draws double-sided foliage / flat cards as normal triangles, so we
// need a runtime flip to avoid reversed shading on the back side.
public float3 orientNormal(float3 normal, bool isFrontFace)
{
    return isFrontFace ? normal : -normal;
}
