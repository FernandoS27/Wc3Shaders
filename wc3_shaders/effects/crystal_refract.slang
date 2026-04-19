// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026, Fernando Sahmkow
// See LICENSE in the repository root for full terms.
//
// Crystal refraction helpers.
//
// The crystal material is a variant of the HD mesh shader for transparent
// / refractive props (gem surfaces, crystal formations, glass). It layers
// two pieces of math on top of the standard HD lighting:
//
//   1. An "albedo refraction": the albedo map is sampled at `uv` AND at a
//      second UV offset by the projected world normal, and the two
//      samples are lerped by (1 - refractMask). A fully refracted pixel
//      (refractMask = 0) takes the offset sample entirely — giving the
//      characteristic "smear" of whatever is painted behind the facet —
//      while an opaque pixel (refractMask = 1) keeps the centre sample.
//      Only the rgb channels are blended; the alpha is always the centre
//      sample's .w (since it drives the cloak / fresnel alpha downstream
//      and we don't want the refraction to perturb transparency shape).
//
//   2. A "refract-modulated Fresnel alpha": the usual fresnel-overlay
//      weight (fresnelAlpha · (1 - NdotV)²) is remapped so fully-refracted
//      pixels use a flat CRYSTAL_FRESNEL_NOREFRACT (0.2) floor, ensuring
//      a transparent body still picks up a visible hero rim. Opaque
//      pixels still use the authored fresnelAlpha.
//
// These two helpers are the *only* pieces unique to the crystal path —
// everything else in the crystal pixel shader (TBN, BRDF, IBL, cloak,
// ML blend, fog, colour matrix, debug modes) is reused verbatim from the
// shared HD primitives in math/ effects/ and ps/.

implementing wc3_shaders;

// Per-pixel UV-space radius of the refracted albedo fetch. Grazing-angle
// pixels (small |normalWS.z|) get a larger offset, which produces the
// wider colour smear around a facet's silhouette; near-axis pixels see
// almost no offset. 0.02 matches the legacy crystal shader.
public static const float CRYSTAL_REFRACT_SCALE    = 0.0199999996;

// Fresnel-overlay weight substituted when the surface is fully refracted
// (refractMask = 0). Numerically equal to cloak's base alpha bias but
// used here only as a Fresnel mix endpoint — deliberately given its own
// name so a future tweak to either constant does not drift silently.
public static const float CRYSTAL_FRESNEL_NOREFRACT = 0.200000003;

// Sample the albedo map at the UV and at the refraction-offset UV, lerp
// between them by (1 - refractMask), and preserve the centre sample's
// alpha. Returns rgba ready to drop into the existing HD shading flow.
// `refractMask` is the caller-fetched `t_orm.Sample().x` (possibly
// substituted by the DEBUG_VIS override) — taken as a parameter rather
// than resampled so a single fetch can be reused for both the albedo
// blend and the downstream Fresnel modulation.
public float4 sampleCrystalAlbedo(float2 uv, float3 normalWS, float refractMask)
{
    float4 centre = t_albedo.Sample(s_albedo, uv);

    // Project the world normal into view space by flipping z, then form
    // the UV offset. `abs()` on z keeps the sample direction consistent
    // for both front- and back-facing pixels so a cloaked back face
    // still refracts toward its silhouette.
    float3 viewNrm  = float3(normalWS.xy, -normalWS.z);
    float  scale    = CRYSTAL_REFRACT_SCALE / abs(viewNrm.z);
    float2 uvRefract = viewNrm.xy * scale + uv;

    float3 offset = t_albedo.Sample(s_albedo, uvRefract).xyz;
    float3 rgb    = lerp(centre.xyz, offset, 1.0 - refractMask);
    return float4(rgb, centre.w);
}

// Remap the Fresnel-overlay alpha parameter by the refract mask. At
// refractMask = 1 the result is the authored `fresnelAlpha`; at
// refractMask = 0 it clamps to CRYSTAL_FRESNEL_NOREFRACT (0.2) so the
// rim stays visible through a fully transparent pixel. Plug the return
// value into `applyFresnelOverlay` / `applyFresnelOverlayML` in place of
// `psCB2.fresnelColor.w`.
public float refractModulatedFresnelAlpha(float fresnelAlpha, float refractMask)
{
    return (1.0 - refractMask) * (CRYSTAL_FRESNEL_NOREFRACT - fresnelAlpha) + fresnelAlpha;
}
