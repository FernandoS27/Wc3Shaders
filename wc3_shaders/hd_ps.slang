// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026, Fernando Sahmkow
// See LICENSE in the repository root for full terms.
//
// HD pixel shader entry point.
//
// Specialisation space:
//   TFog              — fog-mode policy.
//   TAlpha            — alpha-test policy.
//   TMat              — material policy (standard vs. multi-layer).
//   HAS_IBL           — route to the full IBL / Cook-Torrance body.
//   HAS_EXTRA_VERTS   — VS passed shadowClip0-2 cascade coords; the IBL
//                        body samples the three shadow cascades and
//                        attenuates the main directional light.
//   IS_DEPTH_PREPASS  — depth-only pass (no color).
//   IS_MRT            — also write the deferred-G-buffer targets.
//   HAS_DEBUG_VIS     — debug visualisation enabled.
//
// The body dispatches to one of {psIBLBody, psMultiLayerBody,
// psStandardBody} based on the flags, then performs post-processing
// (alpha test, fog, color matrix, MRT normal pack, debug overlay) in
// the order the original shader used.

implementing wc3_shaders;

[shader("fragment")]
public PSOutput<IS_MRT> ps_main<
    TFog     : IFogMode,
    TAlpha   : IAlphaTest,
    TMat     : IMaterialLayer,
    let HAS_IBL            : bool,
    let HAS_EXTRA_VERTS    : bool,
    let IS_DEPTH_PREPASS   : bool,
    let IS_MRT             : bool,
    let HAS_DEBUG_VIS      : bool
>(PSInput input, uint frontFace : SV_IsFrontFace)
{
    PSOutput<IS_MRT> output;
    bool isFrontFace = frontFace != 0;

    // Non-IBL / non-prepass draws discard fragments flagged by the VS
    // as "behind the fog-clip plane" before doing any further work.
    // The IBL path delays this until after the body (it needs the
    // world-space normal anyway) and the prepass runs it internally.
    if (!HAS_IBL && !IS_DEPTH_PREPASS)
        backfaceDiscard(input.worldPos.w);

    if (IS_DEPTH_PREPASS)
    {
        // Depth-only path: either run the alpha-tested prepass or the
        // cheap stub, then return a zero color (only depth is needed).
        // Both branches already discard backfaces / failed alpha tests
        // themselves; do NOT follow with an unconditional `discard` —
        // that would kill every surviving fragment and suppress the
        // rasterizer's depth write, defeating the whole prepass.
        if (!TAlpha.enabled)
            depthPrepassStub(input);
        else
            depthPrepassBody<TAlpha, TMat>(input, isFrontFace);
        output.color = float4(0, 0, 0, 0);
        return output;
    }

    // Dispatch to the appropriate lit body.
    float4 result;
    if (HAS_IBL)
        result = psIBLBody<TMat, HAS_EXTRA_VERTS, HAS_DEBUG_VIS>(input, isFrontFace);
    else if (TMat.hasMultiLayer)
        result = psMultiLayerBody<HAS_DEBUG_VIS>(input, isFrontFace);
    else
        result = psStandardBody<HAS_DEBUG_VIS>(input, isFrontFace);

    // MRT path: alpha test, extra "are we opaque enough for the
    // G-buffer" test, and G-buffer payloads.
    float3 mrtNormal = float3(0, 0, 0);
    float preAlpha = 0;
    if (IS_MRT)
    {
        TAlpha.test(result.w, psCB2.alphaRef);

        // Deferred-only alpha gate — avoids writing partial-coverage
        // pixels to the opaque G-buffer.
        float mrtAlpha = t_albedo.Sample(s_albedo, input.uv.xy).w;
        mrtAlphaTest(mrtAlpha, psCB2.inverseSoftness);

        mrtNormal = computeMRTNormal(input, isFrontFace);

        // Store the pre-fog alpha — used by the deferred resolver to
        // reconstruct the surface opacity.
        preAlpha = result.w;
    }

    // Non-MRT path runs the alpha test here since the MRT block ran
    // it above. Running it before fog avoids paying the fog apply()
    // cost on discarded pixels.
    if (!IS_MRT)
        TAlpha.test(result.w, psCB2.alphaRef);

    // When fog is active we pre-multiply by inverseSoftness now so the
    // fog apply() doesn't further modulate alpha. The non-fog branch
    // applies this multiply down below after the color matrix.
    if (TFog.hasFog)
        result.w = psCB2.inverseSoftness * result.w;

    result.xyz = TFog.apply(result.xyz, psCB2.fogColor.xyz,
                            input.worldPos.z, psCB2.fogStart, psCB2.fogEnd, psCB2.fogDensity);

    // The IBL path deferred its backface discard — run it now after
    // the body has finished.
    if (HAS_IBL)
        backfaceDiscard(input.worldPos.w);

    if (!HAS_DEBUG_VIS)
    {
        // Release: apply the post color-matrix grade and finish the
        // alpha scale if fog was off.
        result.xyz = applyColorMatrix(result.xyz);
        if (!TFog.hasFog)
            result.w = psCB2.inverseSoftness * result.w;
    }

    if (HAS_DEBUG_VIS)
    {
        // Debug overlay needs the world-space normal + its pre-
        // normalisation length; recompute from the normal map rather
        // than storing it through the lit bodies.
        float3 nSample = t_normal.Sample(s_normal, input.uv.xy).xyw;
        float3 tsN = decodeNormalMap(nSample);
        float3 wn = tangentToWorld(tsN, input.tangentWS.xyz, input.normalWS.xyz, input.tangentWS.w);
        float normalLen = length(wn);
        wn = normalize(wn);
        wn = orientNormal(wn, isFrontFace);

        // Effective albedo for DEBUG mode 1: the centre sample after
        // the bit-0 override has been applied, matching what the lit
        // body shaded with.
        float3 effectiveAlbedo = t_albedo.Sample(s_albedo, input.uv.xy).xyz;
        if ((asuint(psCB3.enabledShaders) & 1) != 0)
            effectiveAlbedo = psCB3.overrideAlbedo;

        result = debugVisualize(result, input, wn, normalLen, effectiveAlbedo);
        if (!TFog.hasFog)
            result.w = psCB2.inverseSoftness * result.w;
    }

    output.color = result;
    if (IS_MRT)
    {
        // specular.x carries view-space Z for SSR / resolve effects;
        // normal.w carries the pre-fog alpha for coverage.
        output.specular = float4(input.worldPos.z, 0, 0, 1);
        output.normal = float4(mrtNormal, preAlpha);
    }

    return output;
}
