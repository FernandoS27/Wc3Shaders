// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026, Fernando Sahmkow
// See LICENSE in the repository root for full terms.
//
// Toon-HD pixel shader entry point.
//
// Shares its specialisation space and post-processing (depth prepass,
// alpha test, fog, MRT pack, color matrix, debug visualisation) with the
// HD PBR entry point (hd_ps.slang). The only difference is the lit body:
// this entry dispatches into the psToon* family in ps/ps_toon.slang,
// which replaces Cook-Torrance + IBL with a Breath-of-the-Wild style
// 2-band cel-shading model.
//
// Specialisation space (identical to hd_ps):
//   TFog              — fog-mode policy.
//   TAlpha            — alpha-test policy.
//   TMat              — material policy (standard vs. multi-layer).
//   HAS_IBL           — route to the full shadow-cascaded / IBL-hinted toon body.
//   HAS_EXTRA_VERTS   — VS passed shadowClip0-2 cascade coords for the IBL body.
//   IS_DEPTH_PREPASS  — depth-only pass (no color).
//   IS_MRT            — also write the deferred-G-buffer targets.
//   HAS_DEBUG_VIS     — debug visualisation enabled.

implementing wc3_shaders;

[shader("fragment")]
public PSOutput<IS_MRT && !IS_DEPTH_PREPASS, !IS_DEPTH_PREPASS> toon_ps_main<
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
    PSOutput<IS_MRT && !IS_DEPTH_PREPASS, !IS_DEPTH_PREPASS> output;
    bool isFrontFace = frontFace != 0;

    // Early backface discard for the non-IBL / non-prepass paths — the
    // IBL body defers it, the prepass handles it internally.
    if (!HAS_IBL && !IS_DEPTH_PREPASS)
        backfaceDiscard(input.worldPos.w);

    if (IS_DEPTH_PREPASS)
    {
        // The depth prepass only cares about producing an identical
        // alpha mask to the lit pass. Since toon materials run the same
        // fresnel/cloak alpha math as HD, we reuse the HD prepass body
        // unchanged.
        if (!TAlpha.enabled)
            depthPrepassStub(input);
        else
            depthPrepassBody<TAlpha, TMat>(input, isFrontFace);
        output.color = float4(0, 0, 0, 0);
        return output;
    }

    // Toon body dispatch. Three variants paralleling the HD bodies:
    //   - HAS_IBL            → full shadow-cascaded + IBL-ambient toon.
    //   - TMat.hasMultiLayer → cheap ML variant.
    //   - otherwise          → cheap single-layer variant.
    float4 result;
    if (HAS_IBL)
        result = psToonIBLBody<TMat, HAS_EXTRA_VERTS, HAS_DEBUG_VIS>(input, isFrontFace);
    else if (TMat.hasMultiLayer)
        result = psToonMultiLayerBody<HAS_DEBUG_VIS>(input, isFrontFace);
    else
        result = psToonStandardBody<HAS_DEBUG_VIS>(input, isFrontFace);

    // MRT + alpha-test + fog sequencing: byte-identical to hd_ps so the
    // toon path drops into the same render targets / deferred resolver.
    float3 mrtNormal = float3(0, 0, 0);
    float preAlpha = 0;
    if (IS_MRT)
    {
        TAlpha.test(result.w, psCB2.alphaRef);

        float mrtAlpha = t_albedo.Sample(s_albedo, input.uv.xy).w;
        mrtAlphaTest(mrtAlpha, psCB2.inverseSoftness);

        mrtNormal = computeMRTNormal(input, isFrontFace);
        preAlpha = result.w;
    }

    if (!IS_MRT)
        TAlpha.test(result.w, psCB2.alphaRef);

    if (TFog.hasFog)
        result.w = psCB2.inverseSoftness * result.w;

    result.xyz = TFog.apply(result.xyz, psCB2.fogColor.xyz,
                            input.worldPos.z, psCB2.fogStart, psCB2.fogEnd, psCB2.fogDensity);

    if (HAS_IBL)
        backfaceDiscard(input.worldPos.w);

    if (!HAS_DEBUG_VIS)
    {
        result.xyz = applyColorMatrix(result.xyz);
        if (!TFog.hasFog)
            result.w = psCB2.inverseSoftness * result.w;
    }

    if (HAS_DEBUG_VIS)
    {
        float3 nSample = t_normal.Sample(s_normal, input.uv.xy).xyw;
        float3 tsN = decodeNormalMap(nSample);
        float3 wn = tangentToWorld(tsN, input.tangentWS.xyz, input.normalWS.xyz, input.tangentWS.w);
        float normalLen = length(wn);
        wn = normalize(wn);
        wn = orientNormal(wn, isFrontFace);

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
        output.specular = float4(input.worldPos.z, 0, 0, 1);
        output.normal = float4(mrtNormal, preAlpha);
    }

    return output;
}
