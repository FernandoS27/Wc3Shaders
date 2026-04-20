// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026, Fernando Sahmkow
// See LICENSE in the repository root for full terms.
//
// SD-on-HD pixel shader entry point.
//
// Specialisation space:
//   TFog, TAlpha      — shared fog + alpha-test policies.
//   HAS_IBL           — use the full SD IBL body (sd_ps_ibl.slang).
//   HAS_EXTRA_VERTS   — cascade clip coords are available for shadows.
//   IS_DEPTH_PREPASS  — depth-only pass.
//   IS_MRT            — write the deferred G-buffer targets.
//   HAS_SRGB_OUTPUT   — encode to sRGB at the end (render target is
//                        _UNORM rather than _SRGB).
//   HAS_DEBUG_VIS     — debug visualisation enabled.

implementing wc3_shaders;

[shader("fragment")]
public PSOutput<IS_MRT && !IS_DEPTH_PREPASS, !IS_DEPTH_PREPASS> sd_on_hd_ps_main<
    TFog    : IFogMode,
    TAlpha  : IAlphaTest,
    let HAS_IBL          : bool,
    let HAS_EXTRA_VERTS  : bool,
    let IS_DEPTH_PREPASS : bool,
    let IS_MRT           : bool,
    let HAS_SRGB_OUTPUT  : bool,
    let HAS_DEBUG_VIS    : bool
>(PSInput input, uint frontFace : SV_IsFrontFace)
{
    PSOutput<IS_MRT && !IS_DEPTH_PREPASS, !IS_DEPTH_PREPASS> output;
    bool isFrontFace = frontFace != 0;

    // Depth prepass short-circuits everything below.
    if (IS_DEPTH_PREPASS)
    {
        if (!TAlpha.enabled)
            sdDepthPrepassStub(input);
        else
            sdDepthPrepassBody<TAlpha>(input, isFrontFace);
        output.color = float4(0, 0, 0, 0);
        return output;
    }

    // Dispatch to the IBL body or the cheap standard body.
    float4 result;
    if (HAS_IBL)
        result = sdIBLBody<HAS_EXTRA_VERTS, HAS_DEBUG_VIS>(input, isFrontFace);
    else
        result = sdStandardBody(input, isFrontFace);

    float3 mrtNormal = float3(0, 0, 0);
    float preAlpha = 0;
    if (IS_MRT)
    {
        // MRT path: same alpha-test + extra-coverage pattern as the HD
        // entry, plus a packed world-space normal.
        TAlpha.test(result.w, sdPsCB2.alphaRef);

        float mrtAlpha = t_albedo.Sample(s_albedo, input.uv.xy).w;
        mrtAlphaTest(mrtAlpha, sdPsCB2.inverseSoftness);

        mrtNormal = sdComputeMRTNormal(input, isFrontFace);
        preAlpha = result.w;
    }

    if (!IS_MRT)
        TAlpha.test(result.w, sdPsCB2.alphaRef);

    // Always pre-scale alpha by inverseSoftness here — the SD-on-HD
    // pipeline has no "run after color matrix" branch.
    result.w = sdPsCB2.inverseSoftness * result.w;

    result.xyz = TFog.apply(result.xyz, sdPsCB2.fogColor.xyz,
                            input.worldPos.z, sdPsCB2.fogStart, sdPsCB2.fogEnd, sdPsCB2.fogDensity);

    // Backface discard happens after the lit body here because the
    // IBL path needs the world-space normal for the probe sample. The
    // non-IBL path still needs the discard so we run it in both cases.
    if (HAS_IBL)
        backfaceDiscard(input.worldPos.w);
    else if (!IS_DEPTH_PREPASS)
        backfaceDiscard(input.worldPos.w);

    // sRGB encoding for UNORM render targets. Skipped in debug mode
    // because the debug overlay emits already-displayable colors.
    if (HAS_SRGB_OUTPUT && !HAS_DEBUG_VIS)
    {
        result.xyz = linearToSrgb(result.xyz);
    }

    if (HAS_DEBUG_VIS)
    {
        result = sdDebugVisualize(result, input, isFrontFace);
    }

    output.color = result;
    if (IS_MRT)
    {
        output.specular = float4(input.worldPos.z, 0, 0, 1);
        output.normal = float4(mrtNormal, preAlpha);
    }

    return output;
}
