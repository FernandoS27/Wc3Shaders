// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026, Fernando Sahmkow
// See LICENSE in the repository root for full terms.
//
// Crystal HD pixel shader entry point.
//
// Same specialisation space as `hd_ps.slang`:
//   TFog              — fog-mode policy.
//   TAlpha            — alpha-test policy.
//   TMat              — material policy (standard vs. multi-layer).
//   HAS_IBL           — route to the IBL / Cook-Torrance body.
//   IS_DEPTH_PREPASS  — depth-only pass (no colour).
//   IS_MRT            — also write the deferred G-buffer targets.
//   HAS_DEBUG_VIS     — debug visualisation enabled.
//
// Structurally identical to `ps_main` — the only difference is that the
// lit body dispatch calls the crystal variants (`crystalStandardBody`,
// `crystalMultiLayerBody`, `crystalIBLBody`, `crystalDepthPrepassBody`)
// which thread the albedo refraction and refract-modulated Fresnel alpha
// through the shading flow. Everything else (backface discard, alpha
// test, MRT, fog, colour matrix, debug overlay) is the shared HD
// post-processing stack.

implementing wc3_shaders;

[shader("fragment")]
public PSOutput<IS_MRT && !IS_DEPTH_PREPASS, !IS_DEPTH_PREPASS> crystal_ps_main<
    TFog     : IFogMode,
    TAlpha   : IAlphaTest,
    TMat     : IMaterialLayer,
    let HAS_IBL            : bool,
    let IS_DEPTH_PREPASS   : bool,
    let IS_MRT             : bool,
    let HAS_DEBUG_VIS      : bool
>(PSInput input, uint frontFace : SV_IsFrontFace)
{
    PSOutput<IS_MRT && !IS_DEPTH_PREPASS, !IS_DEPTH_PREPASS> output;
    bool isFrontFace = frontFace != 0;

    // Early backface discard for every non-IBL non-prepass draw. The IBL
    // path delays it until the world normal has been produced; the
    // prepass runs it inside its body.
    if (!HAS_IBL && !IS_DEPTH_PREPASS)
        backfaceDiscard(input.worldPos.w);

    if (IS_DEPTH_PREPASS)
    {
        // Both branches discard backfaces themselves; the stub does no
        // more work, the body runs the alpha test. NEVER follow the stub
        // with an unconditional `discard`: that would kill every
        // surviving fragment and suppress the rasterizer's depth write,
        // defeating the whole prepass.
        if (!TAlpha.enabled)
            depthPrepassStub(input);
        else
            crystalDepthPrepassBody<TAlpha, TMat>(input, isFrontFace);
        output.color = float4(0, 0, 0, 0);
        return output;
    }

    // Dispatch to the appropriate lit body.
    float4 result;
    if (HAS_IBL)
        result = crystalIBLBody<TMat, HAS_DEBUG_VIS>(input, isFrontFace);
    else if (TMat.hasMultiLayer)
        result = crystalMultiLayerBody<HAS_DEBUG_VIS>(input, isFrontFace);
    else
        result = crystalStandardBody<HAS_DEBUG_VIS>(input, isFrontFace);

    // MRT path: alpha test + deferred-coverage gate + G-buffer payloads.
    float3 mrtNormal = float3(0, 0, 0);
    float  preAlpha  = 0;
    if (IS_MRT)
    {
        TAlpha.test(result.w, psCB2.alphaRef);

        // Deferred-only alpha gate on the centre albedo sample (matches
        // the HD path — we want the same coverage mask even when the
        // refracted colour is wildly different).
        float mrtAlpha = t_albedo.Sample(s_albedo, input.uv.xy).w;
        mrtAlphaTest(mrtAlpha, psCB2.inverseSoftness);

        mrtNormal = computeMRTNormal(input, isFrontFace);
        preAlpha  = result.w;
    }

    if (!IS_MRT)
        TAlpha.test(result.w, psCB2.alphaRef);

    // When fog is on we pre-multiply alpha by inverseSoftness so the fog
    // blend below doesn't further modulate it; the non-fog branch does
    // this after the colour matrix.
    if (TFog.hasFog)
        result.w = psCB2.inverseSoftness * result.w;

    result.xyz = TFog.apply(result.xyz, psCB2.fogColor.xyz,
                            input.worldPos.z, psCB2.fogStart, psCB2.fogEnd, psCB2.fogDensity);

    // IBL deferred backface discard.
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
        // Recompute the world normal from the normal map for the debug
        // overlay (the lit bodies don't plumb it back out).
        float3 nSample = t_normal.Sample(s_normal, input.uv.xy).xyw;
        float3 tsN = decodeNormalMap(nSample);
        float3 wn = tangentToWorld(tsN, input.tangentWS.xyz, input.normalWS.xyz, input.tangentWS.w);
        float  normalLen = length(wn);
        wn = normalize(wn);
        wn = orientNormal(wn, isFrontFace);

        // Effective albedo for DEBUG mode 1: the refracted-blend sample
        // with the bit-0 override applied. Matches what the crystal lit
        // body shaded with so the albedo view is the one the material
        // actually produced, not the raw t0 texel.
        float  refractMask = t_orm.Sample(s_orm, input.uv.xy).x;
        if ((asuint(psCB3.enabledShaders) & 2) != 0)
            refractMask = psCB3.overrideOrm.x;
        float3 effectiveAlbedo = sampleCrystalAlbedo(input.uv.xy, input.normalWS.xyz, refractMask).xyz;
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
        output.normal   = float4(mrtNormal, preAlpha);
    }

    return output;
}
