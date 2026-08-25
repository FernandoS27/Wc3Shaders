// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026, Fernando Sahmkow
// See LICENSE in the repository root for full terms.
//
// Image family (image.fx) — VS + PS.
//
// The UI compositor: any two-dimensional screen-aligned quad, built from up to
// four texture layers composited back-to-front with premultiplied alpha, then
// run through one of five colour-adjust modes.
//
// `Image_BuildSection` @0x102b53510 registers NINE own axes and links [Common]:
//
//     b_colorAdjustMode         @104 w3  (5 modes)
//     b_iLayerCount             @107 w3  (1..4 — how many layers the PS composites)
//     b_iVertexLayoutLayerCount @110 w3  (1..4 — how many the VERTEX STREAM carries)
//     b_renderingInnerText      @113 w1
//     b_imageUse8BitHDR         @114 w1
//     b_iUseAlphaMask           @115 w1
//     b_layer0AlphaOnly         @116 w1
//     b_layer1AlphaOnly         @117 w1
//     b_layer2AlphaOnly         @118 w1
//
// The two layer counts are INDEPENDENT axes and the vertex layout is always the
// wider of the two (retail pairs: 1/1, 1/4, 2/2, 2/4, 3/3, 3/4, 4/4) — the VS
// reads its stream through `b_iVertexLayoutLayerCount` but writes the transport
// through `b_iLayerCount`, so a narrower layout than the layer count would not
// compile.  Layer 3 is never alpha-only ("the final layer cannot be alpha only"),
// which is why there is no `b_layer3AlphaOnly`.
//
// Own IO (`SImageVtxFmt`), so `inject_preamble: false`.
//
// Retail set: 7 VS perms, 3520 PS perms.

implementing sc2_shaders;

// SC2_AXIS comes from here.  It used to be written out per family, because the
// layout was per family and a global switch would fire this file's CB reads while
// another family's entry point was compiling.  The layout is global now, so one
// macro serves the module and the per-axis SC2_PERMDATA_ flag does the selecting.
#include "types/perm_cb.slangh"

#ifndef b_colorAdjustMode
#define b_colorAdjustMode 0
#endif
#ifndef b_iLayerCount
#define b_iLayerCount 1
#endif
#ifndef b_iVertexLayoutLayerCount
#define b_iVertexLayoutLayerCount 1
#endif
#ifndef b_renderingInnerText
#define b_renderingInnerText 0
#endif
#ifndef b_imageUse8BitHDR
#define b_imageUse8BitHDR 0
#endif
#ifndef b_iUseAlphaMask
#define b_iUseAlphaMask 0
#endif
#ifndef b_layer0AlphaOnly
#define b_layer0AlphaOnly 0
#endif
#ifndef b_layer1AlphaOnly
#define b_layer1AlphaOnly 0
#endif
#ifndef b_layer2AlphaOnly
#define b_layer2AlphaOnly 0
#endif

// image.fx colour-adjust modes.
#define e_imageColorAdjustModeNone       0  // vertex multiply only (always done)
#define e_imageColorAdjustModeDesaturate 1  // legacy MaxChannel x AdjustmentColor
#define e_imageColorAdjustModeColorize   2  // fast hue-shift approximation
#define e_imageColorAdjustModeFill       3  // colour overwriting
#define e_imageColorAdjustModeAddSelf    4  // add the image to itself

// image.fx : SImageVtxFmtGeneric — the vertex STREAM, sized by the layout axis.
public struct ImageVSInput
{
    float4 vPos    : POSITION;
    float4 cColor0 : COLOR0;
#if (b_iVertexLayoutLayerCount > 1)
    float4 cColor1 : COLOR1;
#endif
#if (b_iVertexLayoutLayerCount > 2)
    float4 cColor2 : TEXCOORD4;
#endif
#if (b_iVertexLayoutLayerCount > 3)
    float4 cColor3 : TEXCOORD5;
#endif
    float4 vTC0    : TEXCOORD0;
#if (b_iVertexLayoutLayerCount > 1)
    float4 vTC1    : TEXCOORD1;
#endif
#if (b_iVertexLayoutLayerCount > 2)
    float4 vTC2    : TEXCOORD2;
#endif
#if (b_iVertexLayoutLayerCount > 3)
    float4 vTC3    : TEXCOORD3;
#endif
};

// image.fx : SImageVtxFmt — the transport, sized by the LAYER axis.
public struct ImageTransport
{
    float4 vPos    : SV_Position;
    float4 cColor0 : COLOR0;
#if (b_iLayerCount > 1)
    float4 cColor1 : COLOR1;
#endif
#if (b_iLayerCount > 2)
    float4 cColor2 : TEXCOORD4;
#endif
#if (b_iLayerCount > 3)
    float4 cColor3 : TEXCOORD5;
#endif
    float4 vTC0    : TEXCOORD0;
#if (b_iLayerCount > 1)
    float4 vTC1    : TEXCOORD1;
#endif
#if (b_iLayerCount > 2)
    float4 vTC2    : TEXCOORD2;
#endif
#if (b_iLayerCount > 3)
    float4 vTC3    : TEXCOORD3;
#endif
};

public struct ImagePSInput
{
    float4 vPos    : SV_Position;
    float4 cColor0 : COLOR0;
#if (b_iLayerCount > 1)
    float4 cColor1 : COLOR1;
#endif
#if (b_iLayerCount > 2)
    float4 cColor2 : TEXCOORD4;
#endif
#if (b_iLayerCount > 3)
    float4 cColor3 : TEXCOORD5;
#endif
    float4 vTC0    : TEXCOORD0;
#if (b_iLayerCount > 1)
    float4 vTC1    : TEXCOORD1;
#endif
#if (b_iLayerCount > 2)
    float4 vTC2    : TEXCOORD2;
#endif
#if (b_iLayerCount > 3)
    float4 vTC3    : TEXCOORD3;
#endif
};

[shader("vertex")]
public ImageTransport image_vs_main(ImageVSInput IN)
{
    ImageTransport OUT;
    OUT.vPos = mul(float4(IN.vPos), vsGlobals.mTransform);

    // The .zyxw channel swap is the OpenGL/Metal arm only; D3D takes the colours
    // straight through.
    OUT.cColor0 = IN.cColor0;
#if (b_iLayerCount > 1)
    OUT.cColor1 = IN.cColor1;
#endif
#if (b_iLayerCount > 2)
    OUT.cColor2 = IN.cColor2;
#endif
#if (b_iLayerCount > 3)
    OUT.cColor3 = IN.cColor3;
#endif

    OUT.vTC0 = IN.vTC0;
#if (b_iLayerCount > 1)
    OUT.vTC1 = IN.vTC1;
#endif
#if (b_iLayerCount > 2)
    OUT.vTC2 = IN.vTC2;
#endif
#if (b_iLayerCount > 3)
    OUT.vTC3 = IN.vTC3;
#endif
    return OUT;
}

[shader("pixel")]
public float4 image_ps_main(ImagePSInput IN) : SV_Target0
{
    float alphaMult = 1;
    float4 cColor = p_sLayer0_t.Sample(p_sSamp, IN.vTC0.xy);

    // Rendering InnerText means the image is greyscale and only the opaque white
    // parts are wanted: treat 'g' as grey and fold it into alpha, which cuts the
    // inner white text out of its black outline.
    //
    // `#if` -> `if` on a permutation-CB read.  With SC2_USE_PERM_CB off, SC2_AXIS
    // expands to the `#define` itself, so this folds away exactly as the `#if` did
    // -- which is why the conversion is gated by a bit-identity check, not a sweep.
    // With it on, the predicate is uniform across the draw, so it is a scalar branch
    // and the untaken side is skipped rather than masked.
    [branch] if (SC2_AXIS(b_renderingInnerText))
    {
        cColor.a *= cColor.g;
        cColor.rgb = 1;
    }

    // The per-layer alpha source is a CONSTANT-selected channel of the sample
    // (p_vLayerAlphaChannelIndex[i] is a float the shader truncates to 0..3), then
    // optionally inverted and optionally multiplied back into the original alpha.
    int index  = (int)psGlobals.vLayerAlphaChannelIndex[0];
    float newA = cColor[index];
    newA       = lerp(newA, 1 - newA, psGlobals.vLayerAlphaChannelInverse[0]);
    cColor.a   = lerp(newA, newA * cColor.a, psGlobals.vLayerAlphaChannelMultiply[0]);

    cColor *= IN.cColor0;

    // Alpha-only layers multiply their alpha into the next layer up and contribute
    // no colour to the composite.  The final layer cannot be alpha-only.
    [branch] if (SC2_AXIS(b_layer0AlphaOnly))
    {
        alphaMult = cColor.a;
        cColor.rgb = 0;
        cColor.a = 1;
    }
    else
    {
        cColor.rgb *= cColor.a;
        cColor.a = 1 - cColor.a;
    }

#if (b_iLayerCount > 1)
    {
        float4 cLayer = p_sLayer1_t.Sample(p_sSamp, IN.vTC1.xy);

        index    = (int)psGlobals.vLayerAlphaChannelIndex[1];
        newA     = cLayer[index];
        newA     = lerp(newA, 1 - newA, psGlobals.vLayerAlphaChannelInverse[1]);
        cLayer.a = lerp(newA, newA * cLayer.a, psGlobals.vLayerAlphaChannelMultiply[1]);

        [branch] if (SC2_AXIS(b_layer1AlphaOnly))
        {
            alphaMult *= cLayer.a * IN.cColor1.a;
        }
        else
        {
            cLayer *= IN.cColor1;

            float alpha = 1 - (cLayer.a * alphaMult);
            cColor.rgb = lerp(cLayer.rgb, cColor.rgb, alpha);
            cColor.a *= alpha;
            alphaMult = 1;
        }
    }
#endif
#if (b_iLayerCount > 2)
    {
        float4 cLayer = p_sLayer2_t.Sample(p_sSamp, IN.vTC2.xy);

        index    = (int)psGlobals.vLayerAlphaChannelIndex[2];
        newA     = cLayer[index];
        newA     = lerp(newA, 1 - newA, psGlobals.vLayerAlphaChannelInverse[2]);
        cLayer.a = lerp(newA, newA * cLayer.a, psGlobals.vLayerAlphaChannelMultiply[2]);

        [branch] if (SC2_AXIS(b_layer2AlphaOnly))
        {
            alphaMult *= cLayer.a * IN.cColor2.a;
        }
        else
        {
            cLayer *= IN.cColor2;

            float alpha = 1 - (cLayer.a * alphaMult);
            cColor.rgb = lerp(cLayer.rgb, cColor.rgb, alpha);
            cColor.a *= alpha;
            alphaMult = 1;
        }
    }
#endif
#if (b_iLayerCount > 3)
    {
        float4 cLayer = p_sLayer3_t.Sample(p_sSamp, IN.vTC3.xy);

        index    = (int)psGlobals.vLayerAlphaChannelIndex[3];
        newA     = cLayer[index];
        newA     = lerp(newA, 1 - newA, psGlobals.vLayerAlphaChannelInverse[3]);
        cLayer.a = lerp(newA, newA * cLayer.a, psGlobals.vLayerAlphaChannelMultiply[3]);

        cLayer *= IN.cColor3;

        // To save on shaders, the final layer cannot be alpha only.
        float alpha = 1 - (cLayer.a * alphaMult);
        cColor.rgb = lerp(cLayer.rgb, cColor.rgb, alpha);
        cColor.a *= alpha;
    }
#endif

    // The five colour-adjust modes are pure selectors over the SAME composite --
    // no texture, no output, no discard changes between them -- so the mode is a
    // constant-buffer field and the ladder is a scalar branch.  This one axis is a
    // 5x multiplier on the retail grid.
    const int sc2ColorAdjust = SC2_AXIS(b_colorAdjustMode);
    [branch] if (sc2ColorAdjust == e_imageColorAdjustModeDesaturate)
    {
        float fMax = max(max(cColor.r, cColor.g), cColor.b);

        // Un-premultiply the alpha (the 512 guard prevents a divide by 0).
        fMax *= (cColor.a >= 1.0f) ? 512.0f : (1.0f / (1.0f - cColor.a));
        fMax = saturate(fMax);

        float4 desaturateColor;
        desaturateColor = lerp(psGlobals.vDarkAdjustColor, psGlobals.vLightAdjustColor, fMax);

        // Re-multiply the alpha.
        desaturateColor.rgb *= 1 - cColor.a;

        cColor.rgb = lerp(cColor.rgb, desaturateColor.rgb, desaturateColor.a);
    }
    else [branch] if (sc2ColorAdjust == e_imageColorAdjustModeColorize)
    {
        // 2x greyscale value.
        float gray = cColor.r * .62f + cColor.g * 1.1737f + cColor.b * .2067f;
        gray *= (cColor.a >= 1.0f) ? 512.0f : (1.0f / (1.0f - cColor.a));

        float4 colorizeColor = float4(0, 0, 0, 1);

        // Linear interpolation: black(@0.0) -> colour(@1.0) -> white(@2.0).
        if (gray > 1.0f)
        {
            gray -= 1.0f;
            gray = saturate(gray);      // free clamping to [0,1]

            colorizeColor = lerp(psGlobals.vDarkAdjustColor, psGlobals.vLightAdjustColor, gray);
        }
        else
        {
            colorizeColor = float4(
                (gray * psGlobals.vDarkAdjustColor.r),
                (gray * psGlobals.vDarkAdjustColor.g),
                (gray * psGlobals.vDarkAdjustColor.b),
                psGlobals.vDarkAdjustColor.a);
        }

        // Re-multiply the alpha.
        colorizeColor.rgb *= (1 - cColor.a);

        cColor.rgb = lerp(cColor.rgb, colorizeColor.rgb, colorizeColor.a);
    }
    else [branch] if (sc2ColorAdjust == e_imageColorAdjustModeFill)
    {
        float4 fillColor = psGlobals.vLightAdjustColor;
        fillColor.rgb *= (1 - cColor.a);

        // Interpolate between the composite colour and the fill colour.
        cColor.rgb = lerp(cColor.rgb, fillColor.rgb, fillColor.a);
    }
    else [branch] if (sc2ColorAdjust == e_imageColorAdjustModeAddSelf)
    {
        // Un-premultiply the alpha.
        cColor.rgb *= cColor.a >= 1.0f ? 512.0f : (1.0f / (1.0f - cColor.a));

        // ColorOut.rgb = ColorIn.rgb + 2 * (ColorIn.rgb * AdjustmentColor.rgb)
        float4 addColor = psGlobals.vLightAdjustColor;
        addColor.rgb *= 2;                              // 0..2 meaning for the colour channels
        addColor.rgb *= psGlobals.vLightAdjustColor.a;  // alpha drives the interpolation
        addColor.rgb *= cColor.rgb;                     // which channels to add, and how much
        addColor.rgb += cColor.rgb;                     // do the adding
        addColor.rgb = saturate(addColor.rgb);
        addColor.rgb *= (1 - cColor.a);                 // premultiply the alphas

        // No lerp() required — the alpha interpolation has effectively been done.
        cColor.rgb = addColor.rgb;
    }

#if b_iUseAlphaMask
    {
        float4 cLayer = p_sAlphaMask_t.Sample(p_sSamp, IN.vTC0.xy);
        cColor.rgb *= cLayer.a;
        cColor.a = 1 - ((1 - cColor.a) * cLayer.a);
    }
#endif

    [branch] if (SC2_AXIS(b_imageUse8BitHDR))
        cColor.rgb *= 0.5f;

    // common.fx : AlphaTest(1 - cColor.a).
#if b_iAlphaTest
    sc2_clip((1 - cColor.a) - psGlobals.fAlphaThreshold);
#endif

    return cColor;
}
