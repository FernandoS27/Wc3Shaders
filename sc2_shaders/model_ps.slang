// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026, Fernando Sahmkow
// See LICENSE in the repository root for full terms.
//
// Model-family pixel shader entry (psmainshading.fx : DefaultPixelMain).
//
// Shared verbatim by Model / Particle / Ribbon / Foliage (they differ only on
// the vertex side).  Wraps MainShading, applies the final-colour / alpha emit
// gates, and fills the deferred MRT targets (SPixelOut) from the g-buffer values
// MainShading accumulated.

implementing sc2_shaders;
#include "types/perm_cb.slangh"

#ifndef b_emitFinalColor
#define b_emitFinalColor 0
#endif
#ifndef b_emitAlpha
#define b_emitAlpha 0
#endif
#ifndef b_emitMRTSpecularPower
#define b_emitMRTSpecularPower 0
#endif
#ifndef b_blendMRTNormals
#define b_blendMRTNormals 0
#endif
#ifndef b_iUse8BitHDR
#define b_iUse8BitHDR 0
#endif

// The shared DefaultPixelMain body, factored out so own-root families that tail-
// call DefaultPixelMain (SplatDirect/SplatDeferred) can reuse it verbatim without
// duplicating the emit-gate/MRT stack.
public ModelPSOutput computeModelPixel(ModelPSInput vertOut)
{
    ModelPSOutput result;
    DeferredOutputs def;
    result.cColor = mainShading(vertOut, def);

    // "Final color and alpha are always written out, but we can cull the shader
    // computations" — the emit gates (psmainshading.fx:65-70).  The MRT targets
    // read result.cColor.a AFTER these gates (psmainshading.fx:75/81/90/98).
#if (b_emitFinalColor == 0)
    result.cColor.rgb = 1;
#endif
#if (b_emitAlpha == 0)
    result.cColor.a = 1;
#endif

    // ---- Depth/Normal target (psmainshading.fx:72-77) ----
#if (b_emitMRTDepth == 1 || b_emitMRTNormal == 1)
    result.vNormalDepth = encodeDepthNormal(def.vNormal, outputDepth(vertOut));
#endif

    // ---- Diffuse target (psmainshading.fx:78-96) ----
#if (b_emitMRTDiffuse == 1 || b_emitMRTAO == 1)
    result.cDiffuse = float4(0, 0, 0, 0);
    #if (b_emitMRTDiffuse == 1)
    result.cDiffuse = float4(def.cDiffuse, result.cColor.a);
        #if (b_blendMRTNormals == 1)
    // Terrain path: preserve depth (stored in the normal target's alpha) across
    // the no-depth-write blend by replicating the colour alpha there.
    result.vNormalDepth.a = result.cColor.a;
        #endif
    #endif
    #if (b_emitMRTAO == 1)
    result.cDiffuse.a = def.cAO;
    #endif
#endif

    // ---- Specular target (psmainshading.fx:97-104) ----
#if (b_emitMRTSpecular == 1)
    result.cSpecular = float4(def.cSpecular, result.cColor.a);
    #if (b_emitMRTSpecularPower == 1)
    result.cSpecular.a = def.cSpecularPower;
    [branch] if (SC2_AXIS(b_iUse8BitHDR))
        result.cSpecular.a *= psGlobals.vHDRScaling.y;
    #endif
#endif

    return result;
}

[shader("fragment")]
public ModelPSOutput model_ps_main(ModelPSInput vertOut)
{
    return computeModelPixel(vertOut);
}
