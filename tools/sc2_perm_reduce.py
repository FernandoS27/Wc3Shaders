#!/usr/bin/env python3
"""Permutation reduction for `sc2_shaders`: retail key vector -> (structural key,
constant-buffer payload).

Implements milestone R1 of docs/SC2_PERM_REDUCTION_PLAN.md, which in turn implements
docs/SC2_PERM_REDUCTION_DESIGN.md.  Nothing here compiles or edits a shader; it is
the offline classifier the later milestones are driven by, plus its own tests.

The whole reduction rests on one split:

    an axis stays COMPILE-TIME iff changing it changes something the shader cannot
    change at draw time -- its resource bindings, its output signature, its input
    signature, its discard behaviour, or a loop trip count.  Everything else is
    constant-buffer DATA.

`structural_key` reads the compile-time side; `pack_payload` reads the data side.
The split is declared as NAME SETS (`key_axes` / `payload_axes`), never as control
flow, so `--check-all` can verify mechanically that it covers every axis that
actually varies in the retail corpus (T1) and that every class is compile-time
consistent with all of its member slots (T3).

Usage:
  python tools/sc2_perm_reduce.py --check-all            # T1 + T3 + T-count, all 36
  python tools/sc2_perm_reduce.py --report               # the reduction table
  python tools/sc2_perm_reduce.py --build --family Model # write the class tables
  python tools/sc2_perm_reduce.py --axes Model ps        # explain the split
"""
import argparse
import io
import json
import re
import math
import os
import sys
from collections import Counter, OrderedDict

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
REPO_ROOT = os.path.dirname(HERE)

import sc2_shaders_cfg as cfg

REDUCED_DIR = os.path.join(REPO_ROOT, "sc2_perms", "reduced")
INVENTORY = os.path.join(REPO_ROOT, "docs", "_perm_axis_inventory.json")
CORE_CACHE = os.path.join(HERE, "_sc2_perm_cache", "core_layers.json")


# ===========================================================================
# 1. Material layers
# ===========================================================================
# psmaterial.fx SETUP_LAYER: the 18 "standard" layers every Default-root pixel
# shader can carry...
LAYERS_STD = [
    "Diffuse", "Specular", "Decal", "Emissive", "Emissive2", "AlphaMask",
    "AlphaMask2", "Lightmap", "AmbientOcclusion", "SpecularExponent", "Normal",
    "Envio", "EnvioMask", "NormalBlendMask", "NormalBlendMask2",
    "NormalBlendNormal", "NormalBlendNormal2", "Heightmap",
]
# ...plus the 15 layers that exist only inside a non-STANDARD shading mode.  They
# are ordinary layers with the same 17 properties; they are listed separately only
# because their mode is already a key bit, so they never widen the sampled mask of
# a STANDARD permutation.
LAYERS_MODE = [
    "Displacement", "DisplacementStrength",
    "VolumeColor", "VolumeNoise1", "VolumeNoise2",
    "ReflectionNormal", "ReflectionStrength", "ReflectionBlurMask",
    "CloakingAlphaMask", "CloakingAlphaMaskOriginal", "CloakingAlphaTest",
    "CloakingEnvio", "CloakingEnvioMask", "CloakingNormal",
    "TerrainAlphaMask",
]
ALL_LAYERS = LAYERS_STD + LAYERS_MODE

# psmateriallayer.fx SLayerConfig -- the 17 per-layer axes.  Every one of them is
# already a runtime `int` field of SLayerConfig in
# sc2_shaders/ps/model_material.slang, which is why moving them costs no rewrite of
# the shading maths, only of the setup functions.
LAYER_PROPS = [
    "LayerEnable", "LayerId", "TextureType", "UseConstantColor", "UseMask",
    "ChannelSelect", "TeamColorMode", "Invert", "Clamp", "FresnelMode",
    "FresnelTransformMode", "FresnelSaturate", "Op", "UVEmitter", "UVMapping",
    "IsRTTTexture", "TriPlanarUvId",
]

# Of those 17, exactly three touch the shader's INTERFACE:
#   LayerEnable + UseConstantColor  -> together decide whether the layer's texture
#                                      is sampled at all (an SRV), and
#   TextureType                     -> Texture2D vs TextureCube is a TYPE, not a
#                                      value, so it cannot be selected at runtime.
# The first two are dual-role: they also gate whether the layer CONTRIBUTES, which
# is pure data.  See DUAL_ROLE_PROPS.
LAYER_IFACE_PROPS = ("LayerEnable", "UseConstantColor", "TextureType")
DUAL_ROLE_PROPS = ("LayerEnable", "UseConstantColor")

# Which of the 14 movable properties have actually been MOVED, i.e. which
# SC2_PERM_PROP_<Prop> switches ps/model_layers_axes.slangh is compiled with.
#
# This list is the single control for the R5 increments: it decides both what
# `struct_defines` turns on in the shader and whether `classify` calls the axis
# compile-time or data.  Keeping the two in one place is what makes T3 a real test
# -- a property left out of the shader but treated as data here would produce a
# class whose members disagree on a value the blob has baked in, and T3 says so
# with the axis name rather than with a float.
ENABLED_LAYER_PROPS = [
    "Op", "ChannelSelect", "Invert", "Clamp", "UseMask", "TeamColorMode",
    "FresnelMode", "FresnelTransformMode", "FresnelSaturate",
    "UVEmitter", "UVMapping", "IsRTTTexture", "LayerId", "TriPlanarUvId",
]

# The NON-layer data axes that have actually been converted to read from the
# permutation buffer.  Same contract as ENABLED_LAYER_PROPS: an axis only counts as
# data once the SHADER reads it that way.  Until it is listed here it stays
# compile-time, so a class still pins it and every member of that class agrees on it.
#
# Getting this wrong is silent and total: with `b_useLighting` treated as data but
# still read as a `#define`, the class compiles with it at 0 and the whole lighting
# path vanishes -- the diff is ~1.0, not a ULP.  That is how this list came to exist.
MOVED_NONLAYER_AXES = {
    # image.fx (R4)
    "b_colorAdjustMode", "b_renderingInnerText", "b_imageUse8BitHDR",
    "b_layer0AlphaOnly", "b_layer1AlphaOnly", "b_layer2AlphaOnly",
    # hdr.fx (R4)
    "b_iOperator", "b_useWhiteShift", "b_iScaleBloom", "b_iAALumaMode",
    "b_iUse8BitHDRPostProcess",

    # ---- R5b ----------------------------------------------------------------
    # Everything below is verified by `--check-moved`: an axis may only appear here
    # once NO `#if`/`#elif` in sc2_shaders/ still reads it.  One surviving directive
    # is enough to make the class compile with the axis at its default of 0, because
    # only nonzero axes are passed as -D -- a whole arm vanishing, not a ULP.

    # model_vs_common.fx : skinning, parallax vector, splat projector
    "b_iBlendWeightCount", "b_useIdentityBasis", "b_splatTextureProjectorEnabled",
    # model_vs.fx : local/world triplanar frames, vertex AO source, Z offset
    "b_iUseLocalTriPlanarTangentSpace", "b_iUseLocalTriPlanarWeights",
    "b_iUseWorldTriPlanarTangentSpace", "b_iUseWorldTriPlanarWeights",
    "b_iVertexAOFromVertexColor", "b_iZOffsetFromUB4N_XY",
    # vslighting.fx
    "b_useDblLambert", "b_iDirectionalLightCount",
    # particle_vs.fx
    "b_localSpace", "b_useProceduralPosition", "b_randomFlipBookStart",
    "b_clampedTailLength", "b_fixedTailLength", "b_iParticleColorInterpolation",
    "b_iParticleSizeInterpolation", "b_iParticleRotationInterpolation",
    # ribbon_vs.fx
    "b_ribbonLocalSpace", "b_splineRibbon", "b_gpuSplineRibbon", "b_precomputedAge",
    "b_precomputedTangent", "b_proceduralPosition", "b_computeUFromAge",
    "b_clampTail", "b_iRibbonColorInterpolation", "b_iRibbonSizeInterpolation",
    # psmaterial.fx / psmainshading.fx
    "b_iFakeEnergyConservingSpec", "b_iLightmapShadow", "b_iEnvioMultipliers",
    "b_useVertexAlpha", "b_useBlendAdd", "b_iModAlpha", "b_iClampOutput",
    "b_iNormalBlending", "b_iWriteFogDensityIntoRed", "b_no8BitHDR",
    "b_useTeamColorSpecular", "b_iSpecularType", "b_useSpecular",
    # pslighting.fx
    "b_iNormalizeHalfVector",
    # pscommon.fx : depth/normal g-buffer encoding
    "b_iOutEncoding",
    # psnormal.fx
    "b_DXNStyleNormalMaps",
    # psfog.fx / psfow.fx
    "b_fogScaleByAlpha", "b_FOWEnabled", "b_FOWAdditiveScale",
    # pscreep.fx
    "b_iCreepOutputMode", "b_iCreepUseGroundNormalMap", "b_iTriPlanarCreep",
    # splatdeferred.fx
    "b_iUseMinimumHeight",
    # postprocessquad.fx
    "b_iAAMode",

    # ---- R5b, second batch --------------------------------------------------
    # psvectorui.fx / psvolume.fx / psdisplace.fx / psterraindoodad.fx
    "b_additiveSplat", "b_dashedCircleSplats", "b_lowEndShaders",
    "b_splatAttenuationEnabled", "b_iVolumeFalloffType", "b_iVolumeMod",
    "b_distortionDepthFix", "b_useVertexAlphaStrength",
    "b_iSolidColor", "b_computeHeight", "b_creepEnabled", "b_useVertexColors",
    # psmaterial.fx / psnormal.fx
    "b_iTwoSided", "b_useSimplifiedFresnel",
    # deferredlight.fx
    "b_iLightType", "b_useDeffSpecularPower", "b_iUseSpecular",
    # postprocessquad.fx output tail
    "b_iInputChannelFilter", "b_iOutputChannelFilter", "b_iHaloRastermode",
    "b_iScaleOutput", "b_iAddToOutput", "b_iClampNegativeOutput",
    "b_iPostMaskModulate",
    # postprocessquad.fx blur/halo/DOF value axes.  The four that stay compile-time
    # -- b_iTakeMin, b_iWeightedBlur, b_iDepthScaledBlur, b_iCorrectDOFForDepth --
    # are each the ONLY reader of an SRV, so moving them would make the shader
    # sample a texture the caller need not have bound.
    "b_iBlurAllChannels", "b_iBlurAlphaOnly", "b_iConstrainToViewport",
    "b_iDOFDotBlend", "b_iHaloBlurMode",
}

# READ THROUGH A GENERATED DEFINE, so `--check-moved` cannot see them.
#
# The UV emitters are template arguments -- `vsEmitModelUV<SC2_UVMAPPING##i, i>` --
# and `struct_defines` derives SC2_UVMAPPING<i>/SC2_UVRANDOM<i> from the KEY.  Move
# the axis to the payload and those defines stop being emitted, so every slot
# compiles as mapping 0.  There is no `#if` to find, which is exactly why they are
# listed here rather than left to the gate.
#
# Making them dynamic means turning the specialisation into a runtime switch over
# ~19 arms in 5 slots.  Measured worth: 1.12x module-wide (Model_vs 1.36x,
# Particle_vs 1.63x) -- real, but the most expensive conversion left, so it is its
# own increment.
TEMPLATE_ARG_AXES = frozenset(
    ["b_iUVMapping%d" % i for i in range(8)]
    + ["b_UVRandomOffsetEnable%d" % i for i in range(8)])


# Axes converted in ONE stage's files but still read by `#if` in the other's.
#
# A class is compiled for exactly one (family, stage) and carries its own
# SC2_PERMDATA_ flags, so the split is expressible: a Model_vs class can read
# `b_UseAmbientEnvironment` from the buffer while a Model_ps class keeps pinning it.
# The other stage's file is dead code for that entry point -- it only has to
# preprocess, which it does, because the layout is global.
#
# This is not a shortcut around the gate: `--check-moved` still requires that every
# read site IN THAT STAGE'S FILES is converted.  It is what recovers the vertex-stage
# reduction for axes that are genuinely interface-bearing in the pixel stage --
# `b_useShadows` and friends bind a shadow map there, but only select data here.
MOVED_BY_STAGE = {
    "vs": {
        # K3 in the pixel stage (each binds an SRV); ordinary data in the vertex one.
        "b_UseAmbientEnvironment", "b_UseLightingRegions",
        "b_UseVertexBasedAmbientOcclusion", "b_useParallaxMapping", "b_iUsePOMNew",
        # converted in vs/ but still `#if` in ps/
        "b_useNormalMapping", "b_useLighting", "b_iVertexLightingMode",
        "b_iUseHardwareDepth",
        # Pixel-stage features with no vertex read site at all: free to carry as
        # data there, and b_SampleFOW / b_iUse8BitHDR do vary in vertex manifests.
        "b_SampleFOW", "b_iUse8BitHDR", "b_iBlurEnvironmentMap", "b_envMapPass",
        "b_useCaustics", "b_useCubemapReflections", "b_useWaterDepthEffects",
        "b_iUnSwizzleDXNNormalMaps",
    },
    "ps": set(),
}


# Axes that pass --check-moved (no `#if` left) and STILL do not behave as data.
#
# Found by tools/sc2_perm_axis_gate.py, which compiles the class with exactly one axis
# dynamic and everything else pinned, then compares against the ordinary specialisation.
#
# This list used to hold five entries, all ribbon-physics axes, on the theory that they
# "select between MUTATIONS OF SHARED STATE rather than between values".  FOUR OF THEM
# WERE NOT THAT.  They were dxbc_interp dropping the `-` source modifier on `mov` (see
# bitread() there): fxc folds a compile-time `1 - x` into `mov rN.xyz, -rN.xyzx`, both
# legs of a specialised comparison contain it, both were mis-executed identically and
# cancelled -- so the failure only appeared once one leg went dynamic and stopped
# sharing the fold.  The tell was in the original numbers and went unread: three
# different axes diverging by the SAME 5.154897 at the SAME slot 0, which is one cause
# wearing three names, not three axes with a common shape.
#
# Cleared on 40 strided Ribbon_vs slots each, with a negative control (interpreter put
# back the old way) reproducing 5.154897 / 7.315130 / 2.171465 at slots 0 / 108 / 162:
#   b_gpuSplineRibbon  b_splineRibbon  b_proceduralPosition  b_precomputedTangent
#
# Lesson kept: --check-moved is necessary and NOT sufficient, and a behavioural gate is
# only as good as its ability to FAIL.  Never retire an entry from here on a green run
# alone -- pair it with a control that re-breaks the cause and requires the gate to fire.
UNSAFE_AS_DATA = {
    # The one real entry.  Found only at --slots 40, not at --slots 2: the gate's
    # verdict is as good as its coverage.  ROOT CAUSE (established, and confirmed
    # again WITH the interpreter fixed -- it is not Defect 5): this axis picks the arm
    # of vsInterpolateValue's if/else chain.  Held constant, the chain folds to one
    # arm.  As DATA every arm stays live, and the code generated for the non-folded
    # chain reuses registers as constant-buffer indices AFTER overwriting them --
    # Ribbon_vs slot 54, `if_z r0.w` arm:
    #
    #   mov r3.w, v3.x                          ; r3.w = batch index (uint)
    #   lt  r4.x, v0.x, cb0[r3.w + 506].x       ; r4.x = COMPARISON RESULT
    #   mad r3.w, r4.y, r4.z, cb0[r3.w + 658].x ; r3.w clobbered with a float
    #   add r4.y, v0.x, -cb0[r4.x + 506].x      ; indexes with the compare result
    #   mad r4.y, r4.y, r4.w, cb0[r3.w + 658].y ; indexes with the clobbered float
    #   movc r1.w, r4.x, r3.w, r4.y             ; r4.x is the CONDITION here
    #
    # So the out-of-range index is a float BIT PATTERN, which is exactly what the
    # interpreter reported.  It is not a harness gap: the batch index (TEXCOORD6) is
    # domain-constrained and correct where it is read before the clobber.  Keep the
    # exclusion -- moving this axis ships a miscompiled shader, not merely one the
    # gate cannot check.
    "b_iRibbonSizeInterpolation",
}


def moved_axes(stage):
    return ((MOVED_NONLAYER_AXES | MOVED_BY_STAGE.get(stage, set()))
            - UNSAFE_AS_DATA)


def file_stages(path):
    """Which stage(s) a shader file can be part of.

    `vs/` and `*_vs.slang` are vertex-only, `ps/` and `*_ps.slang` pixel-only, and
    anything else (image.slang, postprocessquad.slang, types/) is shared.  That is
    what makes a per-stage move checkable rather than asserted."""
    rel = os.path.relpath(path, SHADER_ROOT).replace("\\", "/")
    base = os.path.basename(rel)
    if rel.startswith("vs/") or base.endswith("_vs.slang"):
        return ("vs",)
    if rel.startswith("ps/") or base.endswith("_ps.slang"):
        return ("ps",)
    return ("vs", "ps")

# UVMapping selects the layer's UV generation.  Exactly three of its values are the
# triplanar family -- UVMAP_TRIPLANAR_LOCAL / _WORLD / _WORLD_LOCAL_Z, 16..18, per
# IsTriPlanarMappingFX in sc2_shaders/ps/model_material.slang:53.  Those index the
# 18-slot TriPlanarUVs table, and a dynamic index there would force a 108-float
# indexable temp.  Triplanar is 0.3 % of the corpus, so instead of solving that,
# `anyTriPlanar` becomes one key bit selecting a rare, fully static arm (design
# section 2.3).  Widening this set is expensive: taking it to 9..18 pulls ordinary
# atlas/scroll mappings into the static arm and costs Model_ps a 1.39x class blowup.
TRIPLANAR_UVMAPPINGS = frozenset((16, 17, 18))


def _layer_axis(layer, prop):
    return "b_i%s%s" % (layer, prop)


# ===========================================================================
# 2. Non-layer interface axes (K1..K5)
# ===========================================================================
# K1 -- pass / mode.  Disjoint shader bodies the engine already dispatches as
# separate draws; there is nothing to gain from unifying them and a lot of
# instruction cache to lose.
K1_AXES = [
    "b_iShadingMode",           # STANDARD / DISPLACEMENT / VOLUME / ... (8 values)
    "b_iCloakingPass", "b_iVolumePass", "b_iDisplacementPass", "b_iCreepPass",
    "b_blurPass", "b_haloPass", "b_iDrawType",
    # own-root / own-axis family modes
    "b_iMode",                  # PostProcessQuad: 13 unrelated post effects
    "b_iInstanceType",          # Particle: 10 disjoint quad-basis constructions
    "b_iRibbonType",
    "b_iBokehPass",
    "b_stencilFillPass",        # SplatDirect: a depth/stencil-only pass
    "b_iSplatBoxRender",        # SplatDeferred: box vs decal projection
    "b_renderBlack",            # a whole-shader early-out
]

# K2 -- MRT signature.  The `SPixelOut` struct IS the output signature.
K2_AXES = [
    "b_emitFinalColor", "b_emitAlpha", "b_emitMRTDiffuse", "b_emitMRTNormal",
    "b_emitMRTSpecular", "b_emitMRTSpecularPower", "b_emitMRTDepth", "b_emitMRTAO",
    "b_blendMRTNormals",
]

# K3 -- texture set.  Every one of these adds, removes or retypes an SRV.
K3_AXES = [
    # shadow map (+ the transparent pair, + the soft-shadow rotation texture)
    "b_useShadows", "b_useTransparentShadows", "b_iUseSoftShadows",
    # ambient environment cube
    "b_UseAmbientEnvironment",
    # fog of war
    "b_SampleFOW", "b_FastFOW", "b_PostProcessFastFOW",
    # lighting-region weight map
    "b_UseLightingRegions",
    # soft-particle depth blend -> the normal/depth prepass map
    "b_useDepthBlend",
    # SSAO
    "b_iUseDynamicAO", "b_UseVertexBasedAmbientOcclusion",
    # parallax occlusion mapping -> the heightmap layer + a ray-march loop (also K5)
    "b_useParallaxMapping", "b_iUsePOMNew",
    # creep map set
    "b_creepEnabled", "b_iCreepUseGroundNormalMap",
    # which depth source SplatDeferred reads
    "b_iUseHardwareDepth",
    # own-root families
    "b_iLayerCount",            # Image: gates the p_sLayer0..3 sampler set
    "b_iUseAlphaMask",          # Image: adds a sampler
    "b_useBloom",               # HDR: adds p_sBloomMap
    "b_useColorMapping",        # HDR: adds the 3D LUT p_sColorMap
    "b_iUseSeparateBlurMap", "b_iUseSeparateDetailSSAO",
    "b_iParticleScreenTexture",
]

# K4 -- IO signature.  VS output must contain PS input, and an interpolant ARRAY's
# size IS its register count.
K4_AXES = [
    "b_iUVEmitterCount",        # sizes the m_UV[N] interpolant array
    "b_iUVEmitterArraySize",
    "b_iUVCoordCount",          # gates the TEXCOORD0..3 INPUT declarations (VS)
    "b_iVertexLayoutLayerCount",   # Image: gates the vertex stream
    "b_iRibbonSimTech",         # genuinely reshapes RibbonVSStreamIn (4 vs 13 attrs)
    "b_iSampleInterpolantCount",   # sizes the GaussianBlurSample array (also K5)
    # vertex layout axes -- these gate INPUT declarations
    "b_iLayoutHasColor", "b_iLayoutHasCustomUB4N1",
    "b_iLayoutHasNoVertexBlendIndices", "b_iLayoutHasNoVertexBlendWeights",
    "b_useModelInstancing", "b_iUseSignedIntUVs", "b_LayoutUseSignedIntUVs",
    "b_compressedVertex",
]

# K5 -- discard and loop trip counts.  `discard` kills early-Z for the whole
# shader; a dynamic trip count trades a predictable cost for a variable one and
# defeats unrolling, which is the one change in this design that can genuinely
# regress frame time.
K5_AXES = [
    "b_iAlphaTest",
    "b_iSoftShadowTaps",
    "b_iWarpCount", "b_enableVertexWarps",
    "b_iTapCount", "b_iSampleCount", "b_iSampleCountPS", "b_iSSAOSampleCount",
    "b_iColorMapSize",
]

# Two trip counts the design deliberately MOVES anyway, because the loop runs over
# data that is ALWAYS present and the count only decides how much of it counts:
#   b_iBlendWeightCount   -- the skinning loop iterates an always-float4 BLENDWEIGHT
#                            attribute, so a 4-iteration [unroll] with a zero-weight
#                            early-out is equivalent (design section 3).
#   b_iDirectionalLightCount -- the light array is always declared; the count only
#                            bounds how many entries contribute (design section 2.2).
# Together they are most of the VS reduction, so they are named here rather than
# left to look like an oversight in K5.
MOVED_TRIP_COUNTS = ["b_iBlendWeightCount", "b_iDirectionalLightCount"]

# Axes the design deliberately KEEPS compile-time even though they look movable,
# and the reason, so a later reader does not "fix" them.
KEPT_DELIBERATELY = {
    "b_iShadingMode": "disjoint shader bodies; separate engine draws",
    "b_iInstanceType": "10 disjoint quad-basis constructions (particle_vs.slang)",
    "b_iRibbonSimTech": "reshapes RibbonVSStreamIn: 4 attributes vs 13",
    "b_iSampleInterpolantCount": "sizes the GaussianBlurSample interpolant ARRAY "
                                 "and bounds a loop -- K4 and K5 at once",
    "b_iSoftShadowTaps": "loop trip count; a dynamic tap loop defeats unrolling",
    "b_iDirectionalLightCount": "loop trip count over the light array",
}

# Axes that appear in the corpus but are DEAD in the port, so they belong in
# neither the key nor the payload.  Each needs positive evidence.
DEAD_AXES = {
    "b_addMotionBlur": "hdr.fx's motion-blur block is commented out "
                       "(see sc2_shaders/hdr.slang header)",
}


def _mode_layers_for(mode_name):
    return [L for L in LAYERS_MODE if L.startswith(mode_name)]


# ===========================================================================
# 3. The interpolant transport profile (design section 4)
# ===========================================================================
# Profiles are a bitmask over interpolant GROUPS, not over individual names.  With
# fixed canonical TEXCOORD slots (R2) any PS reading a SUBSET of what the VS wrote
# links correctly, so the VS may use a coarser profile than the PS and the subset
# rule holds by construction.
#
# `base` (Normal, FogColor, UV[]) is implicit -- always live -- so it costs no bit.
XPORT_GROUPS = OrderedDict([
    ("tangent",  {"Tangent", "Binormal"}),
    ("light",    {"HalfVec", "ViewPos", "EyeToVertex"}),
    ("shadow",   {"ShadowMapUV"}),
    ("fow",      {"FOWUV"}),
    ("fresnel",  {"EyeToVertexFresnel"}),
    ("vlight",   {"Diffuse", "Specular", "ShadowDiffuse", "ShadowSpecular"}),
    ("vcolor",   {"VertexColor"}),
    ("screen",   {"ScreenPos", "HPosAsUV", "BackBufferUV"}),
    ("tri",      {"TriPlanarWeights", "VectorUI0", "VectorUI1", "VectorUI2",
                  "Vector4"}),
    ("parallax", {"ParallaxVector"}),
    ("terrain",  {"TerrainUV", "WorldPos"}),
])
# The VERTEX stage uses a COARSER partition: 6 groups instead of 12.  Coarsening
# only ever ADDS interpolants, so a 6-group VS profile is always a superset of the
# 12-group PS profile it feeds -- the "PS input is a subset of VS output" rule holds
# by construction, and the VS pays about +6 interpolants for a ~2x class reduction
# on the stage where export cost is per-vertex rather than per-pixel.
# Measured over the Model corpus: 12 groups -> 248 profiles, 6 -> 54, 4 -> 14.
XPORT_GROUPS_VS = OrderedDict([
    ("tangent",   {"Tangent", "Binormal"}),
    ("light",     {"HalfVec", "ViewPos", "EyeToVertex", "EyeToVertexFresnel"}),
    ("shadowfow", {"ShadowMapUV", "FOWUV"}),
    ("vertex",    {"Diffuse", "Specular", "ShadowDiffuse", "ShadowSpecular",
                   "VertexColor"}),
    ("screen",    {"ScreenPos", "HPosAsUV", "BackBufferUV", "ParallaxVector"}),
    ("world",     {"TriPlanarWeights", "VectorUI0", "VectorUI1", "VectorUI2",
                   "Vector4", "TerrainUV", "WorldPos"}),
])
XPORT_BASE = {"Normal", "FogColor", "UV"}
# Interpolants outside every group keep their own bit, so nothing can be dropped
# silently; `--axes` reports them.
XPORT_OTHER = ("DownscaleUV0", "DownscaleUV1", "GaussianBlurSample", "FrontFace",
               "View")


def xport_profile(live, stage="ps"):
    """The transport profile of one live interpolant set: (group_mask, other_tuple).

    Coarsening to groups is what decouples the two stages.  Today
    `cfg.interp_defines` packs the EXACT live set into TEXCOORD0,1,2... sorted by
    name, which makes the register assignment class-dependent and welds each PS
    permutation to one VS permutation -- 970 distinct layouts for Model alone.
    """
    if True:
        # BOTH stages use the EXACT live set, not a group mask.
        #
        # Coarsening a PS profile makes the class declare interpolants the retail
        # slot did not have -- and the reference SUBSTITUTES for exactly those:
        # sc2_interp.gen_preamble reconstructs a dropped Binormal from
        # cross(Normal, Tangent)*Normal.w, recomputes a dropped ShadowMapUV/FOWUV,
        # and zeroes anything else.  The comparator then feeds the candidate an
        # independent random value for a varying the reference derived, so the two
        # legs diverge for a reason that cannot happen in the engine (where the VS
        # writes the real value) and that no amount of shader work would fix.
        #
        # The decoupling this design actually needs comes from the CANONICAL SLOTS,
        # not from coarsening: with fixed semantic indices a PS still links to any
        # VS that writes a superset.  So the VERTEX stage keeps its 6-group profile
        # -- writing extra interpolants is harmless and compare_vs ignores them --
        # and the pixel stage stays exact and fully validatable.  Measured cost of
        # this choice is reported by --report.
        return 0, tuple(sorted(live))
    groups = XPORT_GROUPS_VS
    ls = set(live)
    mask = 0
    for i, (_name, members) in enumerate(groups.items()):
        if ls & members:
            mask |= 1 << i
    other = tuple(n for n in XPORT_OTHER if n in ls)
    # Anything we have not accounted for keeps its identity rather than vanishing.
    known = set(XPORT_BASE) | set(XPORT_OTHER)
    for members in groups.values():
        known |= members
    unknown = tuple(sorted(ls - known))
    return mask, other + unknown


# ===========================================================================
# 4. Core layers (design section 5.2)
# ===========================================================================
# The N most-used layers of a family are ALWAYS compiled in; the engine binds a
# shared 1x1 white texture when a material does not use them.  That turns a wide
# sampled-mask into a few residual key bits.  N is tuned PER FAMILY: the siblings
# saturate at 6, Model does not, so copying Model's 12 would cost a sibling ~5.9
# extra wasted layer evaluations for ~23 shaders.
# What "sound" actually requires here is narrower than the design assumed.  A core
# layer's texture is sampled unconditionally and the RESULT IS DISCARDED when the
# payload says the layer is off, so the only property that has to hold is that the
# discard is real -- which t6_dummy_invariance MEASURES (two different texture models,
# bit-identical SV_Target).  Binding a shared 1x1 white texture is a host-side nicety
# for backends that forbid a null descriptor, not a correctness precondition on D3D11,
# where reading an unbound SRV is defined to return 0.
#
# A layer joins the core set only once its SHADER path is core-capable: the layer
# block compiled in unconditionally, and its CONTRIBUTION gated by a runtime
# `if (cfg.b_iLayerEnable)`.  Same contract as ENABLED_LAYER_PROPS -- until the layer
# is listed here it stays interface-bearing, so a class still pins it.
#
# The three layers deliberately NOT in this set are the structurally awkward ones --
# Normal feeds the whole normal-mapping chain, Envio/EnvioMask nest inside each other
# and carry a cube-vs-2D declaration -- and they are also, measured, the cheap ones:
# adding all three buys 381 more classes out of the 10,148 this set already removes.
CORE_LAYERS_READY = frozenset([
    "Diffuse", "Emissive", "Emissive2", "AlphaMask", "AlphaMask2",
    "Specular", "SpecularExponent", "AmbientOcclusion", "Decal",
])
CORE_N_R7 = {
    "Model": 12,
    "Particle": 6,
    "Ribbon": 6,
    "Foliage": 6,
    "SplatDirect": 6,
    "SplatDeferred": 0,     # varies only one layer axis; N=0 already suffices
}
CORE_N = CORE_N_R7


def layer_sampled(bv, layer):
    """Does this layer sample its texture?  That, not `LayerEnable` alone, is what
    decides the SRV set: a layer with a constant colour reads no texture."""
    return (int(bv.get(_layer_axis(layer, "LayerEnable"), 0)) != 0
            and int(bv.get(_layer_axis(layer, "UseConstantColor"), 0)) == 0)


# ===========================================================================
# 5. Axis-set declaration (T1's subject)
# ===========================================================================
def _widths():
    """axis name -> encoded bit width, from the engine's own schema table.

    Using the schema rather than the observed maximum keeps a payload field wide
    enough for values this corpus happens not to contain."""
    out = {}
    tbl = json.load(open(os.path.join(HERE, "sc2_schema_widths.json")))
    for sect in tbl.values():
        for name, w in sect.get("axes", []):
            out[name] = max(out.get(name, 0), int(w))
    return out


WIDTHS = _widths()


def axis_width(name, values=()):
    """Bits needed for `name`.  Prefers the schema; falls back to the observed
    value range so an axis missing from the schema still packs correctly.

    For a LAYER axis the schema alone decides, so that the packed layer block is
    identical for every family -- which is what lets the offsets live in one shared
    generated header instead of on a per-class command line."""
    w = WIDTHS.get(name)
    obs = max([int(v) for v in values] + [0])
    need = max(1, obs.bit_length())
    if _split_layer_axis(name) is not None and w:
        if need > w:
            raise ValueError("layer axis %s observed %d needs %d bits > schema %d"
                             % (name, obs, need, w))
        return w
    return max(w or 0, need)


IRRELEVANT_CACHE = os.path.join(HERE, "_sc2_perm_cache", "irrelevant.json")


def probe_irrelevant(family, stage, bases=6, jobs=8, verbose=False):
    """Which varying axes provably do not affect this (family, stage)?

    An axis is IRRELEVANT if changing its value leaves the compiled DXBC
    byte-identical.  That is a measurement, not a judgement, and it matters because
    sc2_shaders is one translation unit: every family inherits the shared [Common]
    axes whether or not its entry point reads them.  image.fx never touches
    `b_iInEncoding`, but leaving it in the key would triple Image's class count for
    no behavioural reason.

    Conservative by construction: an axis counts as irrelevant only if it is
    byte-identical at EVERY base permutation probed, so a value that matters on some
    other path keeps the axis compile-time.
    """
    import hashlib
    from concurrent.futures import ThreadPoolExecutor
    import sc2_slang_validate as V

    fc = cfg.family_cfg(family)
    if stage not in fc:
        return set()
    entry = fc[stage]["slang_entry"]
    scratch = os.path.join(HERE, "_sc2_perm_cache", "probe")
    os.makedirs(scratch, exist_ok=True)
    rows = list(iter_slots_fast(family, stage))
    step = max(1, len(rows) // bases)
    picks = rows[::step][:bases]
    varying = inventory()["%s_%s" % (family, stage)]["varying"]

    def blob(bv, live, tag):
        defs = cfg.bv_to_defines(bv)
        if family in cfg._SHARED_TRANSPORT:
            defs += cfg.interp_defines(live, bv, stage)
            if stage == "vs":
                defs += ["SC2_STAGE_VS=1"]
                defs += ["SC2_UVMAPPING%d=%d" % (i, m)
                         for i, m in enumerate(cfg.uv_mappings(bv))]
                defs += ["SC2_UVRANDOM%d=%d" % (i, m)
                         for i, m in enumerate(cfg.uv_random_offsets(bv))]
        path = os.path.join(scratch, tag)
        ok = V.compile_slang(cfg.SC2_MODULE, entry, stage, defs, path,
                             include_dirs=[cfg.SC2_INCLUDE])[0]
        p = path + ".dxbc"
        h = hashlib.md5(open(p, "rb").read()).hexdigest() if os.path.exists(p) else None
        for e in (".dxbc", ".asm"):
            try:
                os.remove(path + e)
            except OSError:
                pass
        return h

    const = const_axes(family, stage)
    jobs_list = []
    for bi, (slot, bvv, live) in enumerate(picks):
        full = dict(const)
        full.update(bvv)
        jobs_list.append(("base%d" % bi, bi, None, full, live))
        for a, vals in sorted(varying.items()):
            alt = [v for v in vals if v != int(full.get(a, 0))]
            if not alt:
                continue
            mod = dict(full)
            mod[a] = alt[-1]
            jobs_list.append(("b%d_%s" % (bi, a), bi, a, mod, live))

    def work(j):
        tag, bi, a, bv, live = j
        return (bi, a, blob(bv, live, "%s_%s_%s" % (family, stage, tag)))

    with ThreadPoolExecutor(max_workers=jobs) as ex:
        res = list(ex.map(work, jobs_list))
    basehash = {bi: h for bi, a, h in res if a is None}
    same, seen = {}, set()
    for bi, a, h in res:
        if a is None:
            continue
        seen.add(a)
        ok = (h is not None and basehash.get(bi) is not None and h == basehash[bi])
        same[a] = same.get(a, True) and ok
    out = sorted(a for a in seen if same.get(a))
    if verbose:
        print("  %s_%s: %d/%d varying axes provably irrelevant"
              % (family, stage, len(out), len(varying)))
    return set(out)


_IRREL = None


def irrelevant_axes(family, stage):
    """Cached probe_irrelevant result; empty when the probe has not been run."""
    global _IRREL
    if _IRREL is None:
        _IRREL = (json.load(open(IRRELEVANT_CACHE))
                  if os.path.exists(IRRELEVANT_CACHE) else {})
    return set(_IRREL.get("%s_%s" % (family, stage), []))


def classify(family, stage, varying):
    """Split `varying` (the axis names that actually vary in this manifest) into
    the compile-time set and the data set.

    Returns (key_axes, payload_axes, dual, dead) as sorted lists.  This is the
    function T1 audits: `key | payload | dead` must cover `varying` exactly, and
    `key & payload` must be contained in the declared dual-role set."""
    iface = set(K1_AXES) | set(K2_AXES) | set(K4_AXES) | set(K5_AXES)
    # K3 is a PIXEL-stage category.  No vertex shader in this module declares a
    # texture, so a "texture set" axis has no interface effect there at all -- its
    # only structural consequence is which interpolants the stage writes, and that
    # is already carried by the transport profile.  Treating K3 as compile-time in
    # the VS is what took Model_vs to 14,493 classes instead of 1,293.
    if stage == "ps":
        iface |= set(K3_AXES)
    core = core_layers(family, stage)
    irrel = irrelevant_axes(family, stage)
    key, payload, dual, dead = set(), set(), set(), set()
    for a in varying:
        if a in DEAD_AXES or a in irrel:
            # Measured to leave the compiled DXBC byte-identical for this family --
            # a [Common] axis its entry point never reads.  Neither key nor payload.
            dead.add(a)
            continue
        lay = _split_layer_axis(a)
        if lay is not None and stage == "ps":
            layer, prop = lay
            if prop in DUAL_ROLE_PROPS and layer in core:
                # Core layer: the texture is bound either way, so LayerEnable and
                # UseConstantColor stop being interface-bearing and become plain data.
                payload.add(a)
            elif prop in DUAL_ROLE_PROPS:
                key.add(a)
                payload.add(a)
                dual.add(a)
            elif prop in LAYER_IFACE_PROPS:
                key.add(a)
            elif prop in ENABLED_LAYER_PROPS:
                payload.add(a)
            else:
                key.add(a)      # not moved yet -- still a compile-time define
            continue
        if a in iface or a not in moved_axes(stage):
            key.add(a)
        else:
            payload.add(a)
    return sorted(key), sorted(payload), sorted(dual), sorted(dead)


_LAYER_BY_LEN = sorted(ALL_LAYERS, key=len, reverse=True)


def _split_layer_axis(a):
    """('Diffuse', 'Op') for b_iDiffuseOp, else None.  Longest layer name first so
    `AlphaMask2` is never parsed as `AlphaMask` + `2...`."""
    if not a.startswith("b_i"):
        return None
    body = a[3:]
    for L in _LAYER_BY_LEN:
        if body.startswith(L):
            prop = body[len(L):]
            if prop in LAYER_PROPS:
                return L, prop
    return None


# ===========================================================================
# 6. The structural key
# ===========================================================================
def structural_key(family, stage, bv, live, core=None):
    """The complete COMPILE-TIME identity of a reduced permutation.

    Two retail slots with the same key compile to the same blob; their difference
    is carried entirely by `pack_payload`.  The key reads only interface-bearing
    axes -- plus three DERIVED coarsenings that are the whole point of the design:

      * `sampled`   one bit per non-core layer (core layers are always compiled in),
      * `xport`     the interpolant transport profile instead of the exact live set,
      * `triplanar` one bit, plus the static slot ids only inside the rare arm.
    """
    if core is None:
        core = core_layers(family, stage)
    fam_layers = family_layers(family, stage)

    # K3 -- which layers are compiled in.  Until the core-layer dummy binding lands
    # (R7) this is not a mask but the exact per-layer LayerEnable / UseConstantColor
    # pair, because those two are what gate the layer's texture sample and the class
    # has to pin them for every member.  With a core set they collapse to a mask over
    # the NON-core layers.
    sampled = tuple((L,
                     int(bv.get(_layer_axis(L, "LayerEnable"), 0)),
                     int(bv.get(_layer_axis(L, "UseConstantColor"), 0)))
                    for L in fam_layers
                    if L not in core
                    and (int(bv.get(_layer_axis(L, "LayerEnable"), 0))
                         or int(bv.get(_layer_axis(L, "UseConstantColor"), 0))))
    # K3 -- 2D vs cube is a declaration, so it stays compile-time for every layer.
    textype = tuple((L, int(bv.get(_layer_axis(L, "TextureType"), 0)))
                    for L in fam_layers
                    if int(bv.get(_layer_axis(L, "TextureType"), 0)))

    # design section 2.3 -- triplanar selects a rare fully static arm.
    tri_layers = tuple(sorted(
        L for L in fam_layers
        if int(bv.get(_layer_axis(L, "UVMapping"), 0)) in TRIPLANAR_UVMAPPINGS))
    if tri_layers:
        tri = tuple((L,
                     int(bv.get(_layer_axis(L, "UVMapping"), 0)),
                     int(bv.get(_layer_axis(L, "LayerId"), 0)),
                     int(bv.get(_layer_axis(L, "TriPlanarUvId"), 0)))
                    for L in tri_layers)
    else:
        tri = ()

    # Layer properties that have NOT been moved to the payload yet are ordinary
    # compile-time axes, so they belong in the key.  As each one is added to
    # ENABLED_LAYER_PROPS this tuple shrinks and the class count drops -- which is
    # exactly the per-increment progress R5 is measured by.
    layerprops = tuple(
        (_layer_axis(L, p), int(bv.get(_layer_axis(L, p), 0)))
        for L in fam_layers for p in LAYER_PROPS
        if p not in ENABLED_LAYER_PROPS and p not in LAYER_IFACE_PROPS)
    layerprops = tuple(kv for kv in layerprops if kv[1])

    mask, other = xport_profile(live, stage)
    iface = tuple((a, int(bv.get(a, 0)))
                  for a in _iface_axes_for(family, stage)
                  if a in bv)
    return ("v1", family, stage, iface, sampled, textype, tri, mask, other,
            layerprops)


_IFACE_CACHE = {}


def _iface_axes_for(family, stage):
    """The non-layer interface axes that VARY for this (family, stage), sorted.

    Restricting to the varying set keeps the key tuple small and makes the class
    count independent of axes the family never exercises."""
    ck = (family, stage)
    if ck not in _IFACE_CACHE:
        inv = inventory()
        varying = inv["%s_%s" % (family, stage)]["varying"]
        key, _p, _d, _dead = classify(family, stage, varying)
        _IFACE_CACHE[ck] = tuple(a for a in key if _split_layer_axis(a) is None)
    return _IFACE_CACHE[ck]


# ===========================================================================
# 7. The payload
# ===========================================================================
LAYER_WORDS = 2         # uint32s per layer in the payload (see payload_fields)


# ---------------------------------------------------------------------------
# The payload layout is MODULE-GLOBAL: one bit offset per axis name, identical for
# every family and both stages.
#
# The alternative -- a layout derived from each family's own payload set -- is what
# made the adoption switch have to be per family, because a shared file compiled
# for family A would reference an offset only family B declared.  With one global
# table every SC2_PERMOFF_ exists in every compile, the whole `SC2_USE_PERM_CB_<F>`
# scoping problem disappears, and the engine gets the single buffer shape design
# section 5.1 asked for.
#
# It is also what makes the layout independent of MOVED_NONLAYER_AXES: the table
# covers every axis that COULD be data, so switching one on shifts nothing.  (That
# invariant is the one whose violation made every layer property look broken at
# once -- see the plan's trap list.)
def _true_iface(stage):
    """The axes that are interface-bearing by CATEGORY, so never payload."""
    s = set(K1_AXES) | set(K2_AXES) | set(K4_AXES) | set(K5_AXES)
    if stage == "ps":
        s |= set(K3_AXES)
    return s


_CONVERTIBLE = None


def convertible_nonlayer_axes():
    """Every non-layer axis that is data SOMEWHERE, whether or not it is switched on.

    Membership is by K-category alone, never by MOVED_NONLAYER_AXES, so the layout
    is stable across the R5b/R6 increments."""
    global _CONVERTIBLE
    if _CONVERTIBLE is None:
        inv = inventory()
        out = {}
        for fam, st in _all_stages():
            ent = inv.get("%s_%s" % (fam, st))
            if not ent:
                continue
            ifc = _true_iface(st)
            for a, vals in ent["varying"].items():
                if (a in ifc or a in DEAD_AXES
                        or _split_layer_axis(a) is not None):
                    continue
                w = axis_width(a, vals)
                out[a] = max(out.get(a, 0), w)
        _CONVERTIBLE = out
    return _CONVERTIBLE


_NONLAYER_LAYOUT = None


def nonlayer_layout():
    """{axis: (bit_offset, width)} -- the global non-layer block, at offset 0."""
    global _NONLAYER_LAYOUT
    if _NONLAYER_LAYOUT is None:
        fields, off = {}, 0
        for a, w in sorted(convertible_nonlayer_axes().items()):
            if (off % 32) + w > 32:         # never straddle a word
                off = ((off // 32) + 1) * 32
            fields[a] = (off, w)
            off += w
        _NONLAYER_LAYOUT = fields
    return _NONLAYER_LAYOUT


def layer_block_base():
    """Bit offset of the material-layer block: after the non-layer block, rounded
    up to a whole uint32 so a layer's two words stay word-aligned."""
    fields = nonlayer_layout()
    bits = max([o + w for o, w in fields.values()] + [0])
    return ((bits + 31) // 32) * 32


def _payload_layers(family, stage):
    """Which layers get a payload slot.

    A DefaultPixelMain family gets the FULL layer table, so the six that share
    ps/model_layers.slang share one buffer layout; anything else gets only the
    layers it actually varies."""
    if stage == "ps" and family in cfg._SHARED_TRANSPORT:
        return ALL_LAYERS
    return family_layers(family, stage)


def payload_fields(family, stage):
    """[(axis, bit_offset, width)] -- the packed layout of this family's data axes.

    Layer properties are grouped per layer so a layer's whole spec lands in one
    word range, which is what lets the shader read `layerSpec[i]` and unpack.
    Non-layer data axes follow in one block."""
    ck = (family, stage)
    if ck in _PAYLOAD_CACHE:
        return _PAYLOAD_CACHE[ck]
    inv = inventory()
    ent = inv["%s_%s" % (family, stage)]
    varying = ent["varying"]
    _key, payload, _dual, _dead = classify(family, stage, varying)
    pset = set(payload)

    fields = []
    off = layer_block_base()
    # Per-layer block: LAYER_WORDS uint32s, so a layer's whole spec is addressed as
    # layerSpec[i] with a fixed stride.  The design sketched one word per layer at
    # 34 packed bits; the engine's OWN schema widths (UVMapping 5, LayerId 5,
    # TriPlanarUvId 5, UVEmitter 4, ...) sum past 32 for a fully-featured layer, so
    # the stride is two words.  Using the schema rather than the observed maximum
    # keeps a field wide enough for a value this corpus happens not to contain.
    #
    # The six DefaultPixelMain families share one layer file, so they must share one
    # LAYOUT: every layer of ALL_LAYERS gets its slot whether or not this particular
    # manifest varies it.  Otherwise a class compiled for a family where a property
    # happens to be constant would have no SC2_PERMOFF_ for a field the shared code
    # still reads.  It also gives the engine exactly one buffer shape to fill, which
    # is what the design asked for.
    for L in _payload_layers(family, stage):
        base = off
        bit = 0
        full = _payload_layers(family, stage) is ALL_LAYERS
        for prop in LAYER_PROPS:
            a = _layer_axis(L, prop)
            # In the shared layout every non-interface property gets a slot; in a
            # family-local layout only the ones that actually vary do.
            # The shared layer block allocates EVERY non-declaration property,
            # whether or not it has been switched on yet.  Making the layout depend
            # on ENABLED_LAYER_PROPS would shift every offset each time a property
            # is flipped, while types/perm_layout.slangh stayed at the old layout --
            # the packer and the shader would silently disagree, and the resulting
            # garbage reads look exactly like a broken property.
            if not (full and prop != "TextureType") and a not in pset:
                continue
            w = axis_width(a, varying.get(a, ()))
            if (bit % 32) + w > 32:     # never straddle a word
                bit = ((bit // 32) + 1) * 32
            fields.append((a, base + bit, w))
            bit += w
        if bit:
            if bit > 32 * LAYER_WORDS:
                raise ValueError("layer %s needs %d bits (> %d) in %s_%s"
                                 % (L, bit, 32 * LAYER_WORDS, family, stage))
            off = base + 32 * LAYER_WORDS
    # Non-layer data axes take their GLOBAL offsets, so the same axis lands in the
    # same bits for every family -- see nonlayer_layout().
    gl = nonlayer_layout()
    for a in sorted(pset):
        if _split_layer_axis(a) is not None and stage == "ps":
            continue
        if a not in gl:
            raise ValueError("%s is payload in %s_%s but absent from the global "
                             "non-layer layout" % (a, family, stage))
        off_a, w = gl[a]
        fields.append((a, off_a, w))
    _PAYLOAD_CACHE[ck] = fields
    return fields


_PAYLOAD_CACHE = {}


def payload_words(family=None, stage=None):
    """uint32 word count of the payload, rounded up to a float4 register.

    ONE size for the whole module, because the layout is global: a per-family size
    would make `permCB.w[]` a different type per family, which is exactly the kind
    of family-dependence this design removed.  The unused tail costs a few bytes on
    a per-material upload."""
    bits = layer_block_base() + 32 * LAYER_WORDS * len(ALL_LAYERS)
    words = (bits + 31) // 32
    return ((words + 3) // 4) * 4          # whole float4 registers


def pack_payload(family, stage, bv):
    """The constant-buffer payload for one retail slot, as a list of uint32 words.

    Reads ONLY data axes, so it can never disagree with `structural_key` about an
    axis.  Words (not bytes) because the harness writes raw words: routing a packed
    bitfield through a float conversion would rewrite every one of them."""
    n = payload_words(family, stage)
    words = [0] * n
    const = const_axes(family, stage)
    for a, off, w in payload_fields(family, stage):
        v = int(bv.get(a, const.get(a, 0))) & ((1 << w) - 1)
        # A field never straddles a word: the layer blocks are 32-bit aligned and
        # every individual axis is <= 8 bits, so a straddle would be a packing bug.
        wi, bit = off // 32, off % 32
        if bit + w > 32:
            raise ValueError("payload field %s straddles a word (%d+%d)" % (a, bit, w))
        words[wi] |= v << bit
    return words


# ===========================================================================
# 8. Compile-time define list for a class
# ===========================================================================
def class_transport_defines(family, stage, key):
    """The interpolant-transport `-D` set implied by a class's PROFILE.

    Today `cfg.interp_defines` emits the EXACT live set, which is what welds a PS
    permutation to one VS permutation.  A class instead declares every member of
    every group its profile has set, at that member's canonical slot -- so one blob
    serves every live set inside the profile, and coarsening can only ever ADD an
    interpolant, never drop one a member slot needed.
    """
    (_v, fam, st, iface, _sampled, _tt, _tri, mask, other, _lp) = key
    if family not in cfg._SHARED_TRANSPORT:
        return []                    # own-IO families carry their own structs
    # exact live set (see xport_profile): `other` IS the list of live names
    names = set(other) & set(cfg.SC2_CANON_SLOT)
    ivals = dict(iface)
    defs = []
    for n in sorted(names):
        if n in cfg.SC2_CANON_SLOT:
            defs += ["SC2_HAS_%s=1" % n,
                     "SC2_SEM_%s=TEXCOORD%d" % (n, cfg.SC2_CANON_SLOT[n])]
    # The two ARRAY interpolants are sized by a count axis, which is why those axes
    # stay compile-time (K4): an array's length IS its register count.
    #
    # Gated on the profile like every interpolant above.  It used to be emitted
    # UNCONDITIONALLY, on the docstring's reasoning that coarsening "can only ever ADD
    # an interpolant, never drop one a member slot needed" -- true as stated, and the
    # wrong safety argument, because it assumes ADDING one is free.  It is not:
    # Model_ps slots 11203 / 47987 / 5457 / 13106 read no UV, and declaring it anyway
    # left them 0.073-0.179 off a reference the specialised build matches exactly.
    # Those four were the last real defects in the 107,976-slot grid.
    if "UV" in other:
        uvc = ivals.get("b_iUVEmitterArraySize" if stage == "vs" else "b_iUVEmitterCount", 0)
        defs += ["SC2_HAS_UV=1",
                 "SC2_SEM_UV=TEXCOORD%d" % cfg.SC2_CANON_UV_BASE,
                 "SC2_UV_COUNT=%d" % max(1, uvc)]
    if "GaussianBlurSample" in other:
        defs += ["SC2_HAS_GaussianBlurSample=1",
                 "SC2_SEM_GaussianBlurSample=TEXCOORD%d" % cfg.SC2_CANON_GBS_BASE,
                 "SC2_GBS_COUNT=%d" % max(1, ivals.get("b_iSampleInterpolantCount", 1))]
    for n in other:
        if n in ("FrontFace",) and stage == "ps":
            defs.append("SC2_HAS_FrontFace=1")
    if stage == "vs":
        defs.append("SC2_STAGE_VS=1")
        # model.fx declares `int b_iUVMapping[8]` / `b_UVRandomOffsetEnable[8]` as
        # per-instance ARRAYS that InitShader fills, so they cannot ride along as
        # ordinary axis defines -- cfg.perm_defines emits them as SC2_UVMAPPING<i> /
        # SC2_UVRANDOM<i>, and a class has to do the same.  Leaving them out silently
        # collapses every emitter to generation mode 0, which is what took Model_vs to
        # 154/250.
        vals = dict(const_axes(family, stage))
        vals.update(dict(iface))
        for i in range(8):
            defs.append("SC2_UVMAPPING%d=%d"
                        % (i, int(vals.get("b_iUVMapping%d" % i, 0))))
            defs.append("SC2_UVRANDOM%d=%d"
                        % (i, int(vals.get("b_UVRandomOffsetEnable%d" % i, 0))))
    return defs


def _data_axes(family, stage):
    """The non-layer axes THIS (family, stage) carries in the payload."""
    return {a for a, _o, _w in payload_fields(family, stage)
            if _split_layer_axis(a) is None}


def struct_defines(family, stage, key, live_repr=None):
    """The slangc `-D` list a CLASS is compiled with, derived from the key ALONE.

    This is what makes the reduction checkable: because it is a pure function of
    the key, every member slot of a class is compiled identically, and T3 can then
    assert that the list is consistent with each member's retail vector."""
    (_ver, fam, st, iface, sampled, textype, tri, mask, other, layerprops) = key
    assert (fam, st) == (family, stage)
    defs = []
    for a, v in iface:
        if v:
            defs.append("%s=%d" % (a, v))
    for p in ENABLED_LAYER_PROPS:
        defs.append("SC2_PERM_PROP_%s=1" % p)
    core = core_layers(family, stage)
    for L in core:
        # A core layer is compiled in unconditionally and its CONTRIBUTION becomes a
        # payload predicate rather than a define.
        #
        # This must NOT be spelled `b_i<L>LayerEnable=1`, tempting though it is: that
        # would compile the layer in through the existing `#if` with no source edit at
        # all, but it also PINS an axis whose members disagree, and T3 would rightly
        # call it a too-coarse key.  The core switch is its own name so the class pins
        # nothing it cannot honour, and the shader reads the real value from b2.
        defs.append("SC2_CORE_LAYER_%s=1" % L)
    for L, enable, useconst in sampled:
        if enable:
            defs.append("%s=%d" % (_layer_axis(L, "LayerEnable"), enable))
        if useconst:
            defs.append("%s=%d" % (_layer_axis(L, "UseConstantColor"), useconst))
    for L, v in textype:
        defs.append("%s=%d" % (_layer_axis(L, "TextureType"), v))
    for a, v in layerprops:
        if v:
            defs.append("%s=%d" % (a, v))
    if tri:
        defs.append("SC2_ANY_TRIPLANAR=1")
        for L, uvm, lid, tid in tri:
            defs.append("%s=%d" % (_layer_axis(L, "UVMapping"), uvm))
            defs.append("%s=%d" % (_layer_axis(L, "LayerId"), lid))
            defs.append("%s=%d" % (_layer_axis(L, "TriPlanarUvId"), tid))
    defs.append("SC2_XPORT_MASK=%d" % mask)
    defs += class_transport_defines(family, stage, key)
    # Two switches, deliberately: the global one means "the b2 buffer is declared",
    # the family-scoped one means "THIS family reads it".  Without the split, one
    # family's CB reads fire while another family's entry point is compiling and its
    # offsets are absent -- the one-TU leak.
    defs.append("SC2_USE_PERM_CB=1")
    defs.append("SC2_USE_PERM_CB_%s=1" % family)
    if stage == "ps" and family in cfg._SHARED_TRANSPORT:
        # The material-layer stack is ONE set of files serving six families, so its
        # adoption switch is shared too -- a per-family name there would need six
        # copies of every guard.
        defs.append("SC2_USE_PERM_CB_Default=1")
    # The BIT LAYOUT is global and lives in the generated types/perm_layout.slangh,
    # so a class passes no offsets at all -- only WHICH axes it carries as data.
    #
    # This has to be per axis rather than per family.  `b_useShadows` is data in a
    # vertex stage and interface-bearing in the pixel stage of the SAME family
    # (it binds a shadow map), so a family-wide "reads the CB" switch would make the
    # pixel stage read a payload field the packer never wrote -- a silent zero.
    for a in sorted(_data_axes(family, stage)):
        defs.append("SC2_PERMDATA_%s=1" % a)
    defs += const_defines(family, stage)
    return sorted(defs)


# ===========================================================================
# 9. Family layer sets and core selection
# ===========================================================================
_FAMLAYER_CACHE = {}


def family_layers(family, stage):
    """The layers this (family, stage) actually varies -- the only ones its key or
    payload can mention.  Derived, not declared, so adding a family needs no edit."""
    ck = (family, stage)
    if ck not in _FAMLAYER_CACHE:
        varying = inventory()["%s_%s" % (family, stage)]["varying"]
        seen = set()
        for a in varying:
            lay = _split_layer_axis(a)
            if lay is not None:
                seen.add(lay[0])
        _FAMLAYER_CACHE[ck] = [L for L in ALL_LAYERS if L in seen]
    return _FAMLAYER_CACHE[ck]


def core_layers(family, stage):
    """The N always-compiled layers, chosen by THIS family's own sample frequency.

    Particle and Ribbon lead with AlphaMask rather than Diffuse, so a copied core
    set would waste evaluations on the wrong layers."""
    n = CORE_N.get(family, 0)
    if not n or stage != "ps":
        return frozenset()
    freq = _core_freq()[("%s_%s" % (family, stage))]
    ranked = sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))
    # `_c` filters layers this family never uses at all: Foliage's 6th-ranked layer
    # is Decal with ZERO samples, and compiling a layer in for a family whose corpus
    # never enables it is pure cost with no class to save.
    return frozenset(L for L, _c in ranked[:n]
                     if _c and L in CORE_LAYERS_READY)


_CORE_FREQ = None


def _core_freq():
    """{family_stage: {layer: sample count}}, cached to disk -- it costs a full
    decode sweep of the corpus."""
    global _CORE_FREQ
    if _CORE_FREQ is not None:
        return _CORE_FREQ
    if os.path.exists(CORE_CACHE):
        _CORE_FREQ = json.load(open(CORE_CACHE))
        return _CORE_FREQ
    out = {}
    for fam in sorted(cfg.load_families()):
        fc = cfg.family_cfg(fam)
        for st in ("vs", "ps"):
            if st not in fc:
                continue
            layers = family_layers(fam, st)
            c = Counter()
            for _slot, bv, _live in iter_slots_fast(fam, st):
                for L in layers:
                    if layer_sampled(bv, L):
                        c[L] += 1
            out["%s_%s" % (fam, st)] = {L: c.get(L, 0) for L in layers}
    os.makedirs(os.path.dirname(CORE_CACHE), exist_ok=True)
    json.dump(out, open(CORE_CACHE, "w"), indent=1)
    _CORE_FREQ = out
    return out


# ===========================================================================
# 10. Inventory
# ===========================================================================
_INV = None


def inventory(rebuild=False):
    """{family_stage: {slots, axes, varying{axis:[values]}, const, live_sets}}."""
    global _INV
    if _INV is not None and not rebuild:
        return _INV
    if os.path.exists(INVENTORY) and not rebuild:
        _INV = json.load(open(INVENTORY))
        return _INV
    out = {}
    for fam in sorted(cfg.load_families()):
        fc = cfg.family_cfg(fam)
        for st in ("vs", "ps"):
            if st not in fc:
                continue
            vals, live_sets, n = {}, set(), 0
            for _slot, bv, live, _dd in cfg.iter_slots(fam, st):
                n += 1
                for k, v in bv.items():
                    vals.setdefault(k, set()).add(int(v))
                live_sets.add(tuple(sorted(live)))
            out["%s_%s" % (fam, st)] = {
                "slots": n, "axes": len(vals),
                "varying": {k: sorted(v) for k, v in vals.items() if len(v) > 1},
                "const": {k: next(iter(v)) for k, v in vals.items() if len(v) == 1},
                "live_sets": len(live_sets),
            }
            print("  inventory %s_%s: %d slots, %d varying"
                  % (fam, st, n, len(out["%s_%s" % (fam, st)]["varying"])), flush=True)
    json.dump(out, open(INVENTORY, "w"), indent=1)
    _INV = out
    return out


# ===========================================================================
# 10b. Decoded-slot cache
# ===========================================================================
# Decoding the corpus is the expensive part of every pass here: Model_ps alone is
# ~170 s through sc2_family_decode, and the builder, T3 and the round-trip test each
# want the same walk.  Cache the VARYING axes as one flat byte blob (every observed
# axis value is < 256) plus the per-stage constants, which turns a 170 s sweep into
# a sub-second load.  decode_perm stays the single source of truth -- this is a
# memo of its output, keyed by the manifest's mtime so a re-decoded manifest
# invalidates it.
# Under tools/_sc2* so the repo's .gitignore already excludes it: this is a pure
# memo of decode_perm, ~25 MB, and is rebuilt on demand.
CACHE_DIR = os.path.join(HERE, "_sc2_perm_cache")


def _cache_path(family, stage):
    return os.path.join(CACHE_DIR, "%s_%s.bin" % (family, stage))


def _manifest_stamp(family, stage):
    p = os.path.join(cfg.PERMS_DIR, "%s_%s.json" % (family, stage))
    st = os.stat(p)
    return "%d:%d" % (st.st_mtime_ns, st.st_size)


def decoded(family, stage):
    """(names, rows, lives, slots, const) for one stage, from cache when possible.

    names  -- the varying axis names, sorted
    rows   -- a bytes blob, len(slots) * len(names), row-major
    lives  -- [tuple(live)] per slot
    slots  -- [retail slot index]
    const  -- {axis: value} for every axis that does NOT vary
    """
    import pickle
    path = _cache_path(family, stage)
    stamp = _manifest_stamp(family, stage)
    if os.path.exists(path):
        try:
            with open(path, "rb") as fp:
                blob = pickle.load(fp)
            if blob.get("stamp") == stamp and blob.get("version") == 1:
                return (blob["names"], blob["rows"], blob["lives"],
                        blob["slots"], blob["const"])
        except Exception:
            pass        # a corrupt cache is a cache miss, never an error
    inv = inventory()
    ent = inv["%s_%s" % (family, stage)]
    names = sorted(ent["varying"])
    const = {k: int(v) for k, v in ent["const"].items()}
    idx = {n: i for i, n in enumerate(names)}
    rows = bytearray()
    lives, slots = [], []
    for slot, bv, live, _dd in cfg.iter_slots(family, stage):
        row = bytearray(len(names))
        for k, v in bv.items():
            i = idx.get(k)
            if i is not None:
                iv = int(v)
                if not 0 <= iv < 256:
                    raise ValueError("axis %s = %d does not fit a byte" % (k, iv))
                row[i] = iv
        rows += row
        lives.append(tuple(live))
        slots.append(slot)
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(path, "wb") as fp:
        pickle.dump({"version": 1, "stamp": stamp, "names": names,
                     "rows": bytes(rows), "lives": lives, "slots": slots,
                     "const": const}, fp, protocol=4)
    return names, bytes(rows), lives, slots, const


def iter_slots_fast(family, stage):
    """Yield (slot, bv_varying, live) with bv_varying holding ONLY the axes that
    vary in this manifest.

    Every classifier here reads only varying axes, so this is complete for them;
    `const_axes()` supplies the rest for anything that needs a full vector."""
    names, rows, lives, slots, _const = decoded(family, stage)
    n = len(names)
    for i, slot in enumerate(slots):
        row = rows[i * n:(i + 1) * n]
        yield slot, dict(zip(names, row)), lives[i]


def const_axes(family, stage):
    """{axis: value} for the axes that do not vary -- the part of a permutation
    vector every class of this stage shares."""
    return decoded(family, stage)[4]


def const_defines(family, stage):
    """slangc `-D` list for the non-varying NONZERO axes.

    They are identical for every class, but they still have to be passed: a
    constant-but-nonzero axis is as load-bearing as a varying one, and leaving it
    out would compile a different shader than any retail slot ever used."""
    return ["%s=%d" % (k, v) for k, v in sorted(const_axes(family, stage).items())
            if v]


# ===========================================================================
# 11. The builder
# ===========================================================================
def build_classes(family, stage, verbose=False):
    """(classes, slot_table) for one (family, stage).

    classes    -- [key], in first-seen order; each becomes exactly one compiled blob
    slot_table -- [(retail slot, class index, payload words)] for every retail slot
    """
    core = core_layers(family, stage)
    classes, index, slots = [], {}, []
    for slot, bv, live in iter_slots_fast(family, stage):
        K = structural_key(family, stage, bv, live, core=core)
        c = index.get(K)
        if c is None:
            c = index[K] = len(classes)
            classes.append(K)
        slots.append((slot, c, pack_payload(family, stage, bv)))
    if verbose:
        print("  %-20s %6d slots -> %5d classes  (%.1fx)"
              % ("%s_%s" % (family, stage), len(slots), len(classes),
                 len(slots) / max(1, len(classes))))
    return classes, slots


def write_tables(family, stage, classes, slots):
    os.makedirs(REDUCED_DIR, exist_ok=True)
    path = os.path.join(REDUCED_DIR, "%s_%s.json" % (family, stage))
    json.dump({
        "family": family, "stage": stage,
        "retail_slots": len(slots), "classes": len(classes),
        "payload_words": payload_words(family, stage),
        "core_layers": sorted(core_layers(family, stage)),
        "class_defines": [struct_defines(family, stage, K) for K in classes],
        "slots": [{"slot": s, "class": c, "payload": p} for s, c, p in slots],
    }, open(path, "w"))
    return path


# ===========================================================================
# 12. Tests
# ===========================================================================
# Measured class counts, and the regression anchor for the key.
#
# PROVENANCE.  These are what the key produces AS IMPLEMENTED, with the axes that
# the SHADER has actually been converted to read from b2 -- see ENABLED_LAYER_PROPS
# and MOVED_NONLAYER_AXES.  They are therefore a snapshot of progress, not of the
# design's ceiling: Image (125.7x) and HDR (80x) are fully converted, the six
# DefaultPixelMain pixel stages have their 14 material-layer properties moved AND
# their nine core layers always compiled in (Model_ps 9.7x, Particle_ps 14.0x,
# Ribbon_ps 5.8x), and the stages still at 1.0x are the ones whose remaining axes
# are genuinely interface-bearing.
#
# Each number should move only when an axis is deliberately added to one of those
# two lists.  Drift without such a change means the key changed MEANING, which is
# exactly what this anchor exists to catch.
# Measured class counts, re-baselined after the R7 core layers.  A TRIPWIRE, not a target:
# these move whenever an axis joins the moved set, and the point is that they never
# move when nothing was supposed to change.
T_COUNT = {
    "Model_ps": 5163,
    "Model_vs": 8204,
    "Particle_ps": 758,
    "Particle_vs": 1952,
    "Image_ps": 28,
    "SplatDirect_ps": 765,
    "Ribbon_ps": 549,
    # 1658 -> 1548 when b_gpuSplineRibbon / b_splineRibbon / b_proceduralPosition /
    # b_precomputedTangent came off UNSAFE_AS_DATA; they were never unsafe, only
    # mis-measured (see the note there).
    "Ribbon_vs": 1548,
    "PostProcessQuad_ps": 222,
    "SplatDeferred_ps": 144,
    "SplatDeferred_vs": 4,
    "Foliage_ps": 164,
    "Foliage_vs": 58,
    "HDR_ps": 4,
    "SplatDirect_vs": 123,
    "DeferredLight_ps": 36,
    "DeferredLight_vs": 4,
    "Water_ps": 61,
    "Water_vs": 7,
    "TerrainBlend_ps": 54,
}


def t1_totality(family, stage):
    """Every axis that VARIES is in the key, the payload, or the declared dead set;
    and key/payload overlap only on the declared dual-role properties.

    This is the test that stops an axis being silently dropped -- the failure mode
    that would make the behavioural gate pass vacuously."""
    varying = inventory()["%s_%s" % (family, stage)]["varying"]
    key, payload, dual, dead = classify(family, stage, varying)
    fails = []
    covered = set(key) | set(payload) | set(dead)
    missing = sorted(set(varying) - covered)
    if missing:
        fails.append("unclassified: %s" % ", ".join(missing[:8]))
    overlap = (set(key) & set(payload)) - set(dual)
    if overlap:
        fails.append("undeclared key/payload overlap: %s"
                     % ", ".join(sorted(overlap)[:8]))
    for a in dual:
        lay = _split_layer_axis(a)
        if lay is None or lay[1] not in DUAL_ROLE_PROPS:
            fails.append("dual axis %s is not a declared dual-role property" % a)
    return fails


def t3_class_consistency(family, stage, classes, slots, sample=0):
    """Every class's compile-time define list must be CONSISTENT with each of its
    member slots: an axis the class pins must hold that value in the member's
    retail vector.

    A too-coarse key fails here with a precise message, long before the expensive
    behavioural sweep would fail with a numeric one."""
    fails = []
    defs = [dict(_kv(d) for d in struct_defines(family, stage, K)) for K in classes]
    # iter_slots_fast yields only the VARYING axes; struct_defines also pins the
    # stage's constant-but-nonzero axes, so the comparison needs the full vector or
    # every constant would read back as 0.
    const = const_axes(family, stage)
    it = list(iter_slots_fast(family, stage))
    if sample and len(it) > sample:
        step = max(1, len(it) // sample)
        it = it[::step][:sample]
    by_slot = {s: c for s, c, _p in slots}
    for slot, bv, _live in it:
        d = defs[by_slot[slot]]
        for name, val in d.items():
            if not name.startswith("b_"):
                continue
            if int(bv.get(name, const.get(name, 0))) != val:
                fails.append("slot %d class %d pins %s=%d but retail has %d"
                             % (slot, by_slot[slot], name, val,
                                int(bv.get(name, const.get(name, 0)))))
                if len(fails) > 20:
                    return fails
    return fails


def t3b_class_delivery(family, stage, classes, slots, sample=0):
    """T3's CONVERSE: every axis a class's members DISAGREE on must be delivered.

    `t3_class_consistency` walks the class's own define list and checks each pinned
    value against the member's retail vector.  That direction only fails on an axis
    the class MENTIONS -- so an axis in neither the key nor the payload is invisible
    to it, which is precisely the bug it was written to catch.  (Same shape as the
    interpolant gate that compared only matched output keys and reported green.)

    An axis whose members disagree fails in one of two ways:

      UNCARRIED  not in `payload_fields`, so nothing writes its bits and the blob
                 keeps whatever constant it happened to compile with;
      UNREAD     the packer writes the bits but the class's `-D` list never enables
                 the shader's data path for it --
                   non-layer axis                 SC2_PERMDATA_<axis>=1
                   layer property                 SC2_PERM_PROP_<Prop>=1
                   LayerEnable / UseConstantColor  SC2_CORE_LAYER_<Layer>=1

    UNREAD is what an entry added to `payload_fields` but forgotten in
    ENABLED_LAYER_PROPS looks like, and it is silent: the bits are packed, the layout
    agrees, and only the pixels are wrong.
    """
    fails = []
    carried = {a for a, _o, _w in payload_fields(family, stage)}
    defs = [set(struct_defines(family, stage, K)) for K in classes]
    const = const_axes(family, stage)
    by_slot = {s: c for s, c, _p in slots}

    it = list(iter_slots_fast(family, stage))
    if sample and len(it) > sample:
        step = max(1, len(it) // sample)
        it = it[::step][:sample]
    # Collect each class's member vectors, then look for axes they disagree on.
    per_class = {}
    for slot, bv, _live in it:
        per_class.setdefault(by_slot[slot], []).append((slot, bv))

    for c, members in per_class.items():
        if len(members) < 2:
            continue
        axes = set()
        for _s, bv in members:
            axes |= set(bv)
        for a in sorted(axes):
            vals = {int(bv.get(a, const.get(a, 0))) for _s, bv in members}
            if len(vals) < 2:
                continue                       # members agree: pinning is safe
            if a not in carried:
                fails.append("class %d members disagree on %s (%s) and it is not "
                             "in payload_fields" % (c, a, sorted(vals)[:4]))
            else:
                lay = _split_layer_axis(a)
                if lay is None:
                    want = "SC2_PERMDATA_%s=1" % a
                elif lay[1] in DUAL_ROLE_PROPS:
                    want = "SC2_CORE_LAYER_%s=1" % lay[0]
                else:
                    want = "SC2_PERM_PROP_%s=1" % lay[1]
                if want not in defs[c]:
                    fails.append("class %d carries %s in the payload but does not "
                                 "define %s, so the blob cannot read it" % (c, a, want))
            if len(fails) > 20:
                return fails
    return fails

def _kv(define):
    if "=" in define:
        n, v = define.split("=", 1)
        try:
            return n, int(v)
        except ValueError:
            return n, v
    return define, 1


def t4_signatures(families=None):
    """T4 -- the fixed-slot contract: every PS input semantic must be produced by
    the VS class that feeds it.

    With canonical slots this is mostly PROVABLE rather than sampled, which is a
    much better test than spot-checking pairs:

      1. the canonical slot map has no collisions, even after slangc's x10 index
         inflation (cfg.canon_slot_check);
      2. the VS group partition is a COARSENING of the PS one -- every PS group is
         contained in exactly one VS group.  Both stages currently key on the EXACT
         live set (see xport_profile), so a VS class already writes precisely what
         its PS reads; this check keeps the coarsening property true for the day the
         VS profile is widened again;
      3. the two array interpolants still need a size check, because their length is
         a count rather than a group: the VS's `b_iUVEmitterArraySize` must cover the
         PS's `b_iUVEmitterCount` on every slot;
      4. and every interpolant a PS actually reads must be one the transport knows
         how to carry at all.
    """
    fails = []
    try:
        cfg.canon_slot_check()
    except ValueError as e:
        fails.append("canonical slots: %s" % e)

    # (2) coarsening: each PS group inside exactly one VS group.
    for pname, pmem in XPORT_GROUPS.items():
        hosts = [vname for vname, vmem in XPORT_GROUPS_VS.items() if pmem & vmem]
        if len(hosts) != 1 or not pmem <= XPORT_GROUPS_VS[hosts[0]]:
            fails.append("PS group %r is not contained in one VS group (hosts=%s)"
                         % (pname, hosts))

    # (3) + (4) over the corpus.
    known = set(XPORT_BASE) | set(XPORT_OTHER)
    for mem in XPORT_GROUPS.values():
        known |= mem
    for fam, st in _all_stages():
        if families and fam not in families:
            continue
        for slot, bv, live in iter_slots_fast(fam, st):
            unknown = set(live) - known
            if unknown:
                fails.append("%s_%s slot %d reads interpolants the transport does "
                             "not model: %s" % (fam, st, slot, sorted(unknown)))
                break
            if st == "vs" and "UV" in live:
                asz = int(bv.get("b_iUVEmitterArraySize", 0))
                cnt = int(bv.get("b_iUVEmitterCount", 0))
                if asz < cnt:
                    fails.append("%s_vs slot %d: UV array size %d < emitter count %d"
                                 % (fam, slot, asz, cnt))
                    break
    return fails


def t_payload_roundtrip(family, stage, sample=400):
    """Packing must be lossless: every data axis read back out of the packed words
    must equal the retail value.  Catches a width or offset mistake immediately,
    which the failure-triage table calls the first suspect for a single-slot diff."""
    fields = payload_fields(family, stage)
    const = const_axes(family, stage)
    it = list(iter_slots_fast(family, stage))
    if sample and len(it) > sample:
        step = max(1, len(it) // sample)
        it = it[::step][:sample]
    fails = []
    for slot, bv, _live in it:
        words = pack_payload(family, stage, bv)
        for a, off, w in fields:
            got = (words[off // 32] >> (off % 32)) & ((1 << w) - 1)
            want = int(bv.get(a, const.get(a, 0)))
            if got != want:
                fails.append("slot %d: %s packed %d != retail %d (off=%d w=%d)"
                             % (slot, a, got, want, off, w))
                if len(fails) > 20:
                    return fails
    return fails


# ===========================================================================
# 13. CLI
# ===========================================================================
def _all_stages():
    out = []
    for fam in sorted(cfg.load_families()):
        fc = cfg.family_cfg(fam)
        for st in ("vs", "ps"):
            if st in fc:
                out.append((fam, st))
    return out


LAYOUT_HEADER = os.path.join(REPO_ROOT, "sc2_shaders", "types", "perm_layout.slangh")


def emit_layout_header(path=LAYOUT_HEADER):
    """Write the shared material-layer payload layout as a `.slangh`.

    The layer block is family-independent by construction (ALL_LAYERS x
    LAYER_PROPS, widths taken from the engine schema alone), so one header serves
    every family -- and the alternative, one `-D` pair per axis, is ~1,150 flags
    that overflow the Windows command line.  Generated, never hand-edited: the
    packer stays the single source of truth."""
    fields = {a: (o, w) for a, o, w in payload_fields("Model", "ps")}
    nl = nonlayer_layout()
    L = ["// SPDX-License-Identifier: BSD-3-Clause",
         "// Copyright (c) 2026, Fernando Sahmkow",
         "// See LICENSE in the repository root for full terms.",
         "//",
         "// GENERATED by tools/sc2_perm_reduce.py --emit-layout.  Do not edit.",
         "//",
         "// The COMPLETE permutation-payload layout (design section 5.1): one bit",
         "// offset per axis, identical for every family and both stages.",
         "//",
         "// Because the table is global, every SC2_PERMOFF_ exists in every compile",
         "// -- which is what lets one SC2_AXIS macro serve the whole module instead",
         "// of a per-family switch, and what makes the layout independent of which",
         "// axes are switched on.  sc2_shaders is ONE translation unit, so both of",
         "// those matter: a file compiled as dead code still has to preprocess.",
         "//",
         "// Layout:  [ non-layer axes ][ %d layers x %d uint32 ]"
         % (len(ALL_LAYERS), LAYER_WORDS),
         "",
         "#ifndef SC2_PERM_LAYOUT_H",
         "#define SC2_PERM_LAYOUT_H",
         "",
         "// One buffer shape for the whole module.",
         "#ifndef SC2_PERM_CB_VEC4S",
         "#define SC2_PERM_CB_VEC4S %d" % (payload_words() // 4),
         "#endif",
         ""]
    n = 0
    L.append("// ==== non-layer axes ====")
    L.append("//")
    L.append("// SC2_PERMDATA_<axis> is 0 unless the CLASS says this axis is carried")
    L.append("// by the payload.  It is a per-class value, not a per-family one: an")
    L.append("// axis can be data in a family's VS and interface-bearing in its PS")
    L.append("// (b_useShadows binds a shadow map, so the pixel stage pins it).")
    for a in sorted(nl):
        off, w = nl[a]
        L.append("#define SC2_PERMOFF_%s %d" % (a, off))
        L.append("#define SC2_PERMW_%s %d" % (a, w))
        L.append("#ifndef SC2_PERMDATA_%s" % a)
        L.append("#define SC2_PERMDATA_%s 0" % a)
        L.append("#endif")
        n += 1
    L.append("")
    L.append("// ==== material layers ====")
    for layer in ALL_LAYERS:
        emitted = False
        for prop in LAYER_PROPS:
            a = _layer_axis(layer, prop)
            if a not in fields:
                continue
            if not emitted:
                L.append("// ---- %s" % layer)
                emitted = True
            off, w = fields[a]
            L.append("#define SC2_PERMOFF_%s %d" % (a, off))
            L.append("#define SC2_PERMW_%s %d" % (a, w))
            n += 1
        if emitted:
            L.append("")
    L.append("#endif // SC2_PERM_LAYOUT_H")
    io.open(path, "w", encoding="utf-8",
            newline=chr(10)).write(chr(10).join(L) + chr(10))
    return n


SHADER_ROOT = os.path.join(REPO_ROOT, "sc2_shaders")


def _shader_sources():
    out = []
    for dp, _dn, fn in os.walk(SHADER_ROOT):
        for f in sorted(fn):
            if f.endswith((".slang", ".slangh")):
                p = os.path.join(dp, f)
                out.append((p, io.open(p, encoding="utf-8",
                                       errors="replace").read().split("\n")))
    return out


def preproc_sites(axis, sources=None):
    """Lines where `axis` is still read by the PREPROCESSOR.

    An axis may only be switched on once every one of these is gone.  With even one
    left, the class compiles with the `#define` at its default of 0 -- and because
    the convention passes `-D` only for nonzero axes, that arm silently vanishes for
    every member.  The diff is ~1.0, not a ULP, which is exactly how this check came
    to exist."""
    pat = re.compile(r"^\s*#\s*(?:if|elif)\b.*(?<![A-Za-z0-9_])%s(?![A-Za-z0-9_])"
                     % re.escape(axis))
    hits = []
    for path, lines in (sources if sources is not None else _shader_sources()):
        for i, ln in enumerate(lines, 1):
            if pat.match(ln):
                hits.append("%s:%d" % (os.path.relpath(path, REPO_ROOT), i))
    return hits


def cmd_check_moved():
    """Gate for the moved sets, plus the list of axes ready to join them."""
    src = _shader_sources()
    per_stage = [(p, ls, file_stages(p)) for p, ls in src]

    def sites(axis, stage):
        """Read sites that MATTER for `stage`: a `#if` in the other stage's files is
        dead code for this entry point, so it blocks that stage and not this one."""
        return preproc_sites(axis, [(p, ls) for p, ls, st in per_stage if stage in st])

    bad = {}
    for a in sorted(MOVED_NONLAYER_AXES):
        h = preproc_sites(a, src)
        if h:
            bad[a] = h
    for stage, axes in sorted(MOVED_BY_STAGE.items()):
        for a in sorted(axes):
            h = sites(a, stage)
            if h:
                bad["%s (%s)" % (a, stage)] = h
    print("moved axes with a surviving #if (each one is a silent-zero bug):")
    for a, h in bad.items():
        print("  %-38s %s" % (a, " ".join(h[:4])))
    if not bad:
        print("  none -- every moved axis is read only as code in its own stage")

    tmpl = sorted(TEMPLATE_ARG_AXES & set(MOVED_NONLAYER_AXES))
    if tmpl:
        print("\nTEMPLATE-ARG axes wrongly moved (no #if to find, still broken): %s"
              % " ".join(tmpl))
        bad["<template>"] = tmpl

    ready, vs_only, ps_only, blocked = [], [], [], {}
    for a in sorted(convertible_nonlayer_axes()):
        if a in MOVED_NONLAYER_AXES or a in TEMPLATE_ARG_AXES:
            continue
        h = preproc_sites(a, src)
        if not h:
            ready.append(a)
            continue
        hv, hp = sites(a, "vs"), sites(a, "ps")
        if not hv and a not in MOVED_BY_STAGE["vs"]:
            vs_only.append(a)
        elif not hp and a not in MOVED_BY_STAGE["ps"]:
            ps_only.append(a)
        elif hv and hp:
            blocked[a] = h
    print("\nREADY for both stages (%d): %s" % (len(ready), " ".join(ready)))
    print("\nREADY for VS only (%d): %s" % (len(vs_only), " ".join(vs_only)))
    print("\nREADY for PS only (%d): %s" % (len(ps_only), " ".join(ps_only)))
    print("\nstill compile-time in both stages (%d):" % len(blocked))
    for a, h in sorted(blocked.items()):
        print("  %-38s %s" % (a, " ".join(h[:3])))
    return 1 if bad else 0


def cmd_axes(family, stage):
    varying = inventory()["%s_%s" % (family, stage)]["varying"]
    key, payload, dual, dead = classify(family, stage, varying)
    core = core_layers(family, stage)
    print("%s_%s: %d varying axes" % (family, stage, len(varying)))
    print("  KEY     (%3d): %s" % (len(key), " ".join(key)))
    print()
    print("  PAYLOAD (%3d): %s" % (len(payload), " ".join(payload)))
    print()
    print("  DUAL    (%3d): %s" % (len(dual), " ".join(dual)))
    print("  DEAD    (%3d): %s" % (len(dead), " ".join(dead)))
    print("  layers      : %s" % " ".join(family_layers(family, stage)))
    print("  core        : %s" % " ".join(sorted(core)))
    print("  payload     : %d words (%d bytes)"
          % (payload_words(family, stage), 4 * payload_words(family, stage)))


def key_parts(family, stage, bv, live, core, iface, drop=()):
    """The key as a dict of named components, so any one can be knocked out.

    `structural_key` is the tuple form of exactly this."""
    fam_layers = family_layers(family, stage)
    parts = {}
    parts["iface"] = tuple((a, int(bv.get(a, 0))) for a in iface if a not in drop)
    parts["sampled"] = tuple((L,
                              int(bv.get(_layer_axis(L, "LayerEnable"), 0)),
                              int(bv.get(_layer_axis(L, "UseConstantColor"), 0)))
                             for L in fam_layers if L not in core)
    parts["textype"] = tuple((L, int(bv.get(_layer_axis(L, "TextureType"), 0)))
                             for L in fam_layers
                             if int(bv.get(_layer_axis(L, "TextureType"), 0)))
    tl = tuple(sorted(L for L in fam_layers
                      if int(bv.get(_layer_axis(L, "UVMapping"), 0))
                      in TRIPLANAR_UVMAPPINGS))
    parts["tri"] = tuple((L,
                          int(bv.get(_layer_axis(L, "UVMapping"), 0)),
                          int(bv.get(_layer_axis(L, "LayerId"), 0)),
                          int(bv.get(_layer_axis(L, "TriPlanarUvId"), 0)))
                         for L in tl)
    parts["layerprops"] = tuple(
        (_layer_axis(L, p), int(bv.get(_layer_axis(L, p), 0)))
        for L in fam_layers for p in LAYER_PROPS
        if p not in ENABLED_LAYER_PROPS and p not in LAYER_IFACE_PROPS
        and int(bv.get(_layer_axis(L, p), 0)))
    parts["xport"] = xport_profile(live, stage)
    for d in drop:
        parts.pop(d, None)
    return tuple(sorted(parts.items()))


def cmd_knockout(family, stage):
    """How much does each key component / each interface axis actually cost?

    The plan's failure-triage says to bisect a surprise by single-axis knockout
    rather than by assuming; this is that bisection, run over class COUNTS.  It is
    also how an axis that does not belong in the key gets caught: if removing it
    changes nothing, it was never structural."""
    core = core_layers(family, stage)
    iface = _iface_axes_for(family, stage)
    rows = list(iter_slots_fast(family, stage))
    base = len({key_parts(family, stage, bv, live, core, iface) for _s, bv, live in rows})
    print("%s_%s: %d slots -> %d classes  (design %s)"
          % (family, stage, len(rows), base, T_COUNT.get("%s_%s" % (family, stage))))
    print("  iface axes (%d): %s" % (len(iface), " ".join(iface)))
    print("")
    print("  component knockout")
    for comp in ("iface", "sampled", "textype", "tri", "xport", "layerprops"):
        n = len({key_parts(family, stage, bv, live, core, iface, drop=(comp,))
                 for _s, bv, live in rows})
        print("    without %-9s -> %5d  (costs %.2fx)" % (comp, n, base / max(1, n)))
    print("")
    print("  single-axis knockout")
    res = []
    for a in iface:
        n = len({key_parts(family, stage, bv, live, core, iface, drop=(a,))
                 for _s, bv, live in rows})
        res.append((base / max(1, n), n, a))
    for f, n, a in sorted(res, reverse=True):
        if f > 1.0005:
            print("    without %-36s -> %5d  (%.3fx)" % (a, n, f))
    zero = [a for f, _n, a in res if f <= 1.0005]
    if zero:
        print("    free (no class cost): %s" % " ".join(zero))
    return 0


def cmd_check(args):
    stages = ([(args.family[0], args.stage)] if args.family and args.stage
              else _all_stages())
    ok = True
    print("T1 totality + T3 consistency/delivery + payload round-trip + T-count + T4")
    f4 = t4_signatures()
    print("  T4 signatures (canonical slots; VS output covers PS input): %s"
          % ("ok" if not f4 else "FAIL"))
    for m in f4[:8]:
        print("      T4: %s" % m)
    ok = ok and not f4
    print("")
    tot_slots = tot_classes = 0
    for fam, st in stages:
        name = "%s_%s" % (fam, st)
        try:
            f1 = t1_totality(fam, st)
            classes, slots = build_classes(fam, st)
            f3 = t3_class_consistency(fam, st, classes, slots, sample=args.sample)
            f3 += t3b_class_delivery(fam, st, classes, slots, sample=args.sample)
            fp = t_payload_roundtrip(fam, st, sample=args.sample or 400)
        except Exception as e:            # one broken stage must not hide the rest
            print("  %-20s EXCEPTION: %s" % (name, e))
            ok = False
            continue
        tot_slots += len(slots)
        tot_classes += len(classes)
        want = T_COUNT.get(name)
        cnt_ok = (want is None or want == len(classes))
        status = "ok" if (not f1 and not f3 and not fp and cnt_ok) else "FAIL"
        print("  %-20s %6d -> %5d classes  %-4s %s"
              % (name, len(slots), len(classes), status,
                 "" if want is None else ("(design %d%s)"
                                          % (want, "" if cnt_ok else "  MISMATCH"))))
        for m in f1[:6]:
            print("      T1: %s" % m)
        for m in f3[:6]:
            print("      T3: %s" % m)
        for m in fp[:6]:
            print("      PACK: %s" % m)
        ok = ok and status == "ok"
    print("\n  TOTAL %d retail slots -> %d classes (%.1fx)"
          % (tot_slots, tot_classes, tot_slots / max(1, tot_classes)))
    return 0 if ok else 1


def cmd_report(args):
    rows = []
    for fam, st in _all_stages():
        classes, slots = build_classes(fam, st)
        rows.append((("%s_%s" % (fam, st)), len(slots), len(classes)))
    rows.sort(key=lambda r: -r[1])
    print("%-22s %8s %8s %7s" % ("stage", "retail", "reduced", "ratio"))
    ts = tc = 0
    for n, s, c in rows:
        ts += s
        tc += c
        print("%-22s %8d %8d %6.1fx" % (n, s, c, s / max(1, c)))
    print("%-22s %8d %8d %6.1fx" % ("TOTAL", ts, tc, ts / max(1, tc)))
    return 0


def cmd_build(args):
    stages = ([(f, s) for f in args.family for s in (["vs", "ps"] if not args.stage
                                                     else [args.stage])
               if s in cfg.family_cfg(f)] if args.family else _all_stages())
    for fam, st in stages:
        classes, slots = build_classes(fam, st, verbose=True)
        print("    -> %s" % write_tables(fam, st, classes, slots))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check-all", action="store_true", help="T1 + T3 + T-count")
    ap.add_argument("--report", action="store_true", help="the reduction table")
    ap.add_argument("--build", action="store_true", help="write the class tables")
    ap.add_argument("--axes", nargs=2, metavar=("FAMILY", "STAGE"),
                    help="explain one stage's key/payload split")
    ap.add_argument("--knockout", nargs=2, metavar=("FAMILY", "STAGE"),
                    help="cost of each key component / interface axis, by class count")
    ap.add_argument("--rebuild-inventory", action="store_true")
    ap.add_argument("--emit-layout", action="store_true",
                    help="regenerate sc2_shaders/types/perm_layout.slangh")
    ap.add_argument("--check-moved", action="store_true",
                    help="every MOVED_NONLAYER_AXES entry must have no #if left")
    ap.add_argument("--probe-irrelevant", action="store_true",
                    help="measure which axes leave a family's DXBC byte-identical")
    ap.add_argument("--family", nargs="*")
    ap.add_argument("--stage", choices=["ps", "vs"])
    ap.add_argument("--sample", type=int, default=0,
                    help="limit T3 / round-trip to ~N slots per stage")
    args = ap.parse_args(argv)

    if args.rebuild_inventory:
        inventory(rebuild=True)
    if args.probe_irrelevant:
        out = {}
        if os.path.exists(IRRELEVANT_CACHE):
            out = json.load(open(IRRELEVANT_CACHE))
        stages = ([(f, s) for f in args.family for s in (["vs", "ps"] if not args.stage
                                                         else [args.stage])
                   if s in cfg.family_cfg(f)] if args.family else _all_stages())
        for fam, st in stages:
            out["%s_%s" % (fam, st)] = sorted(
                probe_irrelevant(fam, st, jobs=8, verbose=True))
        os.makedirs(os.path.dirname(IRRELEVANT_CACHE), exist_ok=True)
        json.dump(out, open(IRRELEVANT_CACHE, "w"), indent=1)
        print("wrote %s" % IRRELEVANT_CACHE)
        return 0
    if args.emit_layout:
        print("wrote %s (%d axes)" % (LAYOUT_HEADER, emit_layout_header()))
        return 0
    if args.check_moved:
        return cmd_check_moved()
    if args.axes:
        cmd_axes(args.axes[0], args.axes[1])
        return 0
    if args.knockout:
        return cmd_knockout(args.knockout[0], args.knockout[1])
    if args.report:
        return cmd_report(args)
    if args.build:
        return cmd_build(args)
    if args.check_all:
        return cmd_check(args)
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
