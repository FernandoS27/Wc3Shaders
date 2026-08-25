// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026, Fernando Sahmkow
// See LICENSE in the repository root for full terms.
//
// Model-family material-layer setup for the layers beyond Diffuse/Specular
// (psmaterial.fx SETUP_LAYER(<layer>)).  Generated to mirror the .fx SETUP_LAYER
// macro field-for-field (tools scratchpad gen_layers.py); each setup<Layer>()
// builds the layer's SLayerConfig from its `b_i<Layer>*` axes + `<layer>` cbuffer
// constants, exactly like the hand-written setupDiffuseLayer/setupSpecularLayer.
//
// Every axis is #defined 0 when the engine didn't set it (the driver passes only
// nonzero defines), so the setup C-expressions always resolve.

implementing sc2_shaders;

// SC2_LAYER_<Prop>: a material-layer property reads from the permutation
// constant buffer once its switch is on, and from the `#define` until then.
#include "ps/model_layers_axes.slangh"

// ---- axis defaults (0 = engine default) ------------------------------------

#ifndef b_iDiffuseLayerId
#define b_iDiffuseLayerId 0
#endif
#ifndef b_iDiffuseTextureType
#define b_iDiffuseTextureType 0
#endif
#ifndef b_iDiffuseTriPlanarUvId
#define b_iDiffuseTriPlanarUvId 0
#endif
#ifndef b_iSpecularLayerId
#define b_iSpecularLayerId 0
#endif
#ifndef b_iSpecularTextureType
#define b_iSpecularTextureType 0
#endif
#ifndef b_iSpecularTriPlanarUvId
#define b_iSpecularTriPlanarUvId 0
#endif
#ifndef b_iDecalLayerEnable
#define b_iDecalLayerEnable 0
#endif
#ifndef b_iDecalLayerId
#define b_iDecalLayerId 0
#endif
#ifndef b_iDecalTextureType
#define b_iDecalTextureType 0
#endif
#ifndef b_iDecalUVMapping
#define b_iDecalUVMapping 0
#endif
#ifndef b_iDecalTeamColorMode
#define b_iDecalTeamColorMode 0
#endif
#ifndef b_iDecalUVEmitter
#define b_iDecalUVEmitter 0
#endif
#ifndef b_iDecalUseMask
#define b_iDecalUseMask 0
#endif
#ifndef b_iDecalChannelSelect
#define b_iDecalChannelSelect 0
#endif
#ifndef b_iDecalInvert
#define b_iDecalInvert 0
#endif
#ifndef b_iDecalClamp
#define b_iDecalClamp 0
#endif
#ifndef b_iDecalFresnelMode
#define b_iDecalFresnelMode 0
#endif
#ifndef b_iDecalFresnelTransformMode
#define b_iDecalFresnelTransformMode 0
#endif
#ifndef b_iDecalFresnelSaturate
#define b_iDecalFresnelSaturate 0
#endif
#ifndef b_iDecalUseConstantColor
#define b_iDecalUseConstantColor 0
#endif
#ifndef b_iDecalOp
#define b_iDecalOp 0
#endif
#ifndef b_iDecalIsRTTTexture
#define b_iDecalIsRTTTexture 0
#endif
#ifndef b_iDecalTriPlanarUvId
#define b_iDecalTriPlanarUvId 0
#endif
#ifndef b_iEmissiveLayerEnable
#define b_iEmissiveLayerEnable 0
#endif
#ifndef b_iEmissiveLayerId
#define b_iEmissiveLayerId 0
#endif
#ifndef b_iEmissiveTextureType
#define b_iEmissiveTextureType 0
#endif
#ifndef b_iEmissiveUVMapping
#define b_iEmissiveUVMapping 0
#endif
#ifndef b_iEmissiveTeamColorMode
#define b_iEmissiveTeamColorMode 0
#endif
#ifndef b_iEmissiveUVEmitter
#define b_iEmissiveUVEmitter 0
#endif
#ifndef b_iEmissiveUseMask
#define b_iEmissiveUseMask 0
#endif
#ifndef b_iEmissiveChannelSelect
#define b_iEmissiveChannelSelect 0
#endif
#ifndef b_iEmissiveInvert
#define b_iEmissiveInvert 0
#endif
#ifndef b_iEmissiveClamp
#define b_iEmissiveClamp 0
#endif
#ifndef b_iEmissiveFresnelMode
#define b_iEmissiveFresnelMode 0
#endif
#ifndef b_iEmissiveFresnelTransformMode
#define b_iEmissiveFresnelTransformMode 0
#endif
#ifndef b_iEmissiveFresnelSaturate
#define b_iEmissiveFresnelSaturate 0
#endif
#ifndef b_iEmissiveUseConstantColor
#define b_iEmissiveUseConstantColor 0
#endif
#ifndef b_iEmissiveOp
#define b_iEmissiveOp 0
#endif
#ifndef b_iEmissiveIsRTTTexture
#define b_iEmissiveIsRTTTexture 0
#endif
#ifndef b_iEmissiveTriPlanarUvId
#define b_iEmissiveTriPlanarUvId 0
#endif
#ifndef b_iEmissive2LayerEnable
#define b_iEmissive2LayerEnable 0
#endif
#ifndef b_iEmissive2LayerId
#define b_iEmissive2LayerId 0
#endif
#ifndef b_iEmissive2TextureType
#define b_iEmissive2TextureType 0
#endif
#ifndef b_iEmissive2UVMapping
#define b_iEmissive2UVMapping 0
#endif
#ifndef b_iEmissive2TeamColorMode
#define b_iEmissive2TeamColorMode 0
#endif
#ifndef b_iEmissive2UVEmitter
#define b_iEmissive2UVEmitter 0
#endif
#ifndef b_iEmissive2UseMask
#define b_iEmissive2UseMask 0
#endif
#ifndef b_iEmissive2ChannelSelect
#define b_iEmissive2ChannelSelect 0
#endif
#ifndef b_iEmissive2Invert
#define b_iEmissive2Invert 0
#endif
#ifndef b_iEmissive2Clamp
#define b_iEmissive2Clamp 0
#endif
#ifndef b_iEmissive2FresnelMode
#define b_iEmissive2FresnelMode 0
#endif
#ifndef b_iEmissive2FresnelTransformMode
#define b_iEmissive2FresnelTransformMode 0
#endif
#ifndef b_iEmissive2FresnelSaturate
#define b_iEmissive2FresnelSaturate 0
#endif
#ifndef b_iEmissive2UseConstantColor
#define b_iEmissive2UseConstantColor 0
#endif
#ifndef b_iEmissive2Op
#define b_iEmissive2Op 0
#endif
#ifndef b_iEmissive2IsRTTTexture
#define b_iEmissive2IsRTTTexture 0
#endif
#ifndef b_iEmissive2TriPlanarUvId
#define b_iEmissive2TriPlanarUvId 0
#endif
#ifndef b_iAlphaMaskLayerEnable
#define b_iAlphaMaskLayerEnable 0
#endif
#ifndef b_iAlphaMaskLayerId
#define b_iAlphaMaskLayerId 0
#endif
#ifndef b_iAlphaMaskTextureType
#define b_iAlphaMaskTextureType 0
#endif
#ifndef b_iAlphaMaskUVMapping
#define b_iAlphaMaskUVMapping 0
#endif
#ifndef b_iAlphaMaskTeamColorMode
#define b_iAlphaMaskTeamColorMode 0
#endif
#ifndef b_iAlphaMaskUVEmitter
#define b_iAlphaMaskUVEmitter 0
#endif
#ifndef b_iAlphaMaskUseMask
#define b_iAlphaMaskUseMask 0
#endif
#ifndef b_iAlphaMaskChannelSelect
#define b_iAlphaMaskChannelSelect 0
#endif
#ifndef b_iAlphaMaskInvert
#define b_iAlphaMaskInvert 0
#endif
#ifndef b_iAlphaMaskClamp
#define b_iAlphaMaskClamp 0
#endif
#ifndef b_iAlphaMaskFresnelMode
#define b_iAlphaMaskFresnelMode 0
#endif
#ifndef b_iAlphaMaskFresnelTransformMode
#define b_iAlphaMaskFresnelTransformMode 0
#endif
#ifndef b_iAlphaMaskFresnelSaturate
#define b_iAlphaMaskFresnelSaturate 0
#endif
#ifndef b_iAlphaMaskUseConstantColor
#define b_iAlphaMaskUseConstantColor 0
#endif
#ifndef b_iAlphaMaskOp
#define b_iAlphaMaskOp 0
#endif
#ifndef b_iAlphaMaskIsRTTTexture
#define b_iAlphaMaskIsRTTTexture 0
#endif
#ifndef b_iAlphaMaskTriPlanarUvId
#define b_iAlphaMaskTriPlanarUvId 0
#endif
#ifndef b_iAlphaMask2LayerEnable
#define b_iAlphaMask2LayerEnable 0
#endif
#ifndef b_iAlphaMask2LayerId
#define b_iAlphaMask2LayerId 0
#endif
#ifndef b_iAlphaMask2TextureType
#define b_iAlphaMask2TextureType 0
#endif
#ifndef b_iAlphaMask2UVMapping
#define b_iAlphaMask2UVMapping 0
#endif
#ifndef b_iAlphaMask2TeamColorMode
#define b_iAlphaMask2TeamColorMode 0
#endif
#ifndef b_iAlphaMask2UVEmitter
#define b_iAlphaMask2UVEmitter 0
#endif
#ifndef b_iAlphaMask2UseMask
#define b_iAlphaMask2UseMask 0
#endif
#ifndef b_iAlphaMask2ChannelSelect
#define b_iAlphaMask2ChannelSelect 0
#endif
#ifndef b_iAlphaMask2Invert
#define b_iAlphaMask2Invert 0
#endif
#ifndef b_iAlphaMask2Clamp
#define b_iAlphaMask2Clamp 0
#endif
#ifndef b_iAlphaMask2FresnelMode
#define b_iAlphaMask2FresnelMode 0
#endif
#ifndef b_iAlphaMask2FresnelTransformMode
#define b_iAlphaMask2FresnelTransformMode 0
#endif
#ifndef b_iAlphaMask2FresnelSaturate
#define b_iAlphaMask2FresnelSaturate 0
#endif
#ifndef b_iAlphaMask2UseConstantColor
#define b_iAlphaMask2UseConstantColor 0
#endif
#ifndef b_iAlphaMask2Op
#define b_iAlphaMask2Op 0
#endif
#ifndef b_iAlphaMask2IsRTTTexture
#define b_iAlphaMask2IsRTTTexture 0
#endif
#ifndef b_iAlphaMask2TriPlanarUvId
#define b_iAlphaMask2TriPlanarUvId 0
#endif
#ifndef b_iLightmapLayerEnable
#define b_iLightmapLayerEnable 0
#endif
#ifndef b_iLightmapLayerId
#define b_iLightmapLayerId 0
#endif
#ifndef b_iLightmapTextureType
#define b_iLightmapTextureType 0
#endif
#ifndef b_iLightmapUVMapping
#define b_iLightmapUVMapping 0
#endif
#ifndef b_iLightmapTeamColorMode
#define b_iLightmapTeamColorMode 0
#endif
#ifndef b_iLightmapUVEmitter
#define b_iLightmapUVEmitter 0
#endif
#ifndef b_iLightmapUseMask
#define b_iLightmapUseMask 0
#endif
#ifndef b_iLightmapChannelSelect
#define b_iLightmapChannelSelect 0
#endif
#ifndef b_iLightmapInvert
#define b_iLightmapInvert 0
#endif
#ifndef b_iLightmapClamp
#define b_iLightmapClamp 0
#endif
#ifndef b_iLightmapFresnelMode
#define b_iLightmapFresnelMode 0
#endif
#ifndef b_iLightmapFresnelTransformMode
#define b_iLightmapFresnelTransformMode 0
#endif
#ifndef b_iLightmapFresnelSaturate
#define b_iLightmapFresnelSaturate 0
#endif
#ifndef b_iLightmapUseConstantColor
#define b_iLightmapUseConstantColor 0
#endif
#ifndef b_iLightmapOp
#define b_iLightmapOp 0
#endif
#ifndef b_iLightmapIsRTTTexture
#define b_iLightmapIsRTTTexture 0
#endif
#ifndef b_iLightmapTriPlanarUvId
#define b_iLightmapTriPlanarUvId 0
#endif
#ifndef b_iAmbientOcclusionLayerEnable
#define b_iAmbientOcclusionLayerEnable 0
#endif
#ifndef b_iAmbientOcclusionLayerId
#define b_iAmbientOcclusionLayerId 0
#endif
#ifndef b_iAmbientOcclusionTextureType
#define b_iAmbientOcclusionTextureType 0
#endif
#ifndef b_iAmbientOcclusionUVMapping
#define b_iAmbientOcclusionUVMapping 0
#endif
#ifndef b_iAmbientOcclusionTeamColorMode
#define b_iAmbientOcclusionTeamColorMode 0
#endif
#ifndef b_iAmbientOcclusionUVEmitter
#define b_iAmbientOcclusionUVEmitter 0
#endif
#ifndef b_iAmbientOcclusionUseMask
#define b_iAmbientOcclusionUseMask 0
#endif
#ifndef b_iAmbientOcclusionChannelSelect
#define b_iAmbientOcclusionChannelSelect 0
#endif
#ifndef b_iAmbientOcclusionInvert
#define b_iAmbientOcclusionInvert 0
#endif
#ifndef b_iAmbientOcclusionClamp
#define b_iAmbientOcclusionClamp 0
#endif
#ifndef b_iAmbientOcclusionFresnelMode
#define b_iAmbientOcclusionFresnelMode 0
#endif
#ifndef b_iAmbientOcclusionFresnelTransformMode
#define b_iAmbientOcclusionFresnelTransformMode 0
#endif
#ifndef b_iAmbientOcclusionFresnelSaturate
#define b_iAmbientOcclusionFresnelSaturate 0
#endif
#ifndef b_iAmbientOcclusionUseConstantColor
#define b_iAmbientOcclusionUseConstantColor 0
#endif
#ifndef b_iAmbientOcclusionOp
#define b_iAmbientOcclusionOp 0
#endif
#ifndef b_iAmbientOcclusionIsRTTTexture
#define b_iAmbientOcclusionIsRTTTexture 0
#endif
#ifndef b_iAmbientOcclusionTriPlanarUvId
#define b_iAmbientOcclusionTriPlanarUvId 0
#endif
#ifndef b_iSpecularExponentLayerEnable
#define b_iSpecularExponentLayerEnable 0
#endif
#ifndef b_iSpecularExponentLayerId
#define b_iSpecularExponentLayerId 0
#endif
#ifndef b_iSpecularExponentTextureType
#define b_iSpecularExponentTextureType 0
#endif
#ifndef b_iSpecularExponentUVMapping
#define b_iSpecularExponentUVMapping 0
#endif
#ifndef b_iSpecularExponentTeamColorMode
#define b_iSpecularExponentTeamColorMode 0
#endif
#ifndef b_iSpecularExponentUVEmitter
#define b_iSpecularExponentUVEmitter 0
#endif
#ifndef b_iSpecularExponentUseMask
#define b_iSpecularExponentUseMask 0
#endif
#ifndef b_iSpecularExponentChannelSelect
#define b_iSpecularExponentChannelSelect 0
#endif
#ifndef b_iSpecularExponentInvert
#define b_iSpecularExponentInvert 0
#endif
#ifndef b_iSpecularExponentClamp
#define b_iSpecularExponentClamp 0
#endif
#ifndef b_iSpecularExponentFresnelMode
#define b_iSpecularExponentFresnelMode 0
#endif
#ifndef b_iSpecularExponentFresnelTransformMode
#define b_iSpecularExponentFresnelTransformMode 0
#endif
#ifndef b_iSpecularExponentFresnelSaturate
#define b_iSpecularExponentFresnelSaturate 0
#endif
#ifndef b_iSpecularExponentUseConstantColor
#define b_iSpecularExponentUseConstantColor 0
#endif
#ifndef b_iSpecularExponentOp
#define b_iSpecularExponentOp 0
#endif
#ifndef b_iSpecularExponentIsRTTTexture
#define b_iSpecularExponentIsRTTTexture 0
#endif
#ifndef b_iSpecularExponentTriPlanarUvId
#define b_iSpecularExponentTriPlanarUvId 0
#endif
#ifndef b_iNormalLayerEnable
#define b_iNormalLayerEnable 0
#endif
#ifndef b_iNormalLayerId
#define b_iNormalLayerId 0
#endif
#ifndef b_iNormalTextureType
#define b_iNormalTextureType 0
#endif
#ifndef b_iNormalUVMapping
#define b_iNormalUVMapping 0
#endif
#ifndef b_iNormalTeamColorMode
#define b_iNormalTeamColorMode 0
#endif
#ifndef b_iNormalUVEmitter
#define b_iNormalUVEmitter 0
#endif
#ifndef b_iNormalUseMask
#define b_iNormalUseMask 0
#endif
#ifndef b_iNormalChannelSelect
#define b_iNormalChannelSelect 0
#endif
#ifndef b_iNormalInvert
#define b_iNormalInvert 0
#endif
#ifndef b_iNormalClamp
#define b_iNormalClamp 0
#endif
#ifndef b_iNormalFresnelMode
#define b_iNormalFresnelMode 0
#endif
#ifndef b_iNormalFresnelTransformMode
#define b_iNormalFresnelTransformMode 0
#endif
#ifndef b_iNormalFresnelSaturate
#define b_iNormalFresnelSaturate 0
#endif
#ifndef b_iNormalUseConstantColor
#define b_iNormalUseConstantColor 0
#endif
#ifndef b_iNormalOp
#define b_iNormalOp 0
#endif
#ifndef b_iNormalIsRTTTexture
#define b_iNormalIsRTTTexture 0
#endif
#ifndef b_iNormalTriPlanarUvId
#define b_iNormalTriPlanarUvId 0
#endif
#ifndef b_iEnvioLayerEnable
#define b_iEnvioLayerEnable 0
#endif
#ifndef b_iEnvioLayerId
#define b_iEnvioLayerId 0
#endif
#ifndef b_iEnvioTextureType
#define b_iEnvioTextureType 0
#endif
#ifndef b_iEnvioUVMapping
#define b_iEnvioUVMapping 0
#endif
#ifndef b_iEnvioTeamColorMode
#define b_iEnvioTeamColorMode 0
#endif
#ifndef b_iEnvioUVEmitter
#define b_iEnvioUVEmitter 0
#endif
#ifndef b_iEnvioUseMask
#define b_iEnvioUseMask 0
#endif
#ifndef b_iEnvioChannelSelect
#define b_iEnvioChannelSelect 0
#endif
#ifndef b_iEnvioInvert
#define b_iEnvioInvert 0
#endif
#ifndef b_iEnvioClamp
#define b_iEnvioClamp 0
#endif
#ifndef b_iEnvioFresnelMode
#define b_iEnvioFresnelMode 0
#endif
#ifndef b_iEnvioFresnelTransformMode
#define b_iEnvioFresnelTransformMode 0
#endif
#ifndef b_iEnvioFresnelSaturate
#define b_iEnvioFresnelSaturate 0
#endif
#ifndef b_iEnvioUseConstantColor
#define b_iEnvioUseConstantColor 0
#endif
#ifndef b_iEnvioOp
#define b_iEnvioOp 0
#endif
#ifndef b_iEnvioIsRTTTexture
#define b_iEnvioIsRTTTexture 0
#endif
#ifndef b_iEnvioTriPlanarUvId
#define b_iEnvioTriPlanarUvId 0
#endif
#ifndef b_iEnvioMaskLayerEnable
#define b_iEnvioMaskLayerEnable 0
#endif
#ifndef b_iEnvioMaskLayerId
#define b_iEnvioMaskLayerId 0
#endif
#ifndef b_iEnvioMaskTextureType
#define b_iEnvioMaskTextureType 0
#endif
#ifndef b_iEnvioMaskUVMapping
#define b_iEnvioMaskUVMapping 0
#endif
#ifndef b_iEnvioMaskTeamColorMode
#define b_iEnvioMaskTeamColorMode 0
#endif
#ifndef b_iEnvioMaskUVEmitter
#define b_iEnvioMaskUVEmitter 0
#endif
#ifndef b_iEnvioMaskUseMask
#define b_iEnvioMaskUseMask 0
#endif
#ifndef b_iEnvioMaskChannelSelect
#define b_iEnvioMaskChannelSelect 0
#endif
#ifndef b_iEnvioMaskInvert
#define b_iEnvioMaskInvert 0
#endif
#ifndef b_iEnvioMaskClamp
#define b_iEnvioMaskClamp 0
#endif
#ifndef b_iEnvioMaskFresnelMode
#define b_iEnvioMaskFresnelMode 0
#endif
#ifndef b_iEnvioMaskFresnelTransformMode
#define b_iEnvioMaskFresnelTransformMode 0
#endif
#ifndef b_iEnvioMaskFresnelSaturate
#define b_iEnvioMaskFresnelSaturate 0
#endif
#ifndef b_iEnvioMaskUseConstantColor
#define b_iEnvioMaskUseConstantColor 0
#endif
#ifndef b_iEnvioMaskOp
#define b_iEnvioMaskOp 0
#endif
#ifndef b_iEnvioMaskIsRTTTexture
#define b_iEnvioMaskIsRTTTexture 0
#endif
#ifndef b_iEnvioMaskTriPlanarUvId
#define b_iEnvioMaskTriPlanarUvId 0
#endif

// ---- per-layer SLayerConfig builders ----------------------------------------
SLayerConfig setupDecalLayer()
{
    SLayerConfig c;
    c.b_iLayerEnable           = SC2_CORE_ENABLE(Decal, b_iDecalLayerEnable);
    c.b_iLayerId               = SC2_LAYER_LayerId(b_iDecalLayerId);
    c.b_iTextureType           = b_iDecalTextureType;
    c.b_iUVMapping             = SC2_LAYER_UVMapping(b_iDecalUVMapping);
    c.b_iTeamColorMode         = SC2_LAYER_TeamColorMode(b_iDecalTeamColorMode);
    c.b_iUVEmitter             = SC2_LAYER_UVEmitter(b_iDecalUVEmitter);
    c.b_iUseMask               = SC2_LAYER_UseMask(b_iDecalUseMask);
    c.b_iChannelSelect         = SC2_LAYER_ChannelSelect(b_iDecalChannelSelect);
    c.b_iInvert                = SC2_LAYER_Invert(b_iDecalInvert);
    c.b_iClamp                 = SC2_LAYER_Clamp(b_iDecalClamp);
    c.b_iFresnelMode           = SC2_LAYER_FresnelMode(b_iDecalFresnelMode);
    c.b_iFresnelTransformMode  = SC2_LAYER_FresnelTransformMode(b_iDecalFresnelTransformMode);
    c.b_iFresnelSaturate       = SC2_LAYER_FresnelSaturate(b_iDecalFresnelSaturate);
    c.b_iUseConstantColor      = SC2_CORE_ENABLE(Decal, b_iDecalUseConstantColor);
    c.b_iOp                    = SC2_LAYER_Op(b_iDecalOp);
    c.b_iIsRTTTexture          = SC2_LAYER_IsRTTTexture(b_iDecalIsRTTTexture);
    c.b_iTriPlanarUvId         = SC2_LAYER_TriPlanarUvId(b_iDecalTriPlanarUvId);
    c.p_vMultiplyAddAlphaTrans = psGlobals.vDecalMultiplyAddAlphaTrans;
    c.p_vAtlasParams           = psGlobals.vDecalAtlasParams;
    c.p_vFresnelExponentBiasScale = psGlobals.vDecalFresnelExponentBiasScale;
    c.p_cConstantColor         = psGlobals.cDecalConstantColor;
    c.p_vRTTTextureOffsetScale = psGlobals.vDecalRTTTextureOffsetScale;
    c.p_mFresnelTransform      = psGlobals.mDecalFresnelTransform;
    c.p_mUVTransform           = psGlobals.mDecalUVTransform;
    return c;
}
SLayerConfig setupEmissiveLayer()
{
    SLayerConfig c;
    c.b_iLayerEnable           = SC2_CORE_ENABLE(Emissive, b_iEmissiveLayerEnable);
    c.b_iLayerId               = SC2_LAYER_LayerId(b_iEmissiveLayerId);
    c.b_iTextureType           = b_iEmissiveTextureType;
    c.b_iUVMapping             = SC2_LAYER_UVMapping(b_iEmissiveUVMapping);
    c.b_iTeamColorMode         = SC2_LAYER_TeamColorMode(b_iEmissiveTeamColorMode);
    c.b_iUVEmitter             = SC2_LAYER_UVEmitter(b_iEmissiveUVEmitter);
    c.b_iUseMask               = SC2_LAYER_UseMask(b_iEmissiveUseMask);
    c.b_iChannelSelect         = SC2_LAYER_ChannelSelect(b_iEmissiveChannelSelect);
    c.b_iInvert                = SC2_LAYER_Invert(b_iEmissiveInvert);
    c.b_iClamp                 = SC2_LAYER_Clamp(b_iEmissiveClamp);
    c.b_iFresnelMode           = SC2_LAYER_FresnelMode(b_iEmissiveFresnelMode);
    c.b_iFresnelTransformMode  = SC2_LAYER_FresnelTransformMode(b_iEmissiveFresnelTransformMode);
    c.b_iFresnelSaturate       = SC2_LAYER_FresnelSaturate(b_iEmissiveFresnelSaturate);
    c.b_iUseConstantColor      = SC2_CORE_ENABLE(Emissive, b_iEmissiveUseConstantColor);
    c.b_iOp                    = SC2_LAYER_Op(b_iEmissiveOp);
    c.b_iIsRTTTexture          = SC2_LAYER_IsRTTTexture(b_iEmissiveIsRTTTexture);
    c.b_iTriPlanarUvId         = SC2_LAYER_TriPlanarUvId(b_iEmissiveTriPlanarUvId);
    c.p_vMultiplyAddAlphaTrans = psGlobals.vEmissiveMultiplyAddAlphaTrans;
    c.p_vAtlasParams           = psGlobals.vEmissiveAtlasParams;
    c.p_vFresnelExponentBiasScale = psGlobals.vEmissiveFresnelExponentBiasScale;
    c.p_cConstantColor         = psGlobals.cEmissiveConstantColor;
    c.p_vRTTTextureOffsetScale = psGlobals.vEmissiveRTTTextureOffsetScale;
    c.p_mFresnelTransform      = psGlobals.mEmissiveFresnelTransform;
    c.p_mUVTransform           = psGlobals.mEmissiveUVTransform;
    return c;
}
SLayerConfig setupEmissive2Layer()
{
    SLayerConfig c;
    c.b_iLayerEnable           = SC2_CORE_ENABLE(Emissive2, b_iEmissive2LayerEnable);
    c.b_iLayerId               = SC2_LAYER_LayerId(b_iEmissive2LayerId);
    c.b_iTextureType           = b_iEmissive2TextureType;
    c.b_iUVMapping             = SC2_LAYER_UVMapping(b_iEmissive2UVMapping);
    c.b_iTeamColorMode         = SC2_LAYER_TeamColorMode(b_iEmissive2TeamColorMode);
    c.b_iUVEmitter             = SC2_LAYER_UVEmitter(b_iEmissive2UVEmitter);
    c.b_iUseMask               = SC2_LAYER_UseMask(b_iEmissive2UseMask);
    c.b_iChannelSelect         = SC2_LAYER_ChannelSelect(b_iEmissive2ChannelSelect);
    c.b_iInvert                = SC2_LAYER_Invert(b_iEmissive2Invert);
    c.b_iClamp                 = SC2_LAYER_Clamp(b_iEmissive2Clamp);
    c.b_iFresnelMode           = SC2_LAYER_FresnelMode(b_iEmissive2FresnelMode);
    c.b_iFresnelTransformMode  = SC2_LAYER_FresnelTransformMode(b_iEmissive2FresnelTransformMode);
    c.b_iFresnelSaturate       = SC2_LAYER_FresnelSaturate(b_iEmissive2FresnelSaturate);
    c.b_iUseConstantColor      = SC2_CORE_ENABLE(Emissive2, b_iEmissive2UseConstantColor);
    c.b_iOp                    = SC2_LAYER_Op(b_iEmissive2Op);
    c.b_iIsRTTTexture          = SC2_LAYER_IsRTTTexture(b_iEmissive2IsRTTTexture);
    c.b_iTriPlanarUvId         = SC2_LAYER_TriPlanarUvId(b_iEmissive2TriPlanarUvId);
    c.p_vMultiplyAddAlphaTrans = psGlobals.vEmissive2MultiplyAddAlphaTrans;
    c.p_vAtlasParams           = psGlobals.vEmissive2AtlasParams;
    c.p_vFresnelExponentBiasScale = psGlobals.vEmissive2FresnelExponentBiasScale;
    c.p_cConstantColor         = psGlobals.cEmissive2ConstantColor;
    c.p_vRTTTextureOffsetScale = psGlobals.vEmissive2RTTTextureOffsetScale;
    c.p_mFresnelTransform      = psGlobals.mEmissive2FresnelTransform;
    c.p_mUVTransform           = psGlobals.mEmissive2UVTransform;
    return c;
}
SLayerConfig setupAlphaMaskLayer()
{
    SLayerConfig c;
    c.b_iLayerEnable           = SC2_CORE_ENABLE(AlphaMask, b_iAlphaMaskLayerEnable);
    c.b_iLayerId               = SC2_LAYER_LayerId(b_iAlphaMaskLayerId);
    c.b_iTextureType           = b_iAlphaMaskTextureType;
    c.b_iUVMapping             = SC2_LAYER_UVMapping(b_iAlphaMaskUVMapping);
    c.b_iTeamColorMode         = SC2_LAYER_TeamColorMode(b_iAlphaMaskTeamColorMode);
    c.b_iUVEmitter             = SC2_LAYER_UVEmitter(b_iAlphaMaskUVEmitter);
    c.b_iUseMask               = SC2_LAYER_UseMask(b_iAlphaMaskUseMask);
    c.b_iChannelSelect         = SC2_LAYER_ChannelSelect(b_iAlphaMaskChannelSelect);
    c.b_iInvert                = SC2_LAYER_Invert(b_iAlphaMaskInvert);
    c.b_iClamp                 = SC2_LAYER_Clamp(b_iAlphaMaskClamp);
    c.b_iFresnelMode           = SC2_LAYER_FresnelMode(b_iAlphaMaskFresnelMode);
    c.b_iFresnelTransformMode  = SC2_LAYER_FresnelTransformMode(b_iAlphaMaskFresnelTransformMode);
    c.b_iFresnelSaturate       = SC2_LAYER_FresnelSaturate(b_iAlphaMaskFresnelSaturate);
    c.b_iUseConstantColor      = SC2_CORE_ENABLE(AlphaMask, b_iAlphaMaskUseConstantColor);
    c.b_iOp                    = SC2_LAYER_Op(b_iAlphaMaskOp);
    c.b_iIsRTTTexture          = SC2_LAYER_IsRTTTexture(b_iAlphaMaskIsRTTTexture);
    c.b_iTriPlanarUvId         = SC2_LAYER_TriPlanarUvId(b_iAlphaMaskTriPlanarUvId);
    c.p_vMultiplyAddAlphaTrans = psGlobals.vAlphaMaskMultiplyAddAlphaTrans;
    c.p_vAtlasParams           = psGlobals.vAlphaMaskAtlasParams;
    c.p_vFresnelExponentBiasScale = psGlobals.vAlphaMaskFresnelExponentBiasScale;
    c.p_cConstantColor         = psGlobals.cAlphaMaskConstantColor;
    c.p_vRTTTextureOffsetScale = psGlobals.vAlphaMaskRTTTextureOffsetScale;
    c.p_mFresnelTransform      = psGlobals.mAlphaMaskFresnelTransform;
    c.p_mUVTransform           = psGlobals.mAlphaMaskUVTransform;
    return c;
}
SLayerConfig setupAlphaMask2Layer()
{
    SLayerConfig c;
    c.b_iLayerEnable           = SC2_CORE_ENABLE(AlphaMask2, b_iAlphaMask2LayerEnable);
    c.b_iLayerId               = SC2_LAYER_LayerId(b_iAlphaMask2LayerId);
    c.b_iTextureType           = b_iAlphaMask2TextureType;
    c.b_iUVMapping             = SC2_LAYER_UVMapping(b_iAlphaMask2UVMapping);
    c.b_iTeamColorMode         = SC2_LAYER_TeamColorMode(b_iAlphaMask2TeamColorMode);
    c.b_iUVEmitter             = SC2_LAYER_UVEmitter(b_iAlphaMask2UVEmitter);
    c.b_iUseMask               = SC2_LAYER_UseMask(b_iAlphaMask2UseMask);
    c.b_iChannelSelect         = SC2_LAYER_ChannelSelect(b_iAlphaMask2ChannelSelect);
    c.b_iInvert                = SC2_LAYER_Invert(b_iAlphaMask2Invert);
    c.b_iClamp                 = SC2_LAYER_Clamp(b_iAlphaMask2Clamp);
    c.b_iFresnelMode           = SC2_LAYER_FresnelMode(b_iAlphaMask2FresnelMode);
    c.b_iFresnelTransformMode  = SC2_LAYER_FresnelTransformMode(b_iAlphaMask2FresnelTransformMode);
    c.b_iFresnelSaturate       = SC2_LAYER_FresnelSaturate(b_iAlphaMask2FresnelSaturate);
    c.b_iUseConstantColor      = SC2_CORE_ENABLE(AlphaMask2, b_iAlphaMask2UseConstantColor);
    c.b_iOp                    = SC2_LAYER_Op(b_iAlphaMask2Op);
    c.b_iIsRTTTexture          = SC2_LAYER_IsRTTTexture(b_iAlphaMask2IsRTTTexture);
    c.b_iTriPlanarUvId         = SC2_LAYER_TriPlanarUvId(b_iAlphaMask2TriPlanarUvId);
    c.p_vMultiplyAddAlphaTrans = psGlobals.vAlphaMask2MultiplyAddAlphaTrans;
    c.p_vAtlasParams           = psGlobals.vAlphaMask2AtlasParams;
    c.p_vFresnelExponentBiasScale = psGlobals.vAlphaMask2FresnelExponentBiasScale;
    c.p_cConstantColor         = psGlobals.cAlphaMask2ConstantColor;
    c.p_vRTTTextureOffsetScale = psGlobals.vAlphaMask2RTTTextureOffsetScale;
    c.p_mFresnelTransform      = psGlobals.mAlphaMask2FresnelTransform;
    c.p_mUVTransform           = psGlobals.mAlphaMask2UVTransform;
    return c;
}
SLayerConfig setupLightmapLayer()
{
    SLayerConfig c;
    c.b_iLayerEnable           = b_iLightmapLayerEnable;
    c.b_iLayerId               = SC2_LAYER_LayerId(b_iLightmapLayerId);
    c.b_iTextureType           = b_iLightmapTextureType;
    c.b_iUVMapping             = SC2_LAYER_UVMapping(b_iLightmapUVMapping);
    c.b_iTeamColorMode         = SC2_LAYER_TeamColorMode(b_iLightmapTeamColorMode);
    c.b_iUVEmitter             = SC2_LAYER_UVEmitter(b_iLightmapUVEmitter);
    c.b_iUseMask               = SC2_LAYER_UseMask(b_iLightmapUseMask);
    c.b_iChannelSelect         = SC2_LAYER_ChannelSelect(b_iLightmapChannelSelect);
    c.b_iInvert                = SC2_LAYER_Invert(b_iLightmapInvert);
    c.b_iClamp                 = SC2_LAYER_Clamp(b_iLightmapClamp);
    c.b_iFresnelMode           = SC2_LAYER_FresnelMode(b_iLightmapFresnelMode);
    c.b_iFresnelTransformMode  = SC2_LAYER_FresnelTransformMode(b_iLightmapFresnelTransformMode);
    c.b_iFresnelSaturate       = SC2_LAYER_FresnelSaturate(b_iLightmapFresnelSaturate);
    c.b_iUseConstantColor      = b_iLightmapUseConstantColor;
    c.b_iOp                    = SC2_LAYER_Op(b_iLightmapOp);
    c.b_iIsRTTTexture          = SC2_LAYER_IsRTTTexture(b_iLightmapIsRTTTexture);
    c.b_iTriPlanarUvId         = SC2_LAYER_TriPlanarUvId(b_iLightmapTriPlanarUvId);
    c.p_vMultiplyAddAlphaTrans = psGlobals.vLightmapMultiplyAddAlphaTrans;
    c.p_vAtlasParams           = psGlobals.vLightmapAtlasParams;
    c.p_vFresnelExponentBiasScale = psGlobals.vLightmapFresnelExponentBiasScale;
    c.p_cConstantColor         = psGlobals.cLightmapConstantColor;
    c.p_vRTTTextureOffsetScale = psGlobals.vLightmapRTTTextureOffsetScale;
    c.p_mFresnelTransform      = psGlobals.mLightmapFresnelTransform;
    c.p_mUVTransform           = psGlobals.mLightmapUVTransform;
    return c;
}
SLayerConfig setupAmbientOcclusionLayer()
{
    SLayerConfig c;
    c.b_iLayerEnable           = SC2_CORE_ENABLE(AmbientOcclusion, b_iAmbientOcclusionLayerEnable);
    c.b_iLayerId               = SC2_LAYER_LayerId(b_iAmbientOcclusionLayerId);
    c.b_iTextureType           = b_iAmbientOcclusionTextureType;
    c.b_iUVMapping             = SC2_LAYER_UVMapping(b_iAmbientOcclusionUVMapping);
    c.b_iTeamColorMode         = SC2_LAYER_TeamColorMode(b_iAmbientOcclusionTeamColorMode);
    c.b_iUVEmitter             = SC2_LAYER_UVEmitter(b_iAmbientOcclusionUVEmitter);
    c.b_iUseMask               = SC2_LAYER_UseMask(b_iAmbientOcclusionUseMask);
    c.b_iChannelSelect         = SC2_LAYER_ChannelSelect(b_iAmbientOcclusionChannelSelect);
    c.b_iInvert                = SC2_LAYER_Invert(b_iAmbientOcclusionInvert);
    c.b_iClamp                 = SC2_LAYER_Clamp(b_iAmbientOcclusionClamp);
    c.b_iFresnelMode           = SC2_LAYER_FresnelMode(b_iAmbientOcclusionFresnelMode);
    c.b_iFresnelTransformMode  = SC2_LAYER_FresnelTransformMode(b_iAmbientOcclusionFresnelTransformMode);
    c.b_iFresnelSaturate       = SC2_LAYER_FresnelSaturate(b_iAmbientOcclusionFresnelSaturate);
    c.b_iUseConstantColor      = SC2_CORE_ENABLE(AmbientOcclusion, b_iAmbientOcclusionUseConstantColor);
    c.b_iOp                    = SC2_LAYER_Op(b_iAmbientOcclusionOp);
    c.b_iIsRTTTexture          = SC2_LAYER_IsRTTTexture(b_iAmbientOcclusionIsRTTTexture);
    c.b_iTriPlanarUvId         = SC2_LAYER_TriPlanarUvId(b_iAmbientOcclusionTriPlanarUvId);
    c.p_vMultiplyAddAlphaTrans = psGlobals.vAmbientOcclusionMultiplyAddAlphaTrans;
    c.p_vAtlasParams           = psGlobals.vAmbientOcclusionAtlasParams;
    c.p_vFresnelExponentBiasScale = psGlobals.vAmbientOcclusionFresnelExponentBiasScale;
    c.p_cConstantColor         = psGlobals.cAmbientOcclusionConstantColor;
    c.p_vRTTTextureOffsetScale = psGlobals.vAmbientOcclusionRTTTextureOffsetScale;
    c.p_mFresnelTransform      = psGlobals.mAmbientOcclusionFresnelTransform;
    c.p_mUVTransform           = psGlobals.mAmbientOcclusionUVTransform;
    return c;
}
SLayerConfig setupSpecularExponentLayer()
{
    SLayerConfig c;
    c.b_iLayerEnable           = SC2_CORE_ENABLE(SpecularExponent, b_iSpecularExponentLayerEnable);
    c.b_iLayerId               = SC2_LAYER_LayerId(b_iSpecularExponentLayerId);
    c.b_iTextureType           = b_iSpecularExponentTextureType;
    c.b_iUVMapping             = SC2_LAYER_UVMapping(b_iSpecularExponentUVMapping);
    c.b_iTeamColorMode         = SC2_LAYER_TeamColorMode(b_iSpecularExponentTeamColorMode);
    c.b_iUVEmitter             = SC2_LAYER_UVEmitter(b_iSpecularExponentUVEmitter);
    c.b_iUseMask               = SC2_LAYER_UseMask(b_iSpecularExponentUseMask);
    c.b_iChannelSelect         = SC2_LAYER_ChannelSelect(b_iSpecularExponentChannelSelect);
    c.b_iInvert                = SC2_LAYER_Invert(b_iSpecularExponentInvert);
    c.b_iClamp                 = SC2_LAYER_Clamp(b_iSpecularExponentClamp);
    c.b_iFresnelMode           = SC2_LAYER_FresnelMode(b_iSpecularExponentFresnelMode);
    c.b_iFresnelTransformMode  = SC2_LAYER_FresnelTransformMode(b_iSpecularExponentFresnelTransformMode);
    c.b_iFresnelSaturate       = SC2_LAYER_FresnelSaturate(b_iSpecularExponentFresnelSaturate);
    c.b_iUseConstantColor      = SC2_CORE_ENABLE(SpecularExponent, b_iSpecularExponentUseConstantColor);
    c.b_iOp                    = SC2_LAYER_Op(b_iSpecularExponentOp);
    c.b_iIsRTTTexture          = SC2_LAYER_IsRTTTexture(b_iSpecularExponentIsRTTTexture);
    c.b_iTriPlanarUvId         = SC2_LAYER_TriPlanarUvId(b_iSpecularExponentTriPlanarUvId);
    c.p_vMultiplyAddAlphaTrans = psGlobals.vSpecularExponentMultiplyAddAlphaTrans;
    c.p_vAtlasParams           = psGlobals.vSpecularExponentAtlasParams;
    c.p_vFresnelExponentBiasScale = psGlobals.vSpecularExponentFresnelExponentBiasScale;
    c.p_cConstantColor         = psGlobals.cSpecularExponentConstantColor;
    c.p_vRTTTextureOffsetScale = psGlobals.vSpecularExponentRTTTextureOffsetScale;
    c.p_mFresnelTransform      = psGlobals.mSpecularExponentFresnelTransform;
    c.p_mUVTransform           = psGlobals.mSpecularExponentUVTransform;
    return c;
}
SLayerConfig setupNormalLayer()
{
    SLayerConfig c;
    c.b_iLayerEnable           = b_iNormalLayerEnable;
    c.b_iLayerId               = SC2_LAYER_LayerId(b_iNormalLayerId);
    c.b_iTextureType           = b_iNormalTextureType;
    c.b_iUVMapping             = SC2_LAYER_UVMapping(b_iNormalUVMapping);
    c.b_iTeamColorMode         = SC2_LAYER_TeamColorMode(b_iNormalTeamColorMode);
    c.b_iUVEmitter             = SC2_LAYER_UVEmitter(b_iNormalUVEmitter);
    c.b_iUseMask               = SC2_LAYER_UseMask(b_iNormalUseMask);
    c.b_iChannelSelect         = SC2_LAYER_ChannelSelect(b_iNormalChannelSelect);
    c.b_iInvert                = SC2_LAYER_Invert(b_iNormalInvert);
    c.b_iClamp                 = SC2_LAYER_Clamp(b_iNormalClamp);
    c.b_iFresnelMode           = SC2_LAYER_FresnelMode(b_iNormalFresnelMode);
    c.b_iFresnelTransformMode  = SC2_LAYER_FresnelTransformMode(b_iNormalFresnelTransformMode);
    c.b_iFresnelSaturate       = SC2_LAYER_FresnelSaturate(b_iNormalFresnelSaturate);
    c.b_iUseConstantColor      = b_iNormalUseConstantColor;
    c.b_iOp                    = SC2_LAYER_Op(b_iNormalOp);
    c.b_iIsRTTTexture          = SC2_LAYER_IsRTTTexture(b_iNormalIsRTTTexture);
    c.b_iTriPlanarUvId         = SC2_LAYER_TriPlanarUvId(b_iNormalTriPlanarUvId);
    c.p_vMultiplyAddAlphaTrans = psGlobals.vNormalMultiplyAddAlphaTrans;
    c.p_vAtlasParams           = psGlobals.vNormalAtlasParams;
    c.p_vFresnelExponentBiasScale = psGlobals.vNormalFresnelExponentBiasScale;
    c.p_cConstantColor         = psGlobals.cNormalConstantColor;
    c.p_vRTTTextureOffsetScale = psGlobals.vNormalRTTTextureOffsetScale;
    c.p_mFresnelTransform      = psGlobals.mNormalFresnelTransform;
    c.p_mUVTransform           = psGlobals.mNormalUVTransform;
    return c;
}
SLayerConfig setupEnvioLayer()
{
    SLayerConfig c;
    c.b_iLayerEnable           = b_iEnvioLayerEnable;
    c.b_iLayerId               = SC2_LAYER_LayerId(b_iEnvioLayerId);
    c.b_iTextureType           = b_iEnvioTextureType;
    c.b_iUVMapping             = SC2_LAYER_UVMapping(b_iEnvioUVMapping);
    c.b_iTeamColorMode         = SC2_LAYER_TeamColorMode(b_iEnvioTeamColorMode);
    c.b_iUVEmitter             = SC2_LAYER_UVEmitter(b_iEnvioUVEmitter);
    c.b_iUseMask               = SC2_LAYER_UseMask(b_iEnvioUseMask);
    c.b_iChannelSelect         = SC2_LAYER_ChannelSelect(b_iEnvioChannelSelect);
    c.b_iInvert                = SC2_LAYER_Invert(b_iEnvioInvert);
    c.b_iClamp                 = SC2_LAYER_Clamp(b_iEnvioClamp);
    c.b_iFresnelMode           = SC2_LAYER_FresnelMode(b_iEnvioFresnelMode);
    c.b_iFresnelTransformMode  = SC2_LAYER_FresnelTransformMode(b_iEnvioFresnelTransformMode);
    c.b_iFresnelSaturate       = SC2_LAYER_FresnelSaturate(b_iEnvioFresnelSaturate);
    c.b_iUseConstantColor      = b_iEnvioUseConstantColor;
    c.b_iOp                    = SC2_LAYER_Op(b_iEnvioOp);
    c.b_iIsRTTTexture          = SC2_LAYER_IsRTTTexture(b_iEnvioIsRTTTexture);
    c.b_iTriPlanarUvId         = SC2_LAYER_TriPlanarUvId(b_iEnvioTriPlanarUvId);
    c.p_vMultiplyAddAlphaTrans = psGlobals.vEnvioMultiplyAddAlphaTrans;
    c.p_vAtlasParams           = psGlobals.vEnvioAtlasParams;
    c.p_vFresnelExponentBiasScale = psGlobals.vEnvioFresnelExponentBiasScale;
    c.p_cConstantColor         = psGlobals.cEnvioConstantColor;
    c.p_vRTTTextureOffsetScale = psGlobals.vEnvioRTTTextureOffsetScale;
    c.p_mFresnelTransform      = psGlobals.mEnvioFresnelTransform;
    c.p_mUVTransform           = psGlobals.mEnvioUVTransform;
    return c;
}
SLayerConfig setupEnvioMaskLayer()
{
    SLayerConfig c;
    c.b_iLayerEnable           = b_iEnvioMaskLayerEnable;
    c.b_iLayerId               = SC2_LAYER_LayerId(b_iEnvioMaskLayerId);
    c.b_iTextureType           = b_iEnvioMaskTextureType;
    c.b_iUVMapping             = SC2_LAYER_UVMapping(b_iEnvioMaskUVMapping);
    c.b_iTeamColorMode         = SC2_LAYER_TeamColorMode(b_iEnvioMaskTeamColorMode);
    c.b_iUVEmitter             = SC2_LAYER_UVEmitter(b_iEnvioMaskUVEmitter);
    c.b_iUseMask               = SC2_LAYER_UseMask(b_iEnvioMaskUseMask);
    c.b_iChannelSelect         = SC2_LAYER_ChannelSelect(b_iEnvioMaskChannelSelect);
    c.b_iInvert                = SC2_LAYER_Invert(b_iEnvioMaskInvert);
    c.b_iClamp                 = SC2_LAYER_Clamp(b_iEnvioMaskClamp);
    c.b_iFresnelMode           = SC2_LAYER_FresnelMode(b_iEnvioMaskFresnelMode);
    c.b_iFresnelTransformMode  = SC2_LAYER_FresnelTransformMode(b_iEnvioMaskFresnelTransformMode);
    c.b_iFresnelSaturate       = SC2_LAYER_FresnelSaturate(b_iEnvioMaskFresnelSaturate);
    c.b_iUseConstantColor      = b_iEnvioMaskUseConstantColor;
    c.b_iOp                    = SC2_LAYER_Op(b_iEnvioMaskOp);
    c.b_iIsRTTTexture          = SC2_LAYER_IsRTTTexture(b_iEnvioMaskIsRTTTexture);
    c.b_iTriPlanarUvId         = SC2_LAYER_TriPlanarUvId(b_iEnvioMaskTriPlanarUvId);
    c.p_vMultiplyAddAlphaTrans = psGlobals.vEnvioMaskMultiplyAddAlphaTrans;
    c.p_vAtlasParams           = psGlobals.vEnvioMaskAtlasParams;
    c.p_vFresnelExponentBiasScale = psGlobals.vEnvioMaskFresnelExponentBiasScale;
    c.p_cConstantColor         = psGlobals.cEnvioMaskConstantColor;
    c.p_vRTTTextureOffsetScale = psGlobals.vEnvioMaskRTTTextureOffsetScale;
    c.p_mFresnelTransform      = psGlobals.mEnvioMaskFresnelTransform;
    c.p_mUVTransform           = psGlobals.mEnvioMaskUVTransform;
    return c;
}

// ---- NormalBlend layers (normalBlending increment) -------------------------
#ifndef b_iNormalBlendMaskLayerEnable
#define b_iNormalBlendMaskLayerEnable 0
#endif
#ifndef b_iNormalBlendMaskLayerId
#define b_iNormalBlendMaskLayerId 0
#endif
#ifndef b_iNormalBlendMaskTextureType
#define b_iNormalBlendMaskTextureType 0
#endif
#ifndef b_iNormalBlendMaskUVMapping
#define b_iNormalBlendMaskUVMapping 0
#endif
#ifndef b_iNormalBlendMaskTeamColorMode
#define b_iNormalBlendMaskTeamColorMode 0
#endif
#ifndef b_iNormalBlendMaskUVEmitter
#define b_iNormalBlendMaskUVEmitter 0
#endif
#ifndef b_iNormalBlendMaskUseMask
#define b_iNormalBlendMaskUseMask 0
#endif
#ifndef b_iNormalBlendMaskChannelSelect
#define b_iNormalBlendMaskChannelSelect 0
#endif
#ifndef b_iNormalBlendMaskInvert
#define b_iNormalBlendMaskInvert 0
#endif
#ifndef b_iNormalBlendMaskClamp
#define b_iNormalBlendMaskClamp 0
#endif
#ifndef b_iNormalBlendMaskFresnelMode
#define b_iNormalBlendMaskFresnelMode 0
#endif
#ifndef b_iNormalBlendMaskFresnelTransformMode
#define b_iNormalBlendMaskFresnelTransformMode 0
#endif
#ifndef b_iNormalBlendMaskFresnelSaturate
#define b_iNormalBlendMaskFresnelSaturate 0
#endif
#ifndef b_iNormalBlendMaskUseConstantColor
#define b_iNormalBlendMaskUseConstantColor 0
#endif
#ifndef b_iNormalBlendMaskOp
#define b_iNormalBlendMaskOp 0
#endif
#ifndef b_iNormalBlendMaskIsRTTTexture
#define b_iNormalBlendMaskIsRTTTexture 0
#endif
#ifndef b_iNormalBlendMaskTriPlanarUvId
#define b_iNormalBlendMaskTriPlanarUvId 0
#endif
#ifndef b_iNormalBlendMask2LayerEnable
#define b_iNormalBlendMask2LayerEnable 0
#endif
#ifndef b_iNormalBlendMask2LayerId
#define b_iNormalBlendMask2LayerId 0
#endif
#ifndef b_iNormalBlendMask2TextureType
#define b_iNormalBlendMask2TextureType 0
#endif
#ifndef b_iNormalBlendMask2UVMapping
#define b_iNormalBlendMask2UVMapping 0
#endif
#ifndef b_iNormalBlendMask2TeamColorMode
#define b_iNormalBlendMask2TeamColorMode 0
#endif
#ifndef b_iNormalBlendMask2UVEmitter
#define b_iNormalBlendMask2UVEmitter 0
#endif
#ifndef b_iNormalBlendMask2UseMask
#define b_iNormalBlendMask2UseMask 0
#endif
#ifndef b_iNormalBlendMask2ChannelSelect
#define b_iNormalBlendMask2ChannelSelect 0
#endif
#ifndef b_iNormalBlendMask2Invert
#define b_iNormalBlendMask2Invert 0
#endif
#ifndef b_iNormalBlendMask2Clamp
#define b_iNormalBlendMask2Clamp 0
#endif
#ifndef b_iNormalBlendMask2FresnelMode
#define b_iNormalBlendMask2FresnelMode 0
#endif
#ifndef b_iNormalBlendMask2FresnelTransformMode
#define b_iNormalBlendMask2FresnelTransformMode 0
#endif
#ifndef b_iNormalBlendMask2FresnelSaturate
#define b_iNormalBlendMask2FresnelSaturate 0
#endif
#ifndef b_iNormalBlendMask2UseConstantColor
#define b_iNormalBlendMask2UseConstantColor 0
#endif
#ifndef b_iNormalBlendMask2Op
#define b_iNormalBlendMask2Op 0
#endif
#ifndef b_iNormalBlendMask2IsRTTTexture
#define b_iNormalBlendMask2IsRTTTexture 0
#endif
#ifndef b_iNormalBlendMask2TriPlanarUvId
#define b_iNormalBlendMask2TriPlanarUvId 0
#endif
#ifndef b_iNormalBlendNormalLayerEnable
#define b_iNormalBlendNormalLayerEnable 0
#endif
#ifndef b_iNormalBlendNormalLayerId
#define b_iNormalBlendNormalLayerId 0
#endif
#ifndef b_iNormalBlendNormalTextureType
#define b_iNormalBlendNormalTextureType 0
#endif
#ifndef b_iNormalBlendNormalUVMapping
#define b_iNormalBlendNormalUVMapping 0
#endif
#ifndef b_iNormalBlendNormalTeamColorMode
#define b_iNormalBlendNormalTeamColorMode 0
#endif
#ifndef b_iNormalBlendNormalUVEmitter
#define b_iNormalBlendNormalUVEmitter 0
#endif
#ifndef b_iNormalBlendNormalUseMask
#define b_iNormalBlendNormalUseMask 0
#endif
#ifndef b_iNormalBlendNormalChannelSelect
#define b_iNormalBlendNormalChannelSelect 0
#endif
#ifndef b_iNormalBlendNormalInvert
#define b_iNormalBlendNormalInvert 0
#endif
#ifndef b_iNormalBlendNormalClamp
#define b_iNormalBlendNormalClamp 0
#endif
#ifndef b_iNormalBlendNormalFresnelMode
#define b_iNormalBlendNormalFresnelMode 0
#endif
#ifndef b_iNormalBlendNormalFresnelTransformMode
#define b_iNormalBlendNormalFresnelTransformMode 0
#endif
#ifndef b_iNormalBlendNormalFresnelSaturate
#define b_iNormalBlendNormalFresnelSaturate 0
#endif
#ifndef b_iNormalBlendNormalUseConstantColor
#define b_iNormalBlendNormalUseConstantColor 0
#endif
#ifndef b_iNormalBlendNormalOp
#define b_iNormalBlendNormalOp 0
#endif
#ifndef b_iNormalBlendNormalIsRTTTexture
#define b_iNormalBlendNormalIsRTTTexture 0
#endif
#ifndef b_iNormalBlendNormalTriPlanarUvId
#define b_iNormalBlendNormalTriPlanarUvId 0
#endif
#ifndef b_iNormalBlendNormal2LayerEnable
#define b_iNormalBlendNormal2LayerEnable 0
#endif
#ifndef b_iNormalBlendNormal2LayerId
#define b_iNormalBlendNormal2LayerId 0
#endif
#ifndef b_iNormalBlendNormal2TextureType
#define b_iNormalBlendNormal2TextureType 0
#endif
#ifndef b_iNormalBlendNormal2UVMapping
#define b_iNormalBlendNormal2UVMapping 0
#endif
#ifndef b_iNormalBlendNormal2TeamColorMode
#define b_iNormalBlendNormal2TeamColorMode 0
#endif
#ifndef b_iNormalBlendNormal2UVEmitter
#define b_iNormalBlendNormal2UVEmitter 0
#endif
#ifndef b_iNormalBlendNormal2UseMask
#define b_iNormalBlendNormal2UseMask 0
#endif
#ifndef b_iNormalBlendNormal2ChannelSelect
#define b_iNormalBlendNormal2ChannelSelect 0
#endif
#ifndef b_iNormalBlendNormal2Invert
#define b_iNormalBlendNormal2Invert 0
#endif
#ifndef b_iNormalBlendNormal2Clamp
#define b_iNormalBlendNormal2Clamp 0
#endif
#ifndef b_iNormalBlendNormal2FresnelMode
#define b_iNormalBlendNormal2FresnelMode 0
#endif
#ifndef b_iNormalBlendNormal2FresnelTransformMode
#define b_iNormalBlendNormal2FresnelTransformMode 0
#endif
#ifndef b_iNormalBlendNormal2FresnelSaturate
#define b_iNormalBlendNormal2FresnelSaturate 0
#endif
#ifndef b_iNormalBlendNormal2UseConstantColor
#define b_iNormalBlendNormal2UseConstantColor 0
#endif
#ifndef b_iNormalBlendNormal2Op
#define b_iNormalBlendNormal2Op 0
#endif
#ifndef b_iNormalBlendNormal2IsRTTTexture
#define b_iNormalBlendNormal2IsRTTTexture 0
#endif
#ifndef b_iNormalBlendNormal2TriPlanarUvId
#define b_iNormalBlendNormal2TriPlanarUvId 0
#endif

SLayerConfig setupNormalBlendMaskLayer()
{
    SLayerConfig c;
    c.b_iLayerEnable           = b_iNormalBlendMaskLayerEnable;
    c.b_iLayerId               = SC2_LAYER_LayerId(b_iNormalBlendMaskLayerId);
    c.b_iTextureType           = b_iNormalBlendMaskTextureType;
    c.b_iUVMapping             = SC2_LAYER_UVMapping(b_iNormalBlendMaskUVMapping);
    c.b_iTeamColorMode         = SC2_LAYER_TeamColorMode(b_iNormalBlendMaskTeamColorMode);
    c.b_iUVEmitter             = SC2_LAYER_UVEmitter(b_iNormalBlendMaskUVEmitter);
    c.b_iUseMask               = SC2_LAYER_UseMask(b_iNormalBlendMaskUseMask);
    c.b_iChannelSelect         = SC2_LAYER_ChannelSelect(b_iNormalBlendMaskChannelSelect);
    c.b_iInvert                = SC2_LAYER_Invert(b_iNormalBlendMaskInvert);
    c.b_iClamp                 = SC2_LAYER_Clamp(b_iNormalBlendMaskClamp);
    c.b_iFresnelMode           = SC2_LAYER_FresnelMode(b_iNormalBlendMaskFresnelMode);
    c.b_iFresnelTransformMode  = SC2_LAYER_FresnelTransformMode(b_iNormalBlendMaskFresnelTransformMode);
    c.b_iFresnelSaturate       = SC2_LAYER_FresnelSaturate(b_iNormalBlendMaskFresnelSaturate);
    c.b_iUseConstantColor      = b_iNormalBlendMaskUseConstantColor;
    c.b_iOp                    = SC2_LAYER_Op(b_iNormalBlendMaskOp);
    c.b_iIsRTTTexture          = SC2_LAYER_IsRTTTexture(b_iNormalBlendMaskIsRTTTexture);
    c.b_iTriPlanarUvId         = SC2_LAYER_TriPlanarUvId(b_iNormalBlendMaskTriPlanarUvId);
    c.p_vMultiplyAddAlphaTrans = psGlobals.vNormalBlendMaskMultiplyAddAlphaTrans;
    c.p_vAtlasParams           = psGlobals.vNormalBlendMaskAtlasParams;
    c.p_vFresnelExponentBiasScale = psGlobals.vNormalBlendMaskFresnelExponentBiasScale;
    c.p_cConstantColor         = psGlobals.cNormalBlendMaskConstantColor;
    c.p_vRTTTextureOffsetScale = psGlobals.vNormalBlendMaskRTTTextureOffsetScale;
    c.p_mFresnelTransform      = psGlobals.mNormalBlendMaskFresnelTransform;
    c.p_mUVTransform           = psGlobals.mNormalBlendMaskUVTransform;
    return c;
}
SLayerConfig setupNormalBlendMask2Layer()
{
    SLayerConfig c;
    c.b_iLayerEnable           = b_iNormalBlendMask2LayerEnable;
    c.b_iLayerId               = SC2_LAYER_LayerId(b_iNormalBlendMask2LayerId);
    c.b_iTextureType           = b_iNormalBlendMask2TextureType;
    c.b_iUVMapping             = SC2_LAYER_UVMapping(b_iNormalBlendMask2UVMapping);
    c.b_iTeamColorMode         = SC2_LAYER_TeamColorMode(b_iNormalBlendMask2TeamColorMode);
    c.b_iUVEmitter             = SC2_LAYER_UVEmitter(b_iNormalBlendMask2UVEmitter);
    c.b_iUseMask               = SC2_LAYER_UseMask(b_iNormalBlendMask2UseMask);
    c.b_iChannelSelect         = SC2_LAYER_ChannelSelect(b_iNormalBlendMask2ChannelSelect);
    c.b_iInvert                = SC2_LAYER_Invert(b_iNormalBlendMask2Invert);
    c.b_iClamp                 = SC2_LAYER_Clamp(b_iNormalBlendMask2Clamp);
    c.b_iFresnelMode           = SC2_LAYER_FresnelMode(b_iNormalBlendMask2FresnelMode);
    c.b_iFresnelTransformMode  = SC2_LAYER_FresnelTransformMode(b_iNormalBlendMask2FresnelTransformMode);
    c.b_iFresnelSaturate       = SC2_LAYER_FresnelSaturate(b_iNormalBlendMask2FresnelSaturate);
    c.b_iUseConstantColor      = b_iNormalBlendMask2UseConstantColor;
    c.b_iOp                    = SC2_LAYER_Op(b_iNormalBlendMask2Op);
    c.b_iIsRTTTexture          = SC2_LAYER_IsRTTTexture(b_iNormalBlendMask2IsRTTTexture);
    c.b_iTriPlanarUvId         = SC2_LAYER_TriPlanarUvId(b_iNormalBlendMask2TriPlanarUvId);
    c.p_vMultiplyAddAlphaTrans = psGlobals.vNormalBlendMask2MultiplyAddAlphaTrans;
    c.p_vAtlasParams           = psGlobals.vNormalBlendMask2AtlasParams;
    c.p_vFresnelExponentBiasScale = psGlobals.vNormalBlendMask2FresnelExponentBiasScale;
    c.p_cConstantColor         = psGlobals.cNormalBlendMask2ConstantColor;
    c.p_vRTTTextureOffsetScale = psGlobals.vNormalBlendMask2RTTTextureOffsetScale;
    c.p_mFresnelTransform      = psGlobals.mNormalBlendMask2FresnelTransform;
    c.p_mUVTransform           = psGlobals.mNormalBlendMask2UVTransform;
    return c;
}
SLayerConfig setupNormalBlendNormalLayer()
{
    SLayerConfig c;
    c.b_iLayerEnable           = b_iNormalBlendNormalLayerEnable;
    c.b_iLayerId               = SC2_LAYER_LayerId(b_iNormalBlendNormalLayerId);
    c.b_iTextureType           = b_iNormalBlendNormalTextureType;
    c.b_iUVMapping             = SC2_LAYER_UVMapping(b_iNormalBlendNormalUVMapping);
    c.b_iTeamColorMode         = SC2_LAYER_TeamColorMode(b_iNormalBlendNormalTeamColorMode);
    c.b_iUVEmitter             = SC2_LAYER_UVEmitter(b_iNormalBlendNormalUVEmitter);
    c.b_iUseMask               = SC2_LAYER_UseMask(b_iNormalBlendNormalUseMask);
    c.b_iChannelSelect         = SC2_LAYER_ChannelSelect(b_iNormalBlendNormalChannelSelect);
    c.b_iInvert                = SC2_LAYER_Invert(b_iNormalBlendNormalInvert);
    c.b_iClamp                 = SC2_LAYER_Clamp(b_iNormalBlendNormalClamp);
    c.b_iFresnelMode           = SC2_LAYER_FresnelMode(b_iNormalBlendNormalFresnelMode);
    c.b_iFresnelTransformMode  = SC2_LAYER_FresnelTransformMode(b_iNormalBlendNormalFresnelTransformMode);
    c.b_iFresnelSaturate       = SC2_LAYER_FresnelSaturate(b_iNormalBlendNormalFresnelSaturate);
    c.b_iUseConstantColor      = b_iNormalBlendNormalUseConstantColor;
    c.b_iOp                    = SC2_LAYER_Op(b_iNormalBlendNormalOp);
    c.b_iIsRTTTexture          = SC2_LAYER_IsRTTTexture(b_iNormalBlendNormalIsRTTTexture);
    c.b_iTriPlanarUvId         = SC2_LAYER_TriPlanarUvId(b_iNormalBlendNormalTriPlanarUvId);
    c.p_vMultiplyAddAlphaTrans = psGlobals.vNormalBlendNormalMultiplyAddAlphaTrans;
    c.p_vAtlasParams           = psGlobals.vNormalBlendNormalAtlasParams;
    c.p_vFresnelExponentBiasScale = psGlobals.vNormalBlendNormalFresnelExponentBiasScale;
    c.p_cConstantColor         = psGlobals.cNormalBlendNormalConstantColor;
    c.p_vRTTTextureOffsetScale = psGlobals.vNormalBlendNormalRTTTextureOffsetScale;
    c.p_mFresnelTransform      = psGlobals.mNormalBlendNormalFresnelTransform;
    c.p_mUVTransform           = psGlobals.mNormalBlendNormalUVTransform;
    return c;
}
SLayerConfig setupNormalBlendNormal2Layer()
{
    SLayerConfig c;
    c.b_iLayerEnable           = b_iNormalBlendNormal2LayerEnable;
    c.b_iLayerId               = SC2_LAYER_LayerId(b_iNormalBlendNormal2LayerId);
    c.b_iTextureType           = b_iNormalBlendNormal2TextureType;
    c.b_iUVMapping             = SC2_LAYER_UVMapping(b_iNormalBlendNormal2UVMapping);
    c.b_iTeamColorMode         = SC2_LAYER_TeamColorMode(b_iNormalBlendNormal2TeamColorMode);
    c.b_iUVEmitter             = SC2_LAYER_UVEmitter(b_iNormalBlendNormal2UVEmitter);
    c.b_iUseMask               = SC2_LAYER_UseMask(b_iNormalBlendNormal2UseMask);
    c.b_iChannelSelect         = SC2_LAYER_ChannelSelect(b_iNormalBlendNormal2ChannelSelect);
    c.b_iInvert                = SC2_LAYER_Invert(b_iNormalBlendNormal2Invert);
    c.b_iClamp                 = SC2_LAYER_Clamp(b_iNormalBlendNormal2Clamp);
    c.b_iFresnelMode           = SC2_LAYER_FresnelMode(b_iNormalBlendNormal2FresnelMode);
    c.b_iFresnelTransformMode  = SC2_LAYER_FresnelTransformMode(b_iNormalBlendNormal2FresnelTransformMode);
    c.b_iFresnelSaturate       = SC2_LAYER_FresnelSaturate(b_iNormalBlendNormal2FresnelSaturate);
    c.b_iUseConstantColor      = b_iNormalBlendNormal2UseConstantColor;
    c.b_iOp                    = SC2_LAYER_Op(b_iNormalBlendNormal2Op);
    c.b_iIsRTTTexture          = SC2_LAYER_IsRTTTexture(b_iNormalBlendNormal2IsRTTTexture);
    c.b_iTriPlanarUvId         = SC2_LAYER_TriPlanarUvId(b_iNormalBlendNormal2TriPlanarUvId);
    c.p_vMultiplyAddAlphaTrans = psGlobals.vNormalBlendNormal2MultiplyAddAlphaTrans;
    c.p_vAtlasParams           = psGlobals.vNormalBlendNormal2AtlasParams;
    c.p_vFresnelExponentBiasScale = psGlobals.vNormalBlendNormal2FresnelExponentBiasScale;
    c.p_cConstantColor         = psGlobals.cNormalBlendNormal2ConstantColor;
    c.p_vRTTTextureOffsetScale = psGlobals.vNormalBlendNormal2RTTTextureOffsetScale;
    c.p_mFresnelTransform      = psGlobals.mNormalBlendNormal2FresnelTransform;
    c.p_mUVTransform           = psGlobals.mNormalBlendNormal2UVTransform;
    return c;
}

// ---- Heightmap layer (parallax/POM increment) ------------------------------
#ifndef b_iHeightmapLayerEnable
#define b_iHeightmapLayerEnable 0
#endif
#ifndef b_iHeightmapLayerId
#define b_iHeightmapLayerId 0
#endif
#ifndef b_iHeightmapTextureType
#define b_iHeightmapTextureType 0
#endif
#ifndef b_iHeightmapUVMapping
#define b_iHeightmapUVMapping 0
#endif
#ifndef b_iHeightmapTeamColorMode
#define b_iHeightmapTeamColorMode 0
#endif
#ifndef b_iHeightmapUVEmitter
#define b_iHeightmapUVEmitter 0
#endif
#ifndef b_iHeightmapUseMask
#define b_iHeightmapUseMask 0
#endif
#ifndef b_iHeightmapChannelSelect
#define b_iHeightmapChannelSelect 0
#endif
#ifndef b_iHeightmapInvert
#define b_iHeightmapInvert 0
#endif
#ifndef b_iHeightmapClamp
#define b_iHeightmapClamp 0
#endif
#ifndef b_iHeightmapFresnelMode
#define b_iHeightmapFresnelMode 0
#endif
#ifndef b_iHeightmapFresnelTransformMode
#define b_iHeightmapFresnelTransformMode 0
#endif
#ifndef b_iHeightmapFresnelSaturate
#define b_iHeightmapFresnelSaturate 0
#endif
#ifndef b_iHeightmapUseConstantColor
#define b_iHeightmapUseConstantColor 0
#endif
#ifndef b_iHeightmapOp
#define b_iHeightmapOp 0
#endif
#ifndef b_iHeightmapIsRTTTexture
#define b_iHeightmapIsRTTTexture 0
#endif
#ifndef b_iHeightmapTriPlanarUvId
#define b_iHeightmapTriPlanarUvId 0
#endif

SLayerConfig setupHeightmapLayer()
{
    SLayerConfig c;
    c.b_iLayerEnable           = b_iHeightmapLayerEnable;
    c.b_iLayerId               = SC2_LAYER_LayerId(b_iHeightmapLayerId);
    c.b_iTextureType           = b_iHeightmapTextureType;
    c.b_iUVMapping             = SC2_LAYER_UVMapping(b_iHeightmapUVMapping);
    c.b_iTeamColorMode         = SC2_LAYER_TeamColorMode(b_iHeightmapTeamColorMode);
    c.b_iUVEmitter             = SC2_LAYER_UVEmitter(b_iHeightmapUVEmitter);
    c.b_iUseMask               = SC2_LAYER_UseMask(b_iHeightmapUseMask);
    c.b_iChannelSelect         = SC2_LAYER_ChannelSelect(b_iHeightmapChannelSelect);
    c.b_iInvert                = SC2_LAYER_Invert(b_iHeightmapInvert);
    c.b_iClamp                 = SC2_LAYER_Clamp(b_iHeightmapClamp);
    c.b_iFresnelMode           = SC2_LAYER_FresnelMode(b_iHeightmapFresnelMode);
    c.b_iFresnelTransformMode  = SC2_LAYER_FresnelTransformMode(b_iHeightmapFresnelTransformMode);
    c.b_iFresnelSaturate       = SC2_LAYER_FresnelSaturate(b_iHeightmapFresnelSaturate);
    c.b_iUseConstantColor      = b_iHeightmapUseConstantColor;
    c.b_iOp                    = SC2_LAYER_Op(b_iHeightmapOp);
    c.b_iIsRTTTexture          = SC2_LAYER_IsRTTTexture(b_iHeightmapIsRTTTexture);
    c.b_iTriPlanarUvId         = SC2_LAYER_TriPlanarUvId(b_iHeightmapTriPlanarUvId);
    c.p_vMultiplyAddAlphaTrans = psGlobals.vHeightmapMultiplyAddAlphaTrans;
    c.p_vAtlasParams           = psGlobals.vHeightmapAtlasParams;
    c.p_vFresnelExponentBiasScale = psGlobals.vHeightmapFresnelExponentBiasScale;
    c.p_cConstantColor         = psGlobals.cHeightmapConstantColor;
    c.p_vRTTTextureOffsetScale = psGlobals.vHeightmapRTTTextureOffsetScale;
    c.p_mFresnelTransform      = psGlobals.mHeightmapFresnelTransform;
    c.p_mUVTransform           = psGlobals.mHeightmapUVTransform;
    return c;
}
