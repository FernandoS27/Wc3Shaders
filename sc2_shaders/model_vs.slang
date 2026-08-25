// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026, Fernando Sahmkow
// See LICENSE in the repository root for full terms.
//
// Model family vertex shader (model.fx : ModelVertexMain) — the vertex root of the
// Model BLS bundle (M4 / implementation-plan M3).
//
// Structure follows the original exactly: the per-vertex Emit* functions, then one
// linear body that translates the vertex stream, applies the local-space
// preprocessing (Z offset / triplanar), skins into world space, applies world-space
// triplanar + instancing + vertex warps, and finally fills the permuted
// `VertexTransport`.
//
// The engine's `GenInterpolant( X, expr )` macro expands to `WRITE_INTERPOLANT_X(expr)`,
// which is a *no-op* for any interpolant the permutation packed out (sc2_interp
// .gen_preamble).  So every write here is wrapped in `#if SC2_HAS_<X>` — the same
// gate that decides whether the field exists in DefaultVSOutput — and a packed-out
// interpolant costs nothing, exactly as in the reference.
//
// The vertex-lighting emitters live in vs/model_vs_lighting.slang (lighting.fx's
// `Lighting()` specialised to isPerPixel = false); everything else is here.

implementing sc2_shaders;

// Compile-time axes + .fx enums (see the header: `__include` merges declarations
// but not #defines, so each VS translation unit includes this itself).
#include "vs/model_vs_defines.slangh"
#include "types/perm_cb.slangh"

// Axes read as CODE need a file-local default: only NONZERO axes are passed
// as -D, so a zero one would otherwise be an undefined identifier.
#ifndef b_computeHeight
#define b_computeHeight 0
#endif
#ifndef b_creepEnabled
#define b_creepEnabled 0
#endif
#ifndef b_iUseLocalTriPlanarTangentSpace
#define b_iUseLocalTriPlanarTangentSpace 0
#endif
#ifndef b_iUseLocalTriPlanarWeights
#define b_iUseLocalTriPlanarWeights 0
#endif
#ifndef b_iUseWorldTriPlanarTangentSpace
#define b_iUseWorldTriPlanarTangentSpace 0
#endif
#ifndef b_iUseWorldTriPlanarWeights
#define b_iUseWorldTriPlanarWeights 0
#endif
#ifndef b_iVertexAOFromVertexColor
#define b_iVertexAOFromVertexColor 0
#endif
#ifndef b_iZOffsetFromUB4N_XY
#define b_iZOffsetFromUB4N_XY 0
#endif
#ifndef b_splatTextureProjectorEnabled
#define b_splatTextureProjectorEnabled 0
#endif

// ---------------------------------------------------------------------------
// model.fx : local helpers
// ---------------------------------------------------------------------------

// model.fx : DecodeZOffset — unpack a 16-bit terrain height from two ubyte4n lanes.
float vsDecodeZOffset(float x, float y)
{
    float z = (x * 255.0f * 256.0f / 65535.0f + y * 255.0f / 65535.0f);
    return z * (c_terrainMaxHeight - c_terrainMinHeight) + c_terrainMinHeight;
}

// model.fx : ComputeTriPlanarTangentSpace — build a tangent frame by blending the
// three axis-aligned projections, and stash the handedness in vNorm.w.
void vsComputeTriPlanarTangentSpace(float3 weights, inout float4 vNorm,
                                    out float3 vTangent, out float3 vBinormal)
{
    const float3 y = float3(0, 1, 0), z = float3(0, 0, 1);

    vTangent = float3(0, 0, 0);
    vBinormal = float3(0, 0, 0);

    // xy
    float3 t = cross(y, vNorm.xyz);
    float3 b = cross(t, vNorm.xyz);
    vTangent += t * weights.z;
    vBinormal += b * weights.z;

    // xz
    t = cross(z, vNorm.xyz);
    b = cross(t, vNorm.xyz);
    vTangent += t * weights.y;
    vBinormal += b * weights.y;

    // yz
    t = cross(z, vNorm.xyz);
    b = cross(t, vNorm.xyz);
    vTangent += t * weights.x;
    vBinormal += b * weights.x;

    vTangent = normalize(vTangent);
    vBinormal = normalize(vBinormal);

    float3 righthandedBinormal = cross(vNorm.xyz, vTangent.xyz);
    float righthanded = dot(vBinormal, righthandedBinormal);
    vNorm.w = sign(righthanded);
}

// ---------------------------------------------------------------------------
// model.fx : VERTEX SHADER EMITTERS
// ---------------------------------------------------------------------------

// Homogeneous position.
float4 vsEmitModelHPos(ModelVSVertex vertIn)
{
    return mul(vertIn.vPosition, vsGlobals.mVPTransform);
}

// World position.
float4 vsEmitModelWorldPos(ModelVSVertex vertIn)
{
    return float4(vertIn.vPosition.xyz, 1.0f);
}

// Camera-space position (p_mViewTransform is stored transposed, as a 3x4).
float3 vsEmitModelViewPos(ModelVSVertex vertIn)
{
    return mul(vsGlobals.mViewTransform, vertIn.vPosition);
}

// Normal; .w is the handedness.  Terrain always uses a right-handed basis.
float4 vsEmitModelNormal(ModelVSVertex vertIn)
{
    if (b_iShadingMode == VS_SHADINGMODE_TERRAIN)
        return float4(vertIn.vNormal.xyz, -1);
    else
        return vertIn.vNormal;
}

// Vertex ambient occlusion: legacy data carries it in the vertex colour alpha, new
// data in the custom ubyte4n stream.
float4 vsEmitModelAOData(ModelVSVertex vertIn)
{
    float4 value = float4(0, 0, 0, 0);
    if (SC2_AXIS(b_iVertexAOFromVertexColor))
    {
#if b_iLayoutHasColor
        value = vertIn.cColor.a;
#endif
    }
    else
    {
#if b_iLayoutHasCustomUB4N1
        value = vertIn.cCustomUB4N1;
#endif
    }
    return value;
}

float4 vsEmitTriPlanarWeights(ModelVSVertex vertIn, float3 vTriPlanarWeightsLocal,
                              float3 vTriPlanarWeightsWorld)
{
    if (b_iShadingMode != VS_SHADINGMODE_TERRAIN)
        return float4(vTriPlanarWeightsLocal.xyz, vTriPlanarWeightsWorld.x);
    else
        return float4(vTriPlanarWeightsWorld.xyz, 0);   // terrain is always world space
}

// Tangent.
float4 vsEmitModelTangent(ModelVSVertex vertIn, float3 vTriPlanarWeightsWorld)
{
    if (b_iShadingMode == VS_SHADINGMODE_TERRAIN)
    {
        return float4(vsComputeGridTangent(vertIn.vNormal.xyz), 1);
    }
    else if (SC2_AXIS(b_splatTextureProjectorEnabled))
    {
        float3 b = cross(vertIn.vNormal.xyz, vsGlobals.vSplatTextureUVector.xyz);
        float3 t = cross(b, vertIn.vNormal.xyz);
        return float4(normalize(t), 1);
    }
    else
    {
        // :TODO: (original) This will break parallax mapping - shouldn't normalize.
        return float4(normalize(vertIn.vTangent.xyz), vTriPlanarWeightsWorld.y);
    }
}

// Binormal.
float3 vsEmitModelBinormal(ModelVSVertex vertIn, float3 vTangent)
{
    if (b_iShadingMode == VS_SHADINGMODE_TERRAIN)
        return normalize(cross(vTangent.xyz, vertIn.vNormal.xyz));
    else if (SC2_AXIS(b_splatTextureProjectorEnabled))
        return normalize(cross(vTangent.xyz, vertIn.vNormal.xyz));
    else
        // sign of INTERPOLANT_Normal.w is the handedness
        return cross(vertIn.vNormal.xyz, vTangent.xyz) * vertIn.vNormal.w;
}

// UVs — one emitter slot.  `M` is b_iUVMapping[I], a per-instance literal the
// engine's InitShader fills in (see types/model_vs_io.slang).
float4 vsEmitModelUV<let M : int, let I : int>(ModelVSVertex vertIn,
                                               float3 vLocalPosition,
                                               float4 vLocalNormal,
                                               float3 vTriPlanarWeightsWorld)
{
    return vsGenUV<M, I>(
        vLocalPosition,
        vertIn.vPosition.xyz,
        vLocalNormal.xyz,
        vertIn.vNormal.xyz,
        vertIn.UVCoord0(),
        vertIn.UVCoord1(),
        vertIn.UVCoord2(),
        vertIn.UVCoord3(),
        // index 0 is forced: splats rendering on models don't batch, so this pins
        // the right slot in the texture-projection matrix array.
        0,
        vTriPlanarWeightsWorld);
}

// Vertex colour.
float4 vsEmitModelVertexColor(ModelVSVertex vertIn)
{
    // vector ui splat
    if (b_iShadingMode == VS_SHADINGMODE_VECTORUI)
    {
        return vsGlobals.vVectorUIColor;
    }
    // triangle id
    else if (b_iShadingMode == VS_SHADINGMODE_MODEL_TRIANGLE_ID)
    {
        float4 id = float4(0, 0, 0, 0);
#if b_iLayoutHasColor
        id = vertIn.cColor;
#endif
        return id;
    }
    // material splat on model
    else if (SC2_AXIS(b_splatTextureProjectorEnabled))
    {
        float a = vsSplatAttenuationProjector(
            vertIn.vPosition.xyz, vsGlobals.vSplatAttenuationPlane[0],
            vsGlobals.fSplatAttenuationScalar_MinHeight[0].x,
            vsGlobals.fSplatAttenuationScalar_MinHeight[0].y);
        // do not use model vertex colour
        return float4(1, 1, 1, a);
    }
    // normal rendering
    else
    {
        float4 value = float4(0, 0, 0, 0);
#if b_iLayoutHasColor
        value = vertIn.cColor;
#endif
        return value;
    }
}

// Eye to vertex.
float3 vsEmitModelEyeToVertex(ModelVSVertex vertIn)
{
    return normalize(vertIn.vPosition.xyz - vsGlobals.vEyePos);
}

// FOW UVs.
float4 vsEmitModelFOWUV(ModelVSVertex vertIn)
{
    float4 vRet = mul(vertIn.vPosition, vsGlobals.mFOWMatrix);
    // fade out creep on vertical triangles
    if (b_iShadingMode == VS_SHADINGMODE_TERRAIN)
        vRet.z = saturate(abs(vertIn.vNormal.z) * 5);
    return vRet;
}

// Terrain blended-texture UVs and creep texture UVs.
float4 vsEmitTerrainCreepTextureUV(ModelVSVertex vertIn)
{
    if (!SC2_AXIS(b_computeHeight))
    {
        float2 t = mul(vertIn.vPosition, vsGlobals.mTerrainTextureMatrix).xy;
        float2x2 rot = float2x2(vsGlobals.vCreepTextureRotation.x,
                                vsGlobals.vCreepTextureRotation.y,
                                vsGlobals.vCreepTextureRotation.z,
                                vsGlobals.vCreepTextureRotation.w);
        return float4(t, mul(rot, float2(t.x, -t.y)));
    }
    else
    {
        return vertIn.vPosition.z;
    }
}

// ---------------------------------------------------------------------------
// UV-emitter unroll.  model.fx loops `for (i = 0; i < b_iUVEmitterCount; i++)` over
// a compile-time count (and the packed variant halves it, pairing two emitters'
// .xy into one float4 slot).  Both bounds are literals, so the loop is expanded
// here; each slot instantiates vsGenUV with its own literal mapping mode.
// ---------------------------------------------------------------------------
// The write is a macro, exactly like the engine's WRITE_INTERPOLANT_UV(i, v): it
// assigns the array field in place (a `[mutating]` helper would lower to a
// whole-struct inout and trip fxc's "output parameter not completely initialized"
// on a half-filled vertOut), and it vanishes entirely when the UV array was packed
// out — the emitter loop still runs in the original, but every store is a no-op.
#if SC2_HAS_UV
  #define SC2_EMIT_UV(dst, i)                                                        \
    vertOut.m_UV[dst] = vsEmitModelUV<SC2_UVMAPPING##i, i>(                          \
        vertIn, vLocalPosition, vLocalNormal, vTriPlanarWeightsWorld)

  #define SC2_EMIT_UV_PAIR(dst, i, j)                                                \
    vertOut.m_UV[dst] = float4(                                                      \
        vsEmitModelUV<SC2_UVMAPPING##i, i>(vertIn, vLocalPosition, vLocalNormal,     \
                                           vTriPlanarWeightsWorld).xy,               \
        vsEmitModelUV<SC2_UVMAPPING##j, j>(vertIn, vLocalPosition, vLocalNormal,     \
                                           vTriPlanarWeightsWorld).xy)
#else
  #define SC2_EMIT_UV(dst, i)
  #define SC2_EMIT_UV_PAIR(dst, i, j)
#endif

// ---------------------------------------------------------------------------
// Main vertex shader body.
// ---------------------------------------------------------------------------
[shader("vertex")]
public DefaultVSOutput model_vs_main(ModelVSStreamIn streamIn)
{
    DefaultVSOutput vertOut;

    // Translate the vertex stream format into our vertex-shader vertex format.
    ModelVSVertex vertIn = vsTranslateVert(streamIn);

    // --- preprocess the vertex's local-space data --------------------------
    // Preserve the local position (for UV mappings based on it).
    float3 vLocalPosition = vertIn.vPosition.xyz;
    if (SC2_AXIS(b_iZOffsetFromUB4N_XY))
    {
#if b_iLayoutHasCustomUB4N1
        vLocalPosition.z -= vsDecodeZOffset(vertIn.cCustomUB4N1.x, vertIn.cCustomUB4N1.y);
#endif
    }
    float3 vTriPlanarWeightsLocal = float3(0, 0, 0);
    if (SC2_AXIS(b_iUseLocalTriPlanarWeights))
        vTriPlanarWeightsLocal = vsComputeTriPlanarBlendingWeights(vertIn.vNormal.xyz);
    // Generate procedural triplanar UVs in local space.
    if (SC2_AXIS(b_iUseLocalTriPlanarTangentSpace))
    {
        float4 vNorm = vertIn.vNormal;
        float3 vTan, vBin;
        vsComputeTriPlanarTangentSpace(vTriPlanarWeightsLocal, vNorm, vTan, vBin);
        vertIn.vNormal = vNorm;
        vertIn.vTangent = vTan;
        vertIn.vBinormal = vBin;
    }
    float4 vLocalNormal = vertIn.vNormal;

    // --- transform the vertex into world space -----------------------------
    {
        float4 vPos = vertIn.vPosition;
        float3 vNrm = vertIn.vNormal.xyz;
        float3 vTan = vertIn.vTangent;
        float3 vBin = vertIn.vBinormal;
#if b_iLayoutHasNoVertexBlendWeights
        vsSkin(vPos, vNrm, vTan, vBin, float4(0, 0, 0, 0), float4(0, 0, 0, 0));
#else
        vsSkin(vPos, vNrm, vTan, vBin, vertIn.BlendWeights(), vertIn.BlendIndices());
#endif
        vertIn.vPosition = vPos;
        vertIn.vNormal.xyz = vNrm;
        vertIn.vTangent = vTan;
        vertIn.vBinormal = vBin;
    }

    // The vertex is in world space now — generate procedural triplanar UVs there.
    float3 vTriPlanarWeightsWorld = float3(0, 0, 0);
    if (SC2_AXIS(b_iUseWorldTriPlanarWeights))
        vTriPlanarWeightsWorld = vsComputeTriPlanarBlendingWeights(vertIn.vNormal.xyz);
    if (SC2_AXIS(b_iUseWorldTriPlanarTangentSpace))
    {
        float4 vNorm = vertIn.vNormal;
        float3 vTan, vBin;
        vsComputeTriPlanarTangentSpace(vTriPlanarWeightsWorld, vNorm, vTan, vBin);
        vertIn.vNormal = vNorm;
        vertIn.vTangent = vTan;
        vertIn.vBinormal = vBin;
    }

    // Transform instance.
    if (b_useModelInstancing)
        vertIn.vPosition.xyz = mul(float4(vertIn.vPosition.xyz, 1),
                                   vsGlobals.mInstanceTransform).xyz;

    if (b_enableVertexWarps == 1)
        vertIn.vPosition = vsApplyWarps(vertIn.vPosition);

    vertOut.HPos = vsEmitModelHPos(vertIn);

    float4 interpTangent = vsEmitModelTangent(vertIn, vTriPlanarWeightsWorld);
    float3 interpBinormal = vsEmitModelBinormal(vertIn, interpTangent.xyz);

#if SC2_HAS_HPosAsUV
    vertOut.m_HPosAsUV = mul(vertIn.vPosition, vsGlobals.mVPTransform);
#endif
#if SC2_HAS_WorldPos
    vertOut.m_WorldPos = vsEmitModelWorldPos(vertIn);
#endif
#if SC2_HAS_ViewPos
    vertOut.m_ViewPos = vsEmitModelViewPos(vertIn);
#endif
#if SC2_HAS_Normal
    vertOut.m_Normal = vsEmitModelNormal(vertIn);
#endif
#if SC2_HAS_Tangent
    vertOut.m_Tangent = interpTangent;
#endif
#if SC2_HAS_Binormal
    vertOut.m_Binormal = interpBinormal;
#endif
#if SC2_HAS_VectorUI1
    vertOut.m_VectorUI1 = vsEmitModelAOData(vertIn);
#endif

    // Creep rotates the texture, so the tangent frame has to rotate with it.
    // (WRITE_INTERPOLANT_SWIZZLE_* — a no-op when that varying was packed out.)
    if (SC2_AXIS(b_creepEnabled))
    {
        float2x2 invRot = float2x2(vsGlobals.InvCreepTextureRotation.x,
                                   vsGlobals.InvCreepTextureRotation.y,
                                   vsGlobals.InvCreepTextureRotation.z,
                                   vsGlobals.InvCreepTextureRotation.w);
#if SC2_HAS_Tangent
        vertOut.m_Tangent.xy = mul(invRot, interpTangent.xy);
#endif
#if SC2_HAS_Binormal
        vertOut.m_Binormal.xy = mul(invRot, interpBinormal.xy);
#endif
    }

    // We may be using vertex alpha for something else, so recompute the 4th region.
    float4 vLightingRegions = float4(0, 0, 0, 0);
#if b_iLayoutHasColor
    vLightingRegions = vertIn.cColor;
    vLightingRegions.a = 1.0 - (vLightingRegions.r + vLightingRegions.g + vLightingRegions.b);
#endif

#if SC2_HAS_VertexColor
    vertOut.m_VertexColor = vsEmitModelVertexColor(vertIn);
#endif

    // --- vertex lighting varyings ------------------------------------------
    // The three scratch values are threaded between the emitters exactly as in the
    // original: EmitDiffuse fills cVertexSpecularLighting (which EmitSpecular
    // returns) and zeroes cVertexShadowSpecularLighting, then EmitShadowDiffuse
    // fills the latter (which EmitShadowSpecular returns).  Each call sits under
    // its own varying's gate, matching GenInterpolant's no-op expansion — so a
    // permutation that packs a varying out never evaluates its emitter.
#if (SC2_HAS_Diffuse || SC2_HAS_Specular || SC2_HAS_ShadowDiffuse || SC2_HAS_ShadowSpecular)
    float3 cVertexSpecularLighting = float3(0, 0, 0);
    float3 cVertexShadowSpecularLighting = float3(0, 0, 0);
    float3 cVertexShadowSpecularLighting2 = float3(0, 0, 0);
  #if SC2_HAS_Diffuse
    vertOut.m_Diffuse = vsEmitDiffuse(vertOut, vertIn.vPosition.xyz,
                                      vertIn.vNormal.xyz, vLightingRegions,
                                      cVertexSpecularLighting,
                                      cVertexShadowSpecularLighting,
                                      cVertexShadowSpecularLighting2);
  #endif
  #if SC2_HAS_Specular
    vertOut.m_Specular = vsEmitSpecular(cVertexSpecularLighting);
  #endif
  #if SC2_HAS_ShadowDiffuse
    vertOut.m_ShadowDiffuse = vsEmitShadowDiffuse(vertIn.vNormal.xyz, vLightingRegions,
                                                  cVertexShadowSpecularLighting,
                                                  cVertexShadowSpecularLighting2);
  #endif
  #if SC2_HAS_ShadowSpecular
    vertOut.m_ShadowSpecular = vsEmitShadowSpecular(cVertexShadowSpecularLighting);
  #endif
#endif

#if SC2_HAS_ShadowMapUV
    vertOut.m_ShadowMapUV = vsEmitShadowMapUV(vertIn.vPosition.xyz);
#endif
#if SC2_HAS_FogColor
    vertOut.m_FogColor = vsEmitFogColor(vertIn.vPosition.xyz, vTriPlanarWeightsWorld);
#endif
#if SC2_HAS_HalfVec
    vertOut.m_HalfVec = vsComputeHalfVector(vertIn.vPosition.xyz);
#endif
#if SC2_HAS_EyeToVertex
    vertOut.m_EyeToVertex = vsEmitModelEyeToVertex(vertIn);
#endif
#if SC2_HAS_EyeToVertexFresnel
    vertOut.m_EyeToVertexFresnel = vsEmitModelEyeToVertex(vertIn);
#endif
#if SC2_HAS_FOWUV
    vertOut.m_FOWUV = vsEmitModelFOWUV(vertIn);
#endif
#if SC2_HAS_TerrainUV
    vertOut.m_TerrainUV = vsEmitTerrainCreepTextureUV(vertIn);
#endif
#if SC2_HAS_VectorUI0
    vertOut.m_VectorUI0 = vLocalNormal;             // EmitModelNormalLocal
#endif

    // --- per-emitter UVs ---------------------------------------------------
#if b_usePackedUVEmitter
  #if (b_iUVEmitterCount / 2 > 0)
    SC2_EMIT_UV_PAIR(0, 0, 1);
  #endif
  #if (b_iUVEmitterCount / 2 > 1)
    SC2_EMIT_UV_PAIR(1, 2, 3);
  #endif
  #if (b_iUVEmitterCount / 2 > 2)
    SC2_EMIT_UV_PAIR(2, 4, 5);
  #endif
  #if (b_iUVEmitterCount / 2 > 3)
    SC2_EMIT_UV_PAIR(3, 6, 7);
  #endif
  // odd tail: slot count/2 takes emitter count-1 whole
  #if (b_iUVEmitterCount == 1)
    SC2_EMIT_UV(0, 0);
  #elif (b_iUVEmitterCount == 3)
    SC2_EMIT_UV(1, 2);
  #elif (b_iUVEmitterCount == 5)
    SC2_EMIT_UV(2, 4);
  #elif (b_iUVEmitterCount == 7)
    SC2_EMIT_UV(3, 6);
  #endif
#else
  #if (b_iUVEmitterCount > 0)
    SC2_EMIT_UV(0, 0);
  #endif
  #if (b_iUVEmitterCount > 1)
    SC2_EMIT_UV(1, 1);
  #endif
  #if (b_iUVEmitterCount > 2)
    SC2_EMIT_UV(2, 2);
  #endif
  #if (b_iUVEmitterCount > 3)
    SC2_EMIT_UV(3, 3);
  #endif
  #if (b_iUVEmitterCount > 4)
    SC2_EMIT_UV(4, 4);
  #endif
  #if (b_iUVEmitterCount > 5)
    SC2_EMIT_UV(5, 5);
  #endif
  #if (b_iUVEmitterCount > 6)
    SC2_EMIT_UV(6, 6);
  #endif
  #if (b_iUVEmitterCount > 7)
    SC2_EMIT_UV(7, 7);
  #endif
#endif

#if SC2_HAS_ParallaxVector
    vertOut.m_ParallaxVector = vsEmitParallaxVector(
        vertIn.vPosition.xyz, vsGlobals.vEyePos,
        vertOut.Normal().xyz, vertOut.Tangent().xyz, vertOut.Binormal());
#endif
#if SC2_HAS_TriPlanarWeights
    vertOut.m_TriPlanarWeights = vsEmitTriPlanarWeights(vertIn, vTriPlanarWeightsLocal,
                                                        vTriPlanarWeightsWorld);
#endif

    // Vector-UI splats overwrite the generic vectors with their shape parameters.
    if (b_iShadingMode == VS_SHADINGMODE_VECTORUI)
    {
#if SC2_HAS_VectorUI0
        vertOut.m_VectorUI0 = vsGlobals.vVectorUIInterpolant0[0];
#endif
#if SC2_HAS_VectorUI1
        vertOut.m_VectorUI1 = vsGlobals.vVectorUIInterpolant1[0];
#endif
#if SC2_HAS_VectorUI2
        vertOut.m_VectorUI2 = vsGlobals.vVectorUIInterpolant2[0];
#endif
#if SC2_HAS_Vector4
        vertOut.m_Vector4 = vsGlobals.vSplatAttenuationPlane[0];
#endif
    }

    return vertOut;
}
