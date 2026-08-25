#!/usr/bin/env python3
"""Shared config + manifest access for the sc2_shaders pipeline.

One place for the three sc2_shaders driver tools (validator / slang-compiler /
BLS bundler) to resolve:
  * the family table (sc2_shaders.json: fx file + per-stage fx/slang entries),
  * each (family, stage) permutation manifest (sc2_perms/<Family>_<stage>.json,
    the cache-order slot list with decoded (bv, live) — see sc2_perm_manifest),
  * the b_* -> slangc `-D` define convention (nonzero axes only, mirroring
    wc3_shaders' `-DFLAG=1`-when-on pattern; the `#if b_*` gate reads absent as 0).
"""
import os
import sys
import json

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import sc2_compile_perms as _cp

SC2_CONFIG = os.path.join(REPO_ROOT, "sc2_shaders.json")
# The module root + include dir default to the live tree, but honour SC2_MODULE /
# SC2_INCLUDE env overrides so a validation run can point at a FROZEN snapshot of
# sc2_shaders/ — isolating it from a concurrent session editing the shared module
# (the whole module is one translation unit, so a mid-edit save fails every in-flight
# compile).  Freeze with a plain copytree, then export both vars.
SC2_MODULE = os.environ.get(
    "SC2_MODULE", os.path.join(REPO_ROOT, "sc2_shaders", "sc2_shaders.slang"))
SC2_INCLUDE = os.environ.get(
    "SC2_INCLUDE", os.path.join(REPO_ROOT, "sc2_shaders"))
PERMS_DIR = os.path.join(REPO_ROOT, "sc2_perms")
FX_SRC = _cp.SRC            # mods/.../shaders


def load_families():
    """{family: {fx, vs:{fx_entry,slang_entry,...}, ps:{...}}}."""
    with open(SC2_CONFIG) as fp:
        return json.load(fp)["families"]


def family_cfg(family):
    fams = load_families()
    if family not in fams:
        raise KeyError("family %r not in %s (have: %s)"
                       % (family, SC2_CONFIG, ", ".join(sorted(fams))))
    return fams[family]


def fx_path(family):
    return os.path.join(FX_SRC, family_cfg(family)["fx"])


def read_manifest(family, stage):
    """The cache-order permutation manifest for (family, stage), or None."""
    path = os.path.join(PERMS_DIR, "%s_%s.json" % (family, stage))
    if not os.path.exists(path):
        return None
    return json.load(open(path))


def bv_to_defines(bv):
    """slangc `-D` list for a decoded b_* vector: `NAME=value` for every nonzero
    axis (the `#if NAME` / `NAME`-valued gates read an absent define as 0, so
    zero axes are omitted — matches wc3_shaders' convention)."""
    return ["%s=%d" % (k, int(v)) for k, v in sorted(bv.items()) if int(v)]


# ---------------------------------------------------------------------------
# Canonical interpolant slots (design section 4)
# ---------------------------------------------------------------------------
# Today the live interpolants are packed into TEXCOORD0,1,2... sorted by name, so
# the register assignment depends on WHICH interpolants are live -- which welds each
# PS permutation to one VS permutation and costs 970 distinct layouts for Model
# alone.  Giving every interpolant a FIXED semantic index decouples the stages: D3D11
# links a VS output to a PS input by semantic NAME, so any PS reading a SUBSET of
# what the VS wrote links correctly, and the VS may then use a coarser transport
# profile than the PS.
#
# The index is not a register number.  Registers are assigned by declaration order,
# so a shader with three live interpolants still uses v0..v2 whatever their indices
# are; only the linkage names change.  That is why indices above 31 are legal and
# why this costs nothing.
#
# Slots must not collide AFTER slangc's x10 index inflation (see
# sc2_slang_validate._norm_idx): a scalar at slot S is emitted as TEXCOORD(S*10) and
# an array element e of a base-S array as TEXCOORD(S*10 + e), both of which
# normalise back to S and S+e.  So the two ARRAYS are given bases far enough apart
# that their element ranges cannot land on a scalar's slot.
SC2_CANON_SLOT = {
    # --- geometry basis -----------------------------------------------------
    "Normal": 0, "Tangent": 1, "Binormal": 2,
    # --- view / lighting vectors -------------------------------------------
    "EyeToVertex": 3, "EyeToVertexFresnel": 4, "HalfVec": 5, "ViewPos": 6,
    # --- shadow / fog-of-war ------------------------------------------------
    "ShadowMapUV": 7, "FOWUV": 8, "FogColor": 9,
    # --- vertex-stage lighting results -------------------------------------
    "Diffuse": 10, "Specular": 11, "ShadowDiffuse": 12, "ShadowSpecular": 13,
    "VertexColor": 14,
    # --- triplanar / vector-UI ---------------------------------------------
    "TriPlanarWeights": 15, "VectorUI0": 16, "VectorUI1": 17, "VectorUI2": 18,
    "Vector4": 19,
    # --- world / terrain ----------------------------------------------------
    "WorldPos": 20, "TerrainUV": 21,
    # --- parallax + screen-space feeds -------------------------------------
    "ParallaxVector": 22, "HPosAsUV": 23, "BackBufferUV": 24,
    "DownscaleUV0": 25, "DownscaleUV1": 26,
}
# The per-emitter UV array (<= 5 entries) and postprocessquad's per-tap blur-offset
# array (<= 8) get bases whose element ranges are disjoint from the scalars and from
# each other.
SC2_CANON_UV_BASE = 32
SC2_CANON_GBS_BASE = 40


def use_canon_slots():
    """Canonical slots are on by default; SC2_CANON_INTERP=0 restores the legacy
    sorted-live-set packing.

    Both legs of the validation harness read this ONE switch, because the
    comparator aligns a varying by (semantic, index): moving only the candidate to
    canonical slots would make each leg draw a different random value for the same
    interpolant, which looks like a shading bug and is not one."""
    return os.environ.get("SC2_CANON_INTERP", "1") != "0"


def canon_slot_check():
    """No two interpolants may share a normalised slot.  Called by the tests rather
    than at import so a bad edit fails loudly instead of at the first compile."""
    used = {}
    for n, s in SC2_CANON_SLOT.items():
        if s in used:
            raise ValueError("canonical slot %d used by both %s and %s"
                             % (s, used[s], n))
        used[s] = n
    for base, count, name in ((SC2_CANON_UV_BASE, 5, "UV"),
                              (SC2_CANON_GBS_BASE, 8, "GaussianBlurSample")):
        for e in range(count):
            if base + e in used:
                raise ValueError("array %s element %d lands on %s's slot %d"
                                 % (name, e, used[base + e], base + e))
            used[base + e] = "%s[%d]" % (name, e)
    return len(used)


def interp_defines(live, bv, stage="ps"):
    """slangc `-D` list describing the live interpolant transport for the shared-
    interpolant (DefaultPixelMain) families.

    Mirrors sc2_interp.gen_preamble's packing EXACTLY so the slang candidate's
    VS->PS interpolant register assignment matches the fxc reference's: the live
    scalar interpolants (INTERP_DIM) are sorted by name and assigned TEXCOORD0,1,2…
    in that order; the per-emitter UV array (if live) follows at the next TEXCOORD,
    sized like gen_preamble (max(1, UV emitter count)).  For each we emit
    `SC2_HAS_<name>=1` (presence gate) and `SC2_SEM_<name>=TEXCOORD<i>` (the exact
    semantic, passed whole so the slang struct needs no token-pasting); UV also
    gets `SC2_UV_COUNT`.

    Both stages share this packing — gen_preamble builds the VS `VertexTransport`
    and the PS `VertexTransportRaw` from the same sorted live set — so the same
    defines drive the VS output struct and the PS input struct.  The `stage` split
    only concerns the non-TEXCOORD specials: SV_IsFrontFace is a pixel-stage system
    value and must never appear in a vertex output signature."""
    import sc2_interp as ip
    scal = sorted(n for n in live if n in ip.INTERP_DIM)
    defs = []
    canon = use_canon_slots()
    for i, n in enumerate(scal):
        slot = SC2_CANON_SLOT[n] if canon else i
        defs += ["SC2_HAS_%s=1" % n, "SC2_SEM_%s=TEXCOORD%d" % (n, slot)]
    tc = len(scal)
    if "UV" in live:
        uvc = max(1, ip._uv_count(bv))
        defs += ["SC2_HAS_UV=1",
                 "SC2_SEM_UV=TEXCOORD%d" % (SC2_CANON_UV_BASE if canon else tc),
                 "SC2_UV_COUNT=%d" % uvc]
        tc += uvc
    # The second array interpolant (postprocessquad.fx's per-tap blur offsets) sits
    # after the UV array — same order gen_preamble uses.
    if "GaussianBlurSample" in live:
        defs += ["SC2_HAS_GaussianBlurSample=1",
                 "SC2_SEM_GaussianBlurSample=TEXCOORD%d"
                 % (SC2_CANON_GBS_BASE if canon else tc),
                 "SC2_GBS_COUNT=%d" % ip._gbs_count(bv)]
    # SV_IsFrontFace is a system-value input (no TEXCOORD slot), so it's gated
    # independently of the scalar packing.  gen_preamble: FrontFace live ->
    # INTERPOLANT_FrontFace = vertOut.FrontFace; else the safety-net `true`.
    # It is a PS-only input (gen_preamble adds it to VertexTransportRaw only).
    if stage == "ps" and "FrontFace" in live:
        defs.append("SC2_HAS_FrontFace=1")
    return defs


def uv_mappings(bv):
    """The engine-filled `int b_iUVMapping[8]` array for one permutation.

    model.fx declares it as a LOCAL array and has `InitShader` fill it (it is a
    per-instance array, not a `#define`), so it cannot ride along in the `-D` set
    like the scalar axes.  The reference leg takes it through
    `gen_preamble(..., uv_mappings=…)`; the candidate leg gets one
    `SC2_UVMAPPING<i>=<mode>` define per slot (see `perm_defines`).  Both read the
    same MainShading-section axes, so the two legs agree by construction."""
    return [int(bv.get("b_iUVMapping%d" % i, 0)) for i in range(8)]


def uv_random_offsets(bv):
    """The engine-filled `int b_UVRandomOffsetEnable[8]` array for one permutation.

    Same shape as `uv_mappings` — an InitShader-filled array, not a `#define`.
    Only particle.fx READS it (it nudges the UV by a per-particle random byte pair
    unpacked from vRotation.w); the other Default roots merely forward it, which is
    why it could safely default to all-zero until the Particle root landed."""
    return [int(bv.get("b_UVRandomOffsetEnable%d" % i, 0)) for i in range(8)]


# Families whose PS uses the shared DefaultPixelMain interpolant transport.
# SplatDirect/SplatDeferred tail-call DefaultPixelMain, so their PS packs the same
# VertexTransportRaw from the same live-interpolant set.  Water is an own-root PS
# body but still builds its VertexTransport via the shared InitShader (its VS writes
# the same INTERPOLANT_* set), so it packs the same live transport.  (TerrainBlend is
# NOT here: it carries its own IO structs, so its live set is empty.)
_SHARED_TRANSPORT = {"Model", "Particle", "Ribbon", "Foliage",
                     "SplatDirect", "SplatDeferred", "Water",
                     # M4: deferredlight.fx and postprocessquad.fx are own-ROOT but
                     # not own-IO — both call the engine's InitShader/VertexTransport,
                     # so their VS output and PS input are packed from the live set
                     # exactly like the Default families'.
                     "DeferredLight", "PostProcessQuad"}


def perm_defines(family, stage, bv, live):
    """Full slangc `-D` set for one permutation slot: the b_* axes, plus (for the
    shared-transport families) the interpolant-transport defines — the VS output
    struct and the PS input struct are packed from the same live set — and, for
    the vertex stage, the per-emitter UV-mapping array."""
    defs = bv_to_defines(bv)
    if family in _SHARED_TRANSPORT:
        defs += interp_defines(live, bv, stage)
        if stage == "vs":
            # The whole module is ONE translation unit, so slangc preprocesses the
            # vertex root even when the pixel entry is requested (and vice versa).
            # SC2_STAGE_VS lets stage-specific preprocessor logic — notably the
            # "feature not transcribed yet" guards — fire only for its own stage.
            defs += ["SC2_STAGE_VS=1"]
            defs += ["SC2_UVMAPPING%d=%d" % (i, m)
                     for i, m in enumerate(uv_mappings(bv))]
            defs += ["SC2_UVRANDOM%d=%d" % (i, m)
                     for i, m in enumerate(uv_random_offsets(bv))]
    return defs


# shadersystem.fx : MAX_SPLATS — the per-splat uniform array bound the batch-index
# remapping table has to land inside.
MAX_SPLATS = 8

# ---------------------------------------------------------------------------
# Engine-legal value domains for the behavioural comparator.
# ---------------------------------------------------------------------------
# Attributes/constants a family uses as ARRAY INDICES rather than as numbers.
# Feeding them uniform noise indexes outside the declared array, which either
# aborts the interpreter or (worse) silently reads a neighbouring constant — and
# because the two legs lay out their cbuffers independently, "a neighbouring
# constant" is a DIFFERENT constant on each leg, i.e. a false failure.
# See sc2_slang_validate._VS_INPUT_DOMAINS for the domain kinds.
VS_INPUT_DOMAINS = {
    # Ribbon/Particle batch the whole system into one draw and select the batch
    # with a per-vertex integer attribute (`BatchIndexType iBatchIndex` = uint4).
    # It indexes the remapping table (32) and, after remapping, the per-batch
    # arrays (MAX_BATCHED_RIBBONS = 8) — so draw from the tighter of the two.
    "Ribbon":   {"TEXCOORD6": ("uint", 8)},
    # Particle puts its batch index at TEXCOORD4 (Ribbon uses TEXCOORD6, which on
    # Particle is vInterpolator2 — a normal float attribute).  It also declares
    # vSize/vRotation (int4) and vOffset (int2) as true integer attributes, whose
    # registers hold integer BIT PATTERNS: float noise there decodes to ~1e38 sizes.
    "Particle": {"TEXCOORD4": ("uint", 8),
                 "TEXCOORD0": ("sint", 4096),     # vSize    (scaled by 1/256)
                 "TEXCOORD2": ("sint", 4096),     # vRotation(scaled by 1/32)
                 "NORMAL":    ("sint", 2)},       # vOffset  (quad corner, -1/0/1)
    # SPLAT_VERTEX_FORMAT makes vBlendIndices a TRUE `uint4` (vsmodelvertexformat.fx),
    # not the D3DCOLOR bytes the shared BLENDINDICES default assumes — the register
    # holds the integer itself.  splatdirect.fx uses [0] two ways: as the index into
    # p_fBatchIndexRemappingTable[32], and (under b_stencilFillPass) directly into
    # p_vSplatVolumeCorner[8*MAX_SPLATS], so the tighter bound of the two applies.
    "SplatDirect": {"BLENDINDICES": ("uint", 32)},
}

def _particle_flipbook_const(rng, nfloat):
    """p_vSystemTime_ElementScale_FlipbookMidKeyTime_FlipbookColumnCount[8].

    A packed float4 whose components are NOT interchangeable:
      .x system time, .y element scale  — any value works;
      .z flipbook mid-key time          — a fraction; particle.fx divides by
                                          (1.0 - z), so it must not be 1, and
                                          keeping it strictly inside (0,1) means
                                          BOTH flipbook arms stay reachable at
                                          runtime (the age is saturated to [0,1]);
      .w flipbook column count          — used as an INTEGER divisor and modulus
                                          (`iCell % (int)w`), so 0 would abort the
                                          interpreter outright.
    """
    out = []
    for i in range(nfloat):
        c = i & 3
        if c == 2:
            out.append(rng.uniform(0.05, 0.95))
        elif c == 3:
            out.append(float(rng.randrange(1, 9)))
        else:
            out.append(rng.uniform(-1, 1))
    return out


# Pixel-stage counterpart of VS_CONST_DOMAINS: constants a PIXEL shader uses as an
# index rather than as a number.  Same failure mode — each leg lays out its own
# cbuffer and its own indexable temp, so an out-of-range index is a DIFFERENT lane
# on each side and the comparison fails for a reason that can never happen in game.
PS_CONST_DOMAINS = {
    # image.fx picks the layer's alpha source with `cColor[(int)p_vLayerAlphaChannelIndex[i]]`
    # — a per-layer CHANNEL selector (r/g/b/a), one component per layer.
    "Image": {"vLayerAlphaChannelIndex": ("index", 4)},
}


VS_CONST_DOMAINS = {
    # `ArrayDecl(float) p_fRibbonBatchIndexRemappingTable[32]` — a float array the
    # shader truncates into a batch index, so its VALUES must be legal indices too.
    "Ribbon":   {"fRibbonBatchIndexRemappingTable": ("index", 8)},
    "Particle": {
        "fParticleBatchIndexRemappingTable": ("index", 8),
        "vSystemTime_ElementScale_FlipbookMidKeyTime_FlipbookColumnCount":
            _particle_flipbook_const,
    },
    # splatdirect.fx: `(int)(p_fBatchIndexRemappingTable[i] + 0.1)` selects the
    # per-splat uniform arrays (all [MAX_SPLATS]) and the splat projection matrix,
    # so the table's VALUES must themselves be legal splat indices.
    "SplatDirect": {"fBatchIndexRemappingTable": ("index", MAX_SPLATS)},
    # water.fx: `for (i = 0; i < p_fNumWaveVectors.x; i++)` over p_vWaveVectors[64].
    # This one is a LOOP BOUND, not just an index — an unconstrained draw asks the
    # interpreter for ~1e38 iterations.  A small count keeps the sweep cheap and both
    # legs still see the same value.
    "Water": {"fNumWaveVectors": ("index", 8)},
}


def iter_slots(family, stage):
    """Yield (slot, bv, live, dedup) per manifest slot in cache order.

    (bv, live) is decoded from each slot's stored KEY on demand (sc2_family_decode
    .decode_perm) rather than read from the manifest — the manifest stores compact
    keys for the big Default families, so this is the one path that scales from
    Simple (7) to Model (~50k).  Raises if the family's schema is unnamed so a
    half-decoded family fails loudly rather than validating the wrong define set."""
    import sc2_cache
    import sc2_family_decode as fd
    man = read_manifest(family, stage)
    if man is None:
        raise FileNotFoundError(
            "no manifest for %s %s (run sc2_perm_manifest.py --family %s)"
            % (family, stage, family))
    if not man.get("decoded"):
        raise ValueError(
            "%s %s manifest is not decoded (schema unnamed in sc2_family_decode)"
            % (family, stage))
    for p in man["perms"]:
        _, vec = sc2_cache.decode_key(bytes.fromhex(p["key"]))
        r = fd.decode_perm(family, stage, vec)
        if r is None:
            raise ValueError("%s %s slot %d failed to decode"
                             % (family, stage, p["slot"]))
        bv, live = r
        yield p["slot"], bv, list(live), int(p.get("dedup", 0))
