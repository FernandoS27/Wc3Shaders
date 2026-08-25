"""Reconstruct the SC2 engine's per-permutation interpolant preamble.

The shared-interpolant families (model / particle / ribbon / foliage / tread /
splat, and the screen-quad families) don't define their own `VertexTransport`;
the C++ shader interface injects, per permutation, a `VertexTransport` struct, a
`VertexTransportRaw` PS-input struct, an `InitShader` unpacker, and the
`WRITE_INTERPOLANT_* / INTERPOLANT_* / READ_INTERPOLANT_*` macro family — with
only the *live* interpolants (those the VS writes AND the PS reads).

We reproduce that here.  Liveness is recovered by really preprocessing the VS
and PS (via `fxc /P`) with the permutation's `b_*` defines and marker macros,
then intersecting the write set with the read set.  The result is a functionally
faithful preamble (one register per live interpolant); it is not required to be
bit-identical to retail's register packing — only to compile to a shader with
the same behaviour, which is what the DXBC verification harness needs.
"""
import os
import re
import subprocess

# interpolant -> component count (from the Emit* return types / retail GLSL varyings)
INTERP_DIM = {
    "HPosAsUV": 4, "WorldPos": 4, "ViewPos": 3, "Normal": 4, "Tangent": 4,
    "Binormal": 3, "VectorUI0": 4, "VectorUI1": 4, "VectorUI2": 4,
    "VertexColor": 4, "Diffuse": 3, "Specular": 3, "ShadowDiffuse": 3,
    "ShadowSpecular": 3, "ShadowMapUV": 4, "FogColor": 4, "HalfVec": 3,
    "EyeToVertex": 3, "EyeToVertexFresnel": 3, "FOWUV": 4, "TerrainUV": 4,
    "ParallaxVector": 3, "TriPlanarWeights": 4, "Vector4": 4,
    # BackBufferUV is a float2 (GetBackBufferUV returns float2, and
    # sc2_model_key_decode.INTERP_TABLE32 records 2 components for it).  It had been
    # 4 here, which nothing noticed until deferredlight.fx — the only root that ever
    # makes it live — failed to compile its own reference:
    #   WRITE_INTERPOLANT_BackBufferUV(GetBackBufferUV(...)) -> X3017 float2 -> float4.
    "DownscaleUV0": 4, "DownscaleUV1": 4, "BackBufferUV": 2,
}
# array interpolants: name -> (dim, count-source is caller-supplied)
ARRAY_INTERP = {"UV": 4, "GaussianBlurSample": 4}
# read-only PS specials that are not TEXCOORD varyings
SPECIAL_READS = {"ScreenPos", "FrontFace", "View"}
HLSL_T = {2: "float2", 3: "float3", 4: "float4"}


def _preprocess(fxc, src_text, tmp_path, extra_args=()):
    open(tmp_path, "w", encoding="latin1").write(src_text)
    out = tmp_path + ".i"
    r = subprocess.run([fxc, "/nologo", "/P", out, *extra_args, tmp_path],
                       capture_output=True, text=True)
    if not os.path.exists(out):
        return None, (r.stdout + r.stderr)
    txt = open(out, encoding="latin1", errors="replace").read()
    os.remove(out)
    return txt, None


def _marker_defs(kind):
    """Marker macros so preprocessing reveals which interpolants survive.

    kind='write': WRITE_INTERPOLANT_X(...) / GenInterpolant name -> __W_X__
    kind='read' : INTERPOLANT_X            -> __R_X__
    """
    names = list(INTERP_DIM) + list(ARRAY_INTERP) + list(SPECIAL_READS)
    lines = []
    if kind == "write":
        for n in names:
            lines.append("#define WRITE_INTERPOLANT_%s(v) __W_%s__" % (n, n))
            lines.append("#define WRITE_INTERPOLANT_SWIZZLE_%s(s,v) __W_%s__" % (n, n))
        lines.append("#define WRITE_INTERPOLANT_UV(i,v) __W_UV__")
        lines.append("#define GenInterpolantUV(i,v) __W_UV__")
    else:
        for n in names:
            lines.append("#define INTERPOLANT_%s __R_%s__" % (n, n))
        lines.append("#define READ_INTERPOLANT_UV(i) __R_UV__")
        lines.append("#define READ_INTERPOLANT_GaussianBlurSample(i) __R_GaussianBlurSample__")
    return "\n".join(lines)


def _uv_count(b_values):
    n = int(b_values.get("b_iUVEmitterCount", 0))
    if b_values.get("b_usePackedUVEmitter", 0):
        n = (n + 1) // 2
    return n


def _gbs_count(b_values):
    """Length of the per-tap `GaussianBlurSample` array interpolant.

    postprocessquad.fx writes it as `for (i = 0; i < b_iSampleInterpolantCount; i++)`
    — the same axis INTERP_TABLE32 records as its gate.  Floored at 1 so a live-but-
    zero-count permutation still declares a well-formed array (mirroring _uv_count)."""
    return max(1, int(b_values.get("b_iSampleInterpolantCount", 0)))


def _initshader_arity(text, entry):
    epos = text.find(entry + "(")
    body = text[epos:] if epos >= 0 else text
    m = re.search(r"InitShader\s*\(([^)]*)\)", body)
    if not m:
        return -1
    inner = m.group(1).strip()
    return 0 if inner == "" else inner.count(",") + 1


def gen_initshader_stub(entry_text, entry):
    """The `InitShader` definition an OWN-IO family still needs (inject_preamble
    off), or "" when it needs none.

    Families that carry their own IO structs skip gen_preamble entirely, but a few
    still call the engine's zero-argument `InitShader()` — hdr.fx and image.fx do,
    purely as a per-draw hook with no interpolant transport behind it.  Emitting
    the empty definition is enough; the transport structs/macros stay out (they
    would only shadow the family's own `VertexOutput`/`SImageVtxFmt`).

    Returns "" for the arity-1/2/3 forms (those families need the FULL preamble,
    i.e. inject_preamble=True) and for entries that never call InitShader at all —
    so this is inert for renderplane/minimapterrain/lensflare/flash/terrainblend."""
    return "void InitShader() {}" if _initshader_arity(entry_text, entry) == 0 else ""


def gen_preamble(stage, live, b_values, entry_text, entry, uv_mappings=None,
                 uv_random_offsets=None):
    """Emit the engine-equivalent interpolant machinery for one permutation.

    `live` is the set of live interpolant names (VS-writes ∩ PS-reads).  The VS
    output struct and PS input struct are generated from the same sorted live
    set so their signatures match; the PS additionally exposes SV_Position (read
    as ScreenPos) and, when read, SV_IsFrontFace (FrontFace)."""
    scal = sorted(n for n in live if n in INTERP_DIM)
    has_uv = "UV" in live
    uvc = max(1, _uv_count(b_values)) if has_uv else 0
    has_gbs = "GaussianBlurSample" in live
    gbsc = _gbs_count(b_values) if has_gbs else 0
    L = []

    # Canonical interpolant slots (design section 4).  The REFERENCE leg has to move
    # with the candidate: the comparator aligns a varying by (semantic, index), so if
    # only one leg renumbered, the two would draw different random values for the
    # same interpolant and every affected permutation would diverge for a reason that
    # has nothing to do with shading.  The struct field ORDER is unchanged, so fxc
    # still assigns the same registers and emits the same code -- only the linkage
    # names differ.
    import sc2_shaders_cfg as _cfg
    _canon = _cfg.use_canon_slots()

    def field_lines(with_sem):
        out = []
        tc = 0
        out.append("  float4 HPos : SV_Position;" if with_sem else "  float4 HPos;")
        for n in scal:
            t = HLSL_T[INTERP_DIM[n]]
            slot = _cfg.SC2_CANON_SLOT[n] if _canon else tc
            out.append("  %s %s%s;" % (t, n, " : TEXCOORD%d" % slot if with_sem else ""))
            tc += 1
        if has_uv:
            uvb = _cfg.SC2_CANON_UV_BASE if _canon else tc
            out.append("  float4 sc2UV[%d]%s;" %
                       (uvc, " : TEXCOORD%d" % uvb if with_sem else ""))
            tc += uvc
        # The second array interpolant, after UV — postprocessquad.fx's per-tap blur
        # offsets.  Keep this ordering in lockstep with sc2_shaders_cfg.interp_defines.
        if has_gbs:
            gbsb = _cfg.SC2_CANON_GBS_BASE if _canon else tc
            out.append("  float4 sc2GaussianBlurSample[%d]%s;" %
                       (gbsc, " : TEXCOORD%d" % gbsb if with_sem else ""))
            tc += gbsc
        return out, tc

    # ---- macros shared by both stages -------------------------------------
    def interp_macros(varname):
        m = []
        for n in INTERP_DIM:
            if n in live:
                m.append("#define INTERPOLANT_%s %s.%s" % (n, varname, n))
                m.append("#define INTERPOLANT_SWIZZLE_%s(s) %s.%s.s" % (n, varname, n))
            elif n == "ShadowMapUV" and stage == "ps":
                # dropped shadow-UV interpolant (out of slots) -> recompute in-PS from
                # ViewPos exactly like EmitPSShadowMapUV, instead of the ((float4)0) that
                # makes fxc emit a 0/0 NaN (X4579) and kills every out-of-slot shadowed perm.
                _e = "(mul(float4(INTERPOLANT_ViewPos,1.0f), p_mViewToShadowTransform))"
                m.append("#define INTERPOLANT_ShadowMapUV %s" % _e)
                m.append("#define INTERPOLANT_SWIZZLE_ShadowMapUV(s) (%s).s" % _e)
            elif n == "Binormal" and "Normal" in live and "Tangent" in live:
                # dropped Binormal varying -> reconstruct from Normal x Tangent (sign in
                # Normal.w), matching the engine, instead of ((float3)0).
                _e = ("cross(%s.Normal.xyz, %s.Tangent.xyz)*%s.Normal.w"
                      % (varname, varname, varname))
                m.append("#define INTERPOLANT_Binormal (%s)" % _e)
                m.append("#define INTERPOLANT_SWIZZLE_Binormal(s) (float4(%s,0)).s" % _e)
            elif n == "FOWUV" and stage == "ps" and "WorldPos" in live:
                # dropped FOWUV varying -> recompute in-PS from WorldPos exactly like
                # EmitPSFOWUV (mul(INTERPOLANT_WorldPos, p_mFOWMatrix)), instead of the
                # ((float4)0) that collapses the FOW sample to a constant coordinate and
                # kills FOW on every perm where FOWUV was packed out of the interpolants.
                _e = "(mul(INTERPOLANT_WorldPos, p_mFOWMatrix))"
                m.append("#define INTERPOLANT_FOWUV %s" % _e)
                m.append("#define INTERPOLANT_SWIZZLE_FOWUV(s) (%s).s" % _e)
            else:  # safety net: referenced only in dead branches
                m.append("#define INTERPOLANT_%s ((%s)0)" %
                         (n, HLSL_T[INTERP_DIM[n]]))
                m.append("#define INTERPOLANT_SWIZZLE_%s(s) ((float4)0).s" % n)
        m.append("#define INTERPOLANT_ScreenPos %s.HPos" % varname)
        m.append("#define INTERPOLANT_View (-%s.EyeToVertex)" % varname
                 if "EyeToVertex" in live else "#define INTERPOLANT_View ((float3)0)")
        m.append("#define INTERPOLANT_FrontFace %s.FrontFace" % varname
                 if "FrontFace" in live else "#define INTERPOLANT_FrontFace true")
        if has_uv:
            m.append("#define READ_INTERPOLANT_UV(i) %s.sc2UV[i]" % varname)
        else:
            m.append("#define READ_INTERPOLANT_UV(i) float4(0,0,0,0)")
        if has_gbs:
            m.append("#define READ_INTERPOLANT_GaussianBlurSample(i) "
                     "%s.sc2GaussianBlurSample[i]" % varname)
        else:
            m.append("#define READ_INTERPOLANT_GaussianBlurSample(i) float4(0,0,0,0)")
        return m

    def write_macros():
        m = []
        for n in INTERP_DIM:
            if n in live:
                m.append("#define WRITE_INTERPOLANT_%s(v) vertOut.%s = (v)" % (n, n))
                m.append("#define WRITE_INTERPOLANT_SWIZZLE_%s(s,v) vertOut.%s.s = (v)" % (n, n))
            else:
                m.append("#define WRITE_INTERPOLANT_%s(v)" % n)
                m.append("#define WRITE_INTERPOLANT_SWIZZLE_%s(s,v)" % n)
        if has_uv:
            m.append("#define WRITE_INTERPOLANT_UV(i,v) vertOut.sc2UV[i] = (v)")
            # The swizzled form the SPLAT roots use — splatdeferred.fx writes only the
            # .xy of each emitter slot (`WRITE_INTERPOLANT_SWIZZLE_UV(i, xy, ...)`).
            # Without it fxc sees an undeclared *function* call and rejects the
            # ORIGINAL, i.e. the family has no reference at all.
            m.append("#define WRITE_INTERPOLANT_SWIZZLE_UV(i,s,v) vertOut.sc2UV[i].s = (v)")
        else:
            m.append("#define WRITE_INTERPOLANT_UV(i,v)")
            m.append("#define WRITE_INTERPOLANT_SWIZZLE_UV(i,s,v)")
        # BackBufferUV / Downscale write no-ops unless live
        for n in ("BackBufferUV", "DownscaleUV0", "DownscaleUV1"):
            if n not in live:
                m.append("#define WRITE_INTERPOLANT_%s(v)" % n)
        if has_gbs:
            m.append("#define WRITE_INTERPOLANT_GaussianBlurSample(i,v) "
                     "vertOut.sc2GaussianBlurSample[i] = (v)")
        else:
            m.append("#define WRITE_INTERPOLANT_GaussianBlurSample(i,v)")
        return m

    if stage == "vs":
        fl, _ = field_lines(True)
        L.append("struct VertexTransport {\n" + "\n".join(fl) + "\n};")
        L += write_macros()
        L += interp_macros("vertOut")
        ar = _initshader_arity(entry_text, entry)
        if ar == 3:      # model/splat: fills the per-emitter UV arrays
            uvm_vals = uv_mappings or [0] * 8
            # b_UVRandomOffsetEnable[] is read ONLY by particle.fx (the other roots
            # just forward it to InitShader), so it defaulted to all-zero for a long
            # time.  It is a real MainShading axis, and on the Particle root it moves
            # the UV, so the caller can now supply the decoded values.
            uvr_vals = uv_random_offsets or [0] * 8
            body = "".join("(uvm)[%d]=%d;(uvr)[%d]=%d;"
                           % (i, uvm_vals[i] if i < len(uvm_vals) else 0,
                              i, uvr_vals[i] if i < len(uvr_vals) else 0)
                           for i in range(8))
            L.append("#define InitShader(v,uvm,uvr) { %s }" % body)
        elif ar == 1:
            L.append("#define InitShader(v)")
        elif ar == 0:
            L.append("void InitShader() {}")
    else:  # ps
        fl_raw, _ = field_lines(True)
        if "FrontFace" in live:
            fl_raw.append("  bool FrontFace : SV_IsFrontFace;")
        L.append("struct VertexTransportRaw {\n" + "\n".join(fl_raw) + "\n};")
        fl_log, _ = field_lines(False)
        if "FrontFace" in live:
            fl_log.append("  bool FrontFace;")
        L.append("struct VertexTransport {\n" + "\n".join(fl_log) + "\n};")
        # InitShader: unpack raw -> logical (field-by-field copy)
        cp = ["(v).HPos=(r).HPos;"] + ["(v).%s=(r).%s;" % (n, n) for n in scal]
        if has_uv:
            cp += ["(v).sc2UV[%d]=(r).sc2UV[%d];" % (i, i) for i in range(uvc)]
        if has_gbs:
            cp += ["(v).sc2GaussianBlurSample[%d]=(r).sc2GaussianBlurSample[%d];"
                   % (i, i) for i in range(gbsc)]
        if "FrontFace" in live:
            cp.append("(v).FrontFace=(r).FrontFace;")
        L.append("#define InitShader(r,v) { %s }" % "".join(cp))
        L += interp_macros("vertOut")
        # WRITE_* macros.  A pixel shader can legitimately MUTATE its interpolants
        # before the shared body reads them: splatdeferred.fx's box-render preamble
        # reconstructs ViewPos/WorldPos from the depth buffer, writes them, then
        # tail-calls DefaultPixelMain (which re-reads them through InitShader).  So a
        # write to a LIVE interpolant must be a real store to `vertOut` (the entry's
        # raw param); writes to dead interpolants stay no-ops so VS-side macros in
        # shared headers remain inert in the PS.  Both the 1-arg and 2-arg SWIZZLE
        # forms are defined (the box-render's dead STANDARD path uses SWIZZLE writes
        # that must still parse under runtime-`if` DCE).
        for n in INTERP_DIM:
            if n in live:
                L.append("#define WRITE_INTERPOLANT_%s(v) vertOut.%s = (v)" % (n, n))
                L.append("#define WRITE_INTERPOLANT_SWIZZLE_%s(s,v) vertOut.%s.s = (v)" % (n, n))
            else:
                L.append("#define WRITE_INTERPOLANT_%s(v)" % n)
                L.append("#define WRITE_INTERPOLANT_SWIZZLE_%s(s,v)" % n)
        for n in ("GaussianBlurSample", "BackBufferUV", "DownscaleUV0", "DownscaleUV1"):
            L.append("#define WRITE_INTERPOLANT_%s(v)" % n)
            L.append("#define WRITE_INTERPOLANT_SWIZZLE_%s(s,v)" % n)
        # UV writes ARE live in the PS parallax path: OffsetUVEmitter (called from
        # ApplyParallax when b_useParallaxMapping) offsets sc2UV[index] and the layer
        # samplers re-read it.  It is the 2-param (i,v) form and must actually store --
        # a 1-param no-op both mis-counts args (compile X1516) AND drops the offset.
        # The 3-param swizzled form (i,s,v) is the splat box-render's UV update.
        if has_uv:
            L.append("#define WRITE_INTERPOLANT_UV(i,v) vertOut.sc2UV[i] = (v)")
            L.append("#define WRITE_INTERPOLANT_SWIZZLE_UV(i,s,v) vertOut.sc2UV[i].s = (v)")
        else:
            L.append("#define WRITE_INTERPOLANT_UV(i,v)")
            L.append("#define WRITE_INTERPOLANT_SWIZZLE_UV(i,s,v)")
    return "\n".join(L)


_BREF = re.compile(r'\bb_[A-Za-z_][A-Za-z0-9_]*')


def discover_undefined_b(fxc, src_dir, assembled_src, tmp_path):
    """Names of b_* tokens that survive preprocessing as bare identifiers.

    Material-layer axes such as `b_iDiffuseUVMapping` are built by token-pasting
    (`b_i##Layer##UVMapping`) and never appear literally in the source, so the
    static scan misses them.  After real preprocessing they show up as
    undeclared identifiers; we collect and default them (to 0) so the compile
    succeeds.  Their exact per-permutation values come from the key schema."""
    txt, err = _preprocess(fxc, assembled_src, tmp_path, ("/I", src_dir))
    if txt is None:
        return set()
    found = set(_BREF.findall(txt))
    # drop anything that is a struct-field access (preceded by '.') — those are
    # SLayerConfig members, not globals
    real = set()
    for m in _BREF.finditer(txt):
        j = m.start() - 1
        while j >= 0 and txt[j] in " \t":
            j -= 1
        if j >= 0 and txt[j] == ".":
            continue
        real.add(m.group(0))
    return real


def liveness(fxc, src_dir, entry_file, stage_defines_vs, stage_defines_ps,
             tmp_dir, tag="l"):
    """Return the set of live interpolant names for one permutation.

    live = {interp : VS writes it} ∩ {interp : PS reads it}, using the family's
    VS-guarded and PS-guarded code respectively (same file, different stage
    defines).  `tag` makes the scratch filenames unique across parallel workers."""
    inc = os.path.join(entry_file).replace("\\", "/")
    tmpw = os.path.join(tmp_dir, "_%s_w.fx" % tag)
    tmpr = os.path.join(tmp_dir, "_%s_r.fx" % tag)
    wsrc = stage_defines_vs + "\n" + _marker_defs("write") + '\n#include "%s"\n' % inc
    rsrc = stage_defines_ps + "\n" + _marker_defs("read") + '\n#include "%s"\n' % inc
    wtxt, we = _preprocess(fxc, wsrc, tmpw, ("/I", src_dir))
    rtxt, re_ = _preprocess(fxc, rsrc, tmpr, ("/I", src_dir))
    writes = set(re.findall(r"__W_(\w+?)__", wtxt or ""))
    reads = set(re.findall(r"__R_(\w+?)__", rtxt or ""))
    return writes, reads, (we, re_)
