"""Behavioral match: does the native `.fx` compile of a permutation reproduce the
retail shader's behavior?

Runs the native DXBC (sc2_compile_perms, given a `b_*` state + explicit live
interpolant set) and the retail-GLSL DXBC (sc2_glsl_bridge) through
`dxbc_interp` on identical inputs, and reports the per-output difference.

Input mapping: native D3D semantics -> retail GLSL attribute location
  (POSITION=0, BLENDWEIGHT=1, NORMAL=2, COLOR0=3, TANGENT=5, BLENDINDICES=7,
   TEXCOORD0=8, TEXCOORD1=9, ...), which spirv-cross exposes as TEXCOORD<loc>.
Cbuffer mapping: by variable name (strip vp_/p_/_NN_), feeding the same logical
value into both layouts at their respective byte offsets.
"""
import os, re, struct, random, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import dxbc_interp as di
import sc2_compile_perms as h
import sc2_interp as ip

ATTR_LOC = {("POSITION", 0): 0, ("BLENDWEIGHT", 0): 1, ("NORMAL", 0): 2,
            ("COLOR", 0): 3, ("COLOR", 1): 4, ("TANGENT", 0): 5,
            ("BINORMAL", 0): 6, ("BLENDINDICES", 0): 7,
            ("TEXCOORD", 0): 8, ("TEXCOORD", 1): 9, ("TEXCOORD", 2): 10,
            ("TEXCOORD", 3): 11}


def _f2b(f):
    return struct.unpack("<I", struct.pack("<f", f))[0]


def parse_insig(asm):
    """reg -> (semantic, index) from the Input signature block."""
    blk = re.search(r"Input signature:(.*?)//\s*\n//\s*\w+ signature", asm, re.S)
    sig = {}
    if blk:
        for m in re.finditer(r"//\s+(\w+)\s+(\d+)\s+[xyzw]+\s+(\d+)\s+\w+",
                             blk.group(1)):
            sig[int(m.group(3))] = (m.group(1), int(m.group(2)))
    return sig


def canon(name):
    """Canonical logical name shared by native and bridge DXBC.

    native uses `p_`/`vp_`; the spirv-cross bridge uses `_NN_pp_` on cbuffer vars
    and `pp_..._t` / `_pp_..._t_sampler` on resources.  Strip all of that so the
    same logical constant/texture maps across both."""
    n = re.sub(r"^_\d+_", "", name)
    n = re.sub(r"^(vp_|pp_|p_)", "", n)
    n = re.sub(r"_sampler$", "", n)
    n = re.sub(r"_t$", "", n)
    return n


def parse_cb(asm):
    """list of (canon_name, byte_offset, byte_size)."""
    out = []
    for m in re.finditer(
            r"^//\s+(?:row_major\s+)?[\w<>]+\s+(?:_\d+_)?(\w+)(?:\[\d+\])?;\s*"
            r"// Offset:\s*(\d+) Size:\s*(\d+)", asm, re.M):
        out.append((canon(m.group(1)), int(m.group(2)), int(m.group(3))))
    return out


def parse_resources(asm):
    """canon texture/sampler name -> slot, from the Resource Bindings table."""
    tex, smp = {}, {}
    blk = re.search(r"Resource Bindings:(.*?)//\s*\n//\s*\n", asm, re.S)
    if not blk:
        return tex, smp
    for m in re.finditer(r"//\s+(\S+)\s+(texture|sampler)\s+\S+\s+\S+\s+[a-z]+(\d+)",
                         blk.group(1)):
        (tex if m.group(2) == "texture" else smp)[canon(m.group(1))] = int(m.group(3))
    return tex, smp


def cb_write(cb, byte_off, floats):
    for k, f in enumerate(floats):
        pos = byte_off + 4 * k
        row, lane = pos // 16, (pos % 16) // 4
        while row >= len(cb):
            cb.append([0, 0, 0, 0])
        cb[row][lane] = _f2b(f)


def cb_write_words(cb, byte_off, words):
    """Write RAW uint32 words into a constant bank at `byte_off`.

    cb_write converts through _f2b, which is right for a value the shader reads as
    a float.  The permutation payload is packed BITFIELDS the shader reads with
    ubfe/and, so pushing it through a float conversion would rewrite every word.
    """
    for k, w in enumerate(words):
        pos = byte_off + 4 * k
        row, lane = pos // 16, (pos % 16) // 4
        while row >= len(cb):
            cb.append([0, 0, 0, 0])
        cb[row][lane] = int(w) & 0xFFFFFFFF


def input_regs(asm):
    return sorted(set(int(m.group(1))
                      for m in re.finditer(r"dcl_input(?:_ps\s+\w+)? v(\d+)", asm)))


def cb_rows(asm, floor=260, slot=0):
    """Rows to allocate for constant bank `slot` = its declared CB<slot>[N] length.
    Model VS with skinning declares large CB0[] (bone-matrix array), so a fixed
    260 under-allocates and dynamically-indexed reads go out of bounds.

    The 260-row floor applies to bank 0 only: it is there because bank 0 is the big
    engine globals block, whose reflection under-reports dynamically-indexed arrays.
    A higher bank (the permutation payload at b2) is sized purely by its own
    declaration, so it allocates exactly what the shader reads.
    """
    n = floor if slot == 0 else 0
    for m in re.finditer(r"dcl_constantbuffer\s+(?:CB|cb)%d\[(\d+)\]" % slot, asm):
        n = max(n, int(m.group(1)))
    return n + 4


def compile_native_vs(entry_file, entry, b_values, live, tmp_path, uv_mappings=None,
                      uv_random_offsets=None, inject_preamble=True):
    allb, nondef = h.scan_b_tokens(entry_file)
    base = {b: 0 for b in allb}
    base.update(b_values)
    etext = open(entry_file, encoding="latin1").read()
    # Own-IO families (renderplane/lensflare/minimapterrain declare their own
    # VertexTransport, or none at all) must NOT get the shared transport spliced
    # in — it would redefine the struct.  Mirrors the PS path's flag.
    pre = (ip.gen_preamble("vs", live, base, etext, entry, uv_mappings=uv_mappings,
                           uv_random_offsets=uv_random_offsets) if inject_preamble
           else ip.gen_initshader_stub(etext, entry))
    inc = entry_file.replace("\\", "/")

    def assemble(bv):
        return h.stage_defines("vs", bv, nondef) + "\n" + pre + '\n#include "%s"\n' % inc
    extra = ip.discover_undefined_b(h.FXC, h.SRC, assemble(base), tmp_path + "_d.fx")
    bv = dict(base)
    for e in extra:
        bv.setdefault(e, 0)
    open(tmp_path, "w", encoding="latin1").write(assemble(bv))
    import subprocess
    r = subprocess.run([h.FXC, "/nologo", "/Gec", "/T", "vs_5_0", "/E", entry,
                        "/I", h.SRC, "/Fc", tmp_path + ".asm", "/Fo", tmp_path + ".dxbc",
                        tmp_path], capture_output=True, text=True)
    if not os.path.exists(tmp_path + ".asm"):
        return None, [l for l in (r.stdout + r.stderr).splitlines() if "error" in l.lower()]
    return open(tmp_path + ".asm").read(), None


def compile_native_ps(entry_file, entry, b_values, live, tmp_path, uv_mappings=None,
                      inject_preamble=True):
    """Compile the native `.fx` pixel shader for one permutation.  `live` is the
    interpolant set the PS reads (driven from the retail PS's `in_` varyings).

    `inject_preamble=False` skips the gen_preamble interpolant-transport injection
    for families that carry their OWN IO structs and name them `VertexTransport`
    (e.g. terrainblend.fx), where the shared struct would be a redefinition.  Such
    families use direct field access (vertOut.vUV0), not READ_INTERPOLANT."""
    allb, nondef = h.scan_b_tokens(entry_file)
    base = {b: 0 for b in allb}
    base.update(b_values)
    etext = open(entry_file, encoding="latin1").read()
    pre = (ip.gen_preamble("ps", live, base, etext, entry,
                           uv_mappings=uv_mappings) if inject_preamble
           else ip.gen_initshader_stub(etext, entry))
    inc = entry_file.replace("\\", "/")

    def assemble(bv):
        return h.stage_defines("ps", bv, nondef) + "\n" + pre + '\n#include "%s"\n' % inc
    extra = ip.discover_undefined_b(h.FXC, h.SRC, assemble(base), tmp_path + "_d.fx")
    bv = dict(base)
    for e in extra:
        bv.setdefault(e, 0)
    open(tmp_path, "w", encoding="latin1").write(assemble(bv))
    import subprocess
    r = subprocess.run([h.FXC, "/nologo", "/Gec", "/T", "ps_5_0", "/E", entry,
                        "/I", h.SRC, "/Fc", tmp_path + ".asm", "/Fo", tmp_path + ".dxbc",
                        tmp_path], capture_output=True, text=True)
    if not os.path.exists(tmp_path + ".asm"):
        return None, [l for l in (r.stdout + r.stderr).splitlines() if "error" in l.lower()]
    return open(tmp_path + ".asm").read(), None


def compile_native_ps_sm3(entry_file, entry, b_values, live, tmp_path,
                          uv_mappings=None):
    """Compile the native `.fx` pixel shader for one permutation to **ps_3_0**
    (D3D9 SM3), returning the binary token-stream bytes for sm3_parse, or
    (None, errors).  Mirrors compile_native_ps but /T ps_3_0 /Fo (no /Gec)."""
    allb, nondef = h.scan_b_tokens(entry_file)
    base = {b: 0 for b in allb}
    base.update(b_values)
    etext = open(entry_file, encoding="latin1").read()
    pre = ip.gen_preamble("ps", live, base, etext, entry, uv_mappings=uv_mappings)
    inc = entry_file.replace("\\", "/")

    def assemble(bv):
        # ps_3_0 must use the SM3 sampler path (sampler2D/tex2D), so force
        # PIXEL_SHADER_VERSION = SHADER_VERSION_PS_30 (11) instead of PS_40 (12);
        # otherwise shadersystem.fx emits SM4 Texture2D.Sample() which does not
        # lower to a ps_3_0 sampler and the material texture reads get dropped.
        defs = h.stage_defines("ps", bv, nondef).replace(
            "PIXEL_SHADER_VERSION %d" % h.SHADER_VERSION_PS_40,
            "PIXEL_SHADER_VERSION 11")
        return defs + "\n" + pre + '\n#include "%s"\n' % inc
    extra = ip.discover_undefined_b(h.FXC, h.SRC, assemble(base), tmp_path + "_d.fx")
    bv = dict(base)
    for e in extra:
        bv.setdefault(e, 0)
    open(tmp_path, "w", encoding="latin1").write(assemble(bv))
    import subprocess
    # fxc does NOT overwrite /Fo on failure, so a stale .o from a PRIOR compile
    # (tmp_path is reused across records) would be returned as this record's binary
    # -> compile-fails silently inherit the previous shader.  Delete it first.
    opath = tmp_path + ".o"
    if os.path.exists(opath):
        os.remove(opath)
    r = subprocess.run([h.FXC, "/nologo", "/T", "ps_3_0", "/E", entry,
                        "/I", h.SRC, "/Fc", tmp_path + ".asm", "/Fo", opath,
                        tmp_path], capture_output=True, text=True)
    if not os.path.exists(opath):
        return None, [l for l in (r.stdout + r.stderr).splitlines()
                      if "error" in l.lower()]
    return open(opath, "rb").read(), None


def parse_outsig(asm):
    """reg -> (semantic, index) from the Output signature block."""
    blk = re.search(r"Output signature:(.*?)(?://\s*\n//\s*\w+ signature|\Z)", asm, re.S)
    sig = {}
    if blk:
        for m in re.finditer(r"//\s+(\w+)\s+(\d+)\s+[xyzw]+\s+(\d+)\s+\w+",
                             blk.group(1)):
            sig[int(m.group(3))] = (m.group(1), int(m.group(2)))
    return sig


def out_ncomp(asm):
    """reg -> component count actually written (len of the Mask column).

    Interpolants declared narrower than 4 (e.g. `out vec3 out_FOWUV`) write only
    xyz; the native VertexTransport carries a float4 whose unused .w is garbage,
    so the comparison must clip to the components the shader really outputs."""
    blk = re.search(r"Output signature:(.*?)(?://\s*\n//\s*\w+ signature|\Z)", asm, re.S)
    nc = {}
    if blk:
        for m in re.finditer(r"//\s+\w+\s+\d+\s+([xyzw]+)\s+(\d+)\s+\w+", blk.group(1)):
            nc[int(m.group(2))] = len(m.group(1))
    return nc


def ref_out_names(glsl):
    """GLSL out location -> interpolant name (out_<Name>)."""
    out = {}
    for m in re.finditer(r"layout\(location = (\d+)\) out \S+ out_(\w+);", glsl):
        out[int(m.group(1))] = m.group(2)
    return out


def mine_out_names(live, uvc=1):
    """my TEXCOORD index -> interpolant name (generator order: sorted scalars, then UV[])."""
    scal = sorted(n for n in live if n in ip.INTERP_DIM)
    d = {i: n for i, n in enumerate(scal)}
    if "UV" in live:
        for i in range(uvc):
            d[len(scal) + i] = "UV%d" % i
    return d


def _reg_to_name(sig, texcoord_to_name):
    out = {}
    for reg, (sem, idx) in sig.items():
        if sem in ("SV_Position", "SV_POSITION", "POSITION"):
            out[reg] = "HPos"
        elif sem == "TEXCOORD":
            out[reg] = texcoord_to_name.get(idx)
    return out


def compare(mine_asm, ref_asm, glsl, live, uvc=1, trials=6, seed=0):
    """Per-interpolant max abs diff (matched by name), across random trials.

    HPos (SV_Position) is compared under the OpenGL clip-space convention the
    retail GLSL bakes in (y*=-1; z=2z-w) so a D3D-native compile matches."""
    rnd = random.Random(seed)
    mp = di.Program.from_text(mine_asm)
    rp = di.Program.from_text(ref_asm)
    m_out = _reg_to_name(parse_outsig(mine_asm), mine_out_names(live, uvc))
    r_out = _reg_to_name(parse_outsig(ref_asm), ref_out_names(glsl))
    m_byname = {v: k for k, v in m_out.items() if v}
    r_byname = {v: k for k, v in r_out.items() if v}
    shared = [n for n in m_byname if n in r_byname]
    # components actually written per interpolant name (clip to ref's output width)
    r_nc = out_ncomp(ref_asm)
    m_nc = out_ncomp(mine_asm)
    ncomp = {n: min(r_nc.get(r_byname[n], 4), m_nc.get(m_byname[n], 4))
             for n in shared}
    mi, ri = parse_insig(mine_asm), parse_insig(ref_asm)
    mcb, rcb = parse_cb(mine_asm), parse_cb(ref_asm)
    rbyname = {n: (o, s) for n, o, s in rcb}
    # ref reg -> loc (ref semantics are TEXCOORD<loc>)
    ref_loc = {r: ri[r][1] for r in ri}
    diffs = {n: 0.0 for n in shared}
    for _ in range(trials):
        # per-attribute value by location
        attrval = {}
        for loc in range(12):
            if loc == 0:
                attrval[loc] = [rnd.uniform(-1, 1), rnd.uniform(-1, 1), rnd.uniform(-1, 1), 1.0]
            elif loc == 7:  # blend indices -> bone 0 (both decode to 0; avoids ubyte4/4n
                attrval[loc] = [0.0, 0.0, 0.0, 0.0]   # + swizzle mismatch on skinned perms)
            elif loc in (8, 9, 10, 11):
                attrval[loc] = [rnd.uniform(-1, 1), rnd.uniform(-1, 1), 0.0, 0.0]
            else:
                attrval[loc] = [rnd.uniform(0, 1) for _ in range(4)]
        # cbuffer values by name (shared)
        cbvals = {n: [rnd.uniform(-1, 1) for _ in range(s // 4)] for n, o, s in rcb}
        # build inputs
        def mk_inputs(sig):
            out = {}
            for reg, (sem, idx) in sig.items():
                loc = ATTR_LOC.get((sem, idx))
                if loc is None and sem == "TEXCOORD":
                    loc = idx     # ref side: semantic index IS the location
                out[reg] = [_f2b(x) for x in attrval.get(loc, [0, 0, 0, 0])]
            return out
        m_in = mk_inputs(mi)
        r_in = {reg: [_f2b(x) for x in attrval.get(ref_loc[reg], [0, 0, 0, 0])] for reg in ri}
        # cbuffers (sized to each shader's declared CB0[] length)
        m_cb = [[0, 0, 0, 0] for _ in range(cb_rows(mine_asm))]
        for n, o, s in mcb:
            if n in cbvals:
                cb_write(m_cb, o, cbvals[n])
        r_cb = [[0, 0, 0, 0] for _ in range(cb_rows(ref_asm))]
        for n, o, s in rcb:
            if n in cbvals:
                cb_write(r_cb, o, cbvals[n])
        try:
            mo = di.execute(mp, m_in, {0: m_cb}, max_loop=4096)
            ro = di.execute(rp, r_in, {0: r_cb}, max_loop=4096)
        except Exception as e:
            return None, "exec: " + str(e)[:120]
        for name in shared:
            mr, rr = m_byname[name], r_byname[name]
            mv = [mo.f(mr, l) for l in range(4)]
            rv = [ro.f(rr, l) for l in range(4)]
            if name == "HPos":                 # apply GL clip-space convention to native
                mv = [mv[0], -mv[1], 2.0 * mv[2] - mv[3], mv[3]]
            for l in range(ncomp.get(name, 4)):  # only the components the ref writes
                a, b = mv[l], rv[l]
                if a == a and b == b:          # not NaN
                    diffs[name] = max(diffs[name], abs(a - b))
    return diffs, None
