"""Behavioral validation harness: original SC2 `.fx` (fxc, the D3D11 *reference*)
vs a slang port (slangc, the D3D11 *candidate*), compared through `dxbc_interp`
on identical random inputs.

Both sides compile to the SAME target — `ps_5_0` / `vs_5_0` DXBC — so the
comparison is entirely within our own toolchain (no Blizzard-fxc drift to fight,
unlike the retail SM3 path).  Alignment:
  * constants  — by cbuffer reflection **name** (fxc and slangc both emit RDEF
                 field names; a value is drawn once per name and written into
                 each side's cbuffer at that side's own offset).
  * inputs     — by input-signature (semantic, index): a varying gets one shared
                 random vec4 regardless of which register each side placed it in.
  * textures   — by resource name -> one shared smooth stand-in field (RemapTex).
  * outputs    — by SV_Target index; LDR targets clamped to UNORM8 before diffing.

Legs:
  reference : sc2_behav_match.compile_native_ps / compile_native_vs  (fxc /Gec)
  candidate : compile_all_slang.invoke_slangc -> .dxbc -> fxc /dumpbin -> .asm
  compare   : compare_d3d11(ref_asm, cand_asm, ...)

Until real SC2 slang exists, `--self-test` proves the comparator is sound by
running a reference permutation against ITSELF (worst must be exactly 0) and,
if a slang file is given, exercises the slangc leg mechanically.
"""
import os
import re
import sys
import random
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dxbc_interp as di
import sc2_behav_match as bm
import sc2_compile_perms as h

REPO_ROOT = os.path.dirname(HERE)


# ---------------------------------------------------------------------------
# shared texture model — each side's slot -> one shared id per logical name
# ---------------------------------------------------------------------------
class RemapTex(di.TextureModel):
    def __init__(self, slot2id):
        self.slot2id = slot2id

    def _id(self, slot):
        return self.slot2id.get(slot, slot + 1000)

    def sample(self, slot, coords):
        return super().sample(self._id(slot), coords)

    # NOTE: TextureModel.sample_compare / sample_lod are implemented in terms of
    # self.sample(...), so delegating to super() with an already-remapped id would
    # remap it a SECOND time (via this class's sample()) and read the wrong texture
    # field.  Remap ONCE here and drive the base field function directly.
    def sample_compare(self, slot, coords, ref):
        return di.TextureModel.sample(
            self, self._id(slot), [coords[0], coords[1], ref, 0.0])[0]

    def sample_lod(self, slot, coords, lod):
        return di.TextureModel.sample(
            self, self._id(slot),
            [coords[0], coords[1], coords[2], coords[3] + lod * 0.01])

    def lod(self, slot, coords):
        return super().lod(self._id(slot), coords)


def _sat(x):
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


# ---------------------------------------------------------------------------
# slang-aware reflection: slangc's DXBC RDEF names differ systematically from
# fxc's — every cbuffer field / resource gets a `_<n>` disambiguation suffix
# (`p_mProjection_0`), cbuffers are wrapped in a `SLANG_ParameterGroup` struct,
# and matrices are wrapped in a `_MatrixStorage_*` struct (`float4 data_N[4]`).
# fxc emits flat `type name; // Offset: N Size: M`.  These helpers normalise
# BOTH toolchains to the same logical (canon) names so the comparator can align
# a value by name regardless of which compiler produced the blob.
# ---------------------------------------------------------------------------
_SLANG_SUFFIX = re.compile(r"_\d+$")


def _uni_canon(name):
    """Logical name shared by fxc + slangc reflections: drop slangc's `_<n>`
    disambiguation suffix, then apply the shared p_/vp_/_t/_sampler canon."""
    return bm.canon(_SLANG_SUFFIX.sub("", name))


def _parse_cb_slang(asm):
    """Parse a slangc-emitted cbuffer reflection into the same
    [(canon_name, byte_offset, byte_size)] shape bm.parse_cb yields for fxc.

    slangc nests fields: the ConstantBuffer<struct> members live inside the
    `..._natural`/`SLANG_ParameterGroup` struct, and each matrix or struct(-array)
    member is wrapped in a FURTHER inner struct (a `_MatrixStorage` holding
    `data_N[]`, or a `DirectionalLight`/`PointLight` holding its fields).  We track
    brace depth and keep only the OUTERMOST field level (the minimum member depth
    within each cbuffer), so a `_MatrixStorage` matrix or a `p_directionalLights[3]`
    light array is captured as ONE opaque byte-region by its instance name — never
    split into its inner `data_N` / `vDirection` / `cColor` fields (splitting them
    would double-write and corrupt the array's random bytes on the candidate leg).
    A member's byte size is its span to the next outermost member (the last spans
    to the group instance's Size)."""
    out = []

    def flush(entries, group_size):
        if not entries:
            return
        top = min(d for _, _, d in entries)
        members = sorted(((n, o) for n, o, d in entries if d == top),
                         key=lambda x: x[1])
        for i, (name, off) in enumerate(members):
            if i + 1 < len(members):
                size = members[i + 1][1] - off
            elif group_size is not None:
                size = group_size - off
            else:
                size = 16
            out.append((_uni_canon(name), off, size))

    entries = []            # (name, offset, brace_depth)
    group_size = None
    depth = 0
    in_cb = False
    for line in asm.splitlines():
        if re.match(r"^//\s*cbuffer\b", line):
            flush(entries, group_size)
            entries, group_size, depth, in_cb = [], None, 0, True
            continue
        if not in_cb:
            continue
        opens = line.count("{")
        closes = line.count("}")
        new_depth = depth + opens - closes
        m = re.search(r"(\w+)\s*(?:\[\d+\])?;\s*//\s*Offset:\s*(\d+)"
                      r"(?:\s*Size:\s*(\d+))?", line)
        if m:
            name, off, sz = m.group(1), int(m.group(2)), m.group(3)
            if sz is not None:          # outermost group/struct instance (has Size)
                group_size = int(sz)
            else:
                # a struct-instance close ("} name[3];") is declared at the depth
                # AFTER its brace closes; a plain field line at the current depth.
                entries.append((name, off, new_depth if closes else depth))
        depth = new_depth
        if depth <= 0:                  # closed out of the cbuffer
            flush(entries, group_size)
            entries, group_size, in_cb = [], None, False
    flush(entries, group_size)
    return out


# fxc keeps a cbuffer struct(-array) as one variable whose declaration is the
# struct's closing brace: "} p_directionalLights[3];  // Offset: 448 Size: 96".
# bm.parse_cb only matches "<type> <name>; // Offset Size", so it misses these;
# the inner fields carry Offset but no Size, so bm.parse_cb already skips them
# (no double count).  This recovers the struct instance as one opaque region.
_FXC_STRUCT_CLOSE = re.compile(
    r"^//\s+}\s+(?:_\d+_)?(\w+)(?:\[\d+\])?;\s*//\s*Offset:\s*(\d+)\s+Size:\s*(\d+)",
    re.M)


def _struct_consts_fxc(asm):
    return [(bm.canon(m.group(1)), int(m.group(2)), int(m.group(3)))
            for m in _FXC_STRUCT_CLOSE.finditer(asm)]


_CB_HEADER = re.compile(r"^//\s*cbuffer\s+(\S+)\s*$", re.M)


def _cb_blocks(asm):
    """[(cbuffer instance name, block text)] over the Buffer Definitions section,
    in declaration order.

    Both toolchains open each bank with a bare `// cbuffer <instance>` line and
    nest everything else inside it, so splitting on that header keeps each bank's
    fields attributable to ITS OWN bank -- which is what makes a second constant
    buffer (the permutation payload at b2) addressable at all.  Scanning the whole
    disassembly with one regex, as this used to, merges every bank into bank 0."""
    m = re.search(r"Buffer Definitions:(.*?)Resource Bindings:", asm, re.S)
    region = m.group(1) if m else asm
    hits = list(_CB_HEADER.finditer(region))
    return [(h.group(1), region[h.start():(hits[i + 1].start()
                                           if i + 1 < len(hits) else len(region))])
            for i, h in enumerate(hits)]


def _cb_slots(asm):
    """cbuffer instance name -> HLSL bind slot, from the Resource Bindings table.

    fxc prints `$Globals  cbuffer  NA  NA  cb0  1`; slangc prints `psGlobals_0
    cbuffer  NA  NA  cb0  1`.  Reading the slot here beats assuming 0: the reduced
    module binds its permutation payload at b2, and a field's OFFSET is only
    meaningful relative to its own bank."""
    out = {}
    blk = re.search(r"Resource Bindings:(.*?)//\s*\n//\s*\n", asm, re.S)
    if not blk:
        return out
    for m in re.finditer(r"//\s+(\S+)\s+cbuffer\s+\S+\s+\S+\s+cb(\d+)\s",
                         blk.group(1)):
        out[m.group(1)] = int(m.group(2))
    return out


# Both toolchains emit the same "Generated by Microsoft (R) HLSL Shader Compiler"
# banner -- slangc produces HLSL that fxc then assembles -- so a disassembly carries
# no inherent marker.  The COMPILE functions do know which one ran, so they stamp it.
TOOLCHAIN_TAG = "// SC2-TOOLCHAIN: %s\n"


def _tag(asm, which):
    return None if asm is None else (TOOLCHAIN_TAG % which) + asm


def _is_slangc(asm):
    """Which toolchain produced this disassembly?

    Decides whether slangc's x10 TEXCOORD index inflation has to be undone.  Two
    earlier versions were wrong, both silently:

      * hand-passed `slangc=True` on the candidate leg -- correct only while every
        index stayed below 10.  With canonical slots an fxc reference legitimately
        carries TEXCOORD32, and un-inflating that yields slot 5, so ref-vs-ref
        identity (`--self-test`) reported 0.75 instead of 0.
      * sniffing for SLANG_ParameterGroup / _MatrixStorage / a `cbuffer name_<n>`
        instance name -- incidental features, not markers.  LensFlare_vs declares no
        cbuffer and uses no matrix, so a genuine slangc blob read as fxc, its
        interpolant indices were left inflated and the varyings misaligned: worst
        1.164 on a shader whose DXBC was byte-identical to a passing baseline.

    So the producer stamps the answer and this only reads it.  The heuristics stay as
    a fallback for asm that came from elsewhere (a file on disk, another tool).
    """
    t = _toolchain(asm)
    if t is not None:
        return t.startswith("slangc")
    return bool("SLANG_ParameterGroup" in asm or "_MatrixStorage" in asm
                or re.search(r"cbuffer \w+_\d+\b", asm))


def _toolchain(asm):
    """The stamped producer, or None if this asm carries no stamp.

    Reads the FIRST LINE rather than a fixed prefix slice: the earlier version
    took asm[:48], which silently truncated any longer stamp and left the
    comparison reading whatever fragment happened to land inside the window.
    """
    if not asm.startswith("// SC2-TOOLCHAIN:"):
        return None
    return asm.split(chr(10), 1)[0].split(":", 1)[1].strip()


def _sig_inflated(asm):
    """Are slangc's x10 semantic indices still present in this disassembly?

    NOT the same question as `_is_slangc`, and conflating them was a real bug.  A
    blob pulled out of a built BLS has been through
    build_sc2_bls._process_dxbc -> build_bls.fix_dxbc_signatures, which divides the
    inflated indices down.  It therefore carries **slangc's cbuffer reflection with
    fxc's signature convention**, and the two questions have opposite answers.

    With one flag for both, tagging a bundled blob truthfully ("not inflated") also
    switched the CONSTANT-BUFFER parser to fxc's, which misread every field name and
    offset; the two legs then drew different constants and Simple_vs reported 1.88
    on a shader whose body was byte-identical to a passing baseline.  Tagging it
    "slangc" instead happens to work only while every index is below 10 -- with
    canonical slots a bundled TEXCOORD32 would un-inflate to 5.  Hence a third
    value, `slangc-bundled`: slangc reflection, signatures already normalised.
    """
    t = _toolchain(asm)
    if t is not None:
        return t == "slangc"
    return _is_slangc(asm)


def _cb(asm):
    """cbuffer fields as [(slot, canon, offset, size)], auto-detecting fxc vs slangc.

    slangc reflections are recognised by their `_<n>`-suffixed cbuffer instance
    name (`cbuffer psGlobals_0`) -- true of both the raw-`cbuffer` and the
    `ConstantBuffer<struct>` forms -- plus the SLANG_ParameterGroup / _MatrixStorage
    wrappers.  fxc names cbuffers `$Globals` / plain, gives every scalar field a
    `Size:` (bm.parse_cb path), and declares struct(-array) constants via their
    closing brace (recovered by _struct_consts_fxc).

    The leading SLOT is what R0 adds.  Every field is parsed inside its own
    `// cbuffer` block and tagged with that block's bind slot, so a candidate that
    declares a second bank keeps its offsets separate from bank 0's."""
    slangc = _is_slangc(asm)
    slots = _cb_slots(asm)
    out = []
    for name, text in _cb_blocks(asm):
        slot = slots.get(name, 0)
        fields = (_parse_cb_slang(text) if slangc
                  else bm.parse_cb(text) + _struct_consts_fxc(text))
        out += [(slot, n, o, s) for n, o, s in fields]
    return out


def _cb0(asm):
    """The pre-R0 shape -- [(canon, offset, size)] for bank 0 only."""
    return [(n, o, s) for slot, n, o, s in _cb(asm) if slot == 0]


def _res(asm):
    """(texture, sampler) canon-name -> slot, auto-normalising slangc suffixes.
    Mirrors bm.parse_resources but through _uni_canon so a slangc `p_sTexture_t_0`
    and an fxc `p_sTexture_t` both canonicalise to `sTexture`."""
    tex, smp = {}, {}
    blk = re.search(r"Resource Bindings:(.*?)//\s*\n//\s*\n", asm, re.S)
    if not blk:
        return tex, smp
    for m in re.finditer(r"//\s+(\S+)\s+(texture|sampler)\s+\S+\s+\S+\s+[a-z]+(\d+)",
                         blk.group(1)):
        (tex if m.group(2) == "texture" else smp)[_uni_canon(m.group(1))] = \
            int(m.group(3))
    return tex, smp


# ---------------------------------------------------------------------------
# candidate leg: slang -> dxbc -> fxc /dumpbin -> fxc-format disassembly
# ---------------------------------------------------------------------------
def disasm_dxbc(dxbc_path, asm_path):
    """fxc /dumpbin -> the same disassembly format dxbc_interp.Program parses.

    Works on ANY DXBC container (fxc- or slangc-produced), so both legs feed
    the interpreter through one parser."""
    if os.path.exists(asm_path):
        os.remove(asm_path)
    subprocess.run([h.FXC, "/nologo", "/dumpbin", "/Fc", asm_path, dxbc_path],
                   capture_output=True, text=True)
    if not os.path.exists(asm_path):
        return None
    return open(asm_path, encoding="latin1").read()


def compile_slang(slang_file, entry, stage, defines, tmp_path,
                  include_dirs=None, specialize=()):
    """Compile a slang entry to ps_5_0/vs_5_0 DXBC, return fxc-format asm.

    `defines` is a list of "NAME=val" permutation defines (same axes as the
    original's b_* set).  Returns (asm, None) or (None, [errors])."""
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)
    import compile_all_slang as cas
    from pathlib import Path
    profile = {"ps": "ps_5_0", "vs": "vs_5_0"}[stage]
    inc = [Path(d) for d in (include_dirs or [os.path.join(REPO_ROOT, "wc3_shaders")])]
    dxbc = tmp_path + ".dxbc"
    ok = cas.invoke_slangc(entry, "dxbc", profile, list(specialize),
                           Path(dxbc), Path(slang_file),
                           include_dirs=inc, defines=list(defines))
    if not ok or not os.path.exists(dxbc):
        return None, ["slangc failed for %s entry=%s" % (slang_file, entry)]
    asm = disasm_dxbc(dxbc, tmp_path + ".asm")
    if asm is None:
        return None, ["fxc /dumpbin failed on %s" % dxbc]
    return _tag(asm, "slangc"), None


# ---------------------------------------------------------------------------
# reference leg: original .fx -> ps_5_0/vs_5_0 (fxc), returns asm
# ---------------------------------------------------------------------------
def compile_reference(entry_file, entry, stage, b_values, live, tmp_path,
                      uv_mappings=None, inject_preamble=True,
                      uv_random_offsets=None):
    if stage == "ps":
        asm, err = bm.compile_native_ps(entry_file, entry, b_values, live, tmp_path,
                                        uv_mappings=uv_mappings,
                                        inject_preamble=inject_preamble)
    else:
        asm, err = bm.compile_native_vs(entry_file, entry, b_values, live, tmp_path,
                                        uv_mappings=uv_mappings,
                                        uv_random_offsets=uv_random_offsets,
                                        inject_preamble=inject_preamble)
    return _tag(asm, "fxc"), err


# ---------------------------------------------------------------------------
# comparator
# ---------------------------------------------------------------------------
def _target_regs(asm):
    """reg -> SV_Target index."""
    out = {}
    for reg, (sem, idx) in bm.parse_outsig(asm).items():
        if sem in ("SV_Target", "SV_TARGET", "COLOR"):
            out[reg] = idx
    return out


def _in_keys(asm, slangc=False):
    """input reg -> a stable cross-shader key ('HPos', 'FRONTFACE', or (sem,idx)).

    Interpolant indices are normalised through _norm_idx so a slangc candidate's
    ×10-inflated TEXCOORD index (scalar TEXCOORD1 -> TEXCOORD10, array element
    sc2UV[1] at base TEXCOORD5 -> TEXCOORD51) aligns with the fxc reference's
    logical slot.  Pass slangc=True for the candidate leg only."""
    out = {}
    for reg, (sem, idx) in bm.parse_insig(asm).items():
        if sem in ("SV_Position", "SV_POSITION"):
            out[reg] = "HPos"
        elif sem in ("SV_IsFrontFace", "SV_ISFRONTFACE"):
            out[reg] = "FRONTFACE"
        else:
            out[reg] = (sem, _norm_idx(idx, slangc))
    return out


def _texmap(ref_asm, cand_asm):
    r_tex, _ = _res(ref_asm)
    c_tex, _ = _res(cand_asm)
    names = sorted(set(r_tex) | set(c_tex))
    ids = {n: i for i, n in enumerate(names)}
    return (RemapTex({slot: ids[n] for n, slot in r_tex.items()}),
            RemapTex({slot: ids[n] for n, slot in c_tex.items()}))


def _make_const_draw(rng, const_domains):
    """`draw(name, nfloat) -> [float]`, honouring any domain declared on `name`.

    A domain is either ("index", n) — a float the shader truncates into an index,
    so draw exact integers in [0, n) — or a callable `fn(rng, nfloat) -> [float]`
    for constants with a per-component shape (e.g. a packed float4 whose .w is a
    divisor and must never be 0).  Shared by the PS and VS comparators."""
    cdom = const_domains or {}

    def draw(name, nfloat):
        d = cdom.get(bm.canon(name))
        if d is None:
            return [rng.uniform(-1, 1) for _ in range(nfloat)]
        if callable(d):
            return list(d(rng, nfloat))
        if d[0] == "index":
            return [float(rng.randrange(0, d[1])) for _ in range(nfloat)]
        raise ValueError("unknown const domain %r" % (d,))
    return draw


def _split_pins(cb_pins):
    """Normalise `cb_pins` into ({(slot, name): words}, {slot: words}).

    A key of `slot` alone pins a WHOLE BANK from byte 0 -- which is what a
    permutation payload actually is, and is robust to how the two toolchains name
    the buffer's members (slangc reflects `ConstantBuffer<SC2PermCBT>` as a single
    member `w`, not as `permCB`).  A `(slot, name)` key still pins one field."""
    fields, banks = {}, {}
    for k, v in (cb_pins or {}).items():
        if isinstance(k, tuple):
            fields[k] = v
        else:
            banks[int(k)] = v
    return fields, banks


def _is_pinned(fields, banks, slot, name):
    return slot in banks or (slot, name) in fields



def _unmatched(ref_by_key, cand_by_key):
    """REFERENCE outputs with no counterpart on the candidate leg, or "".

    Comparing just the INTERSECTION of the two legs' output keys is a silent
    coverage cap: if every interpolant misaligns, `SV_Position` still maps to the
    index-independent key 'HPos', so `shared` is never empty and the slot reports
    green having compared one output out of six.  That is how bundled Model_vs /
    Particle_vs blobs passed while carrying a corrupted output signature.

    The check is deliberately ONE-SIDED.  A reference output the candidate cannot
    supply is a hole in the comparison and must fail.  The reverse -- an output only
    the CANDIDATE declares -- is expected and benign here: a reduced build compiles
    one blob per structural class, so a class necessarily declares the union of the
    interpolants its member permutations need, while the fxc reference is
    specialised to a single member and drops the rest.  Measured on Model_vs slot
    3210: reference writes {HPos, TEXCOORD0, TEXCOORD9}, the class writes those
    three plus TEXCOORD32.  Failing on that would reject the reduction itself.

    A genuine misalignment is still caught, because it strands outputs on BOTH
    sides: the array-element signature bug turned a candidate's TEXCOORD33 into
    TEXCOORD321, leaving reference key 33 unmatched -- ref-only, non-empty, fail.
    """
    ro = sorted(set(ref_by_key) - set(cand_by_key), key=str)
    if not ro:
        return ""
    co = sorted(set(cand_by_key) - set(ref_by_key), key=str)
    return "ref-only=%s cand-only=%s" % (ro[:4], co[:4])

def compare_d3d11(ref_asm, cand_asm, *, trials=8, seed=0, name_map=None,
                  const_domains=None, cb_pins=None):
    """Per-SV_Target max abs diff between the fxc reference and slangc candidate.

    `name_map` optionally maps candidate cbuffer/resource canon-names to the
    reference's (for when the slang port renames a uniform); default identity.
    `const_domains` restricts named constants to engine-legal values — a pixel
    shader can index by constant too (image.fx selects the alpha channel with
    `cColor[(int)p_vLayerAlphaChannelIndex[i]]`), and an out-of-range draw makes
    each leg read a different lane of its own indexable temp.
    `cb_pins` is {(slot, canon_name): [uint32, ...]} - fields written VERBATIM as
    raw words instead of being drawn at random, and excluded from the draw on both
    legs.  This is how a reduced permutation is validated: the candidate is the
    generic shader, and its b2 permutation payload is pinned to the packed value for
    the retail slot the reference was specialized with.  Without pinning, a
    candidate-only cbuffer name silently receives a RANDOM draw (see the
    cbvals.setdefault below), so the payload would be noise.
    Returns ({target_index: worst_abs_diff}, None) or (None, error_string)."""
    nm = name_map or {}
    pins, pinned_banks = _split_pins(cb_pins)

    def cn(n):
        return nm.get(n, n)

    rp = di.Program.from_text(ref_asm)
    cp = di.Program.from_text(cand_asm)

    r_in = _in_keys(ref_asm, slangc=_sig_inflated(ref_asm))
    c_in = _in_keys(cand_asm, slangc=_sig_inflated(cand_asm))
    r_cb, c_cb = _cb(ref_asm), _cb(cand_asm)
    r_tgt, c_tgt = _target_regs(ref_asm), _target_regs(cand_asm)
    r_by_t = {t: reg for reg, t in r_tgt.items()}
    c_by_t = {t: reg for reg, t in c_tgt.items()}
    shared_t = sorted(set(r_by_t) & set(c_by_t))
    if not shared_t:
        return None, "no shared SV_Target"
    _only = _unmatched(r_by_t, c_by_t)
    if _only:
        return None, "unmatched SV_Target %s" % _only
    r_nc, c_nc = bm.out_ncomp(ref_asm), bm.out_ncomp(cand_asm)
    r_tex, c_tex = _texmap(ref_asm, cand_asm)
    rng = random.Random(seed)
    diffs = {t: 0.0 for t in shared_t}
    const_draw = _make_const_draw(rng, const_domains)

    for _ in range(trials):
        # one shared random vec4 per input key and per constant name.  Iterate in a
        # DETERMINISTIC order (sorted): iterating the raw set assigns rng draws in
        # Python's hash-randomised set order, so the same perm would get different
        # random inputs each process run — bit-exact perms still read 0, but any real
        # divergence's magnitude/threshold flips run-to-run (a validation-noise bug).
        keys = sorted(set(v for v in r_in.values()) | set(v for v in c_in.values()),
                      key=lambda k: str(k))
        vals = {}
        for k in keys:
            if k == "HPos":
                vals[k] = [rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(0, 1), 1.0]
            elif k == "FRONTFACE":
                vals[k] = None
            else:
                vals[k] = [rng.uniform(-1, 1) for _ in range(4)]
        cbvals = {}
        for _slot, n, o, s in r_cb:
            if not _is_pinned(pins, pinned_banks, _slot, n):
                cbvals.setdefault(n, const_draw(n, max(1, s // 4)))
        for _slot, n, o, s in c_cb:  # ensure candidate-only names still get a value
            if not _is_pinned(pins, pinned_banks, _slot, cn(n)):
                cbvals.setdefault(cn(n), const_draw(n, max(1, s // 4)))

        def mk_in(regmap):
            out = {}
            for reg, k in regmap.items():
                if k == "FRONTFACE":
                    out[reg] = [0xFFFFFFFF] * 4
                else:
                    out[reg] = [bm._f2b(x) for x in vals[k]]
            return out

        def mk_cb(cb_list, asm, rename):
            """{slot: rows} for every constant bank the shader declares.

            A field's offset is relative to ITS bank, so the banks are built
            separately.  Bank 0 is always materialised even when the shader declares
            no constants at all, matching the pre-R0 behaviour."""
            banks = {}
            for slot, n, o, s in cb_list:
                if slot not in banks:
                    banks[slot] = [[0, 0, 0, 0]
                                   for _ in range(bm.cb_rows(asm, slot=slot))]
                key = rename(n)
                if slot in pinned_banks:
                    continue        # whole bank written below, verbatim
                pin = pins.get((slot, key))
                if pin is not None:
                    bm.cb_write_words(banks[slot], o, pin)
                elif key in cbvals:
                    bm.cb_write(banks[slot], o, cbvals[key])
            for slot, words in pinned_banks.items():
                if slot in banks:
                    bm.cb_write_words(banks[slot], 0, words)
            if 0 not in banks:
                banks[0] = [[0, 0, 0, 0] for _ in range(bm.cb_rows(asm, slot=0))]
            return banks

        r_i, c_i = mk_in(r_in), mk_in(c_in)
        r_cbv = mk_cb(r_cb, ref_asm, lambda n: n)
        c_cbv = mk_cb(c_cb, cand_asm, cn)
        try:
            ro = di.execute(rp, r_i, r_cbv, texture=r_tex, max_loop=4096)
            co = di.execute(cp, c_i, c_cbv, texture=c_tex, max_loop=4096)
        except Exception as e:
            return None, "exec: " + str(e)[:160]
        for t in shared_t:
            rr, cr = r_by_t[t], c_by_t[t]
            nc = min(r_nc.get(rr, 4), c_nc.get(cr, 4))
            for l in range(nc):
                a, b = ro.f(rr, l), co.f(cr, l)
                an, bn = (a == a), (b == b)
                if an and bn:
                    d = abs(_sat(a) - _sat(b))
                elif an != bn:
                    d = 1.0
                else:
                    d = 0.0
                diffs[t] = max(diffs[t], d)
    return diffs, None


# ---------------------------------------------------------------------------
# VS comparator: align OUTPUT registers by output-signature semantic
# ---------------------------------------------------------------------------
def _norm_idx(idx, slangc=False):
    """Undo slangc's ×10 TEXCOORD-index inflation on the CANDIDATE leg only.

    slangc emits a scalar interpolant declared TEXCOORDn as TEXCOORD(n*10), and an
    array element e of a base-TEXCOORDb array as TEXCOORD(b*10 + e) — e.g. sc2UV[2]
    at TEXCOORD5 becomes TEXCOORD50, TEXCOORD51.  Both recover to the reference's
    logical register index via idx//10 + idx%10 (n*10 -> n; b*10+e -> b+e, e<10).
    The fxc reference keeps its real register indices unchanged — those are the true
    slots and can legitimately exceed 9 on interpolant-heavy perms, so the reference
    leg must NOT be un-inflated (that would collapse its TEXCOORD10+ onto lower
    slots)."""
    return (idx // 10 + idx % 10) if slangc else idx


def _out_keys(asm, slangc=False):
    """output reg -> stable cross-shader key.  SV_Position/POSITION -> 'HPos';
    every interpolant -> (semantic, normalized-index).  Pass slangc=True for the
    candidate leg (un-inflates slangc's ×10 TEXCOORD indices)."""
    out = {}
    for reg, (sem, idx) in bm.parse_outsig(asm).items():
        if sem in ("SV_Position", "SV_POSITION", "POSITION"):
            out[reg] = "HPos"
        else:
            out[reg] = (sem, _norm_idx(idx, slangc))
    return out


# ---------------------------------------------------------------------------
# ENGINE-LEGAL VALUE DOMAINS
# ---------------------------------------------------------------------------
# Some vertex attributes and some constants are not numbers, they are ARRAY
# INDICES.  A uniform [-1,1] draw would send the lookup far outside the declared
# array, and the interpreter (rightly) does not bounds-clamp — so the run either
# dies with "list index out of range" or, worse, silently reads a NEIGHBOURING
# constant.  That last case is a false failure rather than a crash: the two legs
# lay their cbuffers out independently (aligned by NAME), so an out-of-range row
# is a different constant on each leg.
#
# Both legs still see the SAME random draw; this only keeps it inside the array.
#
#   "ubyte4n" — the stream is D3DCOLOR/ubyte4n and the shader decodes it with
#               D3DCOLORtoUBYTE4 (`*255.001953`, truncate), so a legal element is
#               in [0, (n-1)/255].
#   "uint"    — a true integer attribute (`uint4` in the declaration): the
#               register holds the integer's BIT PATTERN, not its float value.
#   "index"   — a float-typed constant the shader truncates to an index (SC2's
#               batch-index remapping tables are `float p_f...[32]`).
# Keys are matched against BOTH "<SEMANTIC><INDEX>" (e.g. "TEXCOORD6") and the
# bare "<SEMANTIC>", so a domain can pin one indexed slot or a whole semantic.
_VS_INPUT_DOMAINS = {
    "BLENDINDICES": ("ubyte4n", 64),        # p_mBlendMatrices[64]
}


def _domain_draw(kind, n, rng):
    """One 4-component raw-register value honouring an index domain."""
    if kind == "ubyte4n":
        return [bm._f2b(rng.uniform(0.0, (n - 1) / 255.0)) for _ in range(4)]
    if kind == "uint":
        return [rng.randrange(0, n) for _ in range(4)]
    if kind == "sint":
        # A signed integer attribute (`int4`/`int2` in the declaration): the
        # register holds the integer's two's-complement bit pattern.  Not an array
        # index — the range just keeps the decoded value physically plausible
        # instead of the ~1e38 a random float's bits would decode to.
        return [rng.randrange(-n, n) & 0xFFFFFFFF for _ in range(4)]
    raise ValueError("unknown input domain %r" % (kind,))


def _vs_input_value(key, rng, domains=None):
    """RAW 32-bit register contents for one vertex attribute.

    Returns bit patterns, not floats, so integer-typed attributes can carry a
    small integer rather than the bits of a float."""
    if key == "FRONTFACE":
        return [0xFFFFFFFF] * 4
    if isinstance(key, tuple):
        sem, idx = key[0].upper(), key[1]
        for tbl in (domains or {}, _VS_INPUT_DOMAINS):
            d = tbl.get("%s%d" % (sem, idx)) or tbl.get(sem)
            if d is not None:
                return _domain_draw(d[0], d[1], rng)
    return [bm._f2b(rng.uniform(-1, 1)) for _ in range(4)]


def compare_vs(ref_asm, cand_asm, *, trials=8, seed=0, name_map=None,
               input_domains=None, const_domains=None, cb_pins=None):
    """Per-VS-output max abs diff between the fxc reference and slangc candidate.

    Mirrors compare_d3d11's constant(by name)/input(by signature)/texture(by
    name) alignment, but compares the *vertex outputs* — SV_Position and the
    TEXCOORD/COLOR interpolants — matched by output-signature semantic, over the
    components each side actually writes.  Both legs are D3D-native (no OpenGL
    clip-space convention), so positions compare directly.  Returns
    ({output_key: worst_abs_diff}, None) or (None, error_string).

    `input_domains` / `const_domains` restrict specific attributes / constants to
    engine-legal ARRAY INDICES — see _VS_INPUT_DOMAINS.  `cb_pins` pins a bank's
    fields to raw words (see compare_d3d11)."""
    nm = name_map or {}
    pins, pinned_banks = _split_pins(cb_pins)

    def cn(n):
        return nm.get(n, n)

    rp = di.Program.from_text(ref_asm)
    cp = di.Program.from_text(cand_asm)

    # The candidate leg needs slangc=True on its INPUTS too, not just its outputs:
    # the ×10 index inflation hits every indexed semantic, so a vertex attribute
    # declared COLOR1 arrives as COLOR10.  Without this the two legs key that
    # attribute differently and each gets its own random value — a divergence in
    # whatever varying reads it, and nowhere else (index-0 attributes are unaffected
    # because 0*10 == 0, which is why the pixel path never surfaced it).
    r_in = _in_keys(ref_asm, slangc=_sig_inflated(ref_asm))
    c_in = _in_keys(cand_asm, slangc=_sig_inflated(cand_asm))
    r_cb, c_cb = _cb(ref_asm), _cb(cand_asm)
    r_out = _out_keys(ref_asm, slangc=_sig_inflated(ref_asm))
    c_out = _out_keys(cand_asm, slangc=_sig_inflated(cand_asm))
    r_by_k = {k: reg for reg, k in r_out.items()}
    c_by_k = {k: reg for reg, k in c_out.items()}
    shared = sorted(set(r_by_k) & set(c_by_k), key=lambda k: str(k))
    if not shared:
        return None, "no shared VS output"
    _only = _unmatched(r_by_k, c_by_k)
    if _only:
        return None, "unmatched VS output %s" % _only
    r_nc, c_nc = bm.out_ncomp(ref_asm), bm.out_ncomp(cand_asm)
    r_tex, c_tex = _texmap(ref_asm, cand_asm)
    rng = random.Random(seed)
    diffs = {k: 0.0 for k in shared}

    _const_draw = _make_const_draw(rng, const_domains)

    for _ in range(trials):
        keys = set(r_in.values()) | set(c_in.values())
        vals = {}
        for k in sorted(keys, key=str):
            vals[k] = _vs_input_value(k, rng, input_domains)
        cbvals = {}
        for _slot, n, o, s in r_cb:
            if not _is_pinned(pins, pinned_banks, _slot, n):
                cbvals.setdefault(n, _const_draw(n, max(1, s // 4)))
        for _slot, n, o, s in c_cb:
            if not _is_pinned(pins, pinned_banks, _slot, cn(n)):
                cbvals.setdefault(cn(n), _const_draw(n, max(1, s // 4)))

        def mk_in(regmap):
            # vals[] already holds RAW register bit patterns (_vs_input_value).
            return {reg: list(vals[k]) for reg, k in regmap.items()}

        def mk_cb(cb_list, asm, rename):
            """{slot: rows} for every constant bank the shader declares.

            A field's offset is relative to ITS bank, so the banks are built
            separately.  Bank 0 is always materialised even when the shader declares
            no constants at all, matching the pre-R0 behaviour."""
            banks = {}
            for slot, n, o, s in cb_list:
                if slot not in banks:
                    banks[slot] = [[0, 0, 0, 0]
                                   for _ in range(bm.cb_rows(asm, slot=slot))]
                key = rename(n)
                if slot in pinned_banks:
                    continue        # whole bank written below, verbatim
                pin = pins.get((slot, key))
                if pin is not None:
                    bm.cb_write_words(banks[slot], o, pin)
                elif key in cbvals:
                    bm.cb_write(banks[slot], o, cbvals[key])
            for slot, words in pinned_banks.items():
                if slot in banks:
                    bm.cb_write_words(banks[slot], 0, words)
            if 0 not in banks:
                banks[0] = [[0, 0, 0, 0] for _ in range(bm.cb_rows(asm, slot=0))]
            return banks

        r_i, c_i = mk_in(r_in), mk_in(c_in)
        r_cbv = mk_cb(r_cb, ref_asm, lambda n: n)
        c_cbv = mk_cb(c_cb, cand_asm, cn)
        try:
            ro = di.execute(rp, r_i, r_cbv, texture=r_tex, max_loop=4096)
            co = di.execute(cp, c_i, c_cbv, texture=c_tex, max_loop=4096)
        except Exception as e:
            return None, "exec: " + str(e)[:160]
        for k in shared:
            rr, cr = r_by_k[k], c_by_k[k]
            nc = min(r_nc.get(rr, 4), c_nc.get(cr, 4))
            for l in range(nc):
                a, b = ro.f(rr, l), co.f(cr, l)
                an, bn = (a == a), (b == b)
                if an and bn:
                    d = abs(a - b)
                elif an != bn:
                    d = 1.0
                else:
                    d = 0.0
                diffs[k] = max(diffs[k], d)
    return diffs, None


# ---------------------------------------------------------------------------
# self-test: identity (ref vs ref -> 0) + optional slangc leg smoke
# ---------------------------------------------------------------------------
_MODEL_40 = ("40000000030bda290200011d0cc085ff045040040400012920ff050404000960ff"
             "1904502129ff0604140009ff030120ff2092dbb266ff1dde80ff07c0ff2f0800"
             "11ff080100000001000814")


def _self_test(slang_file=None, slang_entry=None):
    import sc2_d3d_decode as D, sc2_cache, sc2_model_key_decode as K, sc2_interp as ip
    all_b, _ = h.scan_b_tokens(h.SRC + r"\model.fx")
    KNOWN = set(ip.INTERP_DIM) | set(ip.ARRAY_INTERP) | set(ip.SPECIAL_READS)
    scratch = os.environ.get("SC2_SCRATCH", os.path.join(HERE, "_slang_tmp"))
    os.makedirs(scratch, exist_ok=True)
    cache = D.D3DCache(); _d, _v, recs = sc2_cache.parse_cache(D.D3D_CACHE)
    r = next(r for r in recs if r["key"] == bytes.fromhex(_MODEL_40))
    (_a, _b), vec = sc2_cache.decode_key(r["key"])
    bv, live = K.decode_ps(vec, all_b, KNOWN)
    ref, err = compile_reference(h.SRC + r"\model.fx", "DefaultPixelMain", "ps",
                                 bv, live, os.path.join(scratch, "ref.fx"))
    if ref is None:
        print("REFERENCE COMPILE FAILED:", err); return 1
    print("reference: ps_5_0 OK  (%d instrs)" % len(di.Program.from_text(ref).insns))
    diffs, cerr = compare_d3d11(ref, ref, trials=8)
    if cerr:
        print("IDENTITY COMPARE ERROR:", cerr); return 1
    worst = max(diffs.values())
    print("identity (ref vs ref): per-target=%s  worst=%.6g  %s" %
          ({t: round(v, 6) for t, v in diffs.items()}, worst,
           "PASS" if worst == 0.0 else "FAIL"))
    rc = 0 if worst == 0.0 else 1
    if slang_file:
        casm, e2 = compile_slang(slang_file, slang_entry or "main", "ps", [],
                                 os.path.join(scratch, "cand"))
        if casm is None:
            print("slangc leg: FAILED -", e2)
        else:
            p = di.Program.from_text(casm)
            print("slangc leg: %s -> %s OK  (model=%s, %d instrs)" %
                  (os.path.basename(slang_file), slang_entry, p.model, len(p.insns)))
    return rc


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="slang<->original D3D11 validation harness")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--slang", help="optional .slang file to exercise the slangc leg")
    ap.add_argument("--entry", default="main")
    args = ap.parse_args()
    if args.self_test:
        sys.exit(_self_test(args.slang, args.entry))
    ap.print_help()
