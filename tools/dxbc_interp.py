"""A small bit-accurate interpreter for disassembled D3D shader bytecode.

It executes the *text* disassembly that 3Dmigoto's ``cmd_Decompiler -d``
emits (``ps_5_0`` / ``vs_5_0`` … assembly) one invocation at a time, with
every register modelled as four 32-bit lanes. The point is **differential
testing**: run a hand-written (slang-compiled) shader and the retail shader
it reimplements on the *same* random inputs and check the outputs agree to a
floating-point tolerance — far stronger evidence of equivalence than eyeballing
two assembly listings, and it pinpoints which input/branch diverges.

What it does NOT model (single-invocation limits — treat residuals here as
harness noise, not bugs):
  * Real texture contents. ``TextureModel`` returns a smooth, deterministic,
    Lipschitz stand-in so that *both* shaders sampling the same coord get the
    same texel; channel ordering / swizzles are honoured so genuine
    channel-mixups still surface.
  * Screen-space derivatives (``deriv_*`` / ``sample`` LOD). A single pixel has
    no neighbours; we emit a smooth deterministic stand-in so both shaders stay
    in lockstep. Anything that depends on the *real* gradient (specular-AA) will
    show a residual — that is a limit, not a divergence.

Scope: Shader Model 4/5 (``ps_4_0`` … ``ps_5_0`` / ``vs_*``). Covers scalar /
vector ALU, flow control (if/else, loop, switch, break[c], discard), int/uint
ops, two-destination ops (``imul``, ``sincos``), and the texture ops the
Warcraft-III Reforged shader set uses. NOT covered (these raise, so a shader
that needs them fails loudly rather than miscomputing): UAV / typed-load /
gather / bitfield-heavy compute ops (``ld``, ``gather4``, ``bfi``, ``store_uav``,
``countbits``) and legacy SM1-3 mnemonics (``texld``, ``texkill``, ``lrp``,
``cmp``). Run ``python tools/dxbc_interp.py`` to execute the built-in self-test.

Typical use (see ``tools/shader_diff.py`` for the batteries-included harness)::

    from dxbc_interp import Program, execute
    prog = Program.from_file("perm_0192.asm")
    out  = execute(prog, inputs={1: [..4 uints..]}, cbufs={2: [[..],..]})
    rgba = [out.f(0, k) for k in range(4)]   # SV_Target0 as floats
"""

import math
import re
import struct
from collections import defaultdict

# --------------------------------------------------------------------------
# bit helpers — registers are stored as raw uint32 lanes; reinterpret per op
# --------------------------------------------------------------------------

def f2b(f):
    """float -> uint32 bit pattern (saturating to +/-inf on overflow)."""
    try:
        return struct.unpack('<I', struct.pack('<f', f))[0]
    except (OverflowError, struct.error):
        return 0x7F800000 if f > 0 else 0xFF800000

def b2f(b):
    """uint32 bit pattern -> float."""
    return struct.unpack('<f', struct.pack('<I', b & 0xFFFFFFFF))[0]

def b2i(b):
    """uint32 bit pattern -> signed int32."""
    b &= 0xFFFFFFFF
    return b - 0x100000000 if b >= 0x80000000 else b

def i2b(i):
    """signed/unsigned int -> uint32 bit pattern."""
    return i & 0xFFFFFFFF

_SW = {'x': 0, 'y': 1, 'z': 2, 'w': 3}

# --------------------------------------------------------------------------
# operand parsing
# --------------------------------------------------------------------------

def _win_float(n):
    """float() that also accepts MSVC/fxc special printouts (1.#INF, -1.#IND, #QNAN)."""
    try:
        return float(n)
    except ValueError:
        low = n.lower()
        sign = -1.0 if low.startswith('-') else 1.0
        if '#inf' in low:
            return sign * float('inf')
        if '#ind' in low or '#qnan' in low or '#snan' in low or '#nan' in low:
            return float('nan')
        raise


def _parse_lit(t):
    """Parse an ``l(a, b, c, d)`` immediate into four uint32 lanes."""
    nums = t[2:t.rindex(')')].split(',')
    bits = []
    for n in nums:
        n = n.strip()
        if ('.' in n) or ('e' in n) or ('E' in n) and 'x' not in n:
            bits.append(f2b(_win_float(n)))
        elif n.startswith('0x') or n.startswith('-0x'):
            bits.append(int(n, 16) & 0xFFFFFFFF)
        else:
            bits.append(i2b(int(n)))
    while len(bits) < 4:
        bits.append(bits[-1] if bits else 0)
    return bits[:4]

def _parse_index(idx):
    """Parse a ``[..]`` register index: constant, or dynamic ``r#.c (+ off)``."""
    if idx is None:
        return None
    e = idx[1:-1].strip()
    mm = re.match(r"^r(\d+)\.([xyzw])\s*\+\s*(\d+)$", e)
    if mm:
        return ('dyn', int(mm.group(1)), _SW[mm.group(2)], int(mm.group(3)))
    mm = re.match(r"^r(\d+)\.([xyzw])$", e)
    if mm:
        return ('dyn', int(mm.group(1)), _SW[mm.group(2)], 0)
    return ('const', int(e))

def _parse_icb(text):
    """Parse a ``dcl_immediateConstantBuffer { {a,b,c,d}, ... }`` body.

    fxc emits the icb when a shader indexes a small constant table dynamically —
    notably a dynamic VECTOR subscript, which it lowers to ``dp4 dst, src,
    icb[i]`` against the four basis vectors (image.fx's per-layer alpha CHANNEL
    selector is exactly this).

    Each dword prints either as a float (``1.000000``), a bare integer, or hex.
    A bare ``0`` is the same 32 bits either way, so only the decimal-point /
    exponent form needs the float reading."""
    rows = []
    for row in re.findall(r"\{([^{}]*)\}", text[text.find("{") + 1:]):
        lanes = []
        for tok in row.split(","):
            tok = tok.strip()
            if not tok:
                continue
            if tok.lower().startswith("0x"):
                lanes.append(int(tok, 16) & 0xFFFFFFFF)
            elif "." in tok or "e" in tok.lower():
                lanes.append(f2b(float(tok)))
            else:
                lanes.append(i2b(int(tok)))
        while len(lanes) < 4:
            lanes.append(0)
        rows.append(lanes[:4])
    return rows


_PRECISE_RE = re.compile(r"^\[precise(?:\([xyzw]+\))?\]\s*")

def _strip_precise(t):
    """Drop fxc's ``[precise(xyz)]`` or ``[precise]`` dest annotation -- it
    constrains the COMPILER's reassociation, not the arithmetic, so it is
    inert here."""
    return _PRECISE_RE.sub("", t.strip())


def _parse_operand(t):
    """Parse a source operand -> tagged tuple (literal or register reference)."""
    t = _strip_precise(t)
    neg = absf = False
    if t.startswith('-'):
        neg = True; t = t[1:]
    if t.startswith('|') and t.endswith('|'):
        absf = True; t = t[1:-1]
    if t.startswith('l('):
        return ('lit', _parse_lit(t), neg, absf)
    m = re.match(r"^([a-z_]+)(\d*)(\[[^\]]+\])?(?:\.([xyzw]+))?$", t)
    base, num, idx, sw = m.group(1), m.group(2), m.group(3), m.group(4)
    swz = [_SW[c] for c in sw] if sw else [0, 1, 2, 3]
    return ('reg', base, int(num) if num else 0, _parse_index(idx), swz, neg, absf)

#: pixel-shader system-value destinations, which fxc prints in camel case
#: (``oMask``, ``oDepthLE``) and which carry no register number.
_SV_DESTS = {'omask': 'omask', 'odepth': 'odepth',
             'odepthle': 'odepth', 'odepthge': 'odepth'}

def _parse_dest(t):
    """Parse a destination operand -> (base, num, index, written-component-list)."""
    t = _strip_precise(t)
    sv = _SV_DESTS.get(t.split('.')[0].lower())
    if sv:
        return (sv, 0, None, [0, 1, 2, 3])
    m = re.match(r"^([a-z_]+)(\d*)(\[[^\]]+\])?(?:\.([xyzw]+))?$", t)
    base, num, idx, sw = m.group(1), m.group(2), m.group(3), m.group(4)
    comps = [_SW[c] for c in sw] if sw else [0, 1, 2, 3]
    return (base, int(num) if num else 0, _parse_index(idx), comps)

def _split_opcode(s):
    """Split ``"opcode operands"`` at the first TOP-LEVEL space.

    Resource-typed opcodes carry parenthesised annotations that themselves
    contain spaces and commas -- e.g.
    ``ld_structured_indexable(structured_buffer, stride=192)(mixed,...) r6.xyz, ...``
    -- so a plain ``split(None, 1)`` would cut the opcode in half.
    """
    depth = 0
    for i, ch in enumerate(s):
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        elif ch.isspace() and depth == 0:
            return s[:i], s[i:].strip()
    return s, ""


def _split_operands(s):
    """Split an operand list on top-level commas (ignoring those inside ()/[])."""
    out = []; depth = 0; cur = ""
    for ch in s:
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        if ch == ',' and depth == 0:
            out.append(cur.strip()); cur = ""
        else:
            cur += ch
    if cur.strip():
        out.append(cur.strip())
    return out

# --------------------------------------------------------------------------
# Program — parse a disassembly file into a signature + flat instruction list
# + a nested control-flow AST ready to execute.
# --------------------------------------------------------------------------

_MODEL_RE = re.compile(r"^(ps|vs|gs|hs|ds|cs)_\d_\d$")
_SIG_RE   = re.compile(r"//\s+(\w+)\s+(\d+)\s+(\S+)\s+(\d+)\s+")

class Program:
    """A parsed shader: signatures, system-value inputs, temps, and AST.

    Attributes:
        model:        e.g. ``"ps_5_0"``.
        input_sig:    list of ``(semantic_name, index, mask, register)``.
        output_sig:   same shape, for outputs.
        sysval_regs:  ``{name: register}`` for ``dcl_input_ps_sgv`` (e.g.
                      ``{"is_front_face": 1}``) — set these from the harness.
        num_temps:    ``dcl_temps`` count.
        insns:        flat ``[(op, operand_string), ...]``.
        ast:          nested control-flow tree (see ``execute``).
    """

    def __init__(self):
        self.model = None
        self.input_sig = []
        self.output_sig = []
        self.sysval_regs = {}
        self.num_temps = 12
        self.insns = []
        self.ast = []
        self.icb = []       # dcl_immediateConstantBuffer rows, as 4 uint32 lanes

    @classmethod
    def from_file(cls, path):
        with open(path, encoding='utf-8', errors='replace') as fh:
            return cls.from_text(fh.read())

    @classmethod
    def from_text(cls, text):
        p = cls()
        p._parse(text.splitlines())
        p.ast, _ = _build_ast(p.insns, 0, frozenset())
        return p

    def _parse(self, lines):
        started = False
        section = None   # 'in' | 'out' | None
        icb_acc = None   # accumulating a multi-line dcl_immediateConstantBuffer
        for raw in lines:
            line = raw.rstrip()
            s = line.strip()
            if not started:
                if s.startswith("// Input signature"):
                    section = 'in'; continue
                if s.startswith("// Output signature"):
                    section = 'out'; continue
                if section and s.startswith("//"):
                    m = _SIG_RE.match(line)
                    if m:
                        entry = (m.group(1).upper(), int(m.group(2)), m.group(3), int(m.group(4)))
                        (self.input_sig if section == 'in' else self.output_sig).append(entry)
                    continue
                if _MODEL_RE.match(s):
                    self.model = s; started = True
                    section = None
                continue
            # --- shader body ---
            if s.startswith("//") or s == "":
                continue
            # dcl_immediateConstantBuffer is the one declaration that WRAPS over
            # several lines; its `{ 0, 1.000000, 0, 0},` continuations would
            # otherwise be handed to the instruction parser as opcodes.
            if icb_acc is not None:
                icb_acc += " " + s
                if icb_acc.count("{") == icb_acc.count("}"):
                    self.icb = _parse_icb(icb_acc)
                    icb_acc = None
                continue
            if s.startswith("dcl_immediateConstantBuffer"):
                if s.count("{") == s.count("}"):
                    self.icb = _parse_icb(s)
                else:
                    icb_acc = s
                continue
            if s.startswith("dcl_temps"):
                self.num_temps = int(s.split()[1]); continue
            if s.startswith("dcl_input") and "_sgv" in s:
                mm = re.search(r"v(\d+)", s)
                name = s.split(',')[-1].strip() if ',' in s else 'sgv'
                if mm:
                    self.sysval_regs[name] = int(mm.group(1))
                continue
            if s.startswith("dcl_"):
                continue
            # `ret` is kept as an instruction (not a parse terminator): an early
            # ret inside an if-block (clip-plane discard-and-return) must survive so
            # _build_ast sees the matching endif and the instructions after it.
            if s == "ret":
                self.insns.append(("ret", ""))
                continue
            op, rest = _split_opcode(s)
            self.insns.append((op, rest))


def _build_ast(insns, i, terms):
    """Turn the flat instruction list into a nested control-flow tree.

    Node shapes: ``('op', op, dest, srcs, raw_ops)``, ``('if', want_nz, cond,
    then, else)``, ``('loop', body)``, ``('switch', sel, [(val|None, body)])``,
    ``('break',)``, ``('breakcz', cond, want_zero)``, ``('discard', cond)``.
    """
    nodes = []
    while i < len(insns):
        op, rest = insns[i]
        if op in terms:
            return nodes, i
        if op in ('if_nz', 'if_z'):
            cond = _parse_operand(rest)
            then_n, j = _build_ast(insns, i + 1, frozenset({'else', 'endif'}))
            else_n = []
            if insns[j][0] == 'else':
                else_n, j = _build_ast(insns, j + 1, frozenset({'endif'}))
            nodes.append(('if', op == 'if_nz', cond, then_n, else_n)); i = j + 1
        elif op == 'loop':
            body, j = _build_ast(insns, i + 1, frozenset({'endloop'}))
            nodes.append(('loop', body)); i = j + 1
        elif op == 'switch':
            sel = _parse_operand(rest); j = i + 1; cases = []
            while insns[j][0] != 'endswitch':
                cop, crest = insns[j]
                if cop == 'case':
                    val = _parse_lit(crest if crest.startswith('l(') else 'l(%s)' % crest)[0]; j += 1
                elif cop == 'default':
                    val = None; j += 1
                else:
                    j += 1; continue
                body, j = _build_ast(insns, j, frozenset({'case', 'default', 'endswitch'}))
                cases.append((val, body))
            nodes.append(('switch', sel, cases)); i = j + 1
        elif op == 'break':
            nodes.append(('break',)); i += 1
        elif op == 'breakc_z':
            nodes.append(('breakcz', _parse_operand(rest), True)); i += 1
        elif op == 'breakc_nz':
            nodes.append(('breakcz', _parse_operand(rest), False)); i += 1
        elif op == 'continue':
            nodes.append(('continue',)); i += 1
        elif op == 'continuec_z':
            nodes.append(('continuecz', _parse_operand(rest), True)); i += 1
        elif op == 'continuec_nz':
            nodes.append(('continuecz', _parse_operand(rest), False)); i += 1
        elif op == 'discard_nz':
            nodes.append(('discard', _parse_operand(rest), True)); i += 1
        elif op == 'discard_z':
            nodes.append(('discard', _parse_operand(rest), False)); i += 1
        elif op == 'ret':
            nodes.append(('ret',)); i += 1
        elif op in ('retc_nz', 'retc_z'):
            nodes.append(('retc', _parse_operand(rest), op == 'retc_nz')); i += 1
        else:
            ops = _split_operands(rest)
            dest = _parse_dest(ops[0]) if ops else None
            srcs = [_parse_operand(o) for o in ops[1:]] if dest else []
            nodes.append(('op', op, dest, srcs, ops)); i += 1
    return nodes, i

# --------------------------------------------------------------------------
# texture model — smooth deterministic stand-in for real texture contents
# --------------------------------------------------------------------------

class TextureModel:
    """Coordinate -> texel stand-in. Override ``sample`` for custom contents.

    Must be a *function of the coordinate only* (and the slot), so that two
    shaders sampling the same slot at the same coord get the same texel. The
    default is a smooth (Lipschitz) field, which keeps floating-point error
    well-behaved and makes channel-ordering bugs visible.
    """

    #: reported by ``resinfo`` (width, height, depth, mip-count).
    dims = (1024, 1024, 0, 11)

    def sample(self, slot, coords):
        out = []
        for c in range(4):
            s = slot * 0.71 + c * 1.31
            v = 0.0
            for j, co in enumerate(coords):
                v += math.sin(co * (0.45 + 0.11 * c + 0.07 * j) + s + j * 0.5)
            out.append(0.5 + 0.4 * math.sin(v + s))
        return out

    def sample_lod(self, slot, coords, lod):
        # Fold the LOD into the coord so distinct mips read distinct texels.
        return self.sample(slot, [coords[0], coords[1], coords[2], coords[3] + lod * 0.01])

    def sample_compare(self, slot, coords, ref):
        # PCF result is a single scalar broadcast to all channels.
        return self.sample(slot, [coords[0], coords[1], ref, 0.0])[0]

    def lod(self, slot, coords):
        v = sum(math.sin(co * 0.3 + slot + j) for j, co in enumerate(coords))
        return 2.5 + 2.0 * math.sin(v)   # smooth, ~[0.5, 4.5]


class StructuredModel:
    """``t#`` structured/raw buffer stand-in for ``ld_structured`` / ``ld_raw``.

    Same contract as :class:`TextureModel`: a pure function of
    (slot, element index, byte offset) so two shaders reading the same element
    see identical bits. Values land in a modest float range because these
    buffers hold material/instance parameters that get multiplied into colours.
    """

    def load(self, slot, index, byte_offset):
        out = []
        for c in range(4):
            v = math.sin(slot * 1.7 + index * 0.37 + (byte_offset + 4 * c) * 0.013)
            out.append(f2b(0.5 + 0.4 * v))
        return out


# --------------------------------------------------------------------------
# execution
# --------------------------------------------------------------------------

class _Break(Exception):
    pass

class _Continue(Exception):
    pass

class _Ret(Exception):
    """Early `ret` inside a control-flow block (e.g. a clip-plane discard that
    returns the current output values).  Unwinds to the top-level run()."""
    pass

class Outputs:
    """Result of ``execute``: output registers plus the discard flag."""

    def __init__(self, regs, discarded, coverage=None, depth=None):
        self.regs = regs            # {register_number: [4 uint32 lanes]}
        self.discarded = discarded
        #: SV_Coverage (``oMask``) as a uint32, or None if the shader never wrote it.
        self.coverage = coverage
        #: SV_Depth (``oDepth``/``oDepthLE``/``oDepthGE``) as a float, else None.
        self.depth = depth

    def bits(self, reg, lane):
        return self.regs.get(reg, [0, 0, 0, 0])[lane]

    def f(self, reg, lane):
        return b2f(self.bits(reg, lane))

    def i(self, reg, lane):
        return b2i(self.bits(reg, lane))


class _VM:
    def __init__(self, program, cbufs, inputs, texture, deriv_scale):
        self.r = [[0, 0, 0, 0] for _ in range(max(program.num_temps, 12))]
        self.cb = cbufs
        self.icb = program.icb
        self.v = inputs
        self.x = {}
        self.o = defaultdict(lambda: [0, 0, 0, 0])
        # SV_Coverage / SV_Depth are separate write targets, not o# registers.
        self.omask = [0, 0, 0, 0]
        self.odepth = [0, 0, 0, 0]
        self.wrote_omask = False
        self.wrote_odepth = False
        self.tex = texture
        self.deriv_scale = deriv_scale
        self.discarded = False

    # operand resolution ----------------------------------------------------
    def _idx(self, spec):
        return spec[1] if spec[0] == 'const' else b2i(self.r[spec[1]][spec[2]]) + spec[3]

    def _base(self, base, num, spec):
        if base == 'r':
            return self.r[num]
        if base == 'v':
            # Inputs can be indexed dynamically (`dcl_indexrange` + `v[r1.w + 2]`),
            # which fxc emits for a loop over consecutive interpolants — the
            # postprocessquad.fx gaussian-blur tap loop is exactly that.  In that
            # form the operand carries NO base number, so the register index is the
            # whole index expression; `v2` (no brackets) keeps using its number.
            return self.v.get(self._idx(spec) if spec is not None else num,
                              [0, 0, 0, 0])
        if base == 'o':
            return self.o[self._idx(spec) if spec is not None else num]
        if base == 'x':
            if num not in self.x:
                self.x[num] = [[0, 0, 0, 0] for _ in range(64)]
            return self.x[num][self._idx(spec)]
        if base == 'cb':
            # A bare list subscript raises on overflow but silently WRAPS a negative
            # index to the end of the bank, returning a real, plausible row -- so a
            # shader indexing a constant buffer with garbage could corrupt a
            # comparison instead of failing it.  Raise on both sides.
            #
            # Deliberately NOT returning 0 the way D3D does for an out-of-bounds
            # constant read: `cb_rows` is a padded heuristic (declared + 4, floor 260
            # for bank 0) whose own docstring notes that reflection under-reports
            # dynamically-indexed arrays, so zero-filling would also swallow genuine
            # under-allocation.  A real out-of-range index here has always meant a
            # broken shader or an unconstrained draw -- it is how the
            # b_iRibbonSizeInterpolation miscompile was found -- and both deserve to
            # be loud.
            i = self._idx(spec)
            bank = self.cb[num]
            if not (0 <= i < len(bank)):
                raise IndexError("cb%d index %d outside [0,%d)" % (num, i, len(bank)))
            return bank[i]
        if base == 'icb':
            # Read-only literal table from dcl_immediateConstantBuffer.  An index
            # past the end is a real shader bug (or an out-of-domain constant
            # feeding the subscript), so it must not silently read row 0.
            i = self._idx(spec)
            if not (0 <= i < len(self.icb)):
                raise IndexError("icb index %d outside [0,%d)" % (i, len(self.icb)))
            return self.icb[i]
        if base == 'omask':
            self.wrote_omask = True
            return self.omask
        if base == 'odepth':
            self.wrote_odepth = True
            return self.odepth
        if base == 'null':
            return None
        raise NotImplementedError("operand base " + base)

    def _raw(self, opd):
        if opd[0] == 'lit':
            return opd[1][:]
        _, base, num, spec, swz, _, _ = opd
        arr = self._base(base, num, spec)
        return [arr[swz[k] if k < len(swz) else swz[-1]] for k in range(4)]

    def bitread(self, opd):
        """Raw bits, with any FLOAT source modifier applied.

        `mov` and `movc` are type-agnostic, so reading them through `_raw` looks
        right -- and silently drops `-` and `| |`.  fxc folds a constant
        `1 - cColor` into `mov r2.xyz, -r2.xyzx` followed by `add ..., l(1,1,1,1)`,
        and a plain bit copy turns that into `1 + cColor`.  The same shader with the
        value read from a buffer instead is emitted as `add ..., -r4.xyzw,
        l(1,1,1,1)` -- negate on `add`, which fread already honours -- so the two
        forms disagreed and the DYNAMIC one was the correct leg.

        Modifiers are only ever printed on a float operand, so applying them means
        taking the float path for exactly those operands and staying a bit copy
        otherwise, which is what preserves integer payloads that are not valid
        floats (and NaN bit patterns that a float round-trip would canonicalise).
        """
        neg, absf = (opd[2], opd[3]) if opd[0] == 'lit' else (opd[5], opd[6])
        if not (neg or absf):
            return self._raw(opd)
        return [f2b(x) for x in self.fread(opd)]

    def fread(self, opd):
        vals = [b2f(x) for x in self._raw(opd)]
        neg, absf = (opd[2], opd[3]) if opd[0] == 'lit' else (opd[5], opd[6])
        if absf:
            vals = [abs(x) for x in vals]
        if neg:
            vals = [-x for x in vals]
        return vals

    def iread(self, opd):
        vals = [b2i(x) for x in self._raw(opd)]
        neg = opd[2] if opd[0] == 'lit' else opd[5]
        return [-x for x in vals] if neg else vals

    def uread(self, opd):
        return [x & 0xFFFFFFFF for x in self._raw(opd)]

    # writeback (channel-aligned: dest channel c <- ALU lane c) --------------
    def write_bits(self, dest, lane_bits, sat=False):
        base, num, spec, comps = dest
        if base == 'null':
            return
        arr = self._base(base, num, spec)
        for c in comps:
            b = lane_bits[c]
            if sat:
                b = f2b(min(1.0, max(0.0, b2f(b))))
            arr[c] = b & 0xFFFFFFFF

    def write_f(self, dest, fvals, sat=False):
        if sat:
            fvals = [min(1.0, max(0.0, x)) for x in fvals]
        self.write_bits(dest, [f2b(x) for x in fvals], False)

    def write_i(self, dest, ivals):
        self.write_bits(dest, [i2b(x) for x in ivals], False)


def _cond_lane(opd):
    """Scalar-condition lane: a literal uses lane 0, a register its 1st swizzle."""
    return 0 if opd[0] == 'lit' else opd[4][0]


def execute(program, inputs=None, cbufs=None, *, texture=None, structured=None,
            deriv_scale=1.0, max_loop=1 << 16):
    """Run ``program`` once.

    Args:
        program:  a :class:`Program`.
        inputs:   ``{register: [4 uint32 lanes]}`` for ``v#`` input registers.
        cbufs:    ``{slot: [[4 uint32 lanes], ...]}`` for ``cb#`` constant buffers.
        texture:  a :class:`TextureModel` (defaults to the smooth stand-in).
        structured: a :class:`StructuredModel` for ``ld_structured`` / ``ld_raw``
                  reads (defaults to the deterministic stand-in).
        deriv_scale: multiplier on the synthetic screen-space derivative.
        max_loop: per-loop iteration cap. Exceeding it raises ``RuntimeError``
                  rather than hanging — almost always means a constant buffer
                  feeding a loop count (e.g. light count) was left as garbage
                  random bits instead of being driven to a sane integer.

    Returns:
        :class:`Outputs`.
    """
    vm = _VM(program, cbufs or {}, inputs or {}, texture or TextureModel(), deriv_scale)
    vm.sbuf = structured or StructuredModel()

    def run(nodes):
        for n in nodes:
            t = n[0]
            if t == 'op':
                _do_op(vm, n)
            elif t == 'if':
                _, want_nz, cond, then_n, else_n = n
                c = vm.uread(cond)[_cond_lane(cond)]
                run(then_n if ((c != 0) == want_nz) else else_n)
            elif t == 'loop':
                it = 0
                while True:
                    try:
                        run(n[1])
                    except _Break:
                        break
                    except _Continue:
                        pass    # skip rest of body, fall through to next iteration
                    it += 1
                    if it > max_loop:
                        raise RuntimeError(
                            f"loop exceeded {max_loop} iterations -- is a loop-count "
                            "constant buffer undriven (garbage random bits)?")
            elif t == 'switch':
                _, sel, cases = n
                sv = b2i(vm._raw(sel)[_cond_lane(sel)])
                start = next((k for k, (val, _) in enumerate(cases) if val == sv), None)
                if start is None:
                    start = next((k for k, (val, _) in enumerate(cases) if val is None), None)
                if start is not None:
                    try:
                        for k in range(start, len(cases)):
                            run(cases[k][1])
                    except _Break:
                        pass
            elif t == 'break':
                raise _Break()
            elif t == 'breakcz':
                if (vm.uread(n[1])[_cond_lane(n[1])] == 0) == n[2]:
                    raise _Break()
            elif t == 'continue':
                raise _Continue()
            elif t == 'continuecz':
                if (vm.uread(n[1])[_cond_lane(n[1])] == 0) == n[2]:
                    raise _Continue()
            elif t == 'discard':
                if (vm.uread(n[1])[_cond_lane(n[1])] != 0) == n[2]:
                    vm.discarded = True
            elif t == 'ret':
                raise _Ret()
            elif t == 'retc':
                if (vm.uread(n[1])[_cond_lane(n[1])] != 0) == n[2]:
                    raise _Ret()

    try:
        run(program.ast)
    except _Ret:
        pass                    # early return: keep whatever outputs were written
    return Outputs(dict(vm.o), vm.discarded,
                   coverage=vm.omask[0] if vm.wrote_omask else None,
                   depth=b2f(vm.odepth[0]) if vm.wrote_odepth else None)


def _texslot(tok):
    return int(re.match(r"t(\d+)", tok).group(1))

def _apply_swz(texel, swz):
    """Reorder a sampled texel by the resource operand's swizzle (e.g. t13.wxyz)."""
    return [texel[swz[k] if k < len(swz) else swz[-1]] for k in range(4)]


def _operand_as_dest(opd):
    """Reinterpret a parsed source operand as a destination (for two-dest ops).

    Two-destination instructions (``imul``, ``sincos``, …) print as
    ``op dstA, dstB, src...``; the generic parser keeps ``dstA`` as the node's
    dest and ``dstB`` lands in ``srcs[0]`` — convert it back to a write target.
    """
    # ('reg', base, num, index, swizzle, neg, abs) -> (base, num, index, comps)
    return (opd[1], opd[2], opd[3], opd[4])


def _do_op(vm, node):
    _, op, dest, srcs, raw = node
    sat = op.endswith('_sat')
    base = op[:-4] if sat else op

    # ---- two-destination ops (operand list is [dstA, dstB, src...]) ----
    if base == 'imul':
        # imul destHI, destLO, src0, src1 (destHI is usually null). 64-bit
        # product split into high/low halves.
        lo = _operand_as_dest(srcs[0]); a = vm.iread(srcs[1]); b = vm.iread(srcs[2])
        prod = [a[k] * b[k] for k in range(4)]
        vm.write_bits(lo, [p & 0xFFFFFFFF for p in prod], False)
        vm.write_bits(dest, [(p >> 32) & 0xFFFFFFFF for p in prod], False)   # destHI
        return
    if base == 'sincos':
        # sincos destSin, destCos, src (either dest may be null)
        cos = _operand_as_dest(srcs[0]); a = vm.fread(srcs[1])
        vm.write_f(dest, [math.sin(x) for x in a], sat)
        vm.write_f(cos, [math.cos(x) for x in a], sat)
        return
    if base == 'udiv':
        # udiv destQuotient, destRemainder, src0, src1 (either dest may be null).
        # Per the D3D spec, divide-by-zero yields 0xFFFFFFFF in BOTH destinations
        # rather than faulting — the flipbook UV path (`iCell % columnCount`)
        # relies on this being total.
        rem = _operand_as_dest(srcs[0])
        a = vm.uread(srcs[1]); b = vm.uread(srcs[2])
        q = [(a[k] // b[k]) if b[k] else 0xFFFFFFFF for k in range(4)]
        r = [(a[k] % b[k]) if b[k] else 0xFFFFFFFF for k in range(4)]
        vm.write_bits(dest, q, False)
        vm.write_bits(rem, r, False)
        return

    # ---- texture ops ----
    if base.startswith('ld_structured') or base.startswith('ld_raw'):
        # ld_structured dest, srcElementIndex, srcByteOffset, srcResource
        # ld_raw        dest, srcByteOffset, srcResource
        if base.startswith('ld_structured'):
            idx = vm.iread(srcs[0])[0]; off = vm.iread(srcs[1])[0]; res = srcs[2]
        else:
            idx = 0; off = vm.iread(srcs[0])[0]; res = srcs[1]
        raw = vm.sbuf.load(res[2], idx, off)
        vm.write_bits(dest, _apply_swz(raw, res[4]), False); return
    if base.startswith('ldms'):           # multisample load: sample index ignored
        coord = vm.fread(srcs[0]); res = srcs[1]
        vm.write_f(dest, _apply_swz(vm.tex.sample(res[2], coord), res[4]), False); return
    if base.startswith('ld'):             # typed texel fetch: ld dest, coord, t#
        # Integer texel coords (xyz = u,v,mip). Feed them through the same
        # smooth field as `sample` so a shader that fetches and one that samples
        # the same slot stay comparable.
        coord = [float(x) for x in vm.iread(srcs[0])]; res = srcs[1]
        vm.write_f(dest, _apply_swz(vm.tex.sample(res[2], coord), res[4]), False); return
    if base.startswith('eval_'):
        # eval_sample_index / eval_centroid / eval_snapped: per-sample attribute
        # evaluation. One invocation has a single sample, so the attribute value
        # IS the interpolated value -- pass v# through unchanged.
        vm.write_bits(dest, vm._raw(srcs[0]), False); return
    if base.startswith('sample_c'):       # sample_c / sample_c_lz (PCF compare)
        coord = vm.fread(srcs[0]); slot = srcs[1][2]; ref = vm.fread(srcs[3])[0]
        val = vm.tex.sample_compare(slot, coord, ref)
        vm.write_f(dest, [val] * 4, False); return
    if base.startswith('sample_l'):       # explicit-LOD sample
        coord = vm.fread(srcs[0]); slot = srcs[1][2]; swz = srcs[1][4]; lod = vm.fread(srcs[3])[0]
        vm.write_f(dest, _apply_swz(vm.tex.sample_lod(slot, coord, lod), swz), False); return
    if base.startswith('sample'):         # plain sample (incl. sample_indexable)
        coord = vm.fread(srcs[0]); slot = srcs[1][2]; swz = srcs[1][4]
        vm.write_f(dest, _apply_swz(vm.tex.sample(slot, coord), swz), False); return
    if base.startswith('resinfo'):
        swz = srcs[1][4]
        vm.write_bits(dest, [i2b(vm.tex.dims[swz[k]]) for k in range(4)], False); return
    if base == 'lod':
        coord = vm.fread(srcs[0]); slot = _texslot(raw[2])
        vm.write_f(dest, [vm.tex.lod(slot, coord)] * 4, False); return

    # ---- float ALU ----
    if base == 'mov':
        vm.write_bits(dest, vm.bitread(srcs[0]), sat); return
    if base == 'movc':
        c = vm.uread(srcs[0]); a = vm.bitread(srcs[1]); b = vm.bitread(srcs[2])
        vm.write_bits(dest, [a[k] if c[k] != 0 else b[k] for k in range(4)], sat); return
    if base == 'dp2':
        a = vm.fread(srcs[0]); b = vm.fread(srcs[1])
        vm.write_f(dest, [a[0] * b[0] + a[1] * b[1]] * 4, sat); return
    if base == 'dp3':
        a = vm.fread(srcs[0]); b = vm.fread(srcs[1])
        vm.write_f(dest, [sum(a[k] * b[k] for k in range(3))] * 4, sat); return
    if base == 'dp4':
        a = vm.fread(srcs[0]); b = vm.fread(srcs[1])
        vm.write_f(dest, [sum(a[k] * b[k] for k in range(4))] * 4, sat); return
    if base in ('add', 'mul', 'mad', 'div', 'min', 'max',
                'sqrt', 'rsq', 'exp', 'log', 'frc', 'round_ni', 'round_z',
                'round_ne', 'round_pi', 'rcp'):
        a = vm.fread(srcs[0])
        if base == 'add':
            b = vm.fread(srcs[1]); r = [a[k] + b[k] for k in range(4)]
        elif base == 'mul':
            b = vm.fread(srcs[1]); r = [a[k] * b[k] for k in range(4)]
        elif base == 'mad':
            b = vm.fread(srcs[1]); c = vm.fread(srcs[2]); r = [a[k] * b[k] + c[k] for k in range(4)]
        elif base == 'div':
            b = vm.fread(srcs[1]); r = [_fdiv(a[k], b[k]) for k in range(4)]
        elif base == 'min':
            b = vm.fread(srcs[1]); r = [min(a[k], b[k]) for k in range(4)]
        elif base == 'max':
            b = vm.fread(srcs[1]); r = [max(a[k], b[k]) for k in range(4)]
        elif base == 'sqrt':
            r = [math.sqrt(x) if x >= 0 else float('nan') for x in a]
        elif base == 'rsq':
            r = [(1.0 / math.sqrt(x) if x > 0 else float('inf')) for x in a]
        elif base == 'rcp':
            r = [_fdiv(1.0, x) for x in a]
        elif base == 'exp':
            # exp2; clamp double-overflow to +inf (GPU semantics), NaN passes through.
            r = []
            for x in a:
                try:
                    r.append(2.0 ** x)
                except OverflowError:
                    r.append(float('inf'))
        elif base == 'log':
            r = [(math.log2(x) if x > 0 else -float('inf')) for x in a]
        elif base == 'frc':
            # NaN/inf pass through as NaN (GPU: frac of non-finite is undefined→NaN);
            # math.floor would raise on non-finite.
            r = [(x - math.floor(x)) if math.isfinite(x) else float('nan') for x in a]
        elif base == 'round_ni':
            r = [math.floor(x) if math.isfinite(x) else x for x in a]
        elif base == 'round_z':
            r = [math.trunc(x) if math.isfinite(x) else x for x in a]
        elif base == 'round_ne':
            # round-half-to-EVEN, which is exactly Python's round() for floats
            r = [float(round(x)) if math.isfinite(x) else x for x in a]
        elif base == 'round_pi':
            r = [math.ceil(x) if math.isfinite(x) else x for x in a]
        vm.write_f(dest, r, sat); return
    if base in ('lt', 'ge', 'eq', 'ne'):
        a = vm.fread(srcs[0]); b = vm.fread(srcs[1])
        cmp = {'lt': lambda x, y: x < y, 'ge': lambda x, y: x >= y,
               'eq': lambda x, y: x == y, 'ne': lambda x, y: x != y}[base]
        vm.write_bits(dest, [0xFFFFFFFF if cmp(a[k], b[k]) else 0 for k in range(4)], False); return

    # ---- integer / bitwise ----
    # (imul/sincos are two-destination ops, handled above before the
    #  generic source list is interpreted.)
    if base in ('iadd', 'imad', 'ishl', 'ishr', 'imax', 'imin', 'ieq', 'ige', 'ilt', 'ine', 'ineg'):
        a = vm.iread(srcs[0])
        if base == 'ineg':
            vm.write_i(dest, [-x for x in a]); return
        b = vm.iread(srcs[1])
        if base == 'iadd':
            vm.write_i(dest, [a[k] + b[k] for k in range(4)])
        elif base == 'imad':
            c = vm.iread(srcs[2]); vm.write_i(dest, [a[k] * b[k] + c[k] for k in range(4)])
        elif base == 'ishl':
            vm.write_i(dest, [a[k] << (b[k] & 31) for k in range(4)])
        elif base == 'ishr':
            vm.write_i(dest, [a[k] >> (b[k] & 31) for k in range(4)])
        elif base == 'imax':
            vm.write_i(dest, [max(a[k], b[k]) for k in range(4)])
        elif base == 'imin':
            vm.write_i(dest, [min(a[k], b[k]) for k in range(4)])
        else:
            cmp = {'ieq': lambda x, y: x == y, 'ige': lambda x, y: x >= y,
                   'ilt': lambda x, y: x < y, 'ine': lambda x, y: x != y}[base]
            vm.write_bits(dest, [0xFFFFFFFF if cmp(a[k], b[k]) else 0 for k in range(4)], False)
        return
    if base in ('ult', 'uge', 'ushr', 'umin', 'umax'):
        a = vm.uread(srcs[0]); b = vm.uread(srcs[1])
        if base == 'ult':
            vm.write_bits(dest, [0xFFFFFFFF if a[k] < b[k] else 0 for k in range(4)], False)
        elif base == 'uge':
            vm.write_bits(dest, [0xFFFFFFFF if a[k] >= b[k] else 0 for k in range(4)], False)
        elif base == 'ushr':
            vm.write_bits(dest, [a[k] >> (b[k] & 31) for k in range(4)], False)
        elif base == 'umin':
            vm.write_bits(dest, [min(a[k], b[k]) for k in range(4)], False)
        elif base == 'umax':
            vm.write_bits(dest, [max(a[k], b[k]) for k in range(4)], False)
        return
    if base in ('and', 'or', 'xor'):
        a = vm._raw(srcs[0]); b = vm._raw(srcs[1])
        fn = {'and': lambda x, y: x & y, 'or': lambda x, y: x | y, 'xor': lambda x, y: x ^ y}[base]
        vm.write_bits(dest, [fn(a[k], b[k]) & 0xFFFFFFFF for k in range(4)], False); return
    if base == 'not':
        a = vm._raw(srcs[0]); vm.write_bits(dest, [(~a[k]) & 0xFFFFFFFF for k in range(4)], False); return
    if base == 'bfrev':
        a = vm.uread(srcs[0])
        vm.write_bits(dest, [int('{:032b}'.format(x)[::-1], 2) for x in a], False); return
    if base == 'bfi':
        # bfi dest, width, offset, insert, base:
        #   mask = ((1 << width) - 1) << offset
        #   dest = ((insert << offset) & mask) | (base & ~mask)
        # Both fxc and slangc emit this for `i * 2^k + c` with a small literal c
        # (width/offset/c all immediates), which is how the batched Ribbon/Particle
        # roots address p_mSplineRibbonControlPoints[iBatchIndex*4 + k].
        w = vm.uread(srcs[0]); off = vm.uread(srcs[1])
        ins = vm.uread(srcs[2]); bse = vm.uread(srcs[3])
        out = []
        for k in range(4):
            wk, ok = w[k] & 31, off[k] & 31
            mask = (((1 << wk) - 1) << ok) & 0xFFFFFFFF
            out.append((((ins[k] << ok) & mask) | (bse[k] & ~mask)) & 0xFFFFFFFF)
        vm.write_bits(dest, out, False); return
    if base in ('ubfe', 'ibfe'):
        # ubfe/ibfe dest, width, offset, src: extract `width` bits at `offset`;
        # ibfe sign-extends the extracted field, ubfe zero-extends it.
        w = vm.uread(srcs[0]); off = vm.uread(srcs[1]); s = vm.uread(srcs[2])
        out = []
        for k in range(4):
            wk, ok = w[k] & 31, off[k] & 31
            if wk == 0:
                out.append(0); continue
            v = (s[k] >> ok) & ((1 << wk) - 1) if ok < 32 else 0
            if base == 'ibfe' and (v >> (wk - 1)) & 1:
                v -= (1 << wk)
            out.append(v & 0xFFFFFFFF)
        vm.write_bits(dest, out, False); return

    # ---- conversions ----
    if base == 'ftoi':
        a = vm.fread(srcs[0]); vm.write_i(dest, [int(x) if math.isfinite(x) else 0 for x in a]); return
    if base == 'ftou':
        a = vm.fread(srcs[0])
        vm.write_bits(dest, [int(x) & 0xFFFFFFFF if math.isfinite(x) and x > 0 else 0 for x in a], False); return
    if base == 'itof':
        a = vm.iread(srcs[0]); vm.write_f(dest, [float(x) for x in a], False); return
    if base == 'utof':
        a = vm.uread(srcs[0]); vm.write_f(dest, [float(x) for x in a], False); return

    # ---- bitfield ----
    if base == 'ubfe':
        width = vm.uread(srcs[0]); off = vm.uread(srcs[1]); val = vm.uread(srcs[2])
        out = []
        for k in range(4):
            w = width[k] & 31; o = off[k] & 31
            out.append(0 if w == 0 else (val[k] >> o) & ((1 << w) - 1))
        vm.write_bits(dest, out, False); return
    if base == 'ibfe':
        width = vm.uread(srcs[0]); off = vm.uread(srcs[1]); val = vm.iread(srcs[2])
        out = []
        for k in range(4):
            w = width[k] & 31; o = off[k] & 31
            if w == 0:
                out.append(0)
            else:
                x = (val[k] >> o) & ((1 << w) - 1)
                if x & (1 << (w - 1)):
                    x -= (1 << w)
                out.append(i2b(x))
        vm.write_bits(dest, out, False); return

    # ---- screen-space derivatives (single-pixel stand-in; see module docstring) ----
    if base in ('deriv_rtx', 'deriv_rty', 'deriv_rtx_coarse', 'deriv_rty_coarse',
                'deriv_rtx_fine', 'deriv_rty_fine'):
        a = vm.fread(srcs[0]); ph = 0.7 if 'rtx' in base else 2.1
        vm.write_f(dest, [vm.deriv_scale * 0.03 * math.sin(3.1 * a[k] + 1.3 * k + ph) for k in range(4)], False); return

    raise NotImplementedError("opcode " + op)


def _fdiv(a, b):
    if b != 0:
        return a / b
    return math.inf * (1 if a > 0 else -1 if a < 0 else 0)


# --------------------------------------------------------------------------
# self-test — `python tools/dxbc_interp.py`
# --------------------------------------------------------------------------

def _selftest():
    def run(body):
        return execute(Program.from_text("ps_5_0\ndcl_temps 8\n" + body + "\nret\n"))

    cases = []
    def check(name, got, want):
        cases.append((name, got, want, got == want))

    # channel-aligned masked write: dest.yz <- lanes 1,2; x,w untouched
    o = run("mov r0.xyzw, l(10,20,30,40)\nadd r1.yz, r0.zzzz, r0.wwww\nmov o0.xyzw, r1.xyzw")
    check("masked .yz write", [o.i(0, k) for k in range(4)], [0, 70, 70, 0])
    # source swizzle shorter than 4 replicates its last component
    o = run("mov r0.xyzw, l(10,20,30,40)\nmov r2.xyzw, r0.xy\nmov o0.xyzw, r2.xyzw")
    check("src .xy replicate", [o.i(0, k) for k in range(4)], [10, 20, 20, 20])
    # per-lane movc: c ? src1 : src2
    o = run("mov r0.xyzw, l(0,1,0,1)\nmovc o0.xyzw, r0.xyzw, l(5,5,5,5), l(9,9,9,9)")
    check("movc per-lane", [o.i(0, k) for k in range(4)], [9, 5, 9, 5])
    # saturate clamps to [0,1]
    o = run("mov r0.xyzw, l(-2.0,0.5,3.0,1.0)\nmov_sat o0.xyzw, r0.xyzw")
    check("mov_sat", [round(o.f(0, k), 3) for k in range(4)], [0.0, 0.5, 1.0, 1.0])
    # abs / neg source modifiers
    o = run("mov r0.x, l(-3.0)\nadd r1.x, |r0.x|, l(1.0)\nadd r1.y, -r0.x, l(0.0)\nmov o0.xy, r1.xy")
    check("abs/neg modifiers", [round(o.f(0, k), 3) for k in range(2)], [4.0, 3.0])
    # ...and mov/movc must honour them too.  They are bit copies, so the modifier
    # is easy to drop, and fxc emits `mov rN.xyz, -rN.xyzx` for every constant-folded
    # `1 - c` -- see bitread().
    o = run("mov r0.xyzw, l(1.0,2.0,3.0,4.0)\nmov r1.xyz, -r0.xyzx\nmov o0.xyzw, r1.xyzw")
    check("mov honours neg", [round(o.f(0, k), 3) for k in range(4)], [-1.0, -2.0, -3.0, 0.0])
    o = run("mov r0.xyzw, l(-1.0,2.0,-3.0,4.0)\nmov r1.xyz, |r0.xyzx|\nmov o0.xyzw, r1.xyzw")
    check("mov honours abs", [round(o.f(0, k), 3) for k in range(4)], [1.0, 2.0, 3.0, 0.0])
    o = run("mov r0.xyzw, l(1.0,2.0,3.0,4.0)\nmov r2.xyzw, l(1,1,0,0)\n"
            "movc r1.xyzw, r2.xyzw, -r0.xyzw, r0.xyzw\nmov o0.xyzw, r1.xyzw")
    check("movc honours neg", [round(o.f(0, k), 3) for k in range(4)], [-1.0, -2.0, 3.0, 4.0])
    # in-place negate, the exact shape of a constant-folded invert: r0 = 1 - r0
    o = run("mov r0.xyzw, l(1.0,2.0,3.0,4.0)\nmov r0.xyz, -r0.xyzx\n"
            "add r0.xyzw, r0.xyzw, l(1.0,1.0,1.0,1.0)\nmov o0.xyzw, r0.xyzw")
    check("in-place neg then add", [round(o.f(0, k), 3) for k in range(4)], [0.0, -1.0, -2.0, 5.0])
    # masked write with a NON-uniform source swizzle: dest .zw take the swizzle's
    # 3rd and 4th entries, not its 1st and 2nd.  The .yz case above cannot tell the
    # difference, because its source swizzle is uniform.
    o = run("mov r0.xyzw, l(10,20,30,40)\nmov r1.zw, r0.xxxy\nmov o0.xyzw, r1.xyzw")
    check("masked write, skewed swizzle", [o.i(0, k) for k in range(4)], [0, 0, 10, 20])
    # imul: low / high halves, signed
    o = run("mov r0.xyzw, l(7,3,65536,0)\nimul null, r1.x, r0.x, r0.y\n"
            "imul r2.x, r2.y, r0.z, r0.z\nmov o0.x, r1.x\nmov o0.y, r2.y\nmov o0.z, r2.x")
    check("imul low/high", [o.i(0, k) for k in range(3)], [21, 0, 1])
    o = run("mov r0.xy, l(-3,3,0,0)\nimul null, r1.x, r0.x, r0.y\nmov o0.x, r1.x")
    check("imul signed low", o.bits(0, 0) & 0xFFFFFFFF, 0xFFFFFFF7)  # -9
    # imad: a*b+c (integer)
    o = run("mov r0.xyz, l(7,3,5,0)\nimad r1.x, r0.x, r0.y, r0.z\nmov o0.x, r1.x")
    check("imad", o.i(0, 0), 26)
    # sincos: two destinations
    o = run("mov r0.w, l(1.0)\nsincos r2.x, r2.y, r0.w\nmov o0.x, r2.x\nmov o0.y, r2.y")
    check("sincos", [round(o.f(0, 0), 4), round(o.f(0, 1), 4)],
          [round(math.sin(1), 4), round(math.cos(1), 4)])
    # dp4 broadcasts to all lanes
    o = run("mov r0.xyzw, l(1.0,2.0,3.0,4.0)\ndp4 r1.x, r0.xyzw, r0.xyzw\nmov o0.x, r1.x")
    check("dp4", round(o.f(0, 0), 3), 30.0)
    # ishr is arithmetic, ushr is logical
    o = run("mov r0.x, l(0x80000000)\nishr r1.x, r0.x, l(4)\nushr r1.y, r0.x, l(4)\nmov o0.xy, r1.xy")
    check("ishr arithmetic", o.bits(0, 0) & 0xFFFFFFFF, 0xF8000000)
    check("ushr logical", o.bits(0, 1) & 0xFFFFFFFF, 0x08000000)
    # if/else flow + loop with break
    o = run("mov r0.x, l(0)\nmov r1.x, l(5)\nloop\n  iadd r0.x, r0.x, l(1)\n"
            "  ieq r2.x, r0.x, r1.x\n  breakc_nz r2.x\nendloop\nmov o0.x, r0.x")
    check("loop+breakc", o.i(0, 0), 5)
    o = run("mov r0.x, l(1)\nif_nz r0.x\n  mov o0.x, l(11)\nelse\n  mov o0.x, l(22)\nendif")
    check("if_nz taken", o.i(0, 0), 11)
    # loop with continue: sum only even counters 0..9 -> skip odds via continue
    o = run("mov r0.x, l(0)\nmov r1.x, l(0)\nloop\n  ige r2.x, r0.x, l(10)\n  breakc_nz r2.x\n"
            "  and r2.x, r0.x, l(1)\n  iadd r0.x, r0.x, l(1)\n  continuec_nz r2.x\n"
            "  iadd r1.x, r1.x, r0.x\nendloop\nmov o0.x, r1.x")
    check("loop+continue", o.i(0, 0), 1 + 3 + 5 + 7 + 9)   # counter incremented before the add
    # unsupported opcode must raise, not miscompute
    try:
        run("texkill r0.x")
        check("unsupported raises", False, True)
    except NotImplementedError:
        check("unsupported raises", True, True)

    ok = True
    for name, got, want, passed in cases:
        if not passed:
            ok = False
            print(f"  FAIL {name}: got {got!r}, want {want!r}")
    print(f"dxbc_interp self-test: {sum(c[3] for c in cases)}/{len(cases)} passed"
          + ("" if ok else "  <-- FAILURES"))
    return ok


if __name__ == '__main__':
    import sys
    sys.exit(0 if _selftest() else 1)
