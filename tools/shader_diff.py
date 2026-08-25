"""Differential tester for two disassembled shaders that should agree.

Given two shader disassemblies (a slang-compiled reimplementation and the
retail bytecode it mirrors), feed both the *same* random inputs and constant
buffers over many trials and report the worst output divergence. Built on
:mod:`dxbc_interp`.

The non-obvious part is **input mapping**. The two shaders rarely share an
input-register layout even when they're semantically identical:

  * slang names varyings with a x10 semantic index (``TEXCOORD10`` == retail's
    ``TEXCOORD1``, ``TEXCOORD70`` == ``TEXCOORD7``); retail uses the raw index.
  * retail PACKS several semantics into one register by write-mask
    (``TEXCOORD1.x`` + ``TEXCOORD7.yzw`` -> register 2); slang usually gives
    each its own register.

So inputs are generated **per semantic** (``{(name, index): [4 lanes]}``) and
:func:`map_inputs` places each semantic's components into the right register
*channels* for whichever shader it's feeding. Get this wrong and you get huge
phantom divergences that look like shader bugs but aren't.

CLI::

    python tools/shader_diff.py SLANG.asm RETAIL.asm [--trials N] [--tol T]
    python tools/shader_diff.py a.dxbc b.dxbc --decompiler C:/path/cmd_Decompiler.exe

For non-trivial shaders (driven constant-buffer discriminants, structured
inputs) import :func:`compare` and pass your own ``inputs_fn`` / ``cbufs_fn``
generators — see ``tools/shader_diff_popcorn.py`` for a worked example.
"""

import argparse
import math
import random
import struct
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dxbc_interp import Program, execute, TextureModel, f2b, b2f  # noqa: E402

_CH = {'x': 0, 'y': 1, 'z': 2, 'w': 3}


# --------------------------------------------------------------------------
# loading (optionally disassembling .dxbc on the way in)
# --------------------------------------------------------------------------

def decompile(dxbc_path, decompiler):
    """Disassemble a ``.dxbc`` to ``.asm`` next to it via 3Dmigoto. Returns the path."""
    dxbc_path = Path(dxbc_path)
    asm = dxbc_path.with_suffix('.asm')
    subprocess.run([str(decompiler), '-d', str(dxbc_path)],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if not asm.exists():
        raise RuntimeError(f"decompiler produced no .asm for {dxbc_path}")
    return asm

def load(path, decompiler=None):
    """Load a :class:`Program` from ``.asm`` (or ``.dxbc`` if ``decompiler`` given)."""
    path = Path(path)
    if path.suffix == '.dxbc':
        if not decompiler:
            raise ValueError("a .dxbc needs --decompiler to disassemble first")
        path = decompile(path, decompiler)
    return Program.from_file(path)


# --------------------------------------------------------------------------
# input mapping — semantic values -> per-shader input registers
# --------------------------------------------------------------------------

def map_inputs(program, semantic_values, sysvals=None):
    """Build ``{register: [4 lanes]}`` for ``program`` from per-semantic values.

    Args:
        program:         the :class:`Program` being fed.
        semantic_values: a mapping with ``.get((name, index))`` -> 4 lanes
                         (raw index; the x10 slang naming is handled here).
        sysvals:         ``{sysval_name: bits}`` e.g. ``{"is_front_face": 0xFFFFFFFF}``.
    """
    inp = {}
    for name, idx, mask, reg in program.input_sig:
        v = semantic_values.get((name, idx))
        if v is None and idx >= 10 and idx % 10 == 0:
            v = semantic_values.get((name, idx // 10))   # slang x10 naming
        if v is None:
            v = [0, 0, 0, 0]
        inp.setdefault(reg, [0, 0, 0, 0])
        # place this semantic's components at its masked channels, in order
        # (retail packs multiple semantics per register -> merge, don't overwrite)
        for di, c in enumerate(ch for ch in mask if ch in _CH):
            inp[reg][_CH[c]] = v[di]
    for name, bits in (sysvals or {}).items():
        if name in program.sysval_regs:
            inp[program.sysval_regs[name]] = [bits, bits, bits, bits]
    return inp


# --------------------------------------------------------------------------
# default random generators (work out of the box; override for real coverage)
# --------------------------------------------------------------------------

class RandomSemantics:
    """Lazily returns a deterministic random 4-vec for any ``(name, index)``.

    Same key -> same value within a trial, so both shaders are fed identically.
    Normals/tangents (TEXCOORD4/5 by Wc3 convention) come out unit-length.
    """

    def __init__(self, seed):
        self.seed = seed

    def get(self, key):
        name, idx = key
        rng = random.Random(f"{self.seed}:{name}:{idx}")   # str seed (tuples unhashable as seeds in 3.14)
        if name == 'TEXCOORD' and idx in (4, 5):      # normal / tangent
            v = [rng.uniform(-1, 1) for _ in range(3)]
            m = math.sqrt(sum(c * c for c in v)) or 1.0
            lanes = [f2b(c / m) for c in v]
            lanes.append(f2b(rng.choice([-1.0, 1.0])))  # handedness in .w
            return lanes
        return [f2b(rng.uniform(-2, 2)) for _ in range(4)]


class _RandomRows:
    def __init__(self, seed, slot):
        self.seed, self.slot = seed, slot

    def __len__(self):
        # D3D11 SM5 constant buffer maximum is 4096 float4 rows (256 typical).
        # Returning a large constant satisfies dxbc_interp's bounds check so
        # dynamically-indexed cbuffers do not raise instead of comparing.
        return 4096

    def __getitem__(self, index):
        rng = random.Random(f"{self.seed}:{self.slot}:{index}")
        return [f2b(rng.uniform(-1, 1)) for _ in range(4)]

class RandomCBufs:
    """``cb[slot][row]`` -> deterministic random 4-vec (read-only, lazy)."""

    def __init__(self, seed):
        self.seed = seed

    def __getitem__(self, slot):
        return _RandomRows(self.seed, slot)


def default_sysvals(seed):
    """Random front-face for PS (other sysvals default to 0)."""
    return {'is_front_face': 0xFFFFFFFF if (seed & 1) else 0}


# --------------------------------------------------------------------------
# output comparison
# --------------------------------------------------------------------------

def output_diff(out_a, out_b, regs):
    """Worst per-channel |a-b| over the given output registers.

    NaN==NaN and same-signed inf==inf are treated as equal (they signal the
    same degenerate input, not a divergence). Returns ``(worst, (reg, ch, a, b))``.
    """
    worst = 0.0; where = None
    for reg in regs:
        for k in range(4):
            a = out_a.f(reg, k); b = out_b.f(reg, k)
            if math.isnan(a) and math.isnan(b):
                continue
            if math.isinf(a) and math.isinf(b) and (a > 0) == (b > 0):
                continue
            d = abs(a - b)
            if d > worst:
                worst = d; where = (reg, k, a, b)
    return worst, where


class CompareResult:
    def __init__(self):
        self.trials = 0
        self.worst = 0.0
        self.worst_where = None       # (reg, ch, a_val, b_val)
        self.worst_seed = None
        self.discard_mismatches = 0

    def __repr__(self):
        return (f"CompareResult(trials={self.trials}, worst={self.worst:.3e}, "
                f"discard_mismatches={self.discard_mismatches})")


def compare(prog_a, prog_b, *, trials=200, output_regs=(0,), tol=1e-3,
            inputs_fn=None, cbufs_fn=None, sysvals_fn=None,
            texture=None, deriv_scale=0.0, seed0=0):
    """Run both programs over ``trials`` random draws and collect the worst diff.

    Args:
        prog_a, prog_b: the two :class:`Program` s (order is irrelevant).
        output_regs:    output registers to compare (``0`` == SV_Target0).
        tol:            informational threshold (does not stop the run).
        inputs_fn:      ``seed -> semantic_values`` (default: :class:`RandomSemantics`).
        cbufs_fn:       ``seed -> cbufs`` (default: :class:`RandomCBufs`).
        sysvals_fn:     ``seed -> {name: bits}`` (default: random front-face).
        texture:        a :class:`TextureModel` shared by both shaders.
        deriv_scale:    synthetic-derivative magnitude (keep 0 unless testing
                        specular-AA-style paths, which can't be matched exactly).
    """
    inputs_fn = inputs_fn or (lambda s: RandomSemantics(s))
    cbufs_fn = cbufs_fn or (lambda s: RandomCBufs(s))
    sysvals_fn = sysvals_fn or default_sysvals
    tex = texture or TextureModel()

    res = CompareResult()
    for t in range(trials):
        seed = seed0 + t
        sv = inputs_fn(seed)
        cb = cbufs_fn(seed)
        sysv = sysvals_fn(seed)
        ia = map_inputs(prog_a, sv, sysv)
        ib = map_inputs(prog_b, sv, sysv)
        oa = execute(prog_a, ia, cb, texture=tex, deriv_scale=deriv_scale)
        ob = execute(prog_b, ib, cb, texture=tex, deriv_scale=deriv_scale)
        res.trials += 1
        if oa.discarded != ob.discarded:
            res.discard_mismatches += 1
        w, where = output_diff(oa, ob, output_regs)
        if w > res.worst:
            res.worst = w; res.worst_where = where; res.worst_seed = seed
    return res


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('a', help='first shader (.asm, or .dxbc with --decompiler)')
    ap.add_argument('b', help='second shader')
    ap.add_argument('--trials', type=int, default=200)
    ap.add_argument('--tol', type=float, default=1e-3)
    ap.add_argument('--outputs', default='0',
                    help='comma-separated output registers to compare (default 0)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--deriv-scale', type=float, default=0.0)
    ap.add_argument('--decompiler', default=None,
                    help='path to 3Dmigoto cmd_Decompiler.exe (needed for .dxbc inputs)')
    args = ap.parse_args(argv)

    prog_a = load(args.a, args.decompiler)
    prog_b = load(args.b, args.decompiler)
    regs = tuple(int(x) for x in args.outputs.split(','))

    try:
        res = compare(prog_a, prog_b, trials=args.trials, output_regs=regs,
                      tol=args.tol, seed0=args.seed, deriv_scale=args.deriv_scale)
    except RuntimeError as e:
        print(f"error: {e}", file=sys.stderr)
        print("hint: the default random generators don't drive data-dependent loop\n"
              "      counts (e.g. light count). Write a driver with a cbufs_fn that\n"
              "      sets them -- see tools/shader_diff_popcorn.py.", file=sys.stderr)
        return 2

    print(f"models      : {prog_a.model} vs {prog_b.model}")
    print(f"trials      : {res.trials}")
    print(f"outputs     : {list(regs)}")
    print(f"worst diff  : {res.worst:.3e}" + (f"  (seed {res.worst_seed})" if res.worst_where else ""))
    if res.worst_where:
        reg, ch, av, bv = res.worst_where
        print(f"  at o{reg}.{'xyzw'[ch]}: a={av:.5g} b={bv:.5g}")
    print(f"discard mism: {res.discard_mismatches}")
    ok = res.worst <= args.tol and res.discard_mismatches == 0
    print("RESULT      :", "MATCH" if ok else "DIVERGE")
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(_main())
