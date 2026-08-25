#!/usr/bin/env python
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026, Fernando Sahmkow
# See LICENSE in the repository root for full terms.
"""T5 -- what one class costs its members in INSTRUCTIONS.

The reduction trades shader count for shader size, and that trade is the whole point
of choosing which axes move.  Class counts are reported everywhere; the other half of
the bargain has to be reported too, or the design is being graded on one number.

For a sample of classes:

    class   = the blob the reduction actually ships for that class
    members = each member slot's own specialisation, the shader it replaces
    ratio   = class instructions / max(member instructions)

`max`, not `mean`: the class has to be at least as capable as its most expensive
member, so the max is the honest denominator.  The mean over members says what the
average draw now pays, and both are reported -- the first is the ceiling, the second
is the cost.

Counting comes from the DXBC disassembly itself (every line that is an instruction),
not from fxc's "approximately N instruction slots" comment, which slangc does not
emit.  Declarations, labels and the register-count line are excluded.

  python tools/sc2_perm_icount.py --family Model --classes 24
  python tools/sc2_perm_icount.py --all --classes 12 --jobs 10
"""
import argparse
import os
import re
import sys
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import dxbc_interp as di          # noqa: E402
import sc2_behav_match as bm      # noqa: E402
import sc2_shaders_cfg as cfg       # noqa: E402
import sc2_perm_reduce as R         # noqa: E402
import sc2_slang_validate as V      # noqa: E402

SCRATCH = os.path.join("C:\\", "tmp", "sc2_icount")
_lock = threading.Lock()
_seq = [0]

# Everything that is not an executable instruction.
_SKIP = re.compile(r"^\s*(//|;|$)|^\s*(dcl_|ret\b|label\b)")


def _scratch(tag):
    with _lock:
        _seq[0] += 1
        n = _seq[0]
    return os.path.join(SCRATCH, "%s_%d" % (tag, n))


def icount(asm):
    n = 0
    body = asm.split("\n")
    for ln in body:
        s = ln.strip()
        if not s or s.startswith("//") or s.startswith(";"):
            continue
        if s.startswith("dcl_") or s.startswith("ps_") or s.startswith("vs_"):
            continue
        if s.startswith("ret") or s.endswith(":"):
            continue
        n += 1
    return n


def _draw(asm_list, payload, seed):
    """One shared random draw, materialised per leg.

    Same values behind the same canon names on both legs, so the two shaders take the
    same data-dependent branches and the only difference in the count is the one under
    test.  b2 is written VERBATIM on whichever leg declares it -- that is the whole
    point: the uniform predicates the reduction added are only skippable if the
    payload says so."""
    import random
    rng = random.Random(seed)
    keysets, cbsets = [], []
    for asm in asm_list:
        keysets.append(V._in_keys(asm, slangc=V._is_slangc(asm)))
        cbsets.append(V._cb(asm))
    allk = sorted({k for ks in keysets for k in ks.values()}, key=str)
    vals = {}
    for k in allk:
        if k == "HPos":
            vals[k] = [rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(0, 1), 1.0]
        elif k == "FRONTFACE":
            vals[k] = None
        else:
            vals[k] = [rng.uniform(-1, 1) for _ in range(4)]
    cbvals = {}
    for cbs in cbsets:
        for slot, n, o, sz in cbs:
            cbvals.setdefault(n, [rng.uniform(-1, 1) for _ in range(max(1, sz // 4))])
    out = []
    for asm, keys, cbs in zip(asm_list, keysets, cbsets):
        ins = {}
        for reg, k in keys.items():
            ins[reg] = ([0xFFFFFFFF] * 4 if k == "FRONTFACE"
                        else [bm._f2b(x) for x in vals[k]])
        banks = {}
        for slot, n, o, sz in cbs:
            if slot not in banks:
                banks[slot] = [[0, 0, 0, 0] for _ in range(bm.cb_rows(asm, slot=slot))]
            if slot == 2 and payload is not None:
                continue
            bm.cb_write(banks[slot], o, cbvals[n])
        if 0 not in banks:
            banks[0] = [[0, 0, 0, 0] for _ in range(bm.cb_rows(asm, slot=0))]
        if payload is not None and 2 in banks:
            bm.cb_write_words(banks[2], 0, payload)
        out.append((ins, banks))
    return out


def dyncount(asm, ins, banks):
    """Instructions actually EXECUTED for one draw.

    The static count is the wrong number for frame time and overstates this design
    badly: every predicate the reduction added is UNIFORM, so `[branch]` skips the
    whole block at runtime rather than selecting through it.  Counting executions is
    what says whether the class really costs what its length suggests.

    `di._do_op` is a module global looked up per call, so wrapping it counts every
    executed op -- and it is NOT thread-safe, which is why this runs serially while
    only the compiles are parallel."""
    prog = di.Program.from_text(asm)
    n = [0]
    orig = di._do_op

    def counting(vm, node):
        n[0] += 1
        return orig(vm, node)

    di._do_op = counting
    try:
        di.execute(prog, ins, banks, texture=V.RemapTex({}), max_loop=4096)
    except Exception as e:
        return None, str(e)[:120]
    finally:
        di._do_op = orig
    return n[0], None


def check_stage(family, stage, n_classes=12, jobs=8, verbose=True,
                dynamic=False):
    classes, slots = R.build_classes(family, stage)
    rows = {r[0]: (r[1], r[2]) for r in cfg.iter_slots(family, stage)}
    members = defaultdict(list)
    for slot, ci, _p in slots:
        members[ci].append(slot)
    # The classes worth measuring are the ones that actually MERGED something; a
    # class with one member is its own specialisation and its ratio is 1 by
    # construction, so sampling those would flatter the result.
    merged = sorted((ci for ci, ms in members.items() if len(ms) > 1),
                    key=lambda ci: -len(members[ci]))
    if not merged:
        return None
    step = max(1, len(merged) // n_classes)
    sample = merged[::step][:n_classes]
    entry = cfg.family_cfg(family)[stage]["slang_entry"]

    def one(ci):
        """Compile the class and a few of its members; counting happens on the main
        thread afterwards, because the dynamic counter patches a module global."""
        asm, err = V.compile_slang(cfg.SC2_MODULE, entry, stage,
                                   R.struct_defines(family, stage, classes[ci]),
                                   _scratch("cls"), include_dirs=[cfg.SC2_INCLUDE])
        if asm is None:
            return None
        ms = members[ci]
        step2 = max(1, len(ms) // 6)
        pairs = []
        for slot in ms[::step2][:6]:
            bv, live = rows[slot]
            spec, serr = V.compile_slang(cfg.SC2_MODULE, entry, stage,
                                         cfg.perm_defines(family, stage, bv, live),
                                         _scratch("spec"),
                                         include_dirs=[cfg.SC2_INCLUDE])
            if spec is not None:
                pairs.append((slot, spec, R.pack_payload(family, stage, bv)))
        if not pairs:
            return None
        return asm, pairs, len(ms)

    built = []
    with ThreadPoolExecutor(max_workers=jobs) as ex:
        for r in ex.map(one, sample):
            if r is not None:
                built.append(r)
    if not built:
        return None

    tot_cls, tot_max, tot_mean, worst = [], [], [], (0.0, None)
    dyn_cls, dyn_mean = [], []
    for asm, pairs, nmem in built:
        counts = [icount(spec) for _s, spec, _p in pairs]
        c = icount(asm)
        tot_cls.append(c)
        tot_max.append(max(counts))
        tot_mean.append(sum(counts) / float(len(counts)))
        if c / float(max(counts)) > worst[0]:
            worst = (c / float(max(counts)), (c, max(counts), nmem))
        if dynamic:
            for _slot, spec, payload in pairs:
                (ci_, bi_), (si_, sb_) = _draw([asm, spec], payload, seed=len(dyn_cls))
                a, ea = dyncount(asm, ci_, bi_)
                b, eb = dyncount(spec, si_, sb_)
                if a and b:
                    dyn_cls.append(a)
                    dyn_mean.append(b)
    med = sorted(tot_cls)[len(tot_cls) // 2]
    vs_max = sum(tot_cls) / float(sum(tot_max))
    vs_mean = sum(tot_cls) / float(sum(tot_mean))
    if verbose:
        extra = ""
        if dynamic and dyn_cls:
            extra = ("   EXECUTED %.2fx (%d vs %d, %d draws)"
                     % (sum(dyn_cls) / float(sum(dyn_mean)),
                        sum(dyn_cls) // len(dyn_cls),
                        sum(dyn_mean) // len(dyn_mean), len(dyn_cls)))
        print("  %-22s static median %4d   vs member max %.2fx   vs mean %.2fx"
              "   worst %.2fx (%d vs %d, %d members)%s"
              % ("%s_%s" % (family, stage), med, vs_max, vs_mean, worst[0],
                 worst[1][0], worst[1][1], worst[1][2], extra))
    if dynamic and dyn_cls:
        return sum(dyn_cls) / float(sum(dyn_mean))
    return vs_max


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", nargs="*")
    ap.add_argument("--stage", choices=("vs", "ps"))
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--classes", type=int, default=12)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--no-core", action="store_true",
                    help="re-key with NO core layers, to separate what the core-layer "
                         "increment costs in instructions from what the rest of the "
                         "reduction already cost.  The shader tree is unchanged: with "
                         "no SC2_CORE_LAYER_ define emitted it compiles exactly as it "
                         "did before, which bit-identity proves.")
    ap.add_argument("--dynamic", action="store_true",
                    help="also count instructions EXECUTED per draw through "
                         "dxbc_interp, which is the number that decides frame time: "
                         "the predicates the reduction adds are uniform, so a "
                         "[branch] skips the block instead of selecting through it. "
                         "When given, the budget is applied to this ratio.")
    ap.add_argument("--budget", type=float, default=3.0,
                    help="fail over this class/member-max ratio (design section 6)")
    args = ap.parse_args()
    if args.no_core:
        # Before any of R's caches are warm -- _iface_axes_for memoises a classify()
        # that has already consulted core_layers().
        R.CORE_LAYERS_READY = frozenset()
    os.makedirs(SCRATCH, exist_ok=True)
    if args.all or not args.family:
        stages = R._all_stages()
    else:
        stages = [(f, s) for f in args.family
                  for s in (("vs", "ps") if not args.stage else (args.stage,))
                  if s in cfg.family_cfg(f)]
    print("T5 instruction cost: one class blob vs the specialisations it replaces")
    over = []
    for fam, st in stages:
        r = check_stage(fam, st, n_classes=args.classes, jobs=args.jobs,
                        dynamic=args.dynamic)
        if r is not None and r > args.budget:
            over.append(("%s_%s" % (fam, st), r))
    print("")
    if over:
        print("OVER BUDGET (%.1fx):" % args.budget)
        for n, r in over:
            print("    %-22s %.2fx" % (n, r))
        return 1
    print("every measured stage is within %.1fx of the shaders it replaces"
          % args.budget)
    return 0


if __name__ == "__main__":
    sys.exit(main())
