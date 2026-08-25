#!/usr/bin/env python
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026, Fernando Sahmkow
# See LICENSE in the repository root for full terms.
"""T6 -- dummy-invariance for the CORE material layers (plan milestone R7).

A core layer is compiled into every class of its family, so a class samples that
layer's texture even for member slots whose material never enabled it.  The whole
reduction rests on one property: when the payload says the layer is off, the sampled
value is DEAD.  This measures that instead of assuming it.

    for a member slot with core layer L disabled:
        run the class blob twice, same inputs, same constants,
        with two DIFFERENT texture fields behind L's texture
        -> every SV_Target must be BIT-identical

Bit-identical, not "within a threshold": a dead value is dead, and any tolerance here
would hide exactly the leak the test exists to find.  The two legs are the SAME
program on the SAME draw, so nothing else can move.

The second gate is the complementary one: the same slot's class blob against its
ordinary per-slot SPECIALISATION, which contains no core layer at all.  T6 says the
dummy sample does not leak; the specialisation leg says the class still computes what
the slot's own shader would.  Neither implies the other -- a class could discard the
sample correctly and still have been mis-keyed.

Both legs are candidate-vs-candidate, so every pre-existing reference difference
(fp reassociation, POM ray-march discontinuities) cancels instead of masking.

  python tools/sc2_perm_core_gate.py --family Model --slots 40
  python tools/sc2_perm_core_gate.py --all --slots 24 --jobs 12
"""
import argparse
import os
import random
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import dxbc_interp as di            # noqa: E402
import sc2_behav_match as bm        # noqa: E402
import sc2_shaders_cfg as cfg       # noqa: E402
import sc2_perm_reduce as R         # noqa: E402
import sc2_slang_validate as V      # noqa: E402

SCRATCH = os.path.join("C:\\", "tmp", "sc2_core_gate")
_lock = threading.Lock()
_seq = [0]


def _scratch(tag):
    with _lock:
        _seq[0] += 1
        n = _seq[0]
    return os.path.join(SCRATCH, "%s_%d" % (tag, n))


class ShiftTex(V.RemapTex):
    """RemapTex with a different texture FIELD behind a chosen set of slots.

    Shifting the id rather than returning a constant matters: a constant field would
    make a leaked sample land on a value the enabled path might legitimately produce,
    so a leak could cancel itself.  Two unrelated smooth fields cannot."""

    def __init__(self, slot2id, shifted, delta):
        V.RemapTex.__init__(self, slot2id)
        self.shifted = set(shifted)
        self.delta = delta

    def _id(self, slot):
        i = V.RemapTex._id(self, slot)
        return i + self.delta if slot in self.shifted else i


def _layer_tex_canon(layer):
    return bm.canon("p_s%sSampler_t" % layer)


def t6_asm(asm, tex_slots, payload=None, trials=6, seed=0):
    """(worst_abs_diff, err) between two runs that differ ONLY in those textures.

    `payload` is the member slot's packed permutation buffer, written VERBATIM to
    b2.  Pinning it is not a detail: b2 is where `cfg.b_iLayerEnable` lives, so a
    random draw there turns the layer back ON and the "dummy" sample legitimately
    reaches the output -- a leak the test itself manufactured."""
    prog = di.Program.from_text(asm)
    keys = V._in_keys(asm, slangc=V._is_slangc(asm))
    cbs = V._cb(asm)
    tgt = V._target_regs(asm)
    ncomp = bm.out_ncomp(asm)
    base_tex, _ = V._res(asm)
    ids = {n: i for i, n in enumerate(sorted(base_tex))}
    slot2id = {slot: ids[n] for n, slot in base_tex.items()}
    a_tex = ShiftTex(slot2id, (), 0)
    b_tex = ShiftTex(slot2id, tex_slots, 7919)
    rng = random.Random(seed)
    worst = 0.0
    for _ in range(trials):
        vals = {}
        for k in sorted(set(keys.values()), key=str):
            if k == "HPos":
                vals[k] = [rng.uniform(-1, 1), rng.uniform(-1, 1),
                           rng.uniform(0, 1), 1.0]
            elif k == "FRONTFACE":
                vals[k] = None
            else:
                vals[k] = [rng.uniform(-1, 1) for _ in range(4)]
        ins = {}
        for reg, k in keys.items():
            ins[reg] = ([0xFFFFFFFF] * 4 if k == "FRONTFACE"
                        else [bm._f2b(x) for x in vals[k]])
        banks = {}
        for slot, n, o, sz in cbs:
            if slot not in banks:
                banks[slot] = [[0, 0, 0, 0]
                               for _ in range(bm.cb_rows(asm, slot=slot))]
            if slot == 2 and payload is not None:
                continue
            bm.cb_write(banks[slot], o,
                        [rng.uniform(-1, 1) for _ in range(max(1, sz // 4))])
        if 0 not in banks:
            banks[0] = [[0, 0, 0, 0] for _ in range(bm.cb_rows(asm, slot=0))]
        if payload is not None and 2 in banks:
            bm.cb_write_words(banks[2], 0, payload)
        try:
            oa = di.execute(prog, ins, banks, texture=a_tex, max_loop=4096)
            ob = di.execute(prog, ins, banks, texture=b_tex, max_loop=4096)
        except Exception as e:
            return None, "exec: " + str(e)[:140]
        for reg in tgt:
            for l in range(ncomp.get(reg, 4)):
                x, y = oa.f(reg, l), ob.f(reg, l)
                if x != x and y != y:
                    continue            # NaN on both legs is agreement
                worst = max(worst, abs(x - y))
    return worst, None


def check_family(family, n_slots=24, jobs=8, do_spec=True, verbose=True):
    stage = "ps"
    core = R.core_layers(family, stage)
    name = "%s_%s" % (family, stage)
    if not core:
        return {}
    classes, slots = R.build_classes(family, stage)
    # cfg.iter_slots, NOT R.iter_slots_fast: the fast one yields only the VARYING
    # axes, and a specialisation compiled from those is missing every
    # constant-but-nonzero axis of the family -- a different shader, which shows
    # up here as a flat 1.0 that has nothing to do with core layers.
    rows = {r[0]: (r[1], r[2]) for r in cfg.iter_slots(family, stage)}
    # Only slots that actually EXERCISE the property: at least one core layer the
    # class compiles in that this member's material does not sample.  A uniform
    # sample over all slots would mostly miss them.
    cand = []
    for slot, ci, payload in slots:
        bv = rows[slot][0]
        off = [L for L in sorted(core) if not R.layer_sampled(bv, L)]
        if off:
            cand.append((slot, ci, payload, off))
    if not cand:
        if verbose:
            print("  %-22s no member slot leaves a core layer unsampled" % name)
        return {}
    step = max(1, len(cand) // n_slots)
    sample = cand[::step][:n_slots]
    entry = cfg.family_cfg(family)[stage]["slang_entry"]
    bad = {}

    def one(item):
        slot, ci, payload, off = item
        defines = R.struct_defines(family, stage, classes[ci])
        asm, err = V.compile_slang(cfg.SC2_MODULE, entry, stage, defines,
                                   _scratch("cls"), include_dirs=[cfg.SC2_INCLUDE])
        if asm is None:
            return slot, off, None, None, str(err)[:140]
        tex, _ = V._res(asm)
        tslots = [tex[c] for c in (_layer_tex_canon(L) for L in off) if c in tex]
        if not tslots:
            # slangc already proved it: the SRV is not even declared.
            return slot, off, 0.0, 0.0, None
        w6, e6 = t6_asm(asm, tslots, payload=payload)
        if e6:
            return slot, off, None, None, e6
        wspec = None
        if do_spec:
            bv, live = rows[slot]
            spec, serr = V.compile_slang(
                cfg.SC2_MODULE, entry, stage,
                cfg.perm_defines(family, stage, bv, live),
                _scratch("spec"), include_dirs=[cfg.SC2_INCLUDE])
            if spec is None:
                return slot, off, w6, None, str(serr)[:140]
            d, derr = V.compare_d3d11(spec, asm, trials=6, cb_pins={2: payload},
                                      const_domains=cfg.PS_CONST_DOMAINS.get(family))
            if derr:
                return slot, off, w6, None, derr[:140]
            wspec = max(d.values()) if d else 0.0
        return slot, off, w6, wspec, None

    with ThreadPoolExecutor(max_workers=jobs) as ex:
        for slot, off, w6, wspec, err in ex.map(one, sample):
            if err:
                bad[slot] = ("error", off, err)
            elif w6 != 0.0:
                bad[slot] = ("t6 LEAK", off, w6)
            elif wspec is not None and wspec > 1e-6:
                bad[slot] = ("vs-spec", off, wspec)
    if verbose:
        if bad:
            print("  %-22s %d/%d sampled slots FAIL:" % (name, len(bad), len(sample)))
            for s, (kind, off, v) in sorted(bad.items())[:8]:
                print("      slot %-7d %-8s %-26s %s"
                      % (s, kind, ",".join(off[:3]), v))
        else:
            print("  %-22s %4d slots, core{%s}: sample dead, class == specialisation"
                  % (name, len(sample), " ".join(sorted(core))))
    return bad


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", nargs="*")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--slots", type=int, default=24)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--no-spec", action="store_true",
                    help="T6 only; skip the class-vs-specialisation leg")
    args = ap.parse_args()
    os.makedirs(SCRATCH, exist_ok=True)
    fams = args.family or sorted(R.CORE_N_R7)
    print("T6 dummy-invariance + class-vs-specialisation over the core layers")
    rc = 0
    for fam in fams:
        if "ps" not in cfg.family_cfg(fam):
            continue
        if check_family(fam, n_slots=args.slots, jobs=args.jobs,
                        do_spec=not args.no_spec):
            rc = 1
    print("")
    print("CORE LAYERS LEAK" if rc else "every sampled core-layer slot is invariant")
    return rc


if __name__ == "__main__":
    sys.exit(main())
