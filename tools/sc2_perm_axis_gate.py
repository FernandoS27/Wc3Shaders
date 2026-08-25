#!/usr/bin/env python
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026, Fernando Sahmkow
# See LICENSE in the repository root for full terms.
"""Per-axis BEHAVIOURAL gate for the permutation payload.

`sc2_perm_reduce.py --check-moved` proves that no `#if` still reads a moved axis.
That is necessary and not sufficient: it says the axis is *reachable* as data, not
that reading it as data computes the same thing.  Three axes passed --check-moved and
still broke `Ribbon_vs` -- `b_gpuSplineRibbon`, `b_proceduralPosition` and
`b_precomputedTangent`, each of which gates a branch whose runtime form is not
equivalent to its specialised form under this reference.

It measures that directly, one axis at a time -- where "axis" means either a
non-layer axis name or a whole layer PROPERTY (the switch is per property, so
`prop:Invert` covers b_iDiffuseInvert, b_iEmissiveInvert and the rest at once).
Layer properties are the majority of the varying axes and were NOT covered until the
`Invert` divergence turned up as a float in a 107,976-slot sweep:

    candidate = the class blob with SC2_PERMDATA_<axis>=1 and every OTHER moved axis
                pinned back to its retail `-D` value
    expected  = the ordinary specialised candidate for the same slot
    verdict   = compare the two with the axis's payload bits pinned into b2

If the axis is genuinely a data selection, the two agree exactly for every slot.  A
disagreement is reported with the axis name, which is the whole point: the
alternative is a family-wide sweep failure that names a float instead of a cause.

The comparison is candidate-vs-candidate on purpose.  Both legs come from the same
toolchain and the same source, so the only variable is HOW the axis arrives -- which
isolates the mechanism from every pre-existing reference difference (fp reassociation,
POM ray-march discontinuities) that a reference comparison would fold in.

  python tools/sc2_perm_axis_gate.py --family Ribbon --stage vs --slots 3
  python tools/sc2_perm_axis_gate.py --all --slots 2 --jobs 8
"""
import argparse
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import sc2_shaders_cfg as cfg          # noqa: E402
import sc2_perm_reduce as R            # noqa: E402
import sc2_slang_validate as V         # noqa: E402

SCRATCH = os.path.join("C:\\", "tmp", "sc2_axis_gate")
_lock = threading.Lock()
_seq = [0]


def _scratch(tag):
    with _lock:
        _seq[0] += 1
        n = _seq[0]
    return os.path.join(SCRATCH, "%s_%d" % (tag, n))


def _compare(stage, family, a, b, pins):
    if stage == "vs":
        return V.compare_vs(a, b, trials=6, cb_pins=pins,
                            input_domains=cfg.VS_INPUT_DOMAINS.get(family),
                            const_domains=cfg.VS_CONST_DOMAINS.get(family))
    return V.compare_d3d11(a, b, trials=6, cb_pins=pins,
                           const_domains=cfg.PS_CONST_DOMAINS.get(family))


def check_stage(family, stage, n_slots=10, jobs=8, verbose=True):
    """Returns {axis: (slot, worst)} for every axis whose dynamic form diverges.

    `axis` is either a non-layer axis name or `prop:<Property>` for a layer property,
    since the shader switch is per property rather than per axis."""
    scfg = cfg.family_cfg(family).get(stage)
    if not scfg:
        return {}
    # `_data_axes` is documented as "the NON-layer axes this stage carries", so for
    # most of this tool's life it tested none of the fourteen ENABLED_LAYER_PROPS nor
    # the two core-gated ones -- and those are the large majority of the varying axes.
    # A layer property that does not behave as data therefore passed --check-moved,
    # passed T3 (one-sided), and only showed up as a float in a full sweep.  That is
    # how the `Invert` divergence survived: cover them here, one PROPERTY at a time,
    # because the switch is per property rather than per axis.
    data = sorted(R._data_axes(family, stage))
    layer = sorted(a for a, _o, _w in R.payload_fields(family, stage)
                   if R._split_layer_axis(a) is not None)
    props = sorted({R._split_layer_axis(a)[1] for a in layer})
    if not data and not props:
        return {}
    slots = list(cfg.iter_slots(family, stage))
    entry = scfg["slang_entry"]
    bad = {}

    def one(job):
        axis, (slot, bv, live, _dd) = job
        key = R.structural_key(family, stage, bv, live)
        base = R.struct_defines(family, stage, key)
        # Everything the payload could carry is pinned back to its retail value; only
        # `axis` -- a non-layer axis name, or "prop:<Property>" for a layer property --
        # is left dynamic.  Stripping ALL of SC2_PERMDATA_ / SC2_PERM_PROP_ /
        # SC2_CORE_LAYER_ first is what makes the isolation real: leaving the layer
        # switches on while testing a non-layer axis is why this tool used to report
        # green with a broken layer property in the same shader.
        defs = [d for d in base
                if not d.startswith("SC2_PERMDATA_")
                and not d.startswith("SC2_PERM_PROP_")
                and not d.startswith("SC2_CORE_LAYER_")]
        prop = axis[5:] if axis.startswith("prop:") else None
        if prop is None:
            defs.append("SC2_PERMDATA_%s=1" % axis)
        elif prop in R.DUAL_ROLE_PROPS:
            # LayerEnable / UseConstantColor move per LAYER, through the core gate.
            defs += ["SC2_CORE_LAYER_%s=1" % L for L in R.core_layers(family, stage)]
        else:
            defs.append("SC2_PERM_PROP_%s=1" % prop)
        for o in data:
            if o == axis:
                continue
            v = int(bv.get(o, 0))
            if v:
                defs.append("%s=%d" % (o, v))
        for o in layer:
            if prop is not None and R._split_layer_axis(o)[1] == prop:
                continue
            v = int(bv.get(o, 0))
            if v:
                defs.append("%s=%d" % (o, v))
        defs = sorted(set(defs))
        cand, cerr = V.compile_slang(cfg.SC2_MODULE, entry, stage, defs,
                                     _scratch("cand"), include_dirs=[cfg.SC2_INCLUDE])
        exp, eerr = V.compile_slang(cfg.SC2_MODULE, entry, stage,
                                    cfg.perm_defines(family, stage, bv, live),
                                    _scratch("exp"), include_dirs=[cfg.SC2_INCLUDE])
        if cand is None or exp is None:
            return axis, slot, None, str(cerr or eerr)[:100]
        diffs, derr = _compare(stage, family, exp, cand,
                               {2: R.pack_payload(family, stage, bv)})
        if derr:
            return axis, slot, None, derr[:100]
        return axis, slot, (max(diffs.values()) if diffs else 0.0), None

    axes = data + ["prop:%s" % p for p in props]

    def sample_for(axis):
        """Slots that actually EXERCISE `axis`, chosen to spread across its members.

        Two things have to be right here, and only the first is obvious.

        A uniform `slots[::step]` is a lottery: for Particle_ps NONE of the first
        2, 3, 8 or 20 strided slots sets any layer Invert at all, so the gate could
        gain the property and still never test it.

        And a layer PROPERTY is one switch over ~20 per-layer axes, so "a slot that
        sets some Invert" is not enough either -- the first two such slots both
        invert AlphaMask, whose folded and dynamic forms happen to agree, while the
        divergence lived in Emissive.  Pick greedily for member COVERAGE instead,
        rarest member first, and say how much of the property went untested.

        Returns (slots, uncovered_member_names).
        """
        names = ([a for a in layer if R._split_layer_axis(a)[1] == axis[5:]]
                 if axis.startswith("prop:") else [axis])
        by_axis = {}
        for n in names:
            hits = [s for s in slots if int(s[1].get(n, 0))]
            if hits:
                by_axis[n] = hits
        if not by_axis:
            return None, []
        chosen, covered = [], set()
        for n in sorted(by_axis, key=lambda k: len(by_axis[k])):
            if n in covered or len(chosen) >= n_slots:
                continue
            pick = by_axis[n][0]
            chosen.append(pick)
            covered |= {m for m in by_axis if int(pick[1].get(m, 0))}
        return chosen, sorted(set(by_axis) - covered)

    jobs_list, unexercised, partial = [], [], []
    for a in axes:
        sel, uncovered = sample_for(a)
        if sel is None:
            unexercised.append(a)
            continue
        if uncovered:
            partial.append((a, len(uncovered), len(uncovered) + len(sel)))
        jobs_list += [(a, s) for s in sel]
    with ThreadPoolExecutor(max_workers=jobs) as ex:
        for axis, slot, worst, err in ex.map(one, jobs_list):
            if err:
                bad.setdefault(axis, (slot, err))
            elif worst > 1e-6 and axis not in bad:
                bad[axis] = (slot, worst)
    if verbose:
        name = "%s_%s" % (family, stage)
        if bad:
            print("  %-22s %d/%d axes DIVERGE as data:" % (name, len(bad), len(axes)))
            for a, (slot, w) in sorted(bad.items()):
                print("      %-34s slot %-6s %s" % (a, slot, w))
        else:
            print("  %-22s all %d moved axes behave as data" % (name, len(axes) - len(unexercised)))
        if unexercised:
            # Not a failure: the manifest never sets these non-zero for this stage,
            # so there is nothing to compare.  Said out loud so "all axes pass" is
            # never read as coverage it does not have.
            print("      (%d not exercised by any slot: %s)"
                  % (len(unexercised), " ".join(sorted(unexercised)[:8])))
        if partial:
            print("      (--slots %d too small to cover every member of: %s)"
                  % (n_slots, " ".join("%s[-%d]" % (a, n) for a, n, _t in
                                       sorted(partial, key=lambda x: -x[1])[:6])))
    return bad


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", nargs="*")
    ap.add_argument("--stage", choices=("vs", "ps"))
    ap.add_argument("--all", action="store_true")
    # 10, not 3: with the greedy member cover above, 10 slots reach every member of
    # every layer property but one on Particle_ps -- and the `Invert` divergence that
    # a 107,976-slot sweep took to surface is named at 10 and missed at 2.
    ap.add_argument("--slots", type=int, default=10)
    ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args()

    os.makedirs(SCRATCH, exist_ok=True)
    if args.all or not args.family:
        stages = R._all_stages()
    else:
        stages = [(f, s) for f in args.family
                  for s in (("vs", "ps") if not args.stage else (args.stage,))
                  if s in cfg.family_cfg(f)]

    print("per-axis behavioural gate: dynamic form must equal the specialisation")
    total = {}
    for fam, st in stages:
        bad = check_stage(fam, st, n_slots=args.slots, jobs=args.jobs)
        for a, v in bad.items():
            total.setdefault(a, set()).add("%s_%s" % (fam, st))
    print("")
    if total:
        print("AXES THAT MUST NOT BE MOVED (%d):" % len(total))
        for a, where in sorted(total.items()):
            print('    "%s",   # %s' % (a, " ".join(sorted(where))))
        return 1
    print("every moved axis behaves as data on the sampled slots")
    return 0


if __name__ == "__main__":
    sys.exit(main())
