#!/usr/bin/env python3
"""Compile sc2_shaders permutation slots to a graphics-API target.

For each (family, stage) the cache-order manifest (sc2_perms/<Family>_<stage>.json)
gives one slot per retail permutation; we compile the family's slang entry with
that slot's decoded `b_*` define set and write `perm_<NNN>.<ext>` in slot order.
build_sc2_bls.py then bundles those blobs, in the same order, into the family BLS.

Output: sc2_slang_out/<target>/<Family>_<stage>/perm_<NNN>.<ext>   (mirrors
compile_all_slang.py's slang_out/<target>/<family>/ layout).  D3D11 DXBC is the
default and the only target the BLS bundler and the validation harness consume;
the others exist so an engine port (or a portability check of the module) can get
the same permutation set as DXIL / SPIR-V / Metal / WGSL / GLSL.

Full sweeps are ENORMOUS (~108k permutations), so a portability check should use
`--sample N`: it picks N slots per (family, stage) that between them exercise every
`b_*` axis VALUE the family's manifest contains, then spends whatever budget is
left on an even spread over the rest.  That beats the first N slots or a plain
spread, because the axis values that break a backend are usually the rare ones.
It does NOT cover every axis COMBINATION — that is the full sweep.

Usage:
  python compile_all_sc2.py [--family Simple] [--stage ps|vs] [--jobs 8]
  python compile_all_sc2.py --all --target all --sample 12   # portability sweep
"""
import os
import sys
import json
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools"))
import compile_all_slang as cas
import sc2_shaders_cfg as cfg
import sc2_perm_reduce as R

OUT_ROOT = REPO_ROOT / "sc2_slang_out"
# Targets come from compile_all_slang's table (the same slangc target / profile /
# extra-args triples wc3_shaders sweeps with), so the two projects stay in step.
# DEFAULT_TARGET is the one build_sc2_bls.py and tools/sc2_validate_all.py read.
TARGETS = cas.TARGETS
DEFAULT_TARGET = "d3d11"

# Decoding a manifest costs real time on the big families (Model ps: ~50k slots,
# ~80s) and the greedy pass over it costs more, so a --sample selection - a pure
# function of (manifest, N, algorithm) - is memoised under the (gitignored) output
# tree.  Bump SAMPLE_ALGO whenever sample_slots' selection rule changes, or stale
# caches would silently pin the old picks.
SAMPLE_CACHE = OUT_ROOT / "_samples"
SAMPLE_ALGO = 2


def sample_slots(family, stage, n):
    """N manifest slots picked to exercise as much of the family as N compiles can.

    Two phases.  First a greedy set cover over the (axis, value) pairs in the
    family's manifest, plus the live-interpolant names (they drive the IO structs,
    so a missing interpolant is its own compile path) - that is what actually finds
    backend breakage, because the axis values that break a backend are usually the
    rare ones no linear spread would reach.  Coverage usually saturates well short
    of N, so the remaining budget is spent on an even spread over the slots not yet
    chosen, which buys some coverage of axis COMBINATIONS too.

    Note what this does NOT claim: covering every axis value is not covering every
    axis combination - that is the full ~108k-permutation sweep.  Ties break toward
    the earlier slot, so the selection is deterministic; slots come back in cache
    order."""
    key = SAMPLE_CACHE / ("%s_%s_n%d.json" % (family, stage, n))
    man_mtime = os.path.getmtime(
        os.path.join(cfg.PERMS_DIR, "%s_%s.json" % (family, stage)))
    slots = list(cfg.iter_slots(family, stage))
    if key.exists():
        c = json.loads(key.read_text())
        if (isinstance(c, dict) and c.get("algo") == SAMPLE_ALGO
                and c.get("manifest_mtime") == man_mtime):
            want = set(c["slots"])
            hit = [s for s in slots if s[0] in want]
            if len(hit) == len(want):
                return hit

    # Precompute each slot's pair set ONCE: the greedy pass rescans every remaining
    # slot per pick, and Model ps has ~50k of them with ~100 axes each.
    pv = [{("b", k, int(v)) for k, v in bv.items()} | {("live", x, 1) for x in live}
          for _slot, bv, live, _ in slots]
    rest = list(range(len(slots)))
    covered, chosen = set(), []
    while rest and len(chosen) < n:
        best = max(rest, key=lambda i: (len(pv[i] - covered), -slots[i][0]))
        if not (pv[best] - covered) and chosen:
            break               # coverage saturated; the spread fill takes over
        covered |= pv[best]
        chosen.append(best)
        rest.remove(best)
    # Spread fill over what is left, so --sample N really is N compiles per stage.
    fill = n - len(chosen)
    if fill > 0 and rest:
        fill = min(fill, len(rest))
        chosen += [rest[round(k * (len(rest) - 1) / max(1, fill - 1))]
                   for k in range(fill)] if fill > 1 else [rest[0]]
    picked = sorted({i for i in chosen})
    picked = [slots[i] for i in picked]
    SAMPLE_CACHE.mkdir(parents=True, exist_ok=True)
    key.write_text(json.dumps({"algo": SAMPLE_ALGO, "manifest_mtime": man_mtime,
                               "slots": [it[0] for it in picked]}))
    return picked


def module_mtime():
    """Newest mtime across the slang module — the incremental-skip watermark.

    The module is ONE translation unit, so ANY source edit invalidates every
    permutation of every family; a per-file dependency check would be wrong."""
    newest = 0.0
    for p in Path(cfg.SC2_INCLUDE).rglob("*.slang*"):
        newest = max(newest, p.stat().st_mtime)
    return newest


def compile_stage(family, stage, jobs=8, verbose=False, skip_existing=False,
                  watermark=None, target=None, sample=0, reduced=False):
    """Compile one (family, stage) -- either every retail SLOT, or one blob per CLASS.

    The reduced build is the whole point of the permutation reduction: `struct_defines`
    is a pure function of the structural key, so one blob serves every member of its
    class and the per-slot difference is carried by the b2 payload that
    build_sc2_bls.py writes into the .perm side table.  Output goes to its OWN
    directory (`<Family>_<stage>_reduced`), never mixed with the per-slot build --
    `class_007.dxbc` and `perm_007.dxbc` mean different things and a consumer that
    confused them would silently ship the wrong shader for 20,000 permutations."""
    scfg = cfg.family_cfg(family).get(stage)
    if scfg is None:
        return None
    target = target or DEFAULT_TARGET
    tgt = TARGETS[target]
    entry = scfg["slang_entry"]
    suffix = "_reduced" if reduced else ""
    out_dir = OUT_ROOT / target / ("%s_%s%s" % (family, stage, suffix))
    out_dir.mkdir(parents=True, exist_ok=True)
    if reduced:
        classes, _slots = R.build_classes(family, stage)
        items = list(enumerate(classes))
        if sample and len(items) > sample:
            step = max(1, len(items) // sample)
            items = items[::step][:sample]
    else:
        items = sample_slots(family, stage, sample) if sample else list(
            cfg.iter_slots(family, stage))
    wm = watermark if watermark is not None else (module_mtime() if skip_existing else 0.0)
    skipped = [0]

    def work(item):
        if reduced:
            idx, key = item
            out = out_dir / ("class_%05d.%s" % (idx, tgt["ext"]))
            defines = R.struct_defines(family, stage, key)
        else:
            idx, bv, live, _dedup = item
            out = out_dir / ("perm_%03d.%s" % (idx, tgt["ext"]))
            defines = cfg.perm_defines(family, stage, bv, live)
        if skip_existing and out.exists() and out.stat().st_mtime >= wm:
            skipped[0] += 1
            return idx, True
        # The profile is the TARGET's, not always ps_5_0/vs_5_0: DXIL needs *_6_0
        # and the SPIR-V / GLSL legs want glsl_450 - same table compile_all_slang
        # sweeps wc3_shaders with.
        ok = cas.invoke_slangc(entry, tgt["target"], tgt[stage], [], out,
                               Path(cfg.SC2_MODULE),
                               extra=tgt["extra"],
                               include_dirs=[Path(cfg.SC2_INCLUDE)],
                               defines=defines)
        return idx, ok

    if jobs <= 1:
        res = [work(it) for it in items]
    else:
        with ThreadPoolExecutor(max_workers=jobs) as ex:
            res = list(ex.map(work, items))
    ok = sum(1 for _, o in res if o)
    fails = [s for s, o in res if not o]
    print("%-8s %-16s %-2s: %6d/%-6d %s%s%s"
          % (target, family, stage, ok, len(res),
             "classes compiled" if reduced else "compiled",
             "  (%d up to date)" % skipped[0] if skipped[0] else "",
             "" if not fails else "  FAILED slots: %s" % fails[:20]),
          flush=True)
    return ok, len(res), fails


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Compile sc2_shaders perms to a graphics-API target.")
    ap.add_argument("--family", help="one family; omit with --all")
    ap.add_argument("--all", action="store_true",
                    help="every family in sc2_shaders.json (M5.1)")
    ap.add_argument("--stage", choices=["ps", "vs"], help="default: both")
    ap.add_argument("--target", default=DEFAULT_TARGET,
                    help="output API, or a comma-separated list, or `all`: %s "
                         "(default d3d11 — the target the BLS bundler and the "
                         "validation harness read)" % ", ".join(sorted(TARGETS)))
    ap.add_argument("--sample", type=int, default=0, metavar="N",
                    help="compile only N slots per (family, stage), chosen to cover "
                         "as many distinct b_* axis values as possible — use this "
                         "for portability sweeps; a full one is ~108k perms")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--skip-existing", action="store_true",
                    help="keep perms already newer than the newest module source "
                         "(makes a whole-module sweep resumable)")
    ap.add_argument("--reduced", action="store_true",
                    help="compile one blob per structural CLASS instead of one per "
                         "retail slot (tools/sc2_perm_reduce.py). Writes "
                         "<Family>_<stage>_reduced/class_<NNNNN>.<ext>; pair it with "
                         "build_sc2_bls.py --reduced, which emits the .perm side "
                         "table mapping each retail slot to its class and payload.")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)
    if not args.all and not args.family:
        ap.error("pass --family <name> or --all")

    families = sorted(cfg.load_families()) if args.all else [args.family]
    stages = [args.stage] if args.stage else ["vs", "ps"]
    targets = (sorted(TARGETS) if args.target == "all"
               else [x.strip() for x in args.target.split(",") if x.strip()])
    bad = [x for x in targets if x not in TARGETS]
    if bad:
        ap.error("unknown --target %s (have: %s, or `all`)"
                 % (", ".join(bad), ", ".join(sorted(TARGETS))))
    # One watermark for the whole sweep: a module edit mid-run must not make the
    # families compiled before it look stale relative to the ones after.
    wm = module_mtime() if args.skip_existing else 0.0
    any_fail = False
    per_target = {}
    for fam in families:
        for st in stages:
            # Decoding a manifest is the expensive part on the big families (Model
            # ps: ~50k slots), so walk the targets INSIDE the (family, stage) loop -
            # sample_slots answers the 2nd..Nth target straight from its cache.
            for tg in targets:
                r = compile_stage(fam, st, jobs=args.jobs, verbose=args.verbose,
                                  skip_existing=args.skip_existing, watermark=wm,
                                  target=tg, sample=args.sample,
                                  reduced=args.reduced)
                if r is None:
                    continue
                ok, n, fails = r
                tot = per_target.setdefault(tg, [0, 0])
                tot[0] += ok
                tot[1] += n
                if fails:
                    any_fail = True
    if args.all or len(targets) > 1:
        print()
        for tg in targets:
            ok, n = per_target.get(tg, (0, 0))
            print("%-8s %d/%d permutations compiled across %d families"
                  % (tg, ok, n, len(families)))
    return 1 if any_fail else 0


if __name__ == "__main__":
    sys.exit(main())
