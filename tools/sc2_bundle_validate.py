#!/usr/bin/env python
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026, Fernando Sahmkow
# See LICENSE in the repository root for full terms.
"""Validate the REDUCED BUNDLE the way a consumer would load it.

Every other gate in this project validates the design: it re-derives a class from the
classifier, compiles it, and compares.  That leaves a real gap.  What ships is a pair
of files -- `<family>.bls` holding one blob per class and `<Family>_<stage>.perm`
mapping each retail permutation to (class, payload) -- and between the classifier and
those files sit the bundler, the DXBC signature fixup, the container's compression,
the payload pool and the word mask.  A bug in any of them ships a wrong shader while
every existing gate stays green.

So this one starts from the FILES:

    slot -> .perm -> class index      -> blob pulled out of the .bls
                  -> pooled payload   -> expanded through the word mask into b2
    reference = fxc on the original .fx with that slot's retail defines

and compares them per `SV_Target` / per varying at the same per-bucket thresholds
`sc2_validate_all.py` uses.  Nothing is recompiled: the DXBC under test is the one in
the bundle, hash and signature fixup included.

  python tools/sc2_bundle_validate.py --all --sample 40 --jobs 8
  python tools/sc2_bundle_validate.py --family Model --stage ps --sample 200
"""
import argparse
import os
import struct
import sys
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
for p in (HERE, REPO):
    if p not in sys.path:
        sys.path.insert(0, p)

import build_bls as bb                 # noqa: E402
import build_sc2_bls as B              # noqa: E402
import sc2_shaders_cfg as cfg          # noqa: E402
import sc2_perm_reduce as R            # noqa: E402
import sc2_slang_validate as V         # noqa: E402
import sc2_validate_all as VA          # noqa: E402

SCRATCH = os.path.join("C:\\", "tmp", "sc2_bundle_val")
STAGE_DIR = {"vs": "vertex", "ps": "pixel"}


def read_bls_blobs(path):
    """[DXBC bytes] in slot order, straight out of the v1.14 container."""
    data = Path(path).read_bytes()
    if data[:4] != bb.BLS_MAGIC:
        raise ValueError("%s: bad magic" % path)
    off_perms, num_perms, _off_blobs, num_blobs, off_data = \
        struct.unpack_from("<IIIII", data, 12)
    sizes, cur = [], off_perms + 4
    for _ in range(num_perms):
        sz, = struct.unpack_from("<I", data, cur)
        sizes.append(sz)
        cur += bb.BLS_V14_PERM_ENTRY
    raw = zlib.decompress(data[off_data:]) if num_blobs else b""
    out, pos = [], 0
    for sz in sizes:
        if sz == 0:
            out.append(None)
            continue
        inner = raw[pos:pos + sz]
        pos += sz
        # The DX inner permutation puts the DXBC at +0x60.  Truncate to the DXBC
        # container's OWN length (header u32 at +24), not to the end of the inner
        # record: the record can carry trailing bytes after the container, and fxc
        # /dumpbin then produces nothing at all -- which reads as an empty signature
        # and an unhelpful "no shared VS output" rather than as a bad slice.
        blob = inner[0x60:]
        if len(blob) >= 28 and blob[:4] == b"DXBC":
            n = struct.unpack_from("<I", blob, 24)[0]
            if 0 < n <= len(blob):
                blob = blob[:n]
        out.append(blob)
    return out


def read_perm_table(path):
    """(class_index[], payload_expander) -- the consumer's half of the contract."""
    data = Path(path).read_bytes()
    if data[:4] != B.PERM_MAGIC:
        raise ValueError("%s: bad magic" % path)
    ver, n_slots, n_classes, n_pool, total_words, stored_n, _f = \
        struct.unpack_from("<IIIIIII", data, 4)
    if ver != B.PERM_VERSION:
        raise ValueError("%s: version %d" % (path, ver))
    off = 32
    nmask = (total_words + 31) // 32
    mask = struct.unpack_from("<%dI" % nmask, data, off)
    off += 4 * nmask
    stored = [i for i in range(total_words) if mask[i // 32] >> (i % 32) & 1]
    cls = struct.unpack_from("<%dH" % n_slots, data, off)
    off += 2 * n_slots
    off += (-off) % 4
    pidx = struct.unpack_from("<%dI" % n_slots, data, off)
    off += 4 * n_slots
    pool = [struct.unpack_from("<%dI" % stored_n, data, off + 4 * stored_n * i)
            if stored_n else () for i in range(n_pool)]

    def payload(slot):
        words = [0] * total_words
        for k, w in zip(stored, pool[pidx[slot]]):
            words[k] = w
        return words

    return list(cls), payload, n_classes


def check_stage(family, stage, bundle_root, sample=40, jobs=8, trials=8,
                verbose=True, reduced=True):
    """Validate one stage's SHIPPED bundle against the fxc reference.

    Two bundle shapes, one code path.  A REDUCED bundle holds one blob per
    structural class and needs the `.perm` side table to say which class a retail
    slot maps to and which payload to pin into b2.  A NON-REDUCED bundle holds one
    blob per retail slot in `iter_slots` order, so the mapping is the row index and
    there is nothing to pin -- every axis is already baked into the blob.

    Both shapes go through `_process_dxbc` on the way into the container, which is
    where the container-hash and signature-un-inflation defects lived, so the
    non-reduced tree needs reading from the FILE just as much as the reduced one.
    """
    scfg = cfg.family_cfg(family).get(stage)
    if scfg is None:
        return None
    bls = Path(bundle_root) / "shaders" / STAGE_DIR[stage] / "dx_5_0" / scfg["bls_name"]
    perm = bls.parent / ("%s_%s.perm" % (family, stage))
    if (not bls.exists()) or (reduced and not perm.exists()):
        if verbose:
            print("  %-22s NOT BUILT (%s)" % ("%s_%s" % (family, stage),
                                              bls.name if not bls.exists()
                                              else perm.name))
        return None
    blobs = read_bls_blobs(bls)

    os.makedirs(SCRATCH, exist_ok=True)
    rows = list(enumerate(cfg.iter_slots(family, stage)))
    if reduced:
        cls, payload_of, n_blobs = read_perm_table(perm)

        def blob_index(_i, slot):
            return cls[slot]

        def pins_for(slot):
            return {2: payload_of(slot)}
    else:
        n_blobs = len(rows)

        # build_stage's non-reduced leg names its inputs
        # `[perm_%03d % slot for slot, ... in iter_slots(...)]`, so the container's
        # k-th blob is the k-th ROW -- not the k-th slot NUMBER, which is not dense.
        def blob_index(i, _slot):
            return i

        def pins_for(_slot):
            return None

    if len(blobs) != n_blobs:
        return [("bundle holds %d blobs, expected %d" % (len(blobs), n_blobs))]

    if sample and len(rows) > sample:
        rows = rows[::max(1, len(rows) // sample)][:sample]
    fxfile = cfg.fx_path(family)
    fails, counted = [], [0]

    def one(irow):
        i, row = irow
        slot, bv, live, _dd = row
        tag = "%s_%s_%d" % (family, stage, slot)
        ref, rerr = V.compile_reference(
            fxfile, scfg["fx_entry"], stage, bv, live,
            os.path.join(SCRATCH, tag + "_ref.fx"),
            uv_mappings=cfg.uv_mappings(bv),
            inject_preamble=scfg.get("inject_preamble", True),
            uv_random_offsets=cfg.uv_random_offsets(bv))
        if ref is None:
            # Same exclusion sc2_validate_all makes: fxc rejecting the ORIGINAL .fx
            # has no ground truth, so it is not a bundle defect.
            if any(x in str(rerr) for x in ("X3504", "X3500", "X4014")):
                return None
            return "slot %d: reference %s" % (slot, str(rerr)[:90])
        bi = blob_index(i, slot)
        blob = blobs[bi]
        if blob is None:
            return "slot %d: bundle entry %d is empty" % (slot, bi)
        dx = os.path.join(SCRATCH, tag + ".dxbc")
        with open(dx, "wb") as f:
            f.write(blob)
        asm_path = os.path.join(SCRATCH, tag + ".asm")
        try:
            cand = V.disasm_dxbc(dx, asm_path)
        except Exception as e:
            return "slot %d: disasm %s" % (slot, str(e)[:90])
        if cand is None:
            return "slot %d: disasm produced nothing" % slot
        # "slangc-bundled", not "slangc" and not "fxc".  _process_dxbc has already
        # run fix_dxbc_signatures over this blob, so its SIGNATURE indices are in
        # fxc's convention while its CBUFFER REFLECTION is still slangc's.  Those
        # are two different questions and the comparator now asks them separately
        # (_sig_inflated vs _is_slangc); answering both with one flag is what made
        # this gate report "no shared VS output" and then 1.88 on a shader whose
        # body is byte-identical to a passing baseline.  Only a gate that reads the
        # SHIPPED bytes can see this at all.
        cand = V.TOOLCHAIN_TAG % "slangc-bundled" + cand
        pins = pins_for(slot)
        if stage == "vs":
            diffs, derr = V.compare_vs(ref, cand, trials=trials, cb_pins=pins,
                                       input_domains=cfg.VS_INPUT_DOMAINS.get(family),
                                       const_domains=cfg.VS_CONST_DOMAINS.get(family))
        else:
            diffs, derr = V.compare_d3d11(ref, cand, trials=trials, cb_pins=pins,
                                          const_domains=cfg.PS_CONST_DOMAINS.get(family))
        if derr:
            return "slot %d: %s" % (slot, derr[:90])
        counted[0] += 1
        worst = max(diffs.values()) if diffs else 0.0
        bk = VA._bucket(family, stage, bv, reduced=reduced)
        if worst > VA.THRESH[bk]:
            return "slot %d: worst %.6g > %s threshold %g" % (slot, worst, bk,
                                                              VA.THRESH[bk])
        return None

    with ThreadPoolExecutor(max_workers=jobs) as ex:
        for r in ex.map(one, rows):
            if r:
                fails.append(r)
    if verbose:
        name = "%s_%s" % (family, stage)
        if fails:
            print("  %-22s %d/%d FAIL" % (name, len(fails), len(rows)))
            for f in fails[:4]:
                print("      %s" % f)
        else:
            print("  %-22s %4d slots served by %4d bundled blobs: green"
                  % (name, counted[0], n_blobs))
    return fails


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", nargs="*")
    ap.add_argument("--stage", choices=("vs", "ps"))
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--bundle", default=None,
                    help="bundle root (default: the reduced tree, or the "
                         "non-reduced one with --non-reduced)")
    ap.add_argument("--non-reduced", dest="non_reduced", action="store_true",
                    help="validate the per-slot bundle (no .perm, no b2 pins)")
    ap.add_argument("--sample", type=int, default=40)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--trials", type=int, default=8)
    args = ap.parse_args()
    if args.bundle is None:
        args.bundle = str(Path(REPO) / ("bls_out_sc2_1_14" if args.non_reduced
                                        else "bls_out_sc2_reduced_1_14"))
    if args.all or not args.family:
        stages = R._all_stages()
    else:
        stages = [(f, s) for f in args.family
                  for s in (("vs", "ps") if not args.stage else (args.stage,))
                  if s in cfg.family_cfg(f)]
    print("bundle validation: the SHIPPED %s against the fxc reference\n  %s"
          % (".bls" if args.non_reduced else ".bls + .perm", args.bundle))
    rc = 0
    for fam, st in stages:
        if check_stage(fam, st, args.bundle, sample=args.sample, jobs=args.jobs,
                       trials=args.trials, reduced=not args.non_reduced):
            rc = 1
    print("")
    print("BUNDLE DEFECTS ABOVE" if rc else "every sampled slot loads and matches")
    return rc


if __name__ == "__main__":
    sys.exit(main())
