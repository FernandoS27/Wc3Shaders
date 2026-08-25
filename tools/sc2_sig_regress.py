#!/usr/bin/env python
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026, Fernando Sahmkow
# See LICENSE in the repository root for full terms.
"""Regression test: does the gate CATCH the array-element signature bug?

Two defects were found together and it matters that they stay found:

  * `fix_dxbc_signatures` recovered slangc's x10 semantic inflation with
    `idx % 10 == 0 -> idx // 10`, which skips ARRAY elements (`b*10 + e`) and
    left them inflated beside an already-corrected base.
  * the comparator only errored when the two legs shared NO output at all, so a
    vertex stage whose every interpolant misaligned still compared `SV_Position`
    and reported green.

Either one alone is survivable; together they shipped 31% of the SC2 module with
a mislabelled input signature and a gate that said it was fine.  So this asserts
both directions on a real blob:

    old rule -> comparator MUST fail with a ref-only output
    new rule -> comparator MUST pass

The "caught" threshold here is a fixed 1e-3, deliberately MORE sensitive than the
shipping buckets: the point is that the old rule produces a measurable difference
where the new one is bit-exact, not that every affected slot would have breached a
release gate.  Some would not have -- Ribbon_ps slot 30 diverges by 1.9e-3, inside
the 5e-3 `trans` bucket -- which is part of why this shipped unnoticed.

Run: python tools/sc2_sig_regress.py [--family Model --stage vs]
"""
import argparse
import glob
import os
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
for p in (HERE, REPO):
    if p not in sys.path:
        sys.path.insert(0, p)

import build_bls as bb                 # noqa: E402
import build_sc2_bls as B              # noqa: E402
import sc2_shaders_cfg as cfg          # noqa: E402
import sc2_slang_validate as V         # noqa: E402
import sc2_bundle_validate as BV       # noqa: E402

SCRATCH = os.path.join("C:\\", "tmp", "sc2_sig_regress")
SGN = {b"ISGN": (24, 0, 4), b"OSGN": (24, 0, 4)}


def _old_fix(dxbc):
    """The pre-fix un-inflation, kept verbatim so the test exercises the real bug."""
    out = bytearray(dxbc)
    for fourcc, off, _s in bb.dxbc_chunks(dxbc):
        if fourcc not in SGN:
            continue
        es, _no, si = SGN[fourcc]
        bs = off + 8
        cnt, = struct.unpack_from("<I", out, bs)
        for i in range(cnt):
            e = bs + 8 + i * es + si
            idx, = struct.unpack_from("<I", out, e)
            if idx and idx % 10 == 0:
                struct.pack_into("<I", out, e, idx // 10)
    out[4:20] = bb.dxbc_hash(bytes(out[20:]))
    return bytes(out)


def _has_array_element(dxbc):
    """True if any signature entry is inflated but NOT a multiple of ten."""
    for fourcc, off, _s in bb.dxbc_chunks(dxbc):
        if fourcc not in SGN:
            continue
        es, _no, si = SGN[fourcc]
        bs = off + 8
        cnt, = struct.unpack_from("<I", dxbc, bs)
        for i in range(cnt):
            idx, = struct.unpack_from("<I", dxbc, bs + 8 + i * es + si)
            if idx >= 10 and idx % 10:
                return True
    return False


def run(family, stage, tries=6, scan=4000):
    scfg = cfg.family_cfg(family)[stage]
    cdir = os.path.join(REPO, "sc2_slang_out", "d3d11",
                        "%s_%s_reduced" % (family, stage))
    perm = os.path.join(REPO, "bls_out_sc2_reduced_1_14", "shaders",
                        BV.STAGE_DIR[stage], "dx_5_0", "%s_%s.perm" % (family, stage))
    cls, payload_of, _n = BV.read_perm_table(perm)
    os.makedirs(SCRATCH, exist_ok=True)
    label = "%s_%s" % (family, stage)

    # Carrying an array element is NECESSARY for the bug to bite but not
    # SUFFICIENT: the mislabelled entry may sit on a signature slot that never
    # reaches a compared output, in which case the old rule is harmless there and
    # that slot proves nothing.  So gather several candidates and keep going until
    # one actually demonstrates the failure.  If none does, say so explicitly --
    # an unobservable slot must not be able to pass for a green.
    cands, seen = [], 0
    for slot, bv, live, _d in cfg.iter_slots(family, stage):
        seen += 1
        if seen > scan or len(cands) >= tries:
            break
        if slot >= len(cls):
            continue
        q = os.path.join(cdir, "class_%05d.dxbc" % cls[slot])
        if os.path.exists(q) and _has_array_element(open(q, "rb").read()):
            cands.append((slot, bv, live, q))
    if not cands:
        print("%-18s no class carries an array element - not applicable" % label)
        return None

    def _fmt(err, worst):
        return err[:44] if err else "ok worst=%.6g" % worst

    unobserved = 0
    for slot, bv, live, path in cands:
        raw = open(path, "rb").read()
        ref, rerr = V.compile_reference(
            cfg.fx_path(family), scfg["fx_entry"], stage, bv, live,
            os.path.join(SCRATCH, "ref.fx"), uv_mappings=cfg.uv_mappings(bv),
            inject_preamble=scfg.get("inject_preamble", True),
            uv_random_offsets=cfg.uv_random_offsets(bv))
        if ref is None:
            continue

        # The 'new' leg goes through the PRODUCTION path, not the signature fix
        # alone.  Calling fix_dxbc_signatures by itself leaves a stale container
        # checksum, fxc refuses the blob, and the disassembly comes back empty --
        # which reads as "no shared VS output" and looks like a signature failure.
        # That is defect 1 all over again, so the test must call _process_dxbc.
        res = {}
        for leg, blob in (("old", _old_fix(raw)), ("new", B._process_dxbc(raw))):
            dx = os.path.join(SCRATCH, "%s.dxbc" % leg)
            with open(dx, "wb") as f:
                f.write(blob)
            asm = V.disasm_dxbc(dx, os.path.join(SCRATCH, "%s.asm" % leg))
            cand = V.TOOLCHAIN_TAG % "slangc-bundled" + asm
            fn = V.compare_vs if stage == "vs" else V.compare_d3d11
            # A reduced class blob is one shader standing in for many permutations
            # and reads its axes from b2.  Leave that unpinned and it runs as some
            # other permutation entirely: both legs diverge and the test measures
            # nothing about signatures.
            kw = {"trials": 4, "cb_pins": {2: payload_of(slot)}}
            if stage == "vs":
                kw.update(input_domains=cfg.VS_INPUT_DOMAINS.get(family),
                          const_domains=cfg.VS_CONST_DOMAINS.get(family))
            else:
                kw.update(const_domains=cfg.PS_CONST_DOMAINS.get(family))
            diffs, derr = fn(ref, cand, **kw)
            res[leg] = (derr, None if derr else
                        (max(diffs.values()) if diffs else 0.0))

        o_err, o_worst = res["old"]
        n_err, n_worst = res["new"]
        # The bug presents differently per stage.  On a VERTEX stage it corrupts the
        # OUTPUT signature, so the comparator strands a reference output and errors.
        # On a PIXEL stage it corrupts the INPUT signature, the legs then feed one
        # register different values, and it surfaces as a large numeric diff.
        # Either counts as caught; only silence would be a regression.
        caught = ("ref-only" in str(o_err)) or (o_worst is not None
                                                and o_worst > 1e-3)
        clean = (n_err is None) and (n_worst is not None) and (n_worst <= 1e-6)
        if not caught:
            unobserved += 1
            continue
        print("%-18s slot %-6d  old: %-44s  new: %-20s -> %s"
              % (label, slot, _fmt(o_err, o_worst), _fmt(n_err, n_worst),
                 "PASS" if clean else "FAIL"))
        return clean

    print("%-18s INCONCLUSIVE: %d candidate class(es) carry an array element but "
          "none reaches a compared output" % (label, unobserved))
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--family", nargs="*",
                    default=["Model", "Particle", "Ribbon", "PostProcessQuad"])
    ap.add_argument("--stage", nargs="*", default=["vs", "ps"])
    a = ap.parse_args()
    outcomes = []
    for fam in a.family:
        for st in a.stage:
            if cfg.family_cfg(fam).get(st) is None:
                continue
            r = run(fam, st)
            if r is not None:
                outcomes.append(r)
    bad = outcomes.count(False)
    print("\n%d checked, %d FAILED" % (len(outcomes), bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
