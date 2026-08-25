#!/usr/bin/env python3
"""Bundle compiled sc2_shaders permutations into per-family BLS files.

SC2 ships no BLS templates (unlike Wc3), so this writes the template-less
**v1.14 DX50** container: each family+stage becomes one BLS whose permutation
slots are the family's retail permutation set, in retail cache order (the
manifest slot order), split VS/PS.  Each slot carries its slang-compiled DXBC
(dedup slots simply carry their own — byte-identical — blob); the v1.14 inner
resource-binding chunk is zero-filled (a consumer reads DXBC reflection).

Reads:  sc2_slang_out/d3d11/<Family>_<stage>/perm_<NNN>.dxbc  (compile_all_sc2.py)
Writes: bls_out_sc2_1_14/shaders/<pixel|vertex>/dx_5_0/<bls_name>

Usage:
  python build_sc2_bls.py [--family Simple] [--slang-out sc2_slang_out]
                          [--output bls_out_sc2] [--verify]
"""
import os
import sys
import struct
import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools"))
import build_bls as bb
import sc2_shaders_cfg as cfg
import sc2_perm_reduce as R

STAGE_DIR = {"vs": "vertex", "ps": "pixel"}


def _process_dxbc(dxbc):
    """Fix slangc's ATTRn ×10 semantic bug, then recompute the DXBC hash.
    No ISGN template alignment (SC2 ships none) — the signature is kept as
    slangc emitted it, only the numeric-suffix bug corrected."""
    dxbc = bb.fix_dxbc_signatures(dxbc)
    out = bytearray(dxbc)
    out[4:20] = bb.dxbc_hash(bytes(out[20:]))
    return bytes(out)


# ---------------------------------------------------------------------------
# The .perm side table (plan milestone R7 step 3)
# ---------------------------------------------------------------------------
# A reduced bundle holds one blob per structural CLASS, so a consumer can no longer
# index it by retail permutation number.  The side table restores that: retail slot
# -> (class index, b2 payload).  Both halves are needed and neither is derivable from
# the other -- the class says WHICH shader, the payload says which permutation it is
# being run AS, and a reduced shader with an unwritten b2 reads every axis as 0.
#
# Two encodings that matter, both measured rather than assumed:
#   * dedup    -- slots per stage far outnumber DISTINCT payloads, so payloads are
#                 pooled and each slot stores a u32 index into the pool.
#   * word mask-- the b2 buffer is ONE shape module-wide (every family's axes at the
#                 same global offsets), but no single family sets more than a
#                 fraction of it.  Storing only the words this stage ever makes
#                 nonzero, plus the mask saying which, drops the pool by the same
#                 fraction.  The consumer zero-fills the rest, which is exactly what
#                 the packer did.
PERM_MAGIC = b"SC2P"
PERM_VERSION = 1


def _perm_payload_pool(slots, total_words):
    """(word_mask_words, stored_word_indices, pool, per_slot_index)."""
    used = set()
    for _s, _c, p in slots:
        for i, w in enumerate(p):
            if w:
                used.add(i)
    stored = sorted(used)
    pool, index, per_slot = {}, [], []
    for _s, _c, p in slots:
        k = tuple(p[i] for i in stored)
        j = pool.get(k)
        if j is None:
            j = pool[k] = len(index)
            index.append(k)
        per_slot.append(j)
    mask = [0] * ((total_words + 31) // 32)
    for i in stored:
        mask[i // 32] |= 1 << (i % 32)
    return mask, stored, index, per_slot


def write_perm_table(family, stage, out_dir, verbose=True):
    """Write <family>_<stage>.perm next to the reduced bundle."""
    classes, slots = R.build_classes(family, stage)
    total_words = R.payload_words()
    if len(classes) > 0xFFFF:
        raise ValueError("%s_%s has %d classes; the u16 class index would overflow"
                         % (family, stage, len(classes)))
    mask, stored, pool, per_slot = _perm_payload_pool(slots, total_words)
    buf = bytearray()
    buf += PERM_MAGIC
    buf += struct.pack("<IIIIIII", PERM_VERSION, len(slots), len(classes), len(pool),
                       total_words, len(stored), 0)
    buf += struct.pack("<%dI" % len(mask), *mask)
    # Retail slot order IS manifest/cache order, so the two arrays are indexed by the
    # permutation number the engine already computes -- no lookup structure needed.
    buf += struct.pack("<%dH" % len(slots), *[c for _s, c, _p in slots])
    while len(buf) % 4:
        buf += bytes(1)
    buf += struct.pack("<%dI" % len(per_slot), *per_slot)
    for words in pool:
        buf += struct.pack("<%dI" % len(stored), *words) if stored else b""
    out = Path(out_dir) / ("%s_%s.perm" % (family, stage))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(bytes(buf))
    if verbose:
        print("    %-22s %6d slots -> %5d classes, %5d payloads x %2d/%d words"
              "  (%.2f MB)" % (out.name, len(slots), len(classes), len(pool),
                               len(stored), total_words, len(buf) / 1e6))
    return out, len(buf)


def verify_perm_table(path, family, stage, sample=0):
    """Read the table back and re-derive every slot from the classifier.

    This is the test that the bundle is USABLE, not merely written: for each retail
    slot the table must name the same class `build_classes` does and expand to
    exactly the words `pack_payload` produces.  Checking the file against the code
    that wrote it would prove nothing, so both halves are recomputed from the
    manifest instead."""
    data = Path(path).read_bytes()
    if data[:4] != PERM_MAGIC:
        return False, "bad magic"
    (ver, n_slots, n_classes, n_pool, total_words, stored_n, _f) =         struct.unpack_from("<IIIIIII", data, 4)
    if ver != PERM_VERSION:
        return False, "version %d" % ver
    off = 32
    nmask = (total_words + 31) // 32
    mask = struct.unpack_from("<%dI" % nmask, data, off)
    off += 4 * nmask
    stored = [i for i in range(total_words) if mask[i // 32] >> (i % 32) & 1]
    if len(stored) != stored_n:
        return False, "mask popcount %d != stored_words %d" % (len(stored), stored_n)
    cls = struct.unpack_from("<%dH" % n_slots, data, off)
    off += 2 * n_slots
    off += (-off) % 4
    pidx = struct.unpack_from("<%dI" % n_slots, data, off)
    off += 4 * n_slots
    pool = []
    for i in range(n_pool):
        pool.append(struct.unpack_from("<%dI" % stored_n, data, off + 4 * stored_n * i)
                    if stored_n else ())
    off += 4 * stored_n * n_pool
    if off != len(data):
        return False, "trailing %d bytes" % (len(data) - off)

    classes, slots = R.build_classes(family, stage)
    if len(classes) != n_classes or len(slots) != n_slots:
        return False, "shape %d/%d != %d/%d" % (n_classes, n_slots, len(classes),
                                                len(slots))
    it = slots
    if sample and len(it) > sample:
        it = it[::max(1, len(it) // sample)][:sample]
    for slot, ci, payload in it:
        if cls[slot] != ci:
            return False, "slot %d: class %d != %d" % (slot, cls[slot], ci)
        got = [0] * total_words
        for k, w in zip(stored, pool[pidx[slot]]):
            got[k] = w
        if got != list(payload):
            bad = [k for k in range(total_words) if got[k] != payload[k]]
            return False, "slot %d: payload words %s differ" % (slot, bad[:6])
    return True, "%d slots, %d classes, %d payloads, %d/%d words stored" % (
        n_slots, n_classes, n_pool, stored_n, total_words)


def build_stage(family, stage, slang_out, out_root, verbose=True, reduced=False):
    scfg = cfg.family_cfg(family).get(stage)
    if scfg is None:
        return None
    suffix = "_reduced" if reduced else ""
    in_dir = Path(slang_out) / "d3d11" / ("%s_%s%s" % (family, stage, suffix))

    dxbcs = []
    if reduced:
        classes, _slots = R.build_classes(family, stage)
        names = [("class_%05d.dxbc" % i) for i in range(len(classes))]
    else:
        names = [("perm_%03d.dxbc" % slot)
                 for slot, _bv, _live, _dedup in cfg.iter_slots(family, stage)]
    for name in names:
        p = in_dir / name
        if not p.exists():
            raise FileNotFoundError(
                "%s (run compile_all_sc2.py --family %s --stage %s%s first)"
                % (p, family, stage, " --reduced" if reduced else ""))
        dxbcs.append(_process_dxbc(p.read_bytes()))

    blob = bb.assemble_dx_v14_bls(dxbcs, bb.PLATFORM_TAG_DX5, bb.FLAGS_DX5)
    out_dir = Path(out_root) / "shaders" / STAGE_DIR[stage] / "dx_5_0"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / scfg["bls_name"]
    out_path.write_bytes(blob)
    if verbose:
        print("  %s %s -> %s  (%d %s, %#x bytes)"
              % (family, stage, out_path, len(dxbcs),
                 "classes" if reduced else "perms", len(blob)))
    if reduced:
        # The bundle alone is not loadable: its slots are classes, and the engine
        # indexes by retail permutation.  Writing the table here, from the same
        # build_classes call, is what keeps the two in step -- a bundle whose .perm
        # came from a different classifier run would map slots to the wrong blobs.
        write_perm_table(family, stage, out_dir, verbose=verbose)
    return out_path, len(dxbcs)


def verify_bls(path, expected_perms):
    """Full round-trip of a written v1.14 BLS: check magic/version + perm count,
    decompress the data blob, split it by the per-perm size table, and confirm
    each non-null slot's inner permutation carries an extractable `DXBC` blob at
    the §3.2 DX inner offset.  Proves the bundle reads back to the same slots we
    packed, in order."""
    import zlib
    data = Path(path).read_bytes()
    if data[:4] != bb.BLS_MAGIC:
        return False, "bad magic"
    minor, major = struct.unpack_from("<HH", data, 4)
    if (major, minor) != (bb.BLS_V14_MAJOR, bb.BLS_V14_MINOR):
        return False, "unexpected version v%d.%d" % (major, minor)
    off_perms, num_perms, off_blobs, num_blobs, off_data = \
        struct.unpack_from("<IIIII", data, 12)
    if num_perms != expected_perms:
        return False, "num_perms=%d expected %d" % (num_perms, expected_perms)

    sizes, prev_cum = [], 0
    cur = off_perms + 4
    for _ in range(num_perms):
        sz, = struct.unpack_from("<I", data, cur)
        cum, = struct.unpack_from("<I", data, cur + 20)
        if cum < prev_cum:
            return False, "non-monotonic cumulative offset"
        prev_cum = cum
        sizes.append(sz)
        cur += bb.BLS_V14_PERM_ENTRY

    decompressed = zlib.decompress(data[off_data:]) if num_blobs else b""
    pos = 0
    dxbc_ok = 0
    for i, sz in enumerate(sizes):
        if sz == 0:
            continue
        inner = decompressed[pos:pos + sz]
        pos += sz
        if inner[0x60:0x64] != b"DXBC":
            return False, "slot %d has no DXBC at inner +0x60" % i
        dxbc_ok += 1
    if pos != len(decompressed):
        return False, "perm sizes (%d) != decompressed length (%d)" % (pos, len(decompressed))
    return True, "%d perms (%d DXBC), %#x bytes" % (num_perms, dxbc_ok, len(data))


def main(argv=None):
    ap = argparse.ArgumentParser(description="Bundle sc2_shaders perms into BLS.")
    ap.add_argument("--family", help="one family; omit with --all")
    ap.add_argument("--all", action="store_true",
                    help="every family in sc2_shaders.json (M5.1)")
    ap.add_argument("--stage", choices=["ps", "vs"], help="default: both")
    ap.add_argument("--slang-out", default=str(REPO_ROOT / "sc2_slang_out"))
    ap.add_argument("--output", default=str(REPO_ROOT / "bls_out_sc2"))
    ap.add_argument("--reduced", action="store_true",
                    help="bundle the per-CLASS build and emit the .perm side table "
                         "(needs compile_all_sc2.py --reduced)")
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args(argv)
    if not args.all and not args.family:
        ap.error("pass --family <name> or --all")

    out_root = args.output + ("_reduced_1_14" if args.reduced else "_1_14")
    families = sorted(cfg.load_families()) if args.all else [args.family]
    stages = [args.stage] if args.stage else ["vs", "ps"]
    print("building %s -> %s" % (", ".join(families), out_root))
    rc = 0
    bundles = perms = 0
    for fam in families:
        for st in stages:
            r = build_stage(fam, st, args.slang_out, out_root,
                            reduced=args.reduced)
            if r is None:
                continue
            path, n = r
            bundles += 1
            perms += n
            if args.verify:
                ok, msg = verify_bls(path, n)
                print("    verify %s: %s" % ("OK" if ok else "FAIL", msg))
                if not ok:
                    rc = 1
                if args.reduced:
                    ok, msg = verify_perm_table(
                        path.parent / ("%s_%s.perm" % (fam, st)), fam, st)
                    print("    verify perm %s: %s" % ("OK" if ok else "FAIL", msg))
                    if not ok:
                        rc = 1
    if args.all:
        print("\n%d bundles, %d permutation slots -> %s" % (bundles, perms, out_root))
    return rc


if __name__ == "__main__":
    sys.exit(main())
