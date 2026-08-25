"""Extract the D3D11 DXBC perms from WoW ``GXSH``/``GFAT`` ``.bls`` bundles.

WoW's shipped shader bundles differ from the War3 ones handled by
``extract_retail_bls.py``: each ``.bls`` is a ``GXSH`` container whose payload
is split into 0x4000-byte blocks, each deflated separately, with a perm table
mapping permutation slots to ``{offset, size}`` records inside the reassembled
payload (slot offset ``0xFFFFFFFF`` = perm not built). Every record ends
exactly at the end of its embedded ``DXBC`` container. Two header versions
exist:

* ``0x0001000E`` — API tag (``DX50``) at +0x08, tables at +0x0C, perm entries
  are 24 bytes ``{u32 offset, u32 size, u8 md5[16]}``
* ``0x0001000C`` — no tag, tables at +0x08, perm entries are 8 bytes
  ``{u32 offset, u32 size}``

The stage-root files (``pixel/material3_*.bls`` etc.) are ``GFAT`` multi-API
wrappers — ``{tag, start, end}`` entries — whose ``DX50`` section is itself a
GXSH container; those are unwrapped transparently.

    python tools/extract_wow_bls.py wow_bls_shaders -o wow_re_shaders

Output: ``<out>/<stage>/<shader>/perm_NNNN.dxbc``, numbered by permutation
SLOT index (gaps = perms the game never built), mirroring the input tree.
"""

import argparse
import struct
import sys
import zlib
from pathlib import Path

GXSH_MAGIC = 0x47585348  # 'HSXG' little-endian
GFAT_MAGIC = 0x47464154  # 'TAFG' little-endian
CHUNK_UNCOMP = 0x4000


def unwrap_gfat(data, want=b'05XD'):
    """Return the ``want`` section of a GFAT wrapper (or ``data`` unchanged)."""
    if struct.unpack_from('<I', data, 0)[0] != GFAT_MAGIC:
        return data
    count = struct.unpack_from('<I', data, 8)[0]
    for i in range(count):
        tag, start, end = struct.unpack_from('<4sII', data, 0x0C + i * 12)
        if tag == want:
            return data[start:end]
    return None


def parse_gxsh(data):
    """Return ``[(slot, record_bytes), ...]`` for every present perm."""
    magic, version = struct.unpack_from('<II', data, 0)
    if magic != GXSH_MAGIC:
        raise ValueError(f"not a GXSH container (magic {magic:#x})")
    if version == 0x1000E:
        base, perm_stride = 0x0C, 24
    elif version == 0x1000C:
        base, perm_stride = 0x08, 8
    else:
        raise ValueError(f"unknown GXSH version {version:#x}")
    perm_off, perm_cnt, chunk_off, chunk_cnt, data_off = \
        struct.unpack_from('<5I', data, base)

    # chunk table: start offsets (relative to data_off) of the zlib streams;
    # a trailing u32 holds the total compressed size
    chunks = struct.unpack_from(f'<{chunk_cnt + 1}I', data, chunk_off)
    payload = b''.join(
        zlib.decompress(data[data_off + chunks[i]:data_off + chunks[i + 1]])
        for i in range(chunk_cnt))

    perms = []
    for slot in range(perm_cnt):
        off, size = struct.unpack_from('<II', data, perm_off + slot * perm_stride)
        if off != 0xFFFFFFFF:
            perms.append((slot, payload[off:off + size]))
    return perms


def slice_dxbc(record):
    """Return the DXBC container inside a perm record (None if absent)."""
    i = record.find(b'DXBC')
    if i < 0 or i + 0x20 > len(record):
        return None
    total = struct.unpack_from('<I', record, i + 0x18)[0]
    if i + total > len(record):
        return None
    return record[i:i + total]


def extract_file(bls_path, out_dir):
    """Extract one .bls into ``out_dir``; return (written, skipped)."""
    data = unwrap_gfat(Path(bls_path).read_bytes())
    if data is None:
        print(f"warning: {bls_path}: GFAT wrapper has no DX50 section",
              file=sys.stderr)
        return 0, 0
    written = skipped = 0
    for slot, record in parse_gxsh(data):
        blob = slice_dxbc(record)
        if blob is None:
            print(f"warning: {bls_path}: perm {slot} has no DXBC blob",
                  file=sys.stderr)
            skipped += 1
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"perm_{slot:04d}.dxbc").write_bytes(blob)
        written += 1
    return written, skipped


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('root', help='wow_bls_shaders tree (stage dirs inside)')
    ap.add_argument('-o', '--out-dir', required=True,
                    help='output tree root (e.g. wow_re_shaders)')
    args = ap.parse_args(argv)

    root, out_root = Path(args.root), Path(args.out_dir)
    targets = sorted(root.glob('*/dx_5_0/*.bls')) + sorted(root.glob('*/*.bls'))
    if not targets:
        print(f"error: no .bls under {root}", file=sys.stderr)
        return 1

    grand_written = grand_skipped = nfiles = 0
    for bls in targets:
        stage = bls.relative_to(root).parts[0]
        written, skipped = extract_file(bls, out_root / stage / bls.stem)
        grand_written += written
        grand_skipped += skipped
        nfiles += 1
    print(f"{root}: {nfiles} .bls -> {grand_written} perms in {out_root}"
          + (f"  ({grand_skipped} records without DXBC skipped)"
             if grand_skipped else ""))
    return 0


if __name__ == '__main__':
    sys.exit(main())
