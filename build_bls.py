"""
Rebuild War3 BLS (v1.8) files from Slang-compiled shader blobs.

This is the inverse of extract_bls.py. For each shader family we:
  1. Open the shipped BLS as a template (DX only — to reuse each perm's
     44-byte "middle chunk" of resource/binding metadata). For Metal the
     template is optional and only consulted for the null-perm pattern.
  2. Load the Slang-compiled blob for every perm index.
  3. For DX: optionally strip RDEF/STAT chunks (not present in shipped
     binaries) and recompute the DXBC hash so the blob matches the shipped
     chunk layout exactly.
  4. Repack into the BLS wire format and write out the file.

The seven shader-family -> Slang-output mapping lives in FAMILY_MAP /
METAL_FAMILY_MAP below. Shipped BLS files live under
war3.w3mod/shaders/{ps,vs,mtlfs,mtlvs}/*.bls. Slang outputs live under:
  <repo>/slang_out/d3d11/<family>/perm_NNN.dxbc      (DX, any platform)
  <repo>/slang_out/metal/<family>/perm_NNN.metallib  (Metal, macOS only)

DX BLS files are always rebuilt; Metal BLS files are rebuilt only when
compile_all_slang.py produced .metallib output for that family (i.e. you
were running on macOS with Apple's Metal compiler available).

Usage:
  build_bls.py --templates war3.w3mod/shaders \\
               --output    out/shaders \\
               [--slang-out slang_out] [--strip] [--family hd_ps]
"""

import argparse
import os
import struct
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_SLANG_OUT = REPO_ROOT / "slang_out"
DXBC_TARGET_SUBDIR  = "d3d11"   # compile_all_slang writes .dxbc under this target.
METAL_TARGET_SUBDIR = "metal"   # ... and .metallib under this one (macOS only).


PERM_INNER_HEADER_SIZE       = 0x50  # bytes before the DXBC blob in each DX perm
METAL_PERM_INNER_HEADER_SIZE = 0x2C  # bytes before the MTLB blob in each Metal perm
BLS_FILE_HEADER_SIZE         = 0x18  # bytes before the cum-size table

BLS_MAGIC       = b'HSXG'
BLS_MINOR       = 8
BLS_MAJOR       = 1
BLS_PRE_META    = 0x14
BLS_DXBC_TAG    = 4             # value at DX perm inner header +0x4C
BLS_METAL_STAGE = 1             # DX perms carry a real stage id; Metal perms
                                # always use stage=1 (MTLB encodes the stage).

# Which Slang output directory feeds which shipped DX BLS, and how many perms
# each family emits. The tuple also pins the VS/PS subfolder under
# --templates so we can find the template BLS automatically.
FAMILY_MAP = {
    'hd_ps':          ('ps', 'hd.bls',          512),
    'crystal_ps':     ('ps', 'crystal.bls',     512),
    'sd_classic_ps':  ('ps', 'sd.bls',          200),
    'sd_on_hd_ps':    ('ps', 'sd_on_hd.bls',    384),
    'hd_vs':          ('vs', 'hd.bls',          144),
    'sd_highspec_vs': ('vs', 'sd_highspec.bls', 162),
    'sd_on_hd_vs':    ('vs', 'sd_on_hd.bls',    144),
}

# Same seven families, re-pointed at the Metal BLS bundles under mtlfs/ and
# mtlvs/. Perm counts match the DX side — DX and Metal share the same
# Slang-level permutation enumeration.
METAL_FAMILY_MAP = {
    'hd_ps':          ('mtlfs', 'hd.bls',          512),
    'crystal_ps':     ('mtlfs', 'crystal.bls',     512),
    'sd_classic_ps':  ('mtlfs', 'sd.bls',          200),
    'sd_on_hd_ps':    ('mtlfs', 'sd_on_hd.bls',    384),
    'hd_vs':          ('mtlvs', 'hd.bls',          144),
    'sd_highspec_vs': ('mtlvs', 'sd_highspec.bls', 162),
    'sd_on_hd_vs':    ('mtlvs', 'sd_on_hd.bls',    144),
}


# ============================================================
# DXBC parsing / chunk manipulation
# ============================================================

def dxbc_chunks(dxbc):
    """Yield (fourcc, offset, size) for every chunk in a DXBC blob."""
    if dxbc[:4] != b'DXBC':
        raise ValueError('not a DXBC blob (bad magic)')
    chunk_count, = struct.unpack_from('<I', dxbc, 28)
    for i in range(chunk_count):
        off, = struct.unpack_from('<I', dxbc, 32 + i * 4)
        fourcc = dxbc[off:off + 4]
        size, = struct.unpack_from('<I', dxbc, off + 4)
        yield fourcc, off, size


def strip_dxbc_chunks(dxbc, drop_fourccs):
    """Return a new DXBC blob with the named chunks removed.

    Preserves the original ordering of the kept chunks. Rewrites the
    file-size field (offset 24) and the chunk offset table, then recomputes
    the 16-byte hash at offset 4.
    """
    keep = []
    for fourcc, off, size in dxbc_chunks(dxbc):
        if fourcc in drop_fourccs:
            continue
        keep.append((fourcc, dxbc[off:off + 8 + size]))  # include tag+size header

    chunk_count = len(keep)
    header_size = 32 + chunk_count * 4
    offsets = []
    body = bytearray()
    cursor = header_size
    for _, chunk_bytes in keep:
        offsets.append(cursor)
        body += chunk_bytes
        cursor += len(chunk_bytes)

    total = header_size + len(body)
    out = bytearray(header_size + len(body))
    out[0:4]   = b'DXBC'
    out[4:20]  = b'\x00' * 16                # placeholder hash, filled below
    out[20:24] = struct.pack('<I', 1)        # version field (constant)
    out[24:28] = struct.pack('<I', total)
    out[28:32] = struct.pack('<I', chunk_count)
    for i, off in enumerate(offsets):
        struct.pack_into('<I', out, 32 + i * 4, off)
    out[header_size:] = body

    # Recompute hash over everything after byte 20.
    out[4:20] = dxbc_hash(bytes(out[20:]))
    return bytes(out)


# ============================================================
# DXBC hash — the modified-MD5 variant used by fxc/d3dcompiler
# ============================================================
# fxc emits a 16-byte digest at bytes 4..20 of every DXBC blob. The digest
# is computed over the rest of the file (bytes 20..end) using a standard
# MD5 core, but with a custom message-length padding rule specific to
# Microsoft's compiler. Implementation mirrors the one in Wine's
# d3dcompiler_43 and the open-source dxbc-hash tools.

def _md5_core(state, block):
    # Stock MD5 compression function, one 64-byte block.
    def F(x, y, z): return (z ^ (x & (y ^ z))) & 0xFFFFFFFF
    def G(x, y, z): return (y ^ (z & (x ^ y))) & 0xFFFFFFFF
    def H(x, y, z): return (x ^ y ^ z) & 0xFFFFFFFF
    def I(x, y, z): return (y ^ (x | (~z & 0xFFFFFFFF))) & 0xFFFFFFFF
    def rol(v, n): return ((v << n) | (v >> (32 - n))) & 0xFFFFFFFF

    K = [
        0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee, 0xf57c0faf, 0x4787c62a,
        0xa8304613, 0xfd469501, 0x698098d8, 0x8b44f7af, 0xffff5bb1, 0x895cd7be,
        0x6b901122, 0xfd987193, 0xa679438e, 0x49b40821, 0xf61e2562, 0xc040b340,
        0x265e5a51, 0xe9b6c7aa, 0xd62f105d, 0x02441453, 0xd8a1e681, 0xe7d3fbc8,
        0x21e1cde6, 0xc33707d6, 0xf4d50d87, 0x455a14ed, 0xa9e3e905, 0xfcefa3f8,
        0x676f02d9, 0x8d2a4c8a, 0xfffa3942, 0x8771f681, 0x6d9d6122, 0xfde5380c,
        0xa4beea44, 0x4bdecfa9, 0xf6bb4b60, 0xbebfbc70, 0x289b7ec6, 0xeaa127fa,
        0xd4ef3085, 0x04881d05, 0xd9d4d039, 0xe6db99e5, 0x1fa27cf8, 0xc4ac5665,
        0xf4292244, 0x432aff97, 0xab9423a7, 0xfc93a039, 0x655b59c3, 0x8f0ccc92,
        0xffeff47d, 0x85845dd1, 0x6fa87e4f, 0xfe2ce6e0, 0xa3014314, 0x4e0811a1,
        0xf7537e82, 0xbd3af235, 0x2ad7d2bb, 0xeb86d391,
    ]
    S = [
        7, 12, 17, 22,  7, 12, 17, 22,  7, 12, 17, 22,  7, 12, 17, 22,
        5,  9, 14, 20,  5,  9, 14, 20,  5,  9, 14, 20,  5,  9, 14, 20,
        4, 11, 16, 23,  4, 11, 16, 23,  4, 11, 16, 23,  4, 11, 16, 23,
        6, 10, 15, 21,  6, 10, 15, 21,  6, 10, 15, 21,  6, 10, 15, 21,
    ]
    a, b, c, d = state
    M = struct.unpack('<16I', block)
    aa, bb, cc, dd = a, b, c, d
    for i in range(64):
        if i < 16:
            f, g = F(b, c, d), i
        elif i < 32:
            f, g = G(b, c, d), (5 * i + 1) % 16
        elif i < 48:
            f, g = H(b, c, d), (3 * i + 5) % 16
        else:
            f, g = I(b, c, d), (7 * i) % 16
        tmp = d
        d = c
        c = b
        b = (b + rol((a + f + K[i] + M[g]) & 0xFFFFFFFF, S[i])) & 0xFFFFFFFF
        a = tmp
    return [(aa + a) & 0xFFFFFFFF,
            (bb + b) & 0xFFFFFFFF,
            (cc + c) & 0xFFFFFFFF,
            (dd + d) & 0xFFFFFFFF]


def dxbc_hash(body):
    """Compute the fxc-style DXBC hash over the bytes that follow offset 20."""
    state = [0x67452301, 0xefcdab89, 0x98badcfe, 0x10325476]
    n = len(body)
    bit_len = n * 8
    # Process full 64-byte blocks.
    full_blocks = n // 64
    for i in range(full_blocks):
        state = _md5_core(state, body[i * 64:(i + 1) * 64])
    tail = body[full_blocks * 64:]
    rem = len(tail)

    # Microsoft's custom padding — see Wine d3dcompiler compute_hash().
    # Layout depends on how much of the final block is occupied.
    if rem >= 56:
        # Block A: tail || 0x80 || zero-pad to 64 bytes
        pad = bytearray(tail) + b'\x80'
        pad += b'\x00' * (64 - len(pad))
        state = _md5_core(state, bytes(pad))
        # Block B: bit_len || 0x80 || 48 zero bytes || 0x40 || zero pad
        final = bytearray(64)
        struct.pack_into('<I', final, 0, bit_len)
        final[4] = 0x80
        # bytes 5..59 already zero
        struct.pack_into('<I', final, 60, 0x80)  # actually write (n*2+1)*bit? See note
        # The documented Microsoft trick: store bit_len at start, byte 0x80 after it,
        # then 0x40 trailing marker. Wine's implementation writes:
        #   padding[0..3] = bit_len
        #   padding[4] = 0x80
        #   padding[60..63] = (n*2+1) packed as u32
        struct.pack_into('<I', final, 60, (n * 2 + 1))
        final[4:60] = b'\x80' + b'\x00' * 55
        state = _md5_core(state, bytes(final))
    else:
        # Single final block: bit_len || tail || 0x80 || zero-pad || (n*2+1)
        final = bytearray(64)
        struct.pack_into('<I', final, 0, bit_len)
        final[4:4 + rem] = tail
        final[4 + rem] = 0x80
        struct.pack_into('<I', final, 60, (n * 2 + 1))
        state = _md5_core(state, bytes(final))

    return struct.pack('<4I', *state)


# ============================================================
# BLS rebuild
# ============================================================

def read_template(bls_path):
    """Parse the shipped BLS and return per-perm middle-chunk bytes + header fields."""
    with open(bls_path, 'rb') as fp:
        data = fp.read()

    if data[:4] != BLS_MAGIC:
        raise ValueError(f'{bls_path}: bad magic')
    minor, major = struct.unpack_from('<HH', data, 4)
    if (major, minor) != (BLS_MAJOR, BLS_MINOR):
        raise ValueError(f'{bls_path}: unsupported v{major}.{minor}')

    pre_meta, num_perms, off_data, pad = struct.unpack_from('<4I', data, 8)
    if pre_meta != BLS_PRE_META or pad != 0:
        raise ValueError(f'{bls_path}: unexpected header')

    cum = struct.unpack_from(f'<{num_perms}I', data, BLS_FILE_HEADER_SIZE)

    middle_chunks = []
    stages = []
    prev = 0
    for i, end in enumerate(cum):
        size = end - prev
        if size == 0:
            middle_chunks.append(None)
            stages.append(None)
        else:
            start = off_data + prev
            middle_chunks.append(bytes(data[start + 0x1C:start + 0x48]))
            stages.append(struct.unpack_from('<I', data, start + 0x18)[0])
        prev = end

    return {
        'num_perms': num_perms,
        'middle_chunks': middle_chunks,
        'stages': stages,
    }


def pack_perm(middle_chunk, stage, dxbc):
    """Serialize one perm: 20 zero bytes || payload_size || stage || middle || dxbc_size || tag=4 || dxbc."""
    dxbc_size = len(dxbc)
    payload_size = 0x38 + dxbc_size          # bytes from +0x18 to end of perm
    buf = bytearray(PERM_INNER_HEADER_SIZE + dxbc_size)
    # bytes 0..0x14 stay zero (pre_meta)
    struct.pack_into('<I', buf, 0x14, payload_size)
    struct.pack_into('<I', buf, 0x18, stage)
    buf[0x1C:0x48] = middle_chunk
    struct.pack_into('<I', buf, 0x48, dxbc_size)
    struct.pack_into('<I', buf, 0x4C, BLS_DXBC_TAG)
    buf[PERM_INNER_HEADER_SIZE:] = dxbc
    return bytes(buf)


def build_bls(template_path, slang_dir, num_perms, strip=False, verbose=False):
    """Return the bytes of a rebuilt BLS for one shader family."""
    tmpl = read_template(template_path)
    if tmpl['num_perms'] != num_perms:
        raise ValueError(
            f'{template_path}: template has {tmpl["num_perms"]} perms, expected {num_perms}')

    width = max(3, len(str(num_perms - 1)))
    perm_blobs = []
    skipped_null = 0

    for i in range(num_perms):
        if tmpl['middle_chunks'][i] is None:
            perm_blobs.append(b'')          # null slot — contributes 0 bytes
            skipped_null += 1
            continue

        dxbc_path = os.path.join(slang_dir, f'perm_{i:0{width}d}.dxbc')
        with open(dxbc_path, 'rb') as fp:
            dxbc = fp.read()

        if strip:
            dxbc = strip_dxbc_chunks(dxbc, {b'RDEF', b'STAT'})

        perm_blobs.append(pack_perm(tmpl['middle_chunks'][i], tmpl['stages'][i], dxbc))

    # Cumulative size table (end offsets, relative to off_data)
    cum = []
    total = 0
    for blob in perm_blobs:
        total += len(blob)
        cum.append(total)

    off_data = BLS_FILE_HEADER_SIZE + num_perms * 4

    file_buf = bytearray(off_data + total)
    file_buf[0:4]   = BLS_MAGIC
    struct.pack_into('<HH', file_buf, 4, BLS_MINOR, BLS_MAJOR)
    struct.pack_into('<4I', file_buf, 8, BLS_PRE_META, num_perms, off_data, 0)
    struct.pack_into(f'<{num_perms}I', file_buf, BLS_FILE_HEADER_SIZE, *cum)

    cursor = off_data
    for blob in perm_blobs:
        file_buf[cursor:cursor + len(blob)] = blob
        cursor += len(blob)

    if verbose:
        print(f'  {os.path.basename(template_path)}: '
              f'{num_perms} perms ({skipped_null} null), file size {len(file_buf):#x}')

    return bytes(file_buf)


# ============================================================
# Metal BLS rebuild (mtlfs / mtlvs)
# ============================================================
# Wc3 Metal BLS files share the v1.8 outer format with the DX variants
# (24-byte file header, u32 cumulative-offset table, uncompressed perm
# blobs). The per-perm inner layout is different — see
# docs/BLS_FILE_FORMAT_SPECIFICATION.md §3.5:
#
#   +0x00..0x14: 20 zero bytes (pre_meta)
#   +0x14: payload_size (= 0x14 + metallib_size)
#   +0x18: stage         (u32, = 1)
#   +0x1C: entry_count   (u32, = 1)
#   +0x20: metallib_size (u32)
#   +0x24: flag          (u32, = 8)
#   +0x28: flag          (u32, = 1)
#   +0x2C..: MTLB blob
#   +(0x2C + metallib_size): single trailing 0x00 byte
#
# Unlike the DX side there is no opaque per-perm metadata chunk to pull
# from the shipped template; the only reason to read the template is to
# mirror the shipped null-permutation pattern.


def read_metal_template_nulls(bls_path, num_perms):
    """Return a list[bool] marking which perms are null in a shipped Metal BLS."""
    with open(bls_path, 'rb') as fp:
        data = fp.read()

    if data[:4] != BLS_MAGIC:
        raise ValueError(f'{bls_path}: bad magic')
    minor, major = struct.unpack_from('<HH', data, 4)
    if (major, minor) != (BLS_MAJOR, BLS_MINOR):
        raise ValueError(f'{bls_path}: unsupported v{major}.{minor}')
    _, tmpl_perms, _, pad = struct.unpack_from('<4I', data, 8)
    if pad != 0:
        raise ValueError(f'{bls_path}: unexpected header')
    if tmpl_perms != num_perms:
        raise ValueError(
            f'{bls_path}: template has {tmpl_perms} perms, expected {num_perms}')

    cum = struct.unpack_from(f'<{num_perms}I', data, BLS_FILE_HEADER_SIZE)
    nulls, prev = [], 0
    for end in cum:
        nulls.append(end == prev)
        prev = end
    return nulls


def pack_metal_perm(metallib):
    """Serialize one Metal perm: 20 zero || payload_size || stage=1 || entry_cnt=1
       || mtlb_size || 8 || 1 || MTLB || 0x00."""
    mtlb_size    = len(metallib)
    payload_size = 0x14 + mtlb_size                 # bytes from +0x18 to end-of-MTLB
    perm_size    = METAL_PERM_INNER_HEADER_SIZE + mtlb_size + 1  # +1 trailing 0x00
    buf = bytearray(perm_size)
    # bytes [0..0x14) stay zero (pre_meta)
    struct.pack_into('<I', buf, 0x14, payload_size)
    struct.pack_into('<I', buf, 0x18, BLS_METAL_STAGE)
    struct.pack_into('<I', buf, 0x1C, 1)             # entry_count
    struct.pack_into('<I', buf, 0x20, mtlb_size)
    struct.pack_into('<I', buf, 0x24, 8)
    struct.pack_into('<I', buf, 0x28, 1)
    buf[METAL_PERM_INNER_HEADER_SIZE:METAL_PERM_INNER_HEADER_SIZE + mtlb_size] = metallib
    # buf[-1] already 0 — trailing padding byte.
    return bytes(buf)


def build_metal_bls(template_path, slang_dir, num_perms, verbose=False):
    """Return the bytes of a rebuilt Metal BLS for one shader family.

    `template_path` is only read to mirror the shipped null-perm pattern.
    If it is missing, every present .metallib becomes a live perm and any
    missing .metallib is treated as a null perm.
    """
    if template_path and os.path.isfile(template_path):
        nulls = read_metal_template_nulls(template_path, num_perms)
    else:
        nulls = [False] * num_perms

    width = max(3, len(str(num_perms - 1)))
    perm_blobs = []
    skipped_null = 0

    for i in range(num_perms):
        metallib_path = os.path.join(slang_dir, f'perm_{i:0{width}d}.metallib')
        if nulls[i] or not os.path.isfile(metallib_path) \
                or os.path.getsize(metallib_path) == 0:
            perm_blobs.append(b'')
            skipped_null += 1
            continue

        with open(metallib_path, 'rb') as fp:
            metallib = fp.read()
        perm_blobs.append(pack_metal_perm(metallib))

    cum, total = [], 0
    for blob in perm_blobs:
        total += len(blob)
        cum.append(total)

    off_data = BLS_FILE_HEADER_SIZE + num_perms * 4
    file_buf = bytearray(off_data + total)
    file_buf[0:4] = BLS_MAGIC
    struct.pack_into('<HH', file_buf, 4, BLS_MINOR, BLS_MAJOR)
    struct.pack_into('<4I', file_buf, 8, BLS_PRE_META, num_perms, off_data, 0)
    struct.pack_into(f'<{num_perms}I', file_buf, BLS_FILE_HEADER_SIZE, *cum)

    cursor = off_data
    for blob in perm_blobs:
        file_buf[cursor:cursor + len(blob)] = blob
        cursor += len(blob)

    if verbose:
        label = os.path.basename(template_path) if template_path else '<no template>'
        print(f'  {label}: {num_perms} perms ({skipped_null} null), '
              f'file size {len(file_buf):#x}')
    return bytes(file_buf)


def has_metallibs(slang_dir):
    """True if `slang_dir` contains at least one non-empty .metallib file."""
    if not os.path.isdir(slang_dir):
        return False
    for name in os.listdir(slang_dir):
        if name.endswith('.metallib'):
            p = os.path.join(slang_dir, name)
            if os.path.isfile(p) and os.path.getsize(p) > 0:
                return True
    return False


# ============================================================
# CLI
# ============================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--slang-out', default=str(DEFAULT_SLANG_OUT),
                    help=f'top-level slang_out directory written by '
                         f'compile_all_slang.py (default: {DEFAULT_SLANG_OUT}). '
                         f'DXBC blobs are read from <slang-out>/{DXBC_TARGET_SUBDIR}/<family>/; '
                         f'.metallib blobs (optional) from <slang-out>/{METAL_TARGET_SUBDIR}/<family>/.')
    ap.add_argument('--templates', required=True,
                    help='directory containing ps/*.bls, vs/*.bls (required) and '
                         'mtlfs/*.bls, mtlvs/*.bls (used when rebuilding Metal BLS)')
    ap.add_argument('--output', required=True, help='output directory for rebuilt BLS')
    ap.add_argument('--family', action='append', choices=list(FAMILY_MAP),
                    help='limit to specific family (default: all 7)')
    ap.add_argument('--strip', action='store_true',
                    help='strip RDEF/STAT chunks from DXBC (match shipped chunk layout) '
                         'and recompute the DXBC hash')
    ap.add_argument('--verbose', '-v', action='store_true')
    args = ap.parse_args()

    families = args.family or list(FAMILY_MAP)
    os.makedirs(args.output, exist_ok=True)

    dxbc_root  = os.path.join(args.slang_out, DXBC_TARGET_SUBDIR)
    metal_root = os.path.join(args.slang_out, METAL_TARGET_SUBDIR)

    for fam in families:
        # ---------- DX (ps/, vs/) ----------
        stage_dir, bls_name, num_perms = FAMILY_MAP[fam]
        template = os.path.join(args.templates, stage_dir, bls_name)
        slang_dir = os.path.join(dxbc_root, fam)

        if not os.path.isfile(template):
            print(f'SKIP {fam}: template missing ({template})', file=sys.stderr)
        elif not os.path.isdir(slang_dir):
            print(f'SKIP {fam}: slang dir missing ({slang_dir}). '
                  f'Run compile_all_slang.py --target d3d11 first.', file=sys.stderr)
        else:
            out_dir = os.path.join(args.output, stage_dir)
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, bls_name)
            try:
                blob = build_bls(template, slang_dir, num_perms,
                                 strip=args.strip, verbose=args.verbose)
                with open(out_path, 'wb') as fp:
                    fp.write(blob)
                print(f'wrote {out_path} ({len(blob):#x} bytes, {num_perms} perms)')
            except Exception as e:
                print(f'FAIL {fam}: {e}', file=sys.stderr)

        # ---------- Metal (mtlfs/, mtlvs/) — only if metallibs were emitted ----
        m_stage_dir, m_bls_name, m_num_perms = METAL_FAMILY_MAP[fam]
        m_slang_dir = os.path.join(metal_root, fam)
        if not has_metallibs(m_slang_dir):
            continue

        m_template = os.path.join(args.templates, m_stage_dir, m_bls_name)
        m_out_dir  = os.path.join(args.output, m_stage_dir)
        os.makedirs(m_out_dir, exist_ok=True)
        m_out_path = os.path.join(m_out_dir, m_bls_name)
        try:
            blob = build_metal_bls(m_template, m_slang_dir, m_num_perms,
                                   verbose=args.verbose)
            with open(m_out_path, 'wb') as fp:
                fp.write(blob)
            print(f'wrote {m_out_path} ({len(blob):#x} bytes, {m_num_perms} perms)')
        except Exception as e:
            print(f'FAIL {fam} [metal]: {e}', file=sys.stderr)


if __name__ == '__main__':
    sys.exit(main())
