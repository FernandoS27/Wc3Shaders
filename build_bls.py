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

The shader-family -> Slang-output mapping lives in ``wc3_shaders.json`` at
the repo root (loaded via ``shader_config.load_families``); the same
config feeds compile_all_slang.py so both scripts stay in sync. Shipped
BLS files live under war3.w3mod/shaders/{ps,vs,mtlfs,mtlvs}/*.bls. Slang
outputs live under:
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

from shader_config import load_families

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_SLANG_OUT = REPO_ROOT / "slang_out"
DXBC_TARGET_SUBDIR  = "d3d11"   # compile_all_slang writes .dxbc under this target.
METAL_TARGET_SUBDIR = "metal"   # ... and .metallib under this one (macOS only).

# Extra (non-shipped) backends: opengl/vulkan/webgpu. Wc3 itself never loads
# these — the engine only ships DX and Metal — but `--build-extra` still
# packages the slangc outputs into BLS containers so users can ship them
# alongside the originals for ports / re-implementations. Each tuple is:
#   (slang_target_subdir, file_ext, vs_outdir, ps_outdir).
# The output subdirectories follow the convention ``<api><stage>`` to
# parallel the shipped ``mtlfs/mtlvs`` naming for Metal.
EXTRA_BACKENDS = {
    "opengl": ("opengl", "glsl", "glslvs", "glslps"),
    "vulkan": ("vulkan", "spv",  "spvvs",  "spvps"),
    "webgpu": ("webgpu", "wgsl", "wgpuvs", "wgpups"),
}


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

# Shader-family configuration is shared with compile_all_slang.py through
# wc3_shaders.json. FamilyConfig exposes stage, perm_count, bls_name, and the
# optional template override — everything this script previously carried
# in FAMILY_MAP / METAL_FAMILY_MAP / TEMPLATE_OVERRIDE. DX and Metal share
# the same basename and perm count for every family; only the containing
# directory differs (ps/vs vs mtlfs/mtlvs), which FamilyConfig derives.
FAMILIES = load_families()


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


def fix_dxbc_signatures(dxbc):
    """Patch ISGN/OSGN semantic indices emitted by slangc.

    The Slang compiler emits HLSL semantics by concatenating the source
    semantic name (already including its numeric suffix, e.g. ``TEXCOORD1``)
    with an additional ``0`` for every field, producing ``TEXCOORD10``.
    When fxc parses that HLSL it strips the trailing digits into the SGN
    `sem_idx` field, so a source semantic of ``TEXCOORD1`` ends up stored
    as ``name="TEXCOORD", sem_idx=10`` in the DXBC — a 10× shift.

    The shipped shaders use semantic indices 0..7, so every bugged index
    is a multiple of 10 and the original is recoverable by a simple
    ``sem_idx //= 10``. The SGN string table and entry sizes don't change,
    so this edit is in-place (just a single u32 per entry).

    Additionally we uppercase system-value names (``SV_Position`` →
    ``SV_POSITION`` etc.) to match the casing in the shipped DXBC blobs;
    D3D resolves SV_* by `sys_val` tag so this is safe but makes the
    rebuilt signature byte-identical to the shipped one aside from the
    program body.
    """
    if dxbc[:4] != b'DXBC':
        return dxbc
    out = bytearray(dxbc)
    for fourcc, off, _size in list(dxbc_chunks(dxbc)):
        if fourcc not in (b'ISGN', b'OSGN', b'PCSG'):
            continue
        body_start = off + 8
        count, = struct.unpack_from('<I', out, body_start)
        for i in range(count):
            e_off = body_start + 8 + i * 24
            sem_idx, = struct.unpack_from('<I', out, e_off + 4)
            if sem_idx and sem_idx % 10 == 0:
                struct.pack_into('<I', out, e_off + 4, sem_idx // 10)
            name_off, = struct.unpack_from('<I', out, e_off)
            s = body_start + name_off
            end = out.find(b'\x00', s)
            if end == -1:
                continue
            name = bytes(out[s:end])
            if name.upper().startswith(b'SV_'):
                out[s:end] = name.upper()
    return bytes(out)


def _shex_declared_input_regs(shex_body):
    """Scan a SHEX program body and return the set of input register
    indices that are actually declared (``dcl_input vN`` and variants).

    Uses the SM5 token stream layout: every instruction is ``opcode ||
    operand || ...`` and ``opcode.bits[24..30]`` is the instruction
    length in dwords. The DCL_INPUT* opcodes (0x5F / 0x60 / 0x61 / 0x62
    / 0x63 / 0x64) all encode the register index as the immediate32
    following the operand token, at ``cursor + 8``.
    """
    INPUT_OPCODES = {0x5F, 0x60, 0x61, 0x62, 0x63, 0x64}
    used = set()
    cursor = 8   # skip program version + length header
    while cursor < len(shex_body):
        tok, = struct.unpack_from('<I', shex_body, cursor)
        op = tok & 0x7FF
        length = (tok >> 24) & 0x7F
        if length == 0:
            # Extended-length (custom-data block): next u32 is the length in dwords.
            length, = struct.unpack_from('<I', shex_body, cursor + 4)
            cursor += length * 4
            continue
        if op in INPUT_OPCODES:
            reg, = struct.unpack_from('<I', shex_body, cursor + 8)
            used.add(reg)
        cursor += length * 4
    return used


def _rewrite_sgn_body(body, keep_regs, remap=None, overrides=None):
    """Rewrite an ISGN/OSGN/PCSG body, keeping only entries whose register
    index is in ``keep_regs`` (or any entry with a non-zero system-value
    tag — SV_Position etc. must stay in the signature even when the
    program body never consumes them, because D3D validates stage-to-stage
    linkage against those entries). Optionally applies a ``remap`` of
    old→new register numbers so the kept entries can be renumbered to
    be contiguous; the SHEX body must be renumbered with the same map.
    ``overrides`` is a ``{old_reg: (sys_val, comp_ty, mask_rw)}`` dict —
    when present, kept entries adopt the template's metadata on those
    fields so the rebuilt signature matches shipped even when our
    compiled body reads a different component subset (mask_rw differs)
    or the template typed an entry differently.
    Preserves entry order, dedupes names in the string table, and pads
    the output to a 4-byte boundary."""
    count, _hdr_dw = struct.unpack_from('<II', body, 0)
    kept = []
    for i in range(count):
        off = 8 + i * 24
        name_off, sem_idx, sys_val, comp_ty, reg, mask_rw = \
            struct.unpack_from('<IIIIII', body, off)
        if reg not in keep_regs and sys_val == 0:
            continue
        original_reg = reg
        if remap is not None and reg in remap:
            reg = remap[reg]
        if overrides is not None and original_reg in overrides:
            o_sv, o_ct, o_mr = overrides[original_reg]
            sys_val = o_sv
            comp_ty = o_ct
            mask_rw = o_mr
        end = body.find(b'\x00', name_off)
        name = bytes(body[name_off:end])
        kept.append((name, sem_idx, sys_val, comp_ty, reg, mask_rw))

    # fxc emits signature entries sorted by (final) register number so
    # the engine can index them by slot. slangc orders them by the
    # original struct layout, which produces the same set but a
    # different sequence once we remap — sort here so the wire bytes
    # match shipped.
    kept.sort(key=lambda e: e[4])

    new_count = len(kept)
    entries_end = 8 + new_count * 24
    str_table = bytearray()
    str_offs = {}
    for name, *_ in kept:
        if name in str_offs:
            continue
        str_offs[name] = entries_end + len(str_table)
        str_table += name + b'\x00'

    out = bytearray()
    out += struct.pack('<II', new_count, 8)
    for name, sem_idx, sys_val, comp_ty, reg, mask_rw in kept:
        out += struct.pack('<IIIIII',
                           str_offs[name], sem_idx, sys_val,
                           comp_ty, reg, mask_rw)
    out += str_table
    # fxc pads the trailing string table to a 4-byte boundary with
    # 0xAB filler rather than 0x00. The engine validates that shape,
    # so rebuild with the same filler for byte-exact parity with shipped.
    while len(out) % 4:
        out.append(0xAB)
    return bytes(out)


def _renumber_shex_input_regs(shex_body, remap):
    """Rewrite SHEX input-file (v<N>) register immediates according to
    ``remap`` (dict old_reg → new_reg).

    SM5 operand tokens encode the operand file type at bits 12..19, index
    dimension at bits 20..21, and the representation of index 0 at bits
    22..24. For input operands indexed 1D via a u32 immediate, those
    three fields form the discriminating 13-bit pattern ``0x101000``
    (type=INPUT=1, dim=1D, rep0=IMMEDIATE32=0). Bits 4..11 carry the
    per-reference mask/swizzle, which differ between ``dcl_input_ps``
    sites (mask mode) and body reads (swizzle mode) but don't affect
    that pattern.

    The immediately-following u32 is the register number we remap.
    ``dcl_input_ps_sgv`` instructions have the same operand shape; their
    trailing system-value tag is just another u32 that we leave alone.

    Scanning is token-by-token; when we identify an input operand at
    position N we advance to N+2 so the immediate can't double-match as
    another operand.
    """
    OP_PATTERN_MASK = 0x00FFF000   # bits 12..23 (file + dim + rep0 low)
    OP_PATTERN_VAL  = 0x00101000   # INPUT (1) | 1D dim (bit 20)
    buf = bytearray(shex_body)
    i = 8
    while i + 8 <= len(buf):
        tok, = struct.unpack_from('<I', buf, i)
        if (tok & OP_PATTERN_MASK) == OP_PATTERN_VAL and (tok & 0x80000000) == 0:
            reg, = struct.unpack_from('<I', buf, i + 4)
            if reg in remap and remap[reg] != reg:
                struct.pack_into('<I', buf, i + 4, remap[reg])
            i += 8
            continue
        i += 4
    return bytes(buf)


def _parse_isgn_entries(dxbc):
    """Return the full list of ``(name, sem_idx, sys_val, comp_ty, reg,
    mask_rw)`` tuples for each ISGN entry in ``dxbc``. Names are
    uppercased so template-vs-compiled comparisons don't fail on
    ``SV_Position`` vs ``SV_POSITION``."""
    for fc, off, size in dxbc_chunks(dxbc):
        if fc != b'ISGN':
            continue
        body = dxbc[off + 8:off + 8 + size]
        cnt, = struct.unpack_from('<I', body, 0)
        out = []
        for k in range(cnt):
            eo = 8 + k * 24
            name_off, sem_idx, sys_val, comp_ty, reg, mask_rw = \
                struct.unpack_from('<IIIIII', body, eo)
            end = body.index(b'\x00', name_off)
            out.append((bytes(body[name_off:end]).upper(), sem_idx,
                        sys_val, comp_ty, reg, mask_rw))
        return out
    return []


def strip_unused_input_signature(dxbc, template_dxbc=None):
    """Align a compiled shader's ISGN and input-register numbering with
    the shipped template.

    Background: slangc's HLSL emits the full input struct even when the
    specialised entry point reads only a subset. fxc then dead-strips
    the unused ``dcl_input`` tokens in SHEX but leaves the corresponding
    ISGN entries in place and keeps the original v<N> register numbers
    for the kept inputs — so e.g. the HD pixel shader's SV_IsFrontFace
    ends up on ``v9`` while v6..v8 are gaps. The shipped game shader
    has these registers packed contiguously (SV_IsFrontFace at ``v6``
    when TEXCOORD4..6 aren't declared) and the engine rejects any
    layout that doesn't match the shipped ISGN.

    When ``template_dxbc`` is provided we use its ISGN as the
    authoritative set of entries + register numbers — an entry is kept
    whenever the template has a matching (semantic, semantic-index),
    regardless of whether the specialised body happens to read it. The
    remap is derived from the template's register assignments so the
    output matches shipped byte-for-byte. Without a template we fall
    back to the legacy heuristic (keep only registers the SHEX actually
    declares, renumber contiguous from 0), which works when the compiled
    body's declarations match shipped but over-prunes otherwise.
    """
    if dxbc[:4] != b'DXBC':
        return dxbc
    chunks = list(dxbc_chunks(dxbc))
    if not any(fc == b'SHEX' for fc, _, _ in chunks):
        return dxbc

    shex_off = next(off for fc, off, _ in chunks if fc == b'SHEX')
    shex_sz, = struct.unpack_from('<I', dxbc, shex_off + 4)
    shex_body = dxbc[shex_off + 8:shex_off + 8 + shex_sz]
    used_in = _shex_declared_input_regs(shex_body)

    # Template-derived overrides: for each of my ISGN entries that has a
    # matching (semantic, semantic-index) in the template, replace the
    # tuple's reg, mask_rw (and comp_ty / sys_val) with the template's.
    # Empty dict if no template is passed in.
    template_overrides = {}
    if template_dxbc is not None:
        # Key by (name, sem_idx) so we can pull the template's metadata
        # regardless of the slangc-side numbering.
        tmpl_map = {(n, si): (sv, ct, r, mr)
                    for (n, si, sv, ct, r, mr) in _parse_isgn_entries(template_dxbc)}
        my_isgn = _parse_isgn_entries(dxbc)
        remap = {}
        keep_regs = set()
        for name, sem_idx, sys_val, comp_ty, reg, mask_rw in my_isgn:
            if (name, sem_idx) in tmpl_map:
                t_sv, t_ct, t_reg, t_mr = tmpl_map[(name, sem_idx)]
                remap[reg] = t_reg
                keep_regs.add(reg)
                template_overrides[reg] = (t_sv, t_ct, t_mr)
            elif sys_val != 0 or reg in used_in:
                # Template-less sys-val entries (shouldn't occur with a
                # correct template) and body-referenced registers are
                # kept in-place to avoid renumbering holes.
                remap[reg] = reg
                keep_regs.add(reg)
    else:
        # Legacy fallback: keep only what the body actually uses, then
        # pack contiguous from 0.
        keep_regs = set()
        for fc, off, size in chunks:
            if fc != b'ISGN':
                continue
            body = dxbc[off + 8:off + 8 + size]
            cnt, = struct.unpack_from('<I', body, 0)
            for k in range(cnt):
                eo = 8 + k * 24
                _, _, sys_val, _, reg, _ = struct.unpack_from('<IIIIII', body, eo)
                if reg in used_in or sys_val != 0:
                    keep_regs.add(reg)
        sorted_regs = sorted(keep_regs)
        remap = {old: new for new, old in enumerate(sorted_regs)}

    rebuilt = []
    changed = False
    for fc, off, size in chunks:
        raw = dxbc[off:off + 8 + size]
        if fc == b'ISGN':
            new_body = _rewrite_sgn_body(raw[8:], keep_regs, remap=remap,
                                         overrides=template_overrides)
            if len(new_body) != size or new_body != raw[8:]:
                raw = fc + struct.pack('<I', len(new_body)) + new_body
                changed = True
        elif fc == b'SHEX':
            new_body = _renumber_shex_input_regs(raw[8:], remap)
            if new_body != raw[8:]:
                raw = fc + struct.pack('<I', len(new_body)) + new_body
                changed = True
        rebuilt.append(raw)

    if not changed:
        return dxbc

    cnt = len(rebuilt)
    header_size = 32 + cnt * 4
    body = bytearray()
    offsets = []
    cursor = header_size
    for raw in rebuilt:
        offsets.append(cursor)
        body += raw
        cursor += len(raw)
    total = header_size + len(body)

    out = bytearray(total)
    out[0:4] = b'DXBC'
    out[4:20] = b'\x00' * 16
    out[20:24] = struct.pack('<I', 1)
    out[24:28] = struct.pack('<I', total)
    out[28:32] = struct.pack('<I', cnt)
    for i, o in enumerate(offsets):
        struct.pack_into('<I', out, 32 + i * 4, o)
    out[header_size:] = body
    out[4:20] = dxbc_hash(bytes(out[20:]))
    return bytes(out)


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
    dxbcs = []
    prev = 0
    for i, end in enumerate(cum):
        size = end - prev
        if size == 0:
            middle_chunks.append(None)
            stages.append(None)
            dxbcs.append(None)
        else:
            start = off_data + prev
            middle_chunks.append(bytes(data[start + 0x1C:start + 0x48]))
            stages.append(struct.unpack_from('<I', data, start + 0x18)[0])
            dxbc_size = struct.unpack_from('<I', data, start + 0x48)[0]
            dxbc_start = start + PERM_INNER_HEADER_SIZE
            dxbcs.append(bytes(data[dxbc_start:dxbc_start + dxbc_size]))
        prev = end

    return {
        'num_perms': num_perms,
        'middle_chunks': middle_chunks,
        'stages': stages,
        'dxbcs': dxbcs,
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

    perm_blobs = []
    skipped_null = 0

    for i in range(num_perms):
        if tmpl['middle_chunks'][i] is None:
            perm_blobs.append(b'')          # null slot — contributes 0 bytes
            skipped_null += 1
            continue

        # Filename format must match compile_all_slang.py's `f"perm_{i:03d}.{ext}"`
        # — 3-digit minimum with natural expansion past 999.
        dxbc_path = os.path.join(slang_dir, f'perm_{i:03d}.dxbc')
        with open(dxbc_path, 'rb') as fp:
            dxbc = fp.read()

        dxbc = fix_dxbc_signatures(dxbc)
        dxbc = strip_unused_input_signature(dxbc, tmpl['dxbcs'][i])
        if strip:
            dxbc = strip_dxbc_chunks(dxbc, {b'RDEF', b'STAT'})
        else:
            dxbc = bytearray(dxbc)
            dxbc[4:20] = dxbc_hash(bytes(dxbc[20:]))
            dxbc = bytes(dxbc)

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


def pack_blob_perm(blob):
    """Serialize one blob-style perm (Metal MTLB or extra-backend blob).

    Wire format — shared by Metal v1.8 (§3.5) and the extra-backend BLS
    variants for opengl / vulkan / webgpu (§3.6):

        20 zero || payload_size || stage=1 || entry_cnt=1
                || blob_size || 8 || 1 || blob || 0x00

    Unlike DX perms there is no opaque per-perm metadata chunk to preserve
    from the shipped templates — the only per-perm wrapper is the 44-byte
    inner header itself, so the same packer works for any opaque blob
    payload.
    """
    blob_size    = len(blob)
    payload_size = 0x14 + blob_size                 # bytes from +0x18 to end-of-blob
    perm_size    = METAL_PERM_INNER_HEADER_SIZE + blob_size + 1  # +1 trailing 0x00
    buf = bytearray(perm_size)
    # bytes [0..0x14) stay zero (pre_meta)
    struct.pack_into('<I', buf, 0x14, payload_size)
    struct.pack_into('<I', buf, 0x18, BLS_METAL_STAGE)
    struct.pack_into('<I', buf, 0x1C, 1)             # entry_count
    struct.pack_into('<I', buf, 0x20, blob_size)
    struct.pack_into('<I', buf, 0x24, 8)
    struct.pack_into('<I', buf, 0x28, 1)
    buf[METAL_PERM_INNER_HEADER_SIZE:METAL_PERM_INNER_HEADER_SIZE + blob_size] = blob
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

    perm_blobs = []
    skipped_null = 0

    for i in range(num_perms):
        # Filename format must match compile_all_slang.py — see build_bls().
        metallib_path = os.path.join(slang_dir, f'perm_{i:03d}.metallib')
        if nulls[i] or not os.path.isfile(metallib_path) \
                or os.path.getsize(metallib_path) == 0:
            perm_blobs.append(b'')
            skipped_null += 1
            continue

        with open(metallib_path, 'rb') as fp:
            metallib = fp.read()
        perm_blobs.append(pack_blob_perm(metallib))

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
    return _has_blobs(slang_dir, '.metallib')


def _has_blobs(slang_dir, suffix):
    """True if `slang_dir` contains at least one non-empty file with `suffix`."""
    if not os.path.isdir(slang_dir):
        return False
    for name in os.listdir(slang_dir):
        if name.endswith(suffix):
            p = os.path.join(slang_dir, name)
            if os.path.isfile(p) and os.path.getsize(p) > 0:
                return True
    return False


# ============================================================
# Extra-backend BLS rebuild — opengl / vulkan / webgpu
# ============================================================
# Wc3 itself never loads these formats; they are emitted on request via
# `--build-extra` so users porting the engine to a non-DX/non-Metal target
# have a ready-made BLS bundle for each shader family. The wire format is
# identical to Metal v1.8 (§3.5 in the BLS spec) — uncompressed v1.8 outer
# header + cumulative offset table, 44-byte per-perm inner header, opaque
# blob, single trailing 0x00 byte. The blob payload depends on the backend:
# raw GLSL / WGSL source text or a SPIR-V binary module.

def build_extra_bls(slang_dir, ext, num_perms, nulls=None, verbose=False):
    """Return the bytes of a rebuilt extra-backend BLS (opengl/vulkan/webgpu).

    `slang_dir` is the per-family slang_out subdirectory (e.g.
    ``slang_out/opengl/hd_vs``); `ext` is the file extension that
    compile_all_slang.py emits for that backend (``glsl``, ``spv``, or
    ``wgsl``). `nulls` is an optional list[bool] marking which perm slots
    should be left empty — typically derived from the DX template so the
    extra bundles mirror the shipped null-perm pattern. When `nulls` is
    None, any missing or zero-byte file is treated as a null perm.
    """
    if nulls is None:
        nulls = [False] * num_perms

    perm_blobs = []
    skipped_null = 0

    for i in range(num_perms):
        # Filename format must match compile_all_slang.py — see build_bls().
        blob_path = os.path.join(slang_dir, f'perm_{i:03d}.{ext}')
        if nulls[i] or not os.path.isfile(blob_path) \
                or os.path.getsize(blob_path) == 0:
            perm_blobs.append(b'')
            skipped_null += 1
            continue

        with open(blob_path, 'rb') as fp:
            blob = fp.read()
        perm_blobs.append(pack_blob_perm(blob))

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
        print(f'  [.{ext}] {num_perms} perms ({skipped_null} null), '
              f'file size {len(file_buf):#x}')
    return bytes(file_buf)


def extra_dir_for(stage, vs_dir, ps_dir):
    """Pick the ``glslvs/glslps``-style output subdirectory for a given stage."""
    return vs_dir if stage == 'vs' else ps_dir


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
    ap.add_argument('--family', action='append', choices=list(FAMILIES),
                    help='limit to specific family (default: all)')
    ap.add_argument('--strip', action='store_true',
                    help='strip RDEF/STAT chunks from DXBC (match shipped chunk layout) '
                         'and recompute the DXBC hash')
    ap.add_argument('--build_extra', '--build-extra', action='store_true',
                    help='also build BLS bundles for the non-shipped backends '
                         '(opengl -> glslvs/glslps, vulkan -> spvvs/spvps, '
                         'webgpu -> wgpuvs/wgpups). Reads per-family blobs from '
                         '<slang-out>/{opengl,vulkan,webgpu}/<family>/. The DX '
                         'template (when present) supplies the null-perm pattern '
                         'so the extra bundles line up with the shipped layout.')
    ap.add_argument('--verbose', '-v', action='store_true')
    args = ap.parse_args()

    family_names = args.family or list(FAMILIES)
    os.makedirs(args.output, exist_ok=True)

    dxbc_root  = os.path.join(args.slang_out, DXBC_TARGET_SUBDIR)
    metal_root = os.path.join(args.slang_out, METAL_TARGET_SUBDIR)

    for fam in family_names:
        cfg = FAMILIES[fam]
        num_perms = cfg.perm_count
        # `toon_hd_*` ships no dedicated BLS — effective_template falls
        # back to the HD template for its resource/binding metadata while
        # the rebuilt file is still written under the family's own name.
        template_name = cfg.effective_template

        # ---------- DX (ps/, vs/) ----------
        template = os.path.join(args.templates, cfg.dx_dir, template_name)
        slang_dir = os.path.join(dxbc_root, fam)

        # Track the DX template's null-perm pattern so the extra-backend
        # passes below can mirror it. Falls back to "all live" when the
        # DX template is missing.
        dx_nulls = None

        if not os.path.isfile(template):
            print(f'SKIP {fam}: template missing ({template})', file=sys.stderr)
        elif not os.path.isdir(slang_dir):
            print(f'SKIP {fam}: slang dir missing ({slang_dir}). '
                  f'Run compile_all_slang.py --target d3d11 first.', file=sys.stderr)
        else:
            out_dir = os.path.join(args.output, cfg.dx_dir)
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, cfg.bls_name)
            try:
                blob = build_bls(template, slang_dir, num_perms,
                                 strip=args.strip, verbose=args.verbose)
                with open(out_path, 'wb') as fp:
                    fp.write(blob)
                print(f'wrote {out_path} ({len(blob):#x} bytes, {num_perms} perms)')
            except Exception as e:
                print(f'FAIL {fam}: {e}', file=sys.stderr)

        # Read the DX template once for its null pattern (cheap; same
        # template was just consumed by build_bls). Used by both the
        # Metal and extra-backend passes.
        if os.path.isfile(template):
            try:
                tmpl = read_template(template)
                dx_nulls = [mc is None for mc in tmpl['middle_chunks']]
            except Exception:
                dx_nulls = None

        # ---------- Metal (mtlfs/, mtlvs/) — only if metallibs were emitted ----
        m_slang_dir = os.path.join(metal_root, fam)
        if has_metallibs(m_slang_dir):
            m_template = os.path.join(args.templates, cfg.metal_dir, template_name)
            m_out_dir  = os.path.join(args.output, cfg.metal_dir)
            os.makedirs(m_out_dir, exist_ok=True)
            m_out_path = os.path.join(m_out_dir, cfg.bls_name)
            try:
                blob = build_metal_bls(m_template, m_slang_dir, num_perms,
                                       verbose=args.verbose)
                with open(m_out_path, 'wb') as fp:
                    fp.write(blob)
                print(f'wrote {m_out_path} ({len(blob):#x} bytes, {num_perms} perms)')
            except Exception as e:
                print(f'FAIL {fam} [metal]: {e}', file=sys.stderr)

        # ---------- Extra backends (opengl/vulkan/webgpu) — opt-in ---------
        if not args.build_extra:
            continue

        for backend, (target_subdir, ext, vs_outdir, ps_outdir) in EXTRA_BACKENDS.items():
            x_slang_dir = os.path.join(args.slang_out, target_subdir, fam)
            if not _has_blobs(x_slang_dir, '.' + ext):
                # Slang didn't produce blobs for this backend — silently
                # skip rather than error, so partial slangc runs still
                # build whatever did succeed.
                continue

            x_out_subdir = extra_dir_for(cfg.stage, vs_outdir, ps_outdir)
            x_out_dir    = os.path.join(args.output, x_out_subdir)
            os.makedirs(x_out_dir, exist_ok=True)
            x_out_path   = os.path.join(x_out_dir, cfg.bls_name)
            try:
                blob = build_extra_bls(x_slang_dir, ext, num_perms,
                                       nulls=dx_nulls, verbose=args.verbose)
                with open(x_out_path, 'wb') as fp:
                    fp.write(blob)
                print(f'wrote {x_out_path} ({len(blob):#x} bytes, {num_perms} perms)')
            except Exception as e:
                print(f'FAIL {fam} [{backend}]: {e}', file=sys.stderr)


if __name__ == '__main__':
    sys.exit(main())
