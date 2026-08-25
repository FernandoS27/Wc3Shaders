"""Heroes of the Storm shader-permutation key decoder.

The HotS (and SC2) shader cache stores one record per compiled permutation,
keyed by a bit-packed "permutation vector". This module reproduces the engine's
packing exactly; the layout was reverse-engineered from the macOS Heroes client
(base54339) -- see docs/HOTS_PERM_KEY_FORMAT.md.

Key layout (identical in SC2 and HotS; only the schema contents differ):

    byte  0     EShaderVersion of the vertex stage (3 = vs_3_0, 4 = vs_4_0)
    byte  1     EShaderVersion of the pixel stage  (11 = ps_3_0, 12 = ps_4_0)
    bytes 2-5   u32 permutation/quality flags  (section+748)
    byte  6     section id == family id
    bytes 7-10  u32 schema hash (PermSchema_ComputeSchemaHash)
    -- axis payload starts at bit 88 --
    root family section's own axes, forward-packed from bit 88
    (root block padded to a whole byte)
    then every child section's block, in link order, each byte-aligned
    then an 8-byte device-specific tail

An axis of width w at bit position p is read as

    v = ((key[p>>3] & maskHi) << 8 | (key[(p>>3)+1] & maskLo)) >> (16 - (p&7) - w)

with mask16 = (((1 << w) - 1) << (16 - w)) >> (p & 7), and

    w = max_value.bit_length()

i.e. the *declared* bit count in the source is not the packed width -- the
engine narrows every axis to exactly the bits its maximum legal value needs.

`decode_key()` from tools/sc2_cache.py returns a vector with two extra leading
bytes, so vec[i] == key[i - 2]; use `key_from_vec()` to strip them.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SCHEMA_PATH = os.path.join(HERE, "hots", "hots_perm_schema.json")

HEADER_BITS = 88          # PermSchema_FinalizeLayout: root sections start here
TAIL_BYTES = 8            # written by the post-build hook at vtable+1904
VEC_PREFIX = 2            # sc2_cache.decode_key emits two leading bytes

STAGE_VS, STAGE_PS = 1, 2


def load_schema(path=SCHEMA_PATH):
    return json.load(open(path))


def axis_width(axis):
    """Packed width: the engine stores bit_length(max), not the declared bits."""
    return int(axis["max"]).bit_length()


def section_bits(schema, name):
    return sum(axis_width(a) for a in schema["sections"][name]["axes"])


def section_bytes(schema, name):
    return (section_bits(schema, name) + 7) // 8


def _place(axes, start_bit):
    """Yield (axis, byte_off, mask_hi, mask_lo, shift) forward-packed from start_bit."""
    p = start_bit
    for a in axes:
        w = axis_width(a)
        m16 = ((((1 << w) - 1) << (16 - w)) & 0xFFFF) >> (p & 7)
        yield a, p >> 3, (m16 >> 8) & 0xFF, m16 & 0xFF, 16 - (p & 7) - w
        p += w


def build_layout(schema, family):
    """Return (entries, key_len). Each entry is a dict with absolute byte offset."""
    fam = schema["families"][family]
    entries = []
    root_axes = schema["sections"][family]["axes"]
    bit = HEADER_BITS
    for a, off, mh, ml, sh in _place(root_axes, bit):
        entries.append(dict(section=family, name=a["name"], stage=a["stage"],
                            max=a["max"], off=off, mask_hi=mh, mask_lo=ml, shift=sh))
    bit += section_bits(schema, family)
    base = (bit + 7) // 8                       # root block size, byte-aligned
    for child in fam["children"]:
        for a, off, mh, ml, sh in _place(schema["sections"][child]["axes"], 0):
            entries.append(dict(section=child, name=a["name"], stage=a["stage"],
                                max=a["max"], off=base + off, mask_hi=mh,
                                mask_lo=ml, shift=sh))
        base += section_bytes(schema, child)
    return entries, base + TAIL_BYTES


def key_len(schema, family):
    return build_layout(schema, family)[1]


def read_axis(key, e):
    hi = key[e["off"]] if e["off"] < len(key) else 0
    lo = key[e["off"] + 1] if e["off"] + 1 < len(key) else 0
    return (((hi & e["mask_hi"]) << 8) | (lo & e["mask_lo"])) >> e["shift"]


def key_from_vec(vec):
    return bytes(vec[VEC_PREFIX:])


def decode(schema, key, family, stage=None):
    """Decode one key into {axis_name: value}. `stage` filters to STAGE_VS/STAGE_PS."""
    entries, _ = build_layout(schema, family)
    out = {}
    for e in entries:
        if stage is not None and not (e["stage"] & stage):
            continue
        name = e["name"]
        if name in out:                          # indexed axis (b_iUVMapping0..7)
            i = 1
            while f"{name}{i}" in out:
                i += 1
            name = f"{name}{i}"
        out[name] = read_axis(key, e)
    return out


def header(key):
    return dict(vs_version=key[0], ps_version=key[1],
                flags=int.from_bytes(key[2:6], "little"),
                section_id=key[6],
                schema_hash=int.from_bytes(key[7:11], "little"))
