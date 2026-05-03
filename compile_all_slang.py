"""Compile all permutations of every shader entry point from the unified
wc3_shaders Slang module, to one of several graphics-API targets.

Runs independently of the current working directory — paths resolve
relative to this script's location.

    --target {d3d11,d3d12,vulkan,opengl,metal,webgpu,all}  (default: d3d11)
    --family {hd_vs,hd_ps,toon_hd_vs,toon_hd_ps,crystal_ps,
              sd_on_hd_vs,sd_on_hd_ps,sd_highspec_vs,sd_classic_ps,
              water_vs,water_ps,tonemap_ps,all}
    --slangc PATH   explicit slangc.exe override
"""

import argparse
import concurrent.futures
import glob
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

from shader_config import load_families

REPO_ROOT = Path(__file__).resolve().parent
SHADER = REPO_ROOT / "wc3_shaders" / "wc3_shaders.slang"
CUSTOM_SHADER = REPO_ROOT / "custom_shaders" / "custom_shaders.slang"
WC3_INCLUDE_DIR = REPO_ROOT / "wc3_shaders"
OUT_BASE = REPO_ROOT / "slang_out"

# Family metadata — stage, entry point, perm_count, which source module
# hosts the entry point — comes from wc3_shaders.json via shader_config. The
# module="custom" families compile from CUSTOM_SHADER with WC3_INCLUDE_DIR
# on the include path so `import wc3_shaders;` resolves.
FAMILY_CONFIGS = load_families()
FAMILIES = tuple(FAMILY_CONFIGS.keys())

# Stage → profile per target + output file extension and optional extra
# slangc args. The metal entry can be switched to metallib (compiled Metal
# bytecode) by `--metallib VERSION`; see main(). metallib generation
# requires Apple's `metal` downstream compiler (Xcode toolchain) and will
# only succeed on macOS.
#
# - d3d11 / d3d12 use the native HLSL profile strings.
# - vulkan (SPIR-V) / opengl (GLSL) use glsl_450.
# - metal and webgpu accept the Slang-universal sm_6_0 profile.
TARGETS = {
    "d3d11":  {"target": "dxbc",  "ext": "dxbc",  "vs": "vs_5_0",   "ps": "ps_5_0",   "extra": []},
    "d3d12":  {"target": "dxil",  "ext": "dxil",  "vs": "vs_6_0",   "ps": "ps_6_0",   "extra": []},
    "vulkan": {"target": "spirv", "ext": "spv",   "vs": "glsl_450", "ps": "glsl_450", "extra": []},
    "opengl": {"target": "glsl",  "ext": "glsl",  "vs": "glsl_450", "ps": "glsl_450", "extra": []},
    "metal":  {"target": "metal", "ext": "metal", "vs": "sm_6_0",   "ps": "sm_6_0",   "extra": []},
    "webgpu": {"target": "wgsl",  "ext": "wgsl",  "vs": "sm_6_0",   "ps": "sm_6_0",   "extra": []},
}

_slangc_path: Optional[str] = None


def resolve_slangc(override: Optional[str] = None) -> str:
    """Locate slangc.exe. Caches the result after the first successful call."""
    global _slangc_path
    if _slangc_path is not None:
        return _slangc_path

    exe = "slangc.exe" if os.name == "nt" else "slangc"

    # Explicit override (CLI flag or SLANGC env var) wins.
    for cand in (override, os.environ.get("SLANGC")):
        if cand and Path(cand).is_file():
            _slangc_path = cand
            return _slangc_path

    # Standard PATH lookup.
    found = shutil.which("slangc")
    if found:
        _slangc_path = found
        return _slangc_path

    # Vulkan SDK installer sets VULKAN_SDK to the active install.
    vulkan_sdk = os.environ.get("VULKAN_SDK")
    if vulkan_sdk:
        candidate = Path(vulkan_sdk) / "Bin" / exe
        if candidate.is_file():
            _slangc_path = str(candidate)
            return _slangc_path

    # Fallback: scan the default Windows install root for the newest SDK.
    if os.name == "nt":
        matches = sorted(glob.glob(r"C:\VulkanSDK\*\Bin\slangc.exe"))
        if matches:
            _slangc_path = matches[-1]
            return _slangc_path

    raise SystemExit(
        "slangc not found. Install the Vulkan SDK, put slangc on PATH, "
        "or pass --slangc / set SLANGC to its full path."
    )


@dataclass
class PermSpec:
    entry: str
    types: List[str]
    label: str


@dataclass
class SweepResult:
    family: str
    count: int
    ok: int = 0
    fail: int = 0
    fail_list: List[str] = field(default_factory=list)


def invoke_slangc(entry: str, target: str, profile: str,
                  specialize: List[str], out_path: Path,
                  shader_path: Path,
                  extra: Optional[List[str]] = None,
                  include_dirs: Optional[List[Path]] = None) -> bool:
    args = [resolve_slangc(), "-entry", entry]
    for t in specialize:
        args += ["-specialize", t]
    for inc in include_dirs or []:
        args += ["-I", str(inc)]
    args += [
        "-profile", profile,
        "-target", target,
        "-o", str(out_path),
        "-warnings-disable", "39001",
    ]
    if extra:
        args += extra
    args.append(str(shader_path))
    subprocess.run(args, capture_output=True, text=True)
    return out_path.exists() and out_path.stat().st_size > 0


def run_sweep(family: str, count: int, mapper: Callable[[int], PermSpec],
              stage: str, target_key: str, jobs: int = 1) -> SweepResult:
    cfg = TARGETS[target_key]
    print()
    print(f"========== [{target_key}] {family} ({count} perms, jobs={jobs}) ==========")
    out_dir = OUT_BASE / target_key / family
    out_dir.mkdir(parents=True, exist_ok=True)

    result = SweepResult(family=family, count=count)
    ext = cfg["ext"]
    target = cfg["target"]
    profile = cfg[stage]
    extra = cfg.get("extra", [])

    # The shipped SD classic pixel shader is compiled to SM4 / SM2 (SHDR
    # + Aon9 chunks) for D3D9 compatibility — every other shipped shader
    # uses SM5 (SHEX). The Wc3 engine binds the SD classic perm via its
    # legacy pipeline and rejects (or mis-binds) SM5 bytecode there, so
    # we override the D3D11 PS profile to ps_4_0 for this family. SM4 is
    # a strict subset of SM5 for the operations the SD classic PS uses
    # (one or two `Sample`s, fixed-function blend, optional fog +
    # `discard`), so the only behaviour change is the chunk type.
    if family == "sd_classic_ps" and target_key == "d3d11":
        profile = "ps_4_0"

    # Custom-shader families compile from their own module file with
    # wc3_shaders on the include path so `import wc3_shaders;` resolves.
    # The explicit -stage flag keeps slangc from trying to validate
    # wc3_shaders' own entry points (vs_main, ps_main, …) against the
    # current profile while it's being imported — without it the
    # import pipeline emits all entry points in the module.
    if FAMILY_CONFIGS[family].module == "custom":
        shader_path = CUSTOM_SHADER
        include_dirs = [WC3_INCLUDE_DIR]
        custom_extra = extra + [
            "-stage", "fragment" if stage == "ps" else "vertex",
            # Suppress slangc's attempt to validate wc3_shaders' own
            # entry points (hd_vs, hd_ps, …) against our profile while
            # it's being imported — they're re-scanned for capability
            # checks by default and error because the profile only
            # matches one stage.
            "-ignore-capabilities",
        ]
    else:
        shader_path = SHADER
        include_dirs = None
        custom_extra = extra

    # Build the per-perm work list once so the executor only has to dispatch.
    perm_specs = [(i, mapper(i), out_dir / f"perm_{i:03d}.{ext}")
                  for i in range(count)]

    def compile_one(item):
        i, spec, out_path = item
        ok = invoke_slangc(spec.entry, target, profile, spec.types, out_path,
                           shader_path, custom_extra, include_dirs)
        return i, spec, ok

    # Each slangc invocation is a long-running subprocess, so a thread
    # pool parallelises well — the GIL is released while we wait on the
    # child process. Use sequential dispatch when jobs<=1 to keep stack
    # traces tidy on single-thread runs.
    if jobs <= 1:
        outcomes = [compile_one(item) for item in perm_specs]
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
            outcomes = list(pool.map(compile_one, perm_specs))

    for i, spec, ok in outcomes:
        if ok:
            result.ok += 1
        else:
            result.fail += 1
            result.fail_list.append(f"perm_{i} ({spec.label})")

    # Keep the fail list in perm-index order so it's stable across runs.
    result.fail_list.sort()

    print()
    print(f"Compile: {result.ok} OK / {result.fail} fail")
    if result.fail_list:
        print("--- First 5 compile failures ---")
        for entry in result.fail_list[:5]:
            print(f"  {entry}")
    return result


def map_hd_vs(idx: int) -> PermSpec:
    tang     = idx % 2
    weight   = (idx // 2) % 3
    color    = (idx // 6) % 2
    texcoord = (idx // 12) % 3
    prepass  = (idx // 36) % 2
    shadows  = (idx // 72) % 2

    skin  = "FourBoneSkinning" if weight == 2 else "Rigid"
    hasT  = "true" if tang == 1 else "false"
    hasC  = "true" if color == 1 else "false"
    hasU0 = "true" if texcoord >= 1 else "false"
    hasU1 = "true" if texcoord >= 2 else "false"
    shad  = "true" if (shadows == 1 and prepass == 0) else "false"

    return PermSpec(
        entry="vs_main",
        types=[skin, f"VertexFormat<{hasT},{hasC},{hasU0},{hasU1}>", shad],
        label=f"{skin}+T={hasT}+C={hasC}+UV={texcoord}+SH={shad}",
    )


def _hd_ps_bits(idx: int):
    """Shared 9-bit feature encoding for hd_ps and crystal_ps. Bit 2 /
    EXTRA_VERTS toggles the shadow cascade inputs the VS writes on
    TEXCOORD4-6; the HD IBL PS consumes them for the 3-cascade PCF
    shadow lookup."""
    return {
        "mrt":  bool(idx & 1),
        "dp":   bool(idx & 2),
        "ev":   bool(idx & 4),
        "ibl":  bool(idx & 8),
        "fogL": bool(idx & 16),
        "fogE": bool(idx & 32),
        "at":   bool(idx & 64),
        "ml":   bool(idx & 128),
        "dbg":  bool(idx & 256),
    }


def _hd_ps_types(b):
    if b["fogL"] and b["fogE"]:
        fog = "FogExp2"
    elif b["fogL"]:
        fog = "FogLinear"
    elif b["fogE"]:
        fog = "FogExponential"
    else:
        fog = "FogNone"
    alpha = "AlphaTestOn" if b["at"] else "AlphaTestOff"
    mat   = "MultiLayerMaterial" if b["ml"] else "StandardMaterial"
    iblS  = "true" if b["ibl"] else "false"
    evS   = "true" if b["ev"] else "false"
    dpS   = "true" if b["dp"] else "false"
    mrtS  = "true" if b["mrt"] else "false"
    dbgS  = "true" if b["dbg"] else "false"
    return fog, alpha, mat, iblS, evS, dpS, mrtS, dbgS


def map_hd_ps(idx: int) -> PermSpec:
    b = _hd_ps_bits(idx)
    fog, alpha, mat, iblS, evS, dpS, mrtS, dbgS = _hd_ps_types(b)
    return PermSpec(
        entry="ps_main",
        types=[fog, alpha, mat, iblS, evS, dpS, mrtS, dbgS],
        label=f"{fog}+{alpha}+{mat}+IBL={iblS}+EV={evS}+DP={dpS}+MRT={mrtS}+DBG={dbgS}",
    )


def map_toon_hd_vs(idx: int) -> PermSpec:
    # Toon-HD shares the HD vertex-format encoding 1:1 — same 144 perms,
    # same specialisation types, different entry point.
    spec = map_hd_vs(idx)
    return PermSpec(entry="toon_vs_main", types=spec.types, label=spec.label)


def map_toon_hd_ps(idx: int) -> PermSpec:
    # Toon-HD shares the HD pixel-shader 9-bit feature encoding 1:1 —
    # same 512 perms, same specialisation types, different entry point.
    spec = map_hd_ps(idx)
    return PermSpec(entry="toon_ps_main", types=spec.types, label=spec.label)


def map_crystal_ps(idx: int) -> PermSpec:
    # Crystal shares hd_ps's 9-bit encoding for most features. Bit 2 /
    # EXTRA_VERTS is unused on crystal's PS signature (crystal doesn't
    # expose a shadow-cascade path yet), so we drop it from the
    # specialisation tuple — `crystal_ps_main` keeps the shorter
    # 7-let form.
    b = _hd_ps_bits(idx)
    fog, alpha, mat, iblS, _evS, dpS, mrtS, dbgS = _hd_ps_types(b)
    return PermSpec(
        entry="crystal_ps_main",
        types=[fog, alpha, mat, iblS, dpS, mrtS, dbgS],
        label=f"{fog}+{alpha}+{mat}+IBL={iblS}+DP={dpS}+MRT={mrtS}+DBG={dbgS}",
    )


def map_sd_on_hd_vs(idx: int) -> PermSpec:
    tang     = idx % 2
    weight   = (idx // 2) % 3
    color    = (idx // 6) % 2
    texcoord = (idx // 12) % 3
    prepass  = (idx // 36) % 2
    shadows  = (idx // 72) % 2

    skin  = "SDFourBoneSkinning" if weight == 2 else "Rigid"
    hasT  = "true" if tang == 1 else "false"
    hasC  = "true" if color == 1 else "false"
    hasU0 = "true" if texcoord >= 1 else "false"
    hasU1 = "true" if texcoord >= 2 else "false"
    shad  = "true" if (shadows == 1 and prepass == 0) else "false"

    return PermSpec(
        entry="sd_on_hd_vs_main",
        types=[skin, f"VertexFormat<{hasT},{hasC},{hasU0},{hasU1}>", shad],
        label=f"{skin}+T={hasT}+C={hasC}+UV={texcoord}+SH={shad}",
    )


def map_sd_on_hd_ps(idx: int) -> PermSpec:
    base       = idx & 0x3F
    high_field = idx // 64

    mrt  = bool(base & 1)
    dp   = bool(base & 2)
    ev   = bool(base & 4)
    ibl  = bool(base & 8)
    fogL = bool(base & 16)
    fogE = bool(base & 32)

    if fogL and fogE:
        fog = "FogExp2"
    elif fogL:
        fog = "FogLinear"
    elif fogE:
        fog = "FogExponential"
    else:
        fog = "FogNone"

    at   = high_field in (1, 4)
    srgb = high_field in (2, 5)
    dbg  = high_field in (3, 4, 5)

    alpha = "AlphaTestOn" if at else "AlphaTestOff"
    iblS  = "true" if ibl else "false"
    evS   = "true" if ev else "false"
    dpS   = "true" if dp else "false"
    mrtS  = "true" if mrt else "false"
    srgbS = "true" if srgb else "false"
    dbgS  = "true" if dbg else "false"

    return PermSpec(
        entry="sd_on_hd_ps_main",
        types=[fog, alpha, iblS, evS, dpS, mrtS, srgbS, dbgS],
        label=(f"{fog}+{alpha}+IBL={iblS}+EV={evS}+DP={dpS}"
               f"+MRT={mrtS}+SRGB={srgbS}+DBG={dbgS}"),
    )


def map_sd_highspec_vs(idx: int) -> PermSpec:
    weight = idx % 3
    color  = (idx // 3) % 2
    uv     = (idx // 6) % 3
    lights = (idx // 18) % 9

    skin  = "SDFourBoneSkinning" if weight == 2 else "Rigid"
    hasC  = "true" if color == 1 else "false"
    hasU0 = "true" if uv >= 1 else "false"
    hasU1 = "true" if uv >= 2 else "false"

    return PermSpec(
        entry="sd_highspec_vs_main",
        types=[skin, f"VertexFormat<false,{hasC},{hasU0},{hasU1}>", str(lights)],
        label=f"{skin}+C={hasC}+UV={uv}+NL={lights}",
    )


STAGE_NAMES = [
    "StageDisabled", "StageModulate", "StageLerp",
    "StageModulateRGB", "StageModulate2X",
]


def map_water_vs(idx: int) -> PermSpec:
    # Single permutation — no feature specialisation.
    return PermSpec(entry="water_vs_main", types=[], label="(only)")


def map_water_ps(idx: int) -> PermSpec:
    # 4 permutations: FogNone / FogLinear / FogExponential / FogExp2.
    fogL = bool(idx & 1)
    fogE = bool(idx & 2)
    if fogL and fogE:
        fog = "FogExp2"
    elif fogL:
        fog = "FogLinear"
    elif fogE:
        fog = "FogExponential"
    else:
        fog = "FogNone"
    return PermSpec(entry="water_ps_main", types=[fog], label=fog)


def map_tonemap_ps(idx: int) -> PermSpec:
    # Single permutation — HDR→LDR resolve has no feature axes.
    return PermSpec(entry="tonemap_ps_main", types=[], label="(only)")


def map_sprite_vs(idx: int) -> PermSpec:
    # Single permutation — pre-projected sprite VS has no feature axes.
    return PermSpec(entry="sprite_vs_main", types=[], label="(only)")


def map_sprite_ps(idx: int) -> PermSpec:
    # 4 permutations driven by 2 independent bools:
    #   bit 0 — HAS_SRGB_ENCODE (linear → sRGB write)
    #   bit 1 — HAS_SRGB_DECODE (sRGB    → linear read)
    # Both bits set is a passthrough (the encode/decode pair cancels),
    # matching the engine's perm_003 == perm_000 collapse.
    enc = "true" if (idx & 1) else "false"
    dec = "true" if (idx & 2) else "false"
    return PermSpec(
        entry="sprite_ps_main",
        types=[enc, dec],
        label=f"ENC={enc}+DEC={dec}",
    )


def map_distortion_ps(idx: int) -> PermSpec:
    # Single permutation — full-screen chromatic-aberration pass has no
    # feature axes.
    return PermSpec(entry="distortion_ps_main", types=[], label="(only)")


def map_terrain_vs(idx: int) -> PermSpec:
    # 8 perms = 3 raw bits, but only 4 functionally distinct outputs:
    #   bit 0 — SHADOW_PASS      (caster pass; suppresses cascade output)
    #   bit 1 — RECEIVE_SHADOWS  (emit cascade UVs)
    #   bit 2 — VERTEX_COLOR     (per-vertex RGBA from ATTR2)
    # Effective HAS_SHADOWS = (RECEIVE_SHADOWS && !SHADOW_PASS) — a draw
    # rendering its own caster pass never emits cascade UVs even when
    # the receive flag is also set. perm_001/003/005/007 collapse onto
    # perm_000/000/004/004 respectively.
    shadow_pass = bool(idx & 1)
    receive     = bool(idx & 2)
    vert_color  = bool(idx & 4)
    has_shadows = receive and not shadow_pass

    vc = "true" if vert_color  else "false"
    sh = "true" if has_shadows else "false"
    return PermSpec(
        entry="terrain_vs_main",
        types=[vc, sh],
        label=f"VC={vc}+SH={sh}",
    )


def map_foliage_vs(idx: int) -> PermSpec:
    # 8 perms = 3 raw bits, but only 4 functionally distinct outputs:
    #   bit 0 — SHADOW_PASS    (caster pass; suppresses cascade output)
    #   bit 1 — RECEIVE_SHADOWS
    #   bit 2 — WIND_ANIMATION
    # Effective HAS_SHADOWS = (RECEIVE_SHADOWS && !SHADOW_PASS) — same
    # collapse rule as terrain_vs.
    shadow_pass = bool(idx & 1)
    receive     = bool(idx & 2)
    wind        = bool(idx & 4)
    has_shadows = receive and not shadow_pass

    sh = "true" if has_shadows else "false"
    wd = "true" if wind        else "false"
    return PermSpec(
        entry="foliage_vs_main",
        types=[sh, wd],
        label=f"SH={sh}+WIND={wd}",
    )


def map_foliage_ps(idx: int) -> PermSpec:
    # 128 perms (7 raw bits) — every bit is functionally distinct:
    #   bit 0 (1)   — MRT_OUTPUTS
    #   bit 1 (2)   — NULL_PASS    (overrides everything; 64/128 collapse)
    #   bit 2 (4)   — RECEIVE_SHADOWS
    #   bit 3 (8)   — FOG_LINEAR
    #   bit 4 (16)  — FOG_EXPONENTIAL  (with bit 3 → FogExp2)
    #   bit 5 (32)  — ALPHA_TEST
    #   bit 6 (64)  — TINT_OVERRIDE
    null_pass = bool(idx & 2)
    mrt       = bool(idx & 1)
    shadows   = bool(idx & 4)
    fogL      = bool(idx & 8)
    fogE      = bool(idx & 16)
    at        = bool(idx & 32)
    tint      = bool(idx & 64)

    if fogL and fogE:
        fog = "FogExp2"
    elif fogL:
        fog = "FogLinear"
    elif fogE:
        fog = "FogExponential"
    else:
        fog = "FogNone"

    npS  = "true" if null_pass else "false"
    mrtS = "true" if mrt       else "false"
    shS  = "true" if shadows   else "false"
    atS  = "true" if at        else "false"
    tnS  = "true" if tint      else "false"
    return PermSpec(
        entry="foliage_ps_main",
        types=[npS, mrtS, shS, fog, atS, tnS],
        label=f"NULL={npS}+MRT={mrtS}+SH={shS}+{fog}+AT={atS}+TINT={tnS}",
    )


def map_terrain_ps(idx: int) -> PermSpec:
    # 128 perms (7 raw bits) but only 4 functionally distinct axes:
    #   bit 0 (1)   — MRT_OUTPUTS
    #   bit 1 (2)   — NULL_PASS    (overrides everything; 64/128 collapse)
    #   bit 2 (4)   — RECEIVE_SHADOWS
    #   bit 3 (8)   — unused (reserved)
    #   bit 4 (16)  — unused (reserved)
    #   bit 5 (32)  — unused (reserved)
    #   bit 6 (64)  — TINT_OVERRIDE
    # The reserved bits collapse to identical bytecode under the same
    # functional axes, so the mapper just ignores them — slangc gets the
    # same -specialize tuple for every collapse-equivalent index.
    null_pass = bool(idx & 2)
    mrt       = bool(idx & 1)
    shadows   = bool(idx & 4)
    tint      = bool(idx & 64)

    npS = "true" if null_pass else "false"
    mrtS = "true" if mrt      else "false"
    shS  = "true" if shadows  else "false"
    tnS  = "true" if tint     else "false"
    return PermSpec(
        entry="terrain_ps_main",
        types=[npS, mrtS, shS, tnS],
        label=f"NULL={npS}+MRT={mrtS}+SH={shS}+TINT={tnS}",
    )


def map_popcorn_vs(idx: int) -> PermSpec:
    # 72 perms = 9 outer blocks × 8 inner bits.
    #   inner bits (0..7):
    #     bit 0 — HAS_RANDOM
    #     bit 1 — HAS_VC
    #     bit 2 — HAS_NT
    #   outer = mode_idx * 3 + uv_variant   (mode 0..2, variant 0..2):
    #     mode 0 = basic, 1 = billboard, 2 = atlas
    #     variant 0 = no UV (collapses any mode to PopcornNoUV)
    #     variant 1/2 = UV stream bound (the engine compiles two
    #                   redundant variants per mode — they map to the
    #                   same Slang specialisation here).
    inner    = idx & 7
    outer    = idx // 8
    mode_idx = outer // 3
    uv_var   = outer %  3

    if uv_var == 0:
        mode = "PopcornNoUV"
    elif mode_idx == 0:
        mode = "PopcornBasicUV"
    elif mode_idx == 1:
        mode = "PopcornBillboard"
    else:
        mode = "PopcornAtlas"

    has_rand = "true" if (inner & 1) else "false"
    has_vc   = "true" if (inner & 2) else "false"
    has_nt   = "true" if (inner & 4) else "false"

    return PermSpec(
        entry="popcorn_vs_main",
        types=[mode, has_rand, has_vc, has_nt],
        label=f"{mode}+R={has_rand}+VC={has_vc}+NT={has_nt}",
    )


def map_popcorn_ps(idx: int) -> PermSpec:
    # 1152 perms = 9 outer blocks × 128 inner bits.
    #   inner bits (0..127):
    #     bit 0 (0x01) — HAS_GBUFFER         (COLOR pass only)
    #     bit 1 (0x02) — FOG_LINEAR          (COLOR pass only)
    #     bit 2 (0x04) — FOG_EXP             (COLOR pass only;
    #                                         set with bit 1 → FogExp2)
    #     bit 3 (0x08) — HAS_SOFT_PARTICLES  (both passes)
    #     bit 4 (0x10) — HAS_ALPHA_LUT       (COLOR pass only)
    #     bit 5 (0x20) — HAS_VC              (both passes)
    #     bit 6 (0x40) — HAS_LIT             (COLOR pass only)
    #
    #   outer (0..8) = mode_idx * 3 + uv_variant:
    #     mode 0 = basic, 1 = billboard, 2 = atlas
    #     variant 0 = no UV         → PopcornNoUV       (COLOR pass)
    #     variant 1 = COLOR pass    → PopcornBasicUV / Billboard / Atlas
    #     variant 2 = MOTION pass   → same modes, IS_MOTION_PASS = true
    #
    # Note: many bit combinations collapse semantically (e.g. all the
    # COLOR-pass-only bits are no-ops in MOTION_PASS) but the engine
    # still compiles every variant so its perm-table stays a fixed grid.
    # We mirror that 1:1 here so the Slang specialisations line up
    # perm-for-perm with the original BLS bundle.
    inner    = idx & 0x7F
    outer    = idx // 128
    mode_idx = outer // 3
    uv_var   = outer %  3

    if uv_var == 0:
        mode = "PopcornNoUV"
    elif mode_idx == 0:
        mode = "PopcornBasicUV"
    elif mode_idx == 1:
        mode = "PopcornBillboard"
    else:
        mode = "PopcornAtlas"

    is_motion = "true" if uv_var == 2 else "false"

    fogL = bool(inner & 0x02)
    fogE = bool(inner & 0x04)
    if fogL and fogE:
        fog = "FogExp2"
    elif fogL:
        fog = "FogLinear"
    elif fogE:
        fog = "FogExponential"
    else:
        fog = "FogNone"

    has_gbuf = "true" if (inner & 0x01) else "false"
    has_sp   = "true" if (inner & 0x08) else "false"
    has_alut = "true" if (inner & 0x10) else "false"
    has_vc   = "true" if (inner & 0x20) else "false"
    has_lit  = "true" if (inner & 0x40) else "false"

    return PermSpec(
        entry="popcorn_ps_main",
        types=[mode, fog, is_motion, has_gbuf, has_sp, has_alut, has_vc, has_lit],
        label=(f"{mode}+M={is_motion}+{fog}+G={has_gbuf}+SP={has_sp}"
               f"+ALUT={has_alut}+VC={has_vc}+LIT={has_lit}"),
    )


def map_sd_classic_ps(idx: int) -> PermSpec:
    low     = idx & 7
    t0stage = (idx // 8) % 5
    t1stage = (idx // 40) % 5

    fogL = bool(low & 1)
    fogE = bool(low & 2)
    at   = bool(low & 4)

    if fogL and fogE:
        fog = "FogExp2"
    elif fogL:
        fog = "FogLinear"
    elif fogE:
        fog = "FogExponential"
    else:
        fog = "FogNone"
    alpha = "AlphaTestOn" if at else "AlphaTestOff"

    return PermSpec(
        entry="sd_classic_ps_main",
        types=[STAGE_NAMES[t0stage], STAGE_NAMES[t1stage], fog, alpha],
        label=f"T0={t0stage}+T1={t1stage}+{fog}+{alpha}",
    )


# Per-family permutation mappers. These are the only piece of family
# metadata that stays as code — each maps a linear perm index to the
# tuple of slangc -specialize types for that perm (bit-packed feature
# axes, conditional type-name selection). Everything else (stage,
# perm_count, entry point, module) comes from wc3_shaders.json.
MAPPERS: dict[str, Callable[[int], PermSpec]] = {
    "hd_vs":          map_hd_vs,
    "hd_ps":          map_hd_ps,
    "toon_hd_vs":     map_toon_hd_vs,
    "toon_hd_ps":     map_toon_hd_ps,
    "crystal_ps":     map_crystal_ps,
    "sd_on_hd_vs":    map_sd_on_hd_vs,
    "sd_on_hd_ps":    map_sd_on_hd_ps,
    "sd_highspec_vs": map_sd_highspec_vs,
    "sd_classic_ps":  map_sd_classic_ps,
    "water_vs":       map_water_vs,
    "water_ps":       map_water_ps,
    "popcorn_vs":     map_popcorn_vs,
    "popcorn_ps":     map_popcorn_ps,
    "tonemap_ps":     map_tonemap_ps,
    "sprite_vs":      map_sprite_vs,
    "sprite_ps":      map_sprite_ps,
    "terrain_vs":     map_terrain_vs,
    "terrain_ps":     map_terrain_ps,
    "foliage_vs":     map_foliage_vs,
    "foliage_ps":     map_foliage_ps,
    "distortion_ps":  map_distortion_ps,
}

# Fail fast if the config and the mapper set drift — every family listed
# in wc3_shaders.json must have a mapper implementation here, and vice versa.
_missing_mappers = set(FAMILY_CONFIGS) - set(MAPPERS)
_orphan_mappers  = set(MAPPERS)        - set(FAMILY_CONFIGS)
if _missing_mappers or _orphan_mappers:
    raise SystemExit(
        f"wc3_shaders.json / MAPPERS mismatch — "
        f"missing mappers for {sorted(_missing_mappers)}, "
        f"orphan mappers for {sorted(_orphan_mappers)}"
    )

# Iteration order matches wc3_shaders.json (i.e. FAMILY_CONFIGS insertion order).
SWEEPS = [
    (name, cfg.perm_count, MAPPERS[name], cfg.stage)
    for name, cfg in FAMILY_CONFIGS.items()
]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--family", choices=(*FAMILIES, "all"), default="all")
    parser.add_argument("--target", choices=(*TARGETS.keys(), "all"),
                        default="d3d11",
                        help="Graphics API target (default: d3d11).")
    parser.add_argument("--metallib", metavar="MACOS_MIN",
                        help="Emit compiled Metal bytecode (.metallib) with the given "
                             "macOS deployment target (e.g. `11` or `11.0`). Requires "
                             "Apple's Metal compiler (Xcode) — macOS only. On macOS "
                             "the metal target is auto-included with macOS min 11 "
                             "unless this flag overrides it.")
    parser.add_argument("--slangc", help="Path to slangc executable "
                                         "(overrides PATH / VULKAN_SDK lookup).")
    parser.add_argument("--jobs", "-j", type=int, default=os.cpu_count() or 1,
                        help="Number of parallel slangc processes to run "
                             "(default: %(default)s = os.cpu_count()). "
                             "Each permutation is an independent slangc "
                             "invocation, so this scales near-linearly until "
                             "you saturate the CPU. Use --jobs 1 for "
                             "deterministic single-thread output.")
    args = parser.parse_args()
    if args.jobs < 1:
        args.jobs = 1

    # On macOS we can drive Apple's metal compiler to emit real .metallib
    # bytecode (not just .metal source), which build_bls.py can then pack
    # into mtlfs/mtlvs BLS files. On other platforms this step is skipped
    # — Apple's toolchain is not available.
    mac_min = args.metallib or ("11" if sys.platform == "darwin" else None)
    if mac_min:
        TARGETS["metal"] = {
            "target": "metallib",
            "ext":    "metallib",
            "vs":     "sm_6_0",
            "ps":     "sm_6_0",
            "extra":  ["-Xmetal", f"-mmacosx-version-min={mac_min}"],
        }

    slangc = resolve_slangc(args.slangc)
    print(f"Using slangc: {slangc}")
    print(f"Shader module: {SHADER}")
    print(f"Parallel jobs: {args.jobs}")
    if mac_min:
        print(f"Metal output: metallib (macOS min={mac_min})")

    OUT_BASE.mkdir(exist_ok=True)

    active_targets = list(TARGETS.keys()) if args.target == "all" else [args.target]
    # macOS bonus: always emit metallibs alongside the primary target so
    # build_bls.py can repackage the mtlfs/mtlvs BLS files.
    if sys.platform == "darwin" and "metal" not in active_targets:
        active_targets.append("metal")

    all_results: List[tuple] = []  # (target_key, SweepResult)
    for target_key in active_targets:
        for name, count, mapper, stage in SWEEPS:
            if args.family != "all" and args.family != name:
                continue
            result = run_sweep(name, count, mapper, stage, target_key, args.jobs)
            all_results.append((target_key, result))

    print()
    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)
    total = SweepResult(family="TOTAL", count=0)
    for target_key, r in all_results:
        total.count += r.count
        total.ok    += r.ok
        total.fail  += r.fail
        label = f"[{target_key}] {r.family}"
        print(f"{label:<28}  total={r.count:>4}  "
              f"compile={r.ok:>4}ok/{r.fail:>4}fail")
    print()
    print(f"{'TOTAL':<28}  total={total.count:>4}  "
          f"compile={total.ok:>4}ok/{total.fail:>4}fail")
    return 0


if __name__ == "__main__":
    sys.exit(main())
