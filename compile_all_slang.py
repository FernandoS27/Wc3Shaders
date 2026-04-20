"""Compile all permutations of every shader entry point from the unified
wc3_shaders Slang module, to one of several graphics-API targets.

Runs independently of the current working directory — paths resolve
relative to this script's location.

    --target {d3d11,d3d12,vulkan,opengl,metal,webgpu,all}  (default: d3d11)
    --family {hd_vs,hd_ps,toon_hd_vs,toon_hd_ps,crystal_ps,
              sd_on_hd_vs,sd_on_hd_ps,sd_highspec_vs,sd_classic_ps,
              water_vs,water_ps,all}
    --slangc PATH   explicit slangc.exe override
"""

import argparse
import glob
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

REPO_ROOT = Path(__file__).resolve().parent
SHADER = REPO_ROOT / "wc3_shaders" / "wc3_shaders.slang"
OUT_BASE = REPO_ROOT / "slang_out"

FAMILIES = (
    "hd_vs", "hd_ps", "crystal_ps",
    "toon_hd_vs", "toon_hd_ps",
    "sd_on_hd_vs", "sd_on_hd_ps",
    "sd_highspec_vs", "sd_classic_ps",
    "water_vs", "water_ps",
)

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
                  extra: Optional[List[str]] = None) -> bool:
    args = [resolve_slangc(), "-entry", entry]
    for t in specialize:
        args += ["-specialize", t]
    args += [
        "-profile", profile,
        "-target", target,
        "-o", str(out_path),
        "-warnings-disable", "39001",
    ]
    if extra:
        args += extra
    args.append(str(SHADER))
    subprocess.run(args, capture_output=True, text=True)
    return out_path.exists() and out_path.stat().st_size > 0


def run_sweep(family: str, count: int, mapper: Callable[[int], PermSpec],
              stage: str, target_key: str) -> SweepResult:
    cfg = TARGETS[target_key]
    print()
    print(f"========== [{target_key}] {family} ({count} perms) ==========")
    out_dir = OUT_BASE / target_key / family
    out_dir.mkdir(parents=True, exist_ok=True)

    result = SweepResult(family=family, count=count)
    ext = cfg["ext"]
    target = cfg["target"]
    profile = cfg[stage]
    extra = cfg.get("extra", [])

    for i in range(count):
        spec = mapper(i)
        out_path = out_dir / f"perm_{i:03d}.{ext}"

        if not invoke_slangc(spec.entry, target, profile, spec.types, out_path, extra):
            result.fail += 1
            result.fail_list.append(f"perm_{i} ({spec.label})")
            continue
        result.ok += 1

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


# family, perm count, mapper, stage ("vs" or "ps" — picks the profile
# from TARGETS for whichever --target is active).
SWEEPS = [
    ("hd_vs",          144, map_hd_vs,          "vs"),
    ("hd_ps",          512, map_hd_ps,          "ps"),
    ("toon_hd_vs",     144, map_toon_hd_vs,     "vs"),
    ("toon_hd_ps",     512, map_toon_hd_ps,     "ps"),
    ("crystal_ps",     512, map_crystal_ps,     "ps"),
    ("sd_on_hd_vs",    144, map_sd_on_hd_vs,    "vs"),
    ("sd_on_hd_ps",    384, map_sd_on_hd_ps,    "ps"),
    ("sd_highspec_vs", 162, map_sd_highspec_vs, "vs"),
    ("sd_classic_ps",  200, map_sd_classic_ps,  "ps"),
    ("water_vs",         1, map_water_vs,       "vs"),
    ("water_ps",         4, map_water_ps,       "ps"),
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
    args = parser.parse_args()

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
            result = run_sweep(name, count, mapper, stage, target_key)
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
