# Wc3Shaders

An open-source recreation of the shaders used by **Warcraft III: Reforged**, reverse-engineered from the game's shipped `.bls` shader bundles.

The project reimplements every shader family the game ships — **SD**, **SD-on-HD**, **HD**, **Crystal**, **water**, and the HDR→LDR **tonemap** — in [Slang](https://shader-slang.com/), then re-packs the compiled bytecode back into the game's `.bls` wire format so patched shaders can be dropped into the game. A separate `custom_shaders` module sits on top of the reconstruction for user-authored variants (currently a toon / cel-shaded HD variant). Both the DirectX bundles (`ps/`, `vs/`) and the Metal bundles (`mtlfs/`, `mtlvs/`) are covered; Metal packing only runs when built on macOS, where Apple's Metal compiler is available.

## What's in the repo

| Path | Contents |
| --- | --- |
| [wc3_shaders/](wc3_shaders/) | Slang source for the faithful reconstruction. One unified module ([wc3_shaders.slang](wc3_shaders/wc3_shaders.slang)) exposes one entry point per shipped family. |
| [custom_shaders/](custom_shaders/) | User-facing Slang module ([custom_shaders.slang](custom_shaders/custom_shaders.slang)) that `import`s `wc3_shaders` and provides drop-in variant shader bodies (e.g. `toon_hd_ps`). |
| [wc3_shaders.json](wc3_shaders.json) | Declarative config for the core families: stage, entry point, permutation count, shipped BLS name. Treat as read-only. |
| [custom_shaders.json](custom_shaders.json) | Declarative config for user-authored variant families. This is the file you edit to add a new variant shader. |
| [shader_config.py](shader_config.py) | Loads and merges both JSON files into a single `FamilyConfig` view used by the build scripts. |
| [compile_all_slang.py](compile_all_slang.py) | Compiles every permutation of every family to a chosen graphics API target (D3D11 by default). |
| [build_bls.py](build_bls.py) | Packs the compiled DXBC blobs back into `.bls` files using the shipped `.bls` files as binding-metadata templates. |

## Shader families

### Core (backed by `wc3_shaders/wc3_shaders.slang`)

Defined in [wc3_shaders.json](wc3_shaders.json). These mirror the shipped BLS layout one-for-one.

| Family | Stage | Permutations | Ships as |
| --- | --- | --- | --- |
| `hd_vs` | Vertex | 144 | `vs/hd.bls` |
| `hd_ps` | Pixel | 512 | `ps/hd.bls` |
| `crystal_ps` | Pixel | 512 | `ps/crystal.bls` |
| `sd_on_hd_vs` | Vertex | 144 | `vs/sd_on_hd.bls` |
| `sd_on_hd_ps` | Pixel | 384 | `ps/sd_on_hd.bls` |
| `sd_highspec_vs` | Vertex | 162 | `vs/sd_highspec.bls` |
| `sd_classic_ps` | Pixel | 200 | `ps/sd.bls` |
| `water_vs` | Vertex | 1 | `vs/water.bls` |
| `water_ps` | Pixel | 4 | `ps/water.bls` |
| `tonemap_ps` | Pixel | 1 | `ps/tonemap.bls` |

### Custom variants (backed by `custom_shaders/custom_shaders.slang`)

Defined in [custom_shaders.json](custom_shaders.json). These clone a core family's BLS template so the rebuilt file remains drop-in compatible with the HD pipeline.

| Family | Stage | Permutations | Ships as | Template |
| --- | --- | --- | --- | --- |
| `toon_hd_vs` | Vertex | 144 | `vs/toon_hd.bls` | `hd.bls` |
| `toon_hd_ps` | Pixel | 512 | `ps/toon_hd.bls` | `hd.bls` |

Each permutation is produced by specializing the corresponding Slang entry point on a set of interface types (skinning model, vertex format, fog mode, alpha-test, material, etc.). The permutation-index → `-specialize` tuple mapping for each family lives in the `map_*` functions of [compile_all_slang.py](compile_all_slang.py).

## Configuration split

Family metadata is split across two JSON files with identical schemas. The build scripts merge them at load time (`shader_config.load_families()`); `custom_shaders.json` is optional and may be emptied to disable all variants.

| File | Purpose | Edit freely? |
| --- | --- | --- |
| [wc3_shaders.json](wc3_shaders.json) | Core wc3 families that mirror the shipped BLS set. | No — changing these breaks the reconstruction. |
| [custom_shaders.json](custom_shaders.json) | User-authored variant families built on top of the reconstruction. | Yes — this is where new variants are registered. |

Per-entry schema:

| Field | Required | Meaning |
| --- | --- | --- |
| `stage` | yes | `"vs"` or `"ps"`. Selects the profile and the BLS output directory. |
| `entry` | yes | slangc entry-point name (e.g. `"toon_ps_main"`). |
| `perm_count` | yes | Total number of permutations — must match the range the family's `map_*` function enumerates. |
| `bls_name` | yes | Output basename under `ps/ vs/ mtlfs/ mtlvs/` (e.g. `"toon_hd.bls"`). |
| `template_override` | no | When the family reuses another shipped BLS as its binding-metadata template. Required whenever the variant's `bls_name` does not itself exist in the shipped `war3.w3mod/shaders/` tree. |

The `module` field is NOT stored in the JSON — it is injected by the loader based on which file the entry came from, so users cannot accidentally desync it.

## Requirements

- **Python 3.8+** (standard library only — no dependencies).
- **`slangc`** from the [Shader Slang](https://github.com/shader-slang/slang) compiler. Install one of:
  - The [Vulkan SDK](https://vulkan.lunarg.com/) (ships with `slangc` under `Bin/`).
  - A standalone Slang release from the Slang GitHub releases page.
- The shipped `.bls` templates under `war3.w3mod/shaders/{ps,vs,mtlfs,mtlvs}/`. These are **not** included in this repo — Extract them from your Warcraft III: Reforged installation with a CASC extractor (e.g. [CASCExplorer](https://github.com/WoW-Tools/CASCExplorer), [CASCView](http://www.zezula.net/en/casc/main.html), or any other CASC-aware tool), pulling the `war3.w3mod/shaders/` tree into this repo root. The build scripts locate them through `--templates` so you can also point at an external directory.

The build scripts resolve `slangc` in this order: `--slangc` flag → `SLANGC` env var → system `PATH` → `VULKAN_SDK/Bin/` → `C:\VulkanSDK\*\Bin\slangc.exe`.

## Building

The workflow is two steps: compile Slang → DXBC, then pack DXBC into `.bls`.

### 1. Compile the shaders

```sh
python compile_all_slang.py
```

This compiles every permutation of every family to D3D11 DXBC and writes them under [slang_out/d3d11/&lt;family&gt;/perm_NNN.dxbc](slang_out/). Core families are read from `wc3_shaders/wc3_shaders.slang`; custom families are read from `custom_shaders/custom_shaders.slang` with `wc3_shaders/` on the include path.

Useful flags:

| Flag | Purpose |
| --- | --- |
| `--family <name>` | Compile only one family (e.g. `--family toon_hd_ps`). |
| `--target <api>` | Pick a target: `d3d11` (default), `d3d12`, `vulkan`, `opengl`, `metal`, `webgpu`, or `all`. Only `d3d11` output can be packed into `.bls`. |
| `--slangc <path>` | Explicit `slangc` path. |
| `--metallib <macos-min>` | When targeting `metal`, emit compiled `.metallib` instead of Metal source. Requires Xcode; macOS only. |

### 2. Pack the patched `.bls` files

```sh
python build_bls.py --templates war3.w3mod/shaders --output bls_out
```

This reads the compiled `.dxbc` blobs from `slang_out/d3d11/` and writes rebuilt `.bls` files to `bls_out/{ps,vs}/`, using the shipped `.bls` files (or the configured `template_override`) as templates for per-permutation metadata (resource bindings, stage flags, etc.).

Useful flags:

| Flag | Purpose |
| --- | --- |
| `--family <name>` | Rebuild only one family (repeatable). |
| `--strip` | Strip `RDEF` / `STAT` chunks from the DXBC and recompute the hash so the output matches the shipped chunk layout byte-for-byte. |
| `--slang-out <dir>` | Alternate location of the compiled DXBC tree (defaults to `./slang_out`). |
| `--verbose` | Print per-family size summaries. |

### Installing the patched shaders

Copy the files from `bls_out/ps/` and `bls_out/vs/` over the originals in your Warcraft III installation's corresponding `shaders/ps` and `shaders/vs` directories (back up the originals first).

## Adding a custom shader

Variant shaders plug into the existing HD pipeline via BLS patching — they do **not** define a new pipeline. That means:

- The variant's VS must produce the same `VSOutput` as `hd_vs`, and its PS must consume the same `PSInput` and produce the same `PSOutput` / MRT layout as `hd_ps`. The engine binds HD-shaped pipeline state (vertex format, descriptor set, render target layout); any divergence breaks the patch.
- Confine stylistic changes to pixel-shader body math (lighting model, color grading, post-lit effects). Extra passes, new textures/samplers, new VS outputs, and new MRT slots are not supported through this path.

The rest is three steps:

### 1. Write the Slang body

Add your entry points to [custom_shaders/](custom_shaders/) and wire them into [custom_shaders.slang](custom_shaders/custom_shaders.slang) with `__include`. The existing [toon_hd_ps.slang](custom_shaders/toon_hd_ps.slang) / [toon_hd_vs.slang](custom_shaders/toon_hd_vs.slang) are the reference example — they reuse `hd_vs`'s vertex transform and replace only the pixel-shader lighting.

### 2. Register the family in `custom_shaders.json`

```json
{
  "families": {
    "my_variant_ps": {
      "stage": "ps",
      "entry": "my_variant_ps_main",
      "perm_count": 512,
      "bls_name": "my_variant.bls",
      "template_override": "hd.bls"
    }
  }
}
```

Use `template_override` whenever `bls_name` does not itself ship under `war3.w3mod/shaders/{ps,vs}/` — the rebuilt file needs a shipped BLS to clone per-permutation binding metadata from.

### 3. Add the permutation mapper

Each family needs a `map_<family>(idx) → PermSpec` function in [compile_all_slang.py](compile_all_slang.py) and an entry in the `MAPPERS` dict. The mapper translates a linear perm index into the tuple of slangc `-specialize` types. Variants that share their base family's permutation axes can delegate: see `map_toon_hd_ps` for an example that reuses `map_hd_ps`'s 9-bit feature encoding with a different entry point.

`compile_all_slang.py` fails fast at startup if `MAPPERS` and the merged JSON disagree on the family set, so forgetting either half surfaces immediately.

## License

Source is released under the **BSD 3-Clause License** — see [LICENSE](LICENSE).

An additional [LICENSE-AI.md](LICENSE-AI.md) notice clarifies that AI-generated derivative works are subject to the same attribution and license conditions as any other derivative work.

This project is an independent, fan-made reimplementation and is not affiliated with or endorsed by Blizzard Entertainment. Warcraft III is a trademark of Blizzard Entertainment.
