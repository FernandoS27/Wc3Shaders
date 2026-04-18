# Wc3Shaders

An open-source recreation of the shaders used by **Warcraft III: Reforged**, reverse-engineered from the game's shipped `.bls` shader bundles.

The project reimplements the **SD**, **SD-on-HD**, and **HD** shader families in [Slang](https://shader-slang.com/), then re-packs the compiled bytecode back into the game's `.bls` wire format so patched shaders can be dropped into the game. Both the DirectX bundles (`ps/`, `vs/`) and the Metal bundles (`mtlfs/`, `mtlvs/`) are covered; Metal packing only runs when compiled on macOS, where Apple's Metal compiler is available.

## What's in the repo

| Path | Contents |
| --- | --- |
| [wc3_shaders/](wc3_shaders/) | Slang source. One unified module ([wc3_shaders.slang](wc3_shaders/wc3_shaders.slang)) exposes six entry points, one per shader family. |
| [compile_all_slang.py](compile_all_slang.py) | Compiles every permutation of every family to a chosen graphics API target (D3D11 by default). |
| [build_bls.py](build_bls.py) | Packs the compiled DXBC blobs back into `.bls` files using the shipped `.bls` files as templates. |

## Shader families

Six families cover every shader the game ships:

| Family | Stage | Permutations |
| --- | --- | --- |
| `hd_vs` | Vertex | 144 |
| `hd_ps` | Pixel | 512 |
| `sd_on_hd_vs` | Vertex | 144 |
| `sd_on_hd_ps` | Pixel | 384 |
| `sd_highspec_vs` | Vertex | 162 |
| `sd_classic_ps` | Pixel | 200 |

Each permutation is produced by specializing the corresponding Slang entry point on a set of interface types (skinning model, vertex format, fog mode, alpha-test, material, etc.). The exact permutation mapping lives in [compile_all_slang.py](compile_all_slang.py).

## Requirements

- **Python 3.8+** (standard library only — no dependencies).
- **`slangc`** from the [Shader Slang](https://github.com/shader-slang/slang) compiler. Install one of:
  - The [Vulkan SDK](https://vulkan.lunarg.com/) (ships with `slangc` under `Bin/`).
  - A standalone Slang release from the Slang GitHub releases page.
- For rebuilding `.bls` files you also need the shipped templates under [war3.w3mod/shaders/](war3.w3mod/shaders/) (included in this repo).

The build scripts resolve `slangc` in this order: `--slangc` flag → `SLANGC` env var → system `PATH` → `VULKAN_SDK/Bin/` → `C:\VulkanSDK\*\Bin\slangc.exe`.

## Building

The workflow is two steps: compile Slang → DXBC, then pack DXBC into `.bls`.

### 1. Compile the shaders

```sh
python compile_all_slang.py
```

This compiles every permutation of every family to D3D11 DXBC and writes them under [slang_out/d3d11/&lt;family&gt;/perm_NNN.dxbc](slang_out/).

Useful flags:

| Flag | Purpose |
| --- | --- |
| `--family <name>` | Compile only one family (e.g. `--family hd_ps`). |
| `--target <api>` | Pick a target: `d3d11` (default), `d3d12`, `vulkan`, `opengl`, `metal`, `webgpu`, or `all`. Only `d3d11` output can be packed into `.bls`. |
| `--slangc <path>` | Explicit `slangc` path. |
| `--metallib <macos-min>` | When targeting `metal`, emit compiled `.metallib` instead of Metal source. Requires Xcode; macOS only. |

### 2. Pack the patched `.bls` files

```sh
python build_bls.py --templates war3.w3mod/shaders --output bls_out
```

This reads the compiled `.dxbc` blobs from `slang_out/d3d11/` and writes rebuilt `.bls` files to `bls_out/{ps,vs}/`, using the shipped `.bls` files as templates for per-permutation metadata (resource bindings, stage flags, etc.).

Useful flags:

| Flag | Purpose |
| --- | --- |
| `--family <name>` | Rebuild only one family (repeatable). |
| `--strip` | Strip `RDEF` / `STAT` chunks from the DXBC and recompute the hash so the output matches the shipped chunk layout byte-for-byte. |
| `--slang-out <dir>` | Alternate location of the compiled DXBC tree (defaults to `./slang_out`). |
| `--verbose` | Print per-family size summaries. |

### Installing the patched shaders

Copy the files from `bls_out/ps/` and `bls_out/vs/` over the originals in your Warcraft III installation's corresponding `shaders/ps` and `shaders/vs` directories (back up the originals first).

## License

Source is released under the **BSD 3-Clause License** — see [LICENSE](LICENSE).

An additional [LICENSE-AI.md](LICENSE-AI.md) notice clarifies that AI-generated derivative works are subject to the same attribution and license conditions as any other derivative work.

This project is an independent, fan-made reimplementation and is not affiliated with or endorsed by Blizzard Entertainment. Warcraft III is a trademark of Blizzard Entertainment.
