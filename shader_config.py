"""Shared loader for shader-family configuration.

Family definitions are split across two JSON files at the repo root:

* ``wc3_shaders.json`` — core Wc3 families backed by
  ``wc3_shaders/wc3_shaders.slang``. Mirrors the shipped BLS layout;
  should not normally be modified.
* ``custom_shaders.json`` — user-editable variant families backed by
  ``custom_shaders/custom_shaders.slang``. Users add or edit entries
  here to introduce new variant shaders without touching the core.

Both files share the same per-entry schema (``stage``, ``entry``,
``perm_count``, ``bls_name``, optional ``template_override``). The
``module`` field is NOT stored in the JSON — the loader injects it based
on which file the entry came from, so users can't accidentally desync it.

``build_bls.py`` and ``compile_all_slang.py`` both call
``load_families()`` to get the merged view. Family names must be unique
across the two files; a collision raises at load time.

The per-family permutation mappers — the Python functions that turn a
linear perm index into the tuple of slangc ``-specialize`` types — stay
in ``compile_all_slang.py``. Those are code (bit-packing + conditional
type-name selection), not data.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

REPO_ROOT = Path(__file__).resolve().parent
WC3_CONFIG_PATH    = REPO_ROOT / "wc3_shaders.json"
CUSTOM_CONFIG_PATH = REPO_ROOT / "custom_shaders.json"


@dataclass(frozen=True)
class FamilyConfig:
    name: str
    stage: str                                  # "vs" or "ps"
    entry: str                                  # slangc entry-point name
    perm_count: int
    module: str                                 # "wc3" or "custom"
    bls_name: str                               # basename, e.g. "hd.bls"
    template_override: Optional[str] = None     # e.g. "hd.bls" for toon_hd_*

    @property
    def dx_dir(self) -> str:
        # DX BLS files live under ps/ or vs/ by stage.
        return self.stage

    @property
    def metal_dir(self) -> str:
        # Metal BLS files live under mtlfs/ or mtlvs/ by stage.
        return "mtlfs" if self.stage == "ps" else "mtlvs"

    @property
    def effective_template(self) -> str:
        return self.template_override or self.bls_name


def _load_one(path: Path, module: str, out: Dict[str, FamilyConfig]) -> None:
    # The custom config is optional — users may delete it to drop all
    # variant families. The core config must exist.
    if module == "custom" and not path.is_file():
        return
    with open(path) as fp:
        data = json.load(fp)
    for name, cfg in data["families"].items():
        if name in out:
            raise ValueError(
                f"shader family {name!r} is defined in both "
                f"wc3_shaders.json and custom_shaders.json"
            )
        out[name] = FamilyConfig(
            name=name,
            stage=cfg["stage"],
            entry=cfg["entry"],
            perm_count=cfg["perm_count"],
            module=module,
            bls_name=cfg["bls_name"],
            template_override=cfg.get("template_override"),
        )


def load_families() -> Dict[str, FamilyConfig]:
    """Merge wc3 + custom family definitions.

    Core families load first so they retain their natural ordering; any
    user-defined variant families follow. Family names must be unique
    across both files.
    """
    out: Dict[str, FamilyConfig] = {}
    _load_one(WC3_CONFIG_PATH,    "wc3",    out)
    _load_one(CUSTOM_CONFIG_PATH, "custom", out)
    return out
