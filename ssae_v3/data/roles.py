"""Access per-dataset column roles defined in the repo-root `config.py`.

`config.py` lives at the repository root (not inside this package) and holds one
`BaselineConfig` per dataset: treatment column, outcome column, columns to drop.
Adapters read roles from here by attribute name (e.g. `AIDS_V1`, `DIUR_V1`) instead
of hard-coding column names. IHDP is intentionally absent from `config.py` (it is a
standalone benchmark); its adapter names its roles directly.
"""

from __future__ import annotations

import importlib.util
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Any


def repo_root() -> Path:
    """Absolute path to the repository root (two levels above this package)."""
    return Path(__file__).resolve().parents[2]


@lru_cache(maxsize=1)
def _config_module() -> ModuleType:
    """Import the repo-root config.py by file path, independent of the CWD."""
    path = repo_root() / "config.py"
    spec = importlib.util.spec_from_file_location("ssae_v3_dataset_roles", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load column-role config from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def baseline_config(name: str) -> Any:
    """Return the BaselineConfig named `name` from the repo-root config.py."""
    module = _config_module()
    try:
        return getattr(module, name)
    except AttributeError as exc:
        raise KeyError(f"no BaselineConfig named {name!r} in config.py") from exc


def resolve_data_path(cfg: Any) -> Path:
    """Resolve a BaselineConfig.data_path (repo-relative) against the repo root."""
    return repo_root() / cfg.data_path
