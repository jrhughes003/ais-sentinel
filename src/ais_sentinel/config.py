"""Configuration loading.

The YAML config is the single source of truth for paths, thresholds and seeds. It is loaded
into a plain nested ``dict`` and wrapped in :class:`Config`, which adds attribute-style access
and a stable hash used to stamp generated artifacts.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"


class Config(dict[str, Any]):
    """A nested dict with attribute access (``cfg.region.lat_min``)."""

    def __getattr__(self, name: str) -> Any:
        try:
            value = self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc
        return Config(value) if isinstance(value, dict) else value

    def hash(self) -> str:
        """Return a short, stable hash of the configuration contents."""
        blob = json.dumps(self, sort_keys=True, default=str).encode()
        return hashlib.sha256(blob).hexdigest()[:12]


def _merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``over`` into a copy of ``base`` (mappings merge, others replace)."""
    out = dict(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path: str | Path | None = None) -> Config:
    """Load a YAML config file (defaults to ``configs/default.yaml``).

    A config may start with ``extends: <file>`` (relative to its own directory). It is then
    deep-merged over that base, so e.g. ``configs/dev.yaml`` only lists what it changes.
    """
    cfg_path = Path(path) if path is not None else DEFAULT_CONFIG
    with cfg_path.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise ValueError(f"Config {cfg_path} must be a mapping")
    parent = raw.pop("extends", None)
    if parent:
        raw = _merge(dict(load_config(cfg_path.parent / str(parent))), raw)
    return Config(raw)


def as_date(value: str | date) -> date:
    """Coerce a YAML date (already a ``date``) or ISO string into a ``date``."""
    return value if isinstance(value, date) else date.fromisoformat(value)
