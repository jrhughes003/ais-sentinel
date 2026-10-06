"""Shared fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

from ais_sentinel.config import Config, load_config


@pytest.fixture
def tmp_cfg(tmp_path: Path) -> Config:
    """Default config with every data path redirected into a temp directory."""
    cfg = load_config()
    cfg["paths"] = {k: str(tmp_path / k) for k in cfg["paths"]}
    cfg["download"]["pause_seconds"] = 0
    return cfg
