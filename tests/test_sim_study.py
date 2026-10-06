"""Simulation-study plumbing (tiny grids and seeds so it runs in seconds)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ais_sentinel.config import Config
from ais_sentinel.tracking import sim_study


def test_sim_study_stage_writes_report_with_criteria(
    tmp_cfg: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sim_study, "CV_GRID", {"q_accel": [0.01, 0.1]})
    monkeypatch.setattr(sim_study, "IMM_GRID", {"q_turn": [0.05]})
    tmp_cfg["tracking"]["sim"] = {"tune_seeds": [0, 3], "eval_seeds": [100, 104]}
    sim_study.stage(tmp_cfg)
    rep = Path(tmp_cfg.paths.reports)
    md = (rep / "tracking.md").read_text(encoding="utf-8")
    res = json.loads((rep / "tracking_sim.json").read_text(encoding="utf-8"))
    assert "Success criteria" in md
    assert len(res["criteria"]) == 3
    assert res["eval"]["imm"]["moving"]["n"] > 100
    assert len(res["tuning_history"]["cv"]) == 2
    nees = res["eval"]["imm"]["nees"]
    assert nees["band"][0] < 4 < nees["band"][1]
