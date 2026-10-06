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
    tmp_cfg["tracking"]["sim"] = {"tune_seeds": [0, 3], "eval_seeds": [100, 104], "mode": "tune"}
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


def test_fixed_mode_keeps_imm_fixed_and_picks_the_better_cv_baseline(
    tmp_cfg: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dataclasses import asdict

    from ais_sentinel.tracking.imm import IMMParams
    from ais_sentinel.tracking.kf import CVParams
    from ais_sentinel.tracking.simulate import SimConfig

    cal = Path(tmp_cfg.paths.reports) / "cal.json"
    cal.parent.mkdir(parents=True, exist_ok=True)
    imm = asdict(IMMParams(r_pos_m=3.0))
    imm["switch"] = imm["switch"].tolist()
    cal.write_text(
        json.dumps(
            {
                "sim_config": asdict(SimConfig(duration_s=1800, log_uniform=True)),
                "cv": asdict(CVParams(q_accel=0.01)),
                "imm": imm,
            }
        ),
        encoding="utf-8",
    )
    tmp_cfg["tracking"]["sim"] = {
        "tune_seeds": [0, 3],
        "eval_seeds": [2000, 2003],
        "mode": "fixed",
        "calibration_file": str(cal),
    }
    monkeypatch.setattr(sim_study, "CV_GRID", {"q_accel": [0.003, 0.1]})
    res = sim_study.run_study(tmp_cfg)
    assert res["mode"] == "fixed"
    assert res["tuning_history"]["imm"] == []  # the IMM is never tuned in simulation
    assert res["tuned"]["imm"]["r_pos_m"] == 3.0
    alt = res["cv_alternatives"]
    best = min(("real_likelihood", "simulator_tuned"), key=lambda k: alt[k]["tune_moving_rmse_m"])
    assert alt["chosen"] == best  # the CV baseline is the better of the two
    assert "Attempt 3 (fixed mode)" in sim_study.report_markdown(res)
