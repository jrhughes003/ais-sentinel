"""Smoke test of the full prediction experiment (evaluate + holdout) on synthetic data."""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from ais_sentinel.config import Config
from ais_sentinel.data import build
from ais_sentinel.data.download import DayRecord, interim_path, save_manifest
from ais_sentinel.data.schema import CANONICAL_SCHEMA
from ais_sentinel.prediction import run
from ais_sentinel.tracking import pipeline as tracking
from tests.test_knn import l_route_voyage

DAYS = [date(2023, 6, 1), date(2023, 6, 2), date(2023, 9, 10), date(2023, 10, 10)]


def _day(d: date, rng: np.random.Generator) -> pl.DataFrame:
    frames = []
    for k in range(4):
        start = datetime(d.year, d.month, d.day, 2 + 5 * k, tzinfo=UTC)
        v = l_route_voyage("x", start, float(rng.uniform(8, 12)), rng)
        frames.append(
            v.with_columns(
                mmsi=pl.lit(316000100 + k + 10 * (d.month == 10)),  # some unseen MMSIs in test
                vessel_name=pl.lit(f"SHIP {k}"),
                vessel_type=pl.lit(70),
                transceiver=pl.lit("A"),
            )
        )
    df = pl.concat(frames).drop("voyage_id")
    missing = {k: pl.lit(None, dtype=t) for k, t in CANONICAL_SCHEMA.items() if k not in df.columns}
    return df.with_columns(**missing).select(
        [pl.col(k).cast(t) for k, t in CANONICAL_SCHEMA.items()]
    )


@pytest.mark.slow
def test_evaluate_and_holdout_end_to_end(tmp_cfg: Config) -> None:
    rng = np.random.default_rng(0)
    for d in DAYS:
        p = interim_path(tmp_cfg, d)
        p.parent.mkdir(parents=True, exist_ok=True)
        _day(d, rng).write_parquet(p)
    save_manifest(
        tmp_cfg, {d.isoformat(): DayRecord(d.isoformat(), "x", 0, 0, 0, "now") for d in DAYS}
    )
    tmp_cfg["tracking"]["workers"] = 1
    tmp_cfg["prediction"]["n_boot"] = 50
    tmp_cfg["prediction"]["knn"] = {"pos_scale_m": [600], "course_scale_deg": [20], "k": [10]}
    tmp_cfg["prediction"]["ml"].update(
        {
            "seeds_gaussian": [0],
            "seeds_mdn": [0],
            "max_epochs": 2,
            "hidden": 8,
            "mlp": 16,
            "threads": 1,
        }
    )
    build.stage(tmp_cfg)
    tracking.stage(tmp_cfg)
    run.evaluate_stage(tmp_cfg)
    run.holdout_stage(tmp_cfg)
    reports = Path(tmp_cfg.paths.reports)
    val_md = (reports / "prediction_val.md").read_text(encoding="utf-8")
    test_md = (reports / "prediction_test.md").read_text(encoding="utf-8")
    assert "B3 kNN route" in val_md
    assert "M1 GRU" in test_md
    assert "Success criteria" in test_md
    log = (reports / "holdout_runs.md").read_text(encoding="utf-8")
    assert log.count("\n| 20") == 1  # exactly one holdout row appended
    fc = pl.read_parquet(Path(tmp_cfg.paths.processed) / "forecasts_test.parquet")
    assert set(fc["model"]) >= {"dead_reckoning", "kf_cv", "imm", "knn_route", "gru", "gru_mdn"}
    assert fc["lat"].is_finite().all()

    import json

    from ais_sentinel.export import web

    tmp_cfg["export"]["n_predictions"] = 3
    web.stage(tmp_cfg)
    site = Path(tmp_cfg.paths.site_data)
    preds = json.loads((site / "predictions.json").read_text(encoding="utf-8"))
    results = json.loads((site / "results.json").read_text(encoding="utf-8"))
    assert preds["split"].startswith("locked test")
    for smp in preds["samples"]:
        assert set(smp["preds"]) >= {"dead_reckoning", "gru"}
        assert len(smp["hist"]["lat"]) > 0
    assert results["prediction"]["rows"]
    assert any(c["area"] == "Prediction" for c in results["criteria"])


@pytest.mark.slow
def test_anomaly_stage_end_to_end(tmp_cfg: Config) -> None:
    from ais_sentinel.anomaly import run as anomaly

    rng = np.random.default_rng(1)
    for d in DAYS:
        p = interim_path(tmp_cfg, d)
        p.parent.mkdir(parents=True, exist_ok=True)
        _day(d, rng).write_parquet(p)
    save_manifest(
        tmp_cfg, {d.isoformat(): DayRecord(d.isoformat(), "x", 0, 0, 0, "now") for d in DAYS}
    )
    tmp_cfg["anomaly"]["eval"].update({"n_per_type": 6, "min_voyage_min": 60})
    build.stage(tmp_cfg)
    anomaly.stage(tmp_cfg)
    md = (Path(tmp_cfg.paths.reports) / "anomaly.md").read_text(encoding="utf-8")
    assert "Success criteria" in md
    assert "| jump |" in md
    assert (Path(tmp_cfg.paths.processed) / "anomalies.parquet").exists()
    import json

    ev = json.loads((Path(tmp_cfg.paths.reports) / "anomaly_eval.json").read_text(encoding="utf-8"))
    kinds = {r["type"] for r in ev["injections"]}
    # Every anomaly type must actually receive injections (an eligibility bug once left
    # gap and deviation with none, silently).
    assert kinds == {"gap", "jump", "loiter", "deviation", "rendezvous"}
