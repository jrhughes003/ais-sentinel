"""End-to-end pipeline on a tiny synthetic dataset: build -> track -> export."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from ais_sentinel.config import Config
from ais_sentinel.data import build
from ais_sentinel.data.download import DayRecord, interim_path, save_manifest
from ais_sentinel.data.schema import CANONICAL_SCHEMA
from ais_sentinel.export import web
from ais_sentinel.geo import enu_to_latlon
from ais_sentinel.tracking import pipeline as tracking

DAY = date(2023, 7, 1)


def synthetic_day(rng: np.random.Generator) -> pl.DataFrame:
    """Two cargo ships crossing the AOI (one with a 45-min silence) and one moored tug."""
    rows = []
    t0 = datetime(2023, 7, 1, 8, tzinfo=UTC)
    for mmsi, name, vt, skip in (
        (316000001, "LAKER ONE", 70, None),
        (366000002, "SALTIE", 70, (60, 105)),
    ):
        e = np.arange(240) * 300.0  # ~10 kn east for 4 h
        n = 200 * np.sin(np.arange(240) / 40)
        lat, lon = enu_to_latlon(e + rng.normal(0, 8, 240), n + rng.normal(0, 8, 240), 42.3, -83.4)
        for i in range(240):
            if skip and skip[0] <= i < skip[1]:
                continue
            rows.append((mmsi, t0 + timedelta(minutes=i), lat[i], lon[i], 9.7, 88.0, name, vt, "A"))
    for i in range(0, 240, 3):
        rows.append(
            (
                367000003,
                t0 + timedelta(minutes=i),
                42.33 + rng.normal(0, 1e-5),
                -83.05,
                0.0,
                None,
                "TUG",
                31,
                "A",
            )
        )
    df = pl.DataFrame(
        rows,
        schema=[
            "mmsi",
            "t",
            "lat",
            "lon",
            "sog_kn",
            "cog_deg",
            "vessel_name",
            "vessel_type",
            "transceiver",
        ],
        orient="row",
    )
    missing = {
        k: pl.lit(None, dtype=v).alias(k)
        for k, v in CANONICAL_SCHEMA.items()
        if k not in df.columns
    }
    return df.with_columns(**missing).select(
        [pl.col(k).cast(v) for k, v in CANONICAL_SCHEMA.items()]
    )


def test_build_track_export(tmp_cfg: Config) -> None:
    tmp_cfg["export"]["n_showcase"] = 2
    path = interim_path(tmp_cfg, DAY)
    path.parent.mkdir(parents=True)
    synthetic_day(np.random.default_rng(0)).write_parquet(path)
    save_manifest(tmp_cfg, {DAY.isoformat(): DayRecord(DAY.isoformat(), "x", 0, 0, 0, "now")})

    build.stage(tmp_cfg)
    voyages = pl.read_parquet(Path(tmp_cfg.paths.processed) / "voyages.parquet")
    # The 45-minute silence splits the second ship into two voyages.
    assert voyages.filter(pl.col("mmsi") == 366000002).height == 2
    assert set(voyages["split"]) == {"train"}
    assert "Usable voyages by split" in (Path(tmp_cfg.paths.reports) / "data_summary.md").read_text(
        encoding="utf-8"
    )

    tracking.stage(tmp_cfg)
    tracks = pl.read_parquet(Path(tmp_cfg.paths.processed) / "tracks.parquet")
    laker = tracks.filter(pl.col("mmsi") == 316000001)
    # Gentle 8 m noise on a smooth path: almost every fix passes the outlier gate.
    assert laker["accepted"].mean() > 0.97  # type: ignore[operator]
    assert np.all(np.isfinite(laker["s_lat"].to_numpy()))

    web.stage(tmp_cfg)
    site = Path(tmp_cfg.paths.site_data)
    tj = json.loads((site / "tracks.json").read_text(encoding="utf-8"))
    aj = json.loads((site / "anomalies.json").read_text(encoding="utf-8"))
    assert "CC0" in tj["attribution"]
    assert {v["group"] for v in tj["voyages"]} == {"cargo"}
    v0 = tj["voyages"][0]
    assert len(v0["lat"]) == len(v0["s_lat"]) == len(v0["dt"])
    assert [e["mmsi"] for e in aj["events"]] == [366000002]
    assert aj["events"][0]["gap_min"] == 46.0
