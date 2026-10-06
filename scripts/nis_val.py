"""Real-data NIS consistency on the validation split with the current tracking config.

Re-filters validation voyages only and writes reports/tracking_nis_attempt4.json. It does
NOT overwrite data/processed/tracks.parquet or filter_states.parquet, which the locked
prediction test and the website were produced with (attempt-1 tracker, DECISIONS D19/D20).

Usage:  .venv\\Scripts\\python scripts\\nis_val.py
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from ais_sentinel.config import load_config
from ais_sentinel.io import write_text
from ais_sentinel.tracking.pipeline import (
    cv_params_from_config,
    imm_params_from_config,
    nis_consistency,
    track_all,
)


def main() -> None:
    cfg = load_config()
    base = Path(cfg.paths.processed)
    voyages = pl.read_parquet(base / "voyages.parquet")
    points = pl.read_parquet(base / "points.parquet")
    val_ids = voyages.filter(pl.col("split") == "val")["voyage_id"].implode()
    pts = points.filter(pl.col("voyage_id").is_in(val_ids))
    pc = cfg.prediction
    tracks, _ = track_all(
        pts,
        cv_params_from_config(cfg),
        imm_params_from_config(cfg),
        float(pc.history_min),
        float(pc.anchor_every_min),
        workers=int(cfg.tracking.get("workers", 4)),
    )
    res = nis_consistency(tracks, voyages)
    dup = tracks.sort("voyage_id", "t").with_columns(
        rep=(pl.col("lat") == pl.col("lat").shift(1).over("voyage_id"))
        & (pl.col("lon") == pl.col("lon").shift(1).over("voyage_id"))
    )
    nodup = dup.filter(~pl.col("rep").fill_null(False) & pl.col("accepted"))["nis"].drop_nans()
    res["imm_no_repeats_frac_above_95"] = float((nodup > 5.991).mean())  # type: ignore[arg-type]
    write_text(Path(cfg.paths.reports) / "tracking_nis_attempt4.json", json.dumps(res, indent=1))
    print(res["criterion"], res["imm_no_repeats_frac_above_95"])


if __name__ == "__main__":
    main()
