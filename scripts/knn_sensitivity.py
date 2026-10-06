"""Validation-only sensitivity check for the kNN baseline's k (run after the pipeline).

The validation grid chose k at its upper edge (80). An under-tuned baseline would flatter
the ML comparison, so this script re-scores larger k on the *validation* split, with the
chosen feature scales fixed. The locked test is not touched. It writes
reports/knn_sensitivity.md.

Usage:  .venv\\Scripts\\python scripts\\knn_sensitivity.py
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import polars as pl

from ais_sentinel.config import load_config
from ais_sentinel.evaluation.metrics import per_sample_scores
from ais_sentinel.io import write_text
from ais_sentinel.prediction.knn import KNNParams, build_library, knn_forecast, region_centre
from ais_sentinel.prediction.run import _inputs, _split_samples


def main() -> None:
    cfg = load_config()
    pc = cfg.prediction
    H = list(pc.horizons_min)
    tracks, voyages, _ = _inputs(cfg)
    val = _split_samples(tracks, voyages, cfg, "val", float(pc.anchor_every_min))
    train_ids = set(voyages.filter(pl.col("split") == "train")["voyage_id"].to_list())
    chosen = json.loads((Path(cfg.paths.models) / "choices.json").read_text(encoding="utf-8"))[
        "knn"
    ]
    base = KNNParams(
        pos_scale_m=float(chosen["pos_scale_m"]), course_scale_deg=float(chosen["course_scale_deg"])
    )
    lib = build_library(tracks, train_ids, *region_centre(dict(cfg.region)), base)
    rows = []
    for k in (40, 80, 160, 320):
        fc, fb = knn_forecast(val, replace(lib, params=replace(base, k=k)), H)
        err = (
            per_sample_scores(fc, val)
            .group_by("horizon_min")
            .agg(pl.col("error_km").mean())
            .sort("horizon_min")
        )
        rows.append({"k": k, **{f"{h} min": float(e) for h, e in err.iter_rows()}, "fallback": fb})
    lines = [
        "# kNN sensitivity to k (validation split only)",
        "",
        f"Scales fixed at the validation choice ({base.pos_scale_m:.0f} m, "
        f"{base.course_scale_deg:.0f}°). Mean error in km; the chosen k was 80 (upper edge of "
        "the tuning grid).",
        "",
        "| k | " + " | ".join(f"{h} min" for h in H) + " | DR fallback |",
        "|---:|" + "---:|" * len(H) + "---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['k']} | "
            + " | ".join(f"{r[f'{h} min']:.3f}" for h in H)
            + f" | {100 * r['fallback']:.1f}% |"
        )
    write_text(Path(cfg.paths.reports) / "knn_sensitivity.md", "\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
