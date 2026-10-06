"""Calibrate the trackers' measurement noise on real data (training split only).

The filters originally assumed 10 m position noise, but real AIS fixes scatter only about
1 m (robust SD) around smoothed tracks. That over-assumption makes both filters
underconfident on real data (DECISIONS D19). This script tries several ``r_pos_m`` values on
a seeded sample of *training* voyages. For each, it measures the share of NIS above
χ²₂(0.95). Fixes that exactly repeat the previous position are excluded: they are a
reporting artefact (mostly moored vessels), not independent measurements. The value
closest to the nominal 5% wins. The validation split (where the criterion is checked) and
the test split are not used.

Writes reports/r_calibration.md and prints the chosen values.
Usage:  .venv\\Scripts\\python scripts\\calibrate_r.py
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import polars as pl

from ais_sentinel.config import load_config
from ais_sentinel.io import write_text
from ais_sentinel.tracking.imm import run_imm
from ais_sentinel.tracking.kf import run_cv_filter
from ais_sentinel.tracking.pipeline import _enu, cv_params_from_config, imm_params_from_config

R_VALUES = (1.5, 2.0, 3.0, 5.0, 10.0)
N_VOYAGES = 400


def frac_above(nis: np.ndarray) -> float:
    v = nis[np.isfinite(nis)]
    return float(np.mean(v > 5.991)) if v.size else float("nan")


def main() -> None:
    cfg = load_config()
    base = Path(cfg.paths.processed)
    voyages = pl.read_parquet(base / "voyages.parquet")
    points = pl.read_parquet(base / "points.parquet")
    ids = voyages.filter((pl.col("split") == "train") & pl.col("usable"))["voyage_id"].to_list()
    rng = np.random.default_rng(int(cfg.seed))
    pick = sorted(rng.choice(len(ids), size=min(N_VOYAGES, len(ids)), replace=False).tolist())
    sample = points.filter(pl.col("voyage_id").is_in([ids[i] for i in pick])).sort("voyage_id", "t")
    voy = sample.partition_by("voyage_id", maintain_order=True)
    cvp, immp = cv_params_from_config(cfg), imm_params_from_config(cfg)
    rows = []
    for r in R_VALUES:
        res = {"r_pos_m": r}
        for name in ("cv", "imm"):
            nis_all, nis_nodup = [], []
            for v in voy:
                z, vel, _, _, _ = _enu(v)
                t = v["t"].dt.epoch("us").to_numpy() / 1e6
                out = (
                    run_cv_filter(t, z, replace(cvp, r_pos_m=r), vel)
                    if name == "cv"
                    else run_imm(t, z, replace(immp, r_pos_m=r), vel)
                )
                ok = out.accepted
                dup = np.concatenate([[False], np.all(z[1:] == z[:-1], axis=1)])
                nis_all.append(out.nis[ok])
                nis_nodup.append(out.nis[ok & ~dup])
            res[f"{name}_all"] = frac_above(np.concatenate(nis_all))
            res[f"{name}_nodup"] = frac_above(np.concatenate(nis_nodup))
        rows.append(res)
        print(res)
    best = {
        name: min(rows, key=lambda x: abs(x[f"{name}_nodup"] - 0.05))["r_pos_m"]
        for name in ("cv", "imm")
    }
    lines = [
        "# Measurement-noise calibration on real AIS (training split)",
        "",
        f"{len(voy)} seeded training voyages. Share of NIS above χ²₂(0.95) (nominal 5%).",
        "'no repeats' excludes fixes identical to the previous one (mostly moored vessels).",
        "",
        "| r_pos_m | CV all | CV no repeats | IMM all | IMM no repeats |",
        "|---:|---:|---:|---:|---:|",
        *[
            f"| {x['r_pos_m']:g} | {100 * x['cv_all']:.1f}% | {100 * x['cv_nodup']:.1f}% | "
            f"{100 * x['imm_all']:.1f}% | {100 * x['imm_nodup']:.1f}% |"
            for x in rows
        ],
        "",
        f"Chosen (closest to 5% without repeats): CV-KF r = {best['cv']:g} m, "
        f"IMM r = {best['imm']:g} m.",
    ]
    write_text(Path(cfg.paths.reports) / "r_calibration.md", "\n".join(lines))
    write_text(
        Path(cfg.paths.reports) / "r_calibration.json",
        json.dumps({"rows": rows, "best": best}, indent=1),
    )
    print(best)


if __name__ == "__main__":
    main()
