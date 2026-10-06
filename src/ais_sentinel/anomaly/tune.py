"""Tune anomaly-detector settings on the **validation** month (v2, DECISIONS D21).

The candidate settings for the gap and route-deviation rules are scored by synthetic
injection into September commercial voyages, together with each candidate's real-data base
alarm rate on September. The locked test period is never used here.

Selection rule, the same for every rule:
- maximise recall in the PLAN §9.4 target bucket;
- subject to precision ≥ 0.9 on injected tests;
- and subject to a commercial base alarm rate no higher than the v1 settings produced on
  the same September data.

A candidate cannot win simply by alarming more.
"""

from __future__ import annotations

import copy
import itertools
import json
import logging
from pathlib import Path
from typing import Any

import polars as pl

from ais_sentinel.anomaly.context import learn_context
from ais_sentinel.anomaly.detectors import learn_vessel_homes
from ais_sentinel.anomaly.run import Detectors, _pr, base_alarm_rate, injection_study
from ais_sentinel.config import Config
from ais_sentinel.data.download import complete_days
from ais_sentinel.data.vessel_types import COMMERCIAL
from ais_sentinel.io import write_text
from ais_sentinel.prediction.knn import region_centre

log = logging.getLogger(__name__)

GAP_GRID: dict[str, list[Any]] = {"min_reception": [0.5, 0.6, 0.7, 0.8]}
DEV_GRID: dict[str, list[Any]] = {
    "traffic_groups": ["all", "commercial"],
    "max_cell_voyages": [2, 5],
    "bridge_min": [0.0, 3.0],
}
V1: dict[str, dict[str, Any]] = {
    "gap": {"min_reception": 0.8},
    "deviation": {"traffic_groups": "all", "max_cell_voyages": 2, "bridge_min": 0.0},
}


def _candidates(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    return [dict(zip(grid, vals, strict=True)) for vals in itertools.product(*grid.values())]


def _cfg_with(cfg: Config, kind: str, cand: dict[str, Any], skip_fishing: bool = True) -> Config:
    c = copy.deepcopy(cfg)
    if kind == "gap":
        c["anomaly"]["gap"].update(cand)
    else:
        c["anomaly"]["deviation"].update(
            {
                "max_cell_voyages": cand["max_cell_voyages"],
                "bridge_min": cand["bridge_min"],
                "skip_groups": ["fishing"] if skip_fishing else [],
            }
        )
        c["anomaly"]["context"]["traffic_groups"] = (
            sorted(COMMERCIAL) if cand["traffic_groups"] == "commercial" else None
        )
    return c


def _score(
    cfg: Config,
    kind: str,
    points: pl.DataFrame,
    voyages: pl.DataFrame,
    n: int,
    ctx_cache: dict[str, Any],
) -> dict[str, Any]:
    """Precision/recall on September injections plus the September commercial base rate."""
    ref = region_centre(dict(cfg.region))
    key = "commercial" if cfg.anomaly["context"].get("traffic_groups") else "all"
    if key not in ctx_cache:
        train = points.filter(pl.col("split") == "train")
        ctx = learn_context(train, *ref, **dict(cfg.anomaly["context"]))
        ctx_cache[key] = (ctx, learn_vessel_homes(train, ctx))
    ctx, homes = ctx_cache[key]
    days, _ = complete_days(cfg)
    det = Detectors(cfg, ctx, homes, available_days=days or None)
    res = injection_study(
        det,
        points,
        voyages,
        n,
        int(cfg.seed),
        float(cfg.anomaly["eval"].get("min_voyage_min", 120)),
        split="val",
        types=(kind,),
    )
    p, r, m = _pr([x for x in res if x["in_target_bucket"]])
    vp = points.filter(
        (pl.col("split") == "val") & pl.col("vessel_group").is_in(sorted(COMMERCIAL))
    ).drop("voyage_id", "split")
    alarms = base_alarm_rate(det.run(kind, vp), vp)
    hours = (
        vp.sort("mmsi", "t")
        .with_columns(
            dt=pl.col("t").diff().over("mmsi").dt.total_seconds().fill_null(0).clip(0, 1800)
        )["dt"]
        .sum()
        / 3600
    )
    rate = 1000 * float(alarms["len"].sum()) / max(float(hours), 1e-9)
    return {"precision": p, "recall": r, "n": m, "base_rate_per_1000h": rate}


def stage(cfg: Config) -> None:
    """CLI stage: score candidate gap/deviation settings on September; write a report."""
    base = Path(cfg.paths.processed)
    points = pl.read_parquet(base / "points.parquet").filter(pl.col("voyage_id").is_not_null())
    points = points.select(
        "mmsi", "t", "lat", "lon", "sog_kn", "cog_deg", "vessel_group", "voyage_id", "split"
    )
    voyages = pl.read_parquet(base / "voyages.parquet")
    n = int(cfg.anomaly.get("tune", {}).get("n_per_type", 150))
    out: dict[str, Any] = {}
    lines = ["# Anomaly-detector tuning on the validation month (September 2023)", ""]
    for kind, grid in (("gap", GAP_GRID), ("deviation", DEV_GRID)):
        cache: dict[str, Any] = {}
        v1_score = _score(
            _cfg_with(cfg, kind, V1[kind], skip_fishing=False), kind, points, voyages, n, cache
        )
        rows = []
        for cand in _candidates(grid):
            sc = _score(_cfg_with(cfg, kind, cand), kind, points, voyages, n, cache)
            ok = (
                sc["precision"] >= 0.9
                and sc["base_rate_per_1000h"] <= v1_score["base_rate_per_1000h"] + 1e-9
            )
            rows.append({"params": cand, **sc, "eligible": ok})
            log.info("tune %s %s -> %s", kind, cand, sc)
        eligible = [r for r in rows if r["eligible"]] or [
            {"params": V1[kind], **v1_score, "eligible": True}
        ]
        best = max(eligible, key=lambda r: (r["recall"], -r["base_rate_per_1000h"]))
        out[kind] = {"v1": {"params": V1[kind], **v1_score}, "candidates": rows, "chosen": best}
        lines += [
            f"## {kind}",
            "",
            f"v1 settings `{json.dumps(V1[kind])}`: P {v1_score['precision']:.2f} / "
            f"R {v1_score['recall']:.2f}, base rate {v1_score['base_rate_per_1000h']:.2f} alarms "
            "per 1,000 commercial vessel-hours.",
            "",
            "| settings | precision | recall | base rate /1000 h | eligible |",
            "|---|---:|---:|---:|---|",
            *[
                f"| `{json.dumps(r['params'])}` | {r['precision']:.2f} | {r['recall']:.2f} | "
                f"{r['base_rate_per_1000h']:.2f} | {'yes' if r['eligible'] else 'no'} |"
                for r in rows
            ],
            "",
            f"**Chosen:** `{json.dumps(best['params'])}`.",
            "",
        ]
    rep = Path(cfg.paths.reports)
    write_text(rep / "anomaly_tuning.json", json.dumps(out, indent=1, default=str))
    write_text(rep / "anomaly_tuning.md", "\n".join(lines))
    log.info(
        "anomaly-tune: chosen gap %s, deviation %s",
        out["gap"]["chosen"]["params"],
        out["deviation"]["chosen"]["params"],
    )
