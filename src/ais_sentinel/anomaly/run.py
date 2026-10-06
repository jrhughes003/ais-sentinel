"""Anomaly stage: learn context, detect on real data, evaluate by synthetic injection.

Outputs:

* ``data/processed/anomalies.parquet``: every detected event in the whole season.
* ``reports/anomaly.md`` and ``reports/anomaly_eval.json``: precision and recall per type
  and magnitude, the base alarm rate on unmodified test data, and checks against the
  PLAN §9.4 criteria.

**Scoring injected anomalies** (PLAN §6.4):

* Detectors run on the vessel's voyage *before* and *after* injection. Only detections that
  are new after injection count, because pre-existing alarms belong to the real data and
  are reported separately as the base alarm rate.
* A new detection *matches* the injection if it has the same type and overlaps the
  injected time span by at least 50% of the shorter of the two spans. Jumps also match
  within ±2 min, because they are nearly instantaneous.
* **Recall** = matched injections / injections.
* **Precision** = matching new detections / all new detections of that type.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from ais_sentinel.anomaly.context import Context, learn_context
from ais_sentinel.anomaly.detectors import (
    VesselHome,
    detect_jumps,
    detect_loitering,
    detect_rendezvous,
    detect_route_deviation,
    empty_events,
    learn_vessel_homes,
)
from ais_sentinel.anomaly.gaps import detect_gaps, edge_distance_km
from ais_sentinel.anomaly.inject import (
    Injection,
    inject_deviation,
    inject_gap,
    inject_jump,
    inject_loiter,
    inject_rendezvous,
)
from ais_sentinel.config import Config
from ais_sentinel.data.vessel_types import COMMERCIAL
from ais_sentinel.io import write_text
from ais_sentinel.prediction.knn import region_centre

log = logging.getLogger(__name__)
TYPES = ("gap", "jump", "loiter", "deviation", "rendezvous")
MAGNITUDES: dict[str, list[float]] = {
    "gap": [20, 30, 45, 60, 90, 180],  # minutes
    "jump": [0.5, 1, 2, 3, 5, 10, 30],  # km
    "loiter": [1, 1.5, 2, 3, 4, 6],  # hours
    "deviation": [0.5, 1, 1.5, 2, 3, 5],  # km peak offset
    "rendezvous": [0.5, 1, 1.5, 2, 3],  # hours
}
# Magnitude buckets the PLAN §9.4 targets apply to ("mid-to-large").
TARGETS: dict[str, tuple[float, float, Callable[[Injection], bool]]] = {
    "gap": (0.90, 0.90, lambda j: j.magnitude >= 45),
    "jump": (0.90, 0.90, lambda j: j.magnitude >= 3),
    "loiter": (0.80, 0.80, lambda j: j.magnitude >= 3),
    "rendezvous": (0.80, 0.80, lambda j: j.magnitude >= 1.5),
    "deviation": (
        0.70,
        0.70,
        lambda j: j.magnitude >= 1.5 and (j.duration_min or 0) >= 20,
    ),
}


class Detectors:
    """Bundle of configured detectors sharing the learned context."""

    def __init__(self, cfg: Config, ctx: Context, homes: VesselHome) -> None:
        self.cfg, self.ctx, self.homes = cfg, ctx, homes
        self.a = cfg.anomaly

    def run(self, kind: str, pts: pl.DataFrame) -> pl.DataFrame:
        """Run one detector type on points."""
        a = self.a
        if pts.is_empty():
            return empty_events()
        if kind == "gap":
            g = a["gap"]
            return detect_gaps(
                pts,
                dict(self.cfg.region),
                float(g["min_gap_min"]),
                float(g["min_sog_kn"]),
                float(g["edge_buffer_km"]),
                ctx=self.ctx,
                min_reception=float(g["min_reception"]),
                max_gap_h=float(g.get("max_gap_h", 24.0)),
                home_cells=self.homes.cells,
            )
        if kind == "jump":
            return detect_jumps(pts, **dict(a["jump"]))
        if kind == "loiter":
            return detect_loitering(pts, self.ctx, self.homes, **dict(a["loiter"]))
        if kind == "deviation":
            return detect_route_deviation(pts, self.ctx, **dict(a["deviation"]))
        if kind == "rendezvous":
            return detect_rendezvous(pts, self.ctx, self.homes, **dict(a["rendezvous"]))
        raise ValueError(kind)

    def all(self, pts: pl.DataFrame) -> pl.DataFrame:
        """Run every detector."""
        return pl.concat([self.run(k, pts) for k in TYPES])


def _overlap_frac(a0: datetime, a1: datetime, b0: datetime, b1: datetime) -> float:
    inter = (min(a1, b1) - max(a0, b0)).total_seconds()
    shorter = max(min((a1 - a0).total_seconds(), (b1 - b0).total_seconds()), 60.0)
    return max(inter, 0.0) / shorter


def _matches(ev: dict[str, Any], inj: Injection) -> bool:
    if inj.type == "jump":
        pad = timedelta(minutes=2)
        return ev["t_start"] - pad <= inj.t_end and ev["t_end"] + pad >= inj.t_start
    return _overlap_frac(ev["t_start"], ev["t_end"], inj.t_start, inj.t_end) >= 0.5


def _new_events(before: pl.DataFrame, after: pl.DataFrame) -> list[dict[str, Any]]:
    """Events in ``after`` that do not overlap any event of ``before``."""
    old = before.to_dicts()
    out = []
    for e in after.to_dicts():
        if not any(
            _overlap_frac(e["t_start"], e["t_end"], o["t_start"], o["t_end"]) > 0.0 for o in old
        ):
            out.append(e)
    return out


def _eligible_t0(
    v: pl.DataFrame,
    kind: str,
    span: timedelta,
    ctx: Context,
    region: dict[str, float],
    rng: np.random.Generator,
) -> datetime | None:
    """Pick an injection time valid for this anomaly type (or None)."""
    t = v["t"].to_list()
    sog = v["sog_kn"].cast(pl.Float64).fill_null(0.0).to_numpy()
    lat, lon = v["lat"].to_numpy(), v["lon"].to_numpy()
    edge = edge_distance_km(
        pl.col("lat"),
        pl.col("lon"),
        **{k: region[k] for k in ("lat_min", "lat_max", "lon_min", "lon_max")},
    )
    edge_km = v.select(edge.alias("e"))["e"].to_numpy()
    cand = []
    for i in range(5, len(t) - 5):
        t0 = t[i]
        if t0 + span + timedelta(minutes=15) > t[-1]:
            break
        if kind in ("gap", "deviation"):
            win = (v["t"] >= t0 - timedelta(minutes=5)) & (
                v["t"] <= t0 + span + timedelta(minutes=5)
            )
            w = win.to_numpy()
            # (Not .min(initial=0): numpy treats `initial` as an extra element.)
            if not w.any() or sog[w].min() < 3.0 or edge_km[w].min() < 5.0:
                continue
        elif kind in ("loiter", "rendezvous"):
            if sog[i] < 2.0 or ctx.in_port(lat[i : i + 1], lon[i : i + 1])[0] or edge_km[i] < 3.0:
                continue
        elif sog[i] < 2.0:
            continue
        cand.append(t0)
    if not cand:
        return None
    return cand[int(rng.integers(len(cand)))]


def injection_study(
    det: Detectors,
    points: pl.DataFrame,
    voyages: pl.DataFrame,
    n_per_type: int,
    seed: int,
    min_voyage_min: float = 120.0,
) -> list[dict[str, Any]]:
    """Inject anomalies into test-period commercial voyages and score the detectors."""
    rng = np.random.default_rng(seed)
    region = dict(det.cfg.region)
    pool = voyages.filter(
        (pl.col("split") == "test")
        & pl.col("usable")
        & pl.col("vessel_group").is_in(sorted(COMMERCIAL))
        & (pl.col("moving_frac") >= 0.5)
        & (pl.col("duration_min") >= min_voyage_min)
    )["voyage_id"].to_list()
    if not pool:
        raise ValueError("no eligible test voyages for injection")
    by_voyage = {
        v["voyage_id"][0]: v.sort("t")
        for v in points.filter(pl.col("voyage_id").is_in(pool)).partition_by("voyage_id")
    }
    results: list[dict[str, Any]] = []
    for kind in TYPES:
        done = attempts = 0
        while done < n_per_type and attempts < n_per_type * 20:
            attempts += 1
            vid = pool[int(rng.integers(len(pool)))]
            v = by_voyage[vid]
            mag = float(rng.choice(MAGNITUDES[kind]))
            minutes = float(rng.uniform(15, 60)) if kind == "deviation" else 0.0
            span = {
                "gap": timedelta(minutes=mag),
                "jump": timedelta(minutes=5),
                "loiter": timedelta(minutes=10),
                "rendezvous": timedelta(minutes=10),
                "deviation": timedelta(minutes=minutes),
            }[kind]
            t0 = _eligible_t0(v, kind, span, det.ctx, region, rng)
            if t0 is None:
                continue
            if kind == "gap":
                mod, inj = inject_gap(v, t0, mag)
            elif kind == "jump":
                mod, inj = inject_jump(v, t0, mag, int(rng.integers(1, 6)), rng)
            elif kind == "loiter":
                mod, inj = inject_loiter(v, t0, mag, rng)
            elif kind == "deviation":
                mod, inj = inject_deviation(v, t0, mag, minutes, rng)
            else:
                mod, inj = inject_rendezvous(v, t0, mag, 999_000_000 + done, rng)
            before = det.run(kind, v.select(mod.columns))
            after = det.run(kind, mod)
            new = _new_events(before, after)
            hits = [e for e in new if _matches(e, inj)]
            results.append(
                {
                    "type": kind,
                    "voyage_id": vid,
                    "magnitude": mag,
                    "duration_min": inj.duration_min,
                    "detected": bool(hits),
                    "new_events": len(new),
                    "matched_events": len(hits),
                    "in_target_bucket": TARGETS[kind][2](inj),
                }
            )
            done += 1
        log.info("injection %s: %d injections (%d attempts)", kind, done, attempts)
    return results


def _pr(rows: list[dict[str, Any]]) -> tuple[float, float, int]:
    n = len(rows)
    if n == 0:
        return float("nan"), float("nan"), 0
    recall = sum(r["detected"] for r in rows) / n
    new = sum(r["new_events"] for r in rows)
    precision = sum(r["matched_events"] for r in rows) / new if new else float("nan")
    return precision, recall, n


def base_alarm_rate(events: pl.DataFrame, points: pl.DataFrame) -> pl.DataFrame:
    """Alarms per 1,000 vessel-hours by type and vessel group on unmodified data."""
    p = points.sort("mmsi", "t").with_columns(
        dt=(pl.col("t").diff().over("mmsi").dt.total_seconds().fill_null(0)).clip(0, 1800)
    )
    hours = p.group_by("vessel_group").agg(vessel_hours=pl.col("dt").sum() / 3600)
    counts = events.group_by("type", "vessel_group").len()
    return (
        counts.join(hours, on="vessel_group", how="left")
        .with_columns(per_1000h=1000 * pl.col("len") / pl.col("vessel_hours"))
        .sort("type", "vessel_group")
    )


def report_markdown(
    results: list[dict[str, Any]], alarms: pl.DataFrame, events: pl.DataFrame
) -> tuple[str, list[dict[str, Any]]]:
    """Render the anomaly evaluation report. Returns (markdown, criteria rows)."""
    lines = [
        "# Anomaly detection: synthetic-injection evaluation",
        "",
        "Real October 2023 (locked test period) commercial voyages with one injected anomaly "
        "each. Precision counts only detections that were *not* present before injection; "
        "see `ais_sentinel/anomaly/run.py` for the matching rule.",
        "",
        "## By type and magnitude",
        "",
        "| type | magnitude | n | precision | recall |",
        "|---|---:|---:|---:|---:|",
    ]
    for kind in TYPES:
        for mag in MAGNITUDES[kind]:
            rows = [r for r in results if r["type"] == kind and r["magnitude"] == mag]
            pr, rc, n = _pr(rows)
            if n:
                lines.append(f"| {kind} | {mag:g} | {n} | {pr:.2f} | {rc:.2f} |")
    lines += [
        "",
        "Magnitude units: gap = minutes of deleted reports, jump = km displacement, loiter and "
        "rendezvous = hours, deviation = km peak sideways offset.",
        "",
        "## Success criteria (PLAN §9.4, set before results)",
        "",
        "| type | bucket | n | precision (target) | recall (target) | met? |",
        "|---|---|---:|---|---|---|",
    ]
    bucket_txt = {
        "gap": "≥ 45 min",
        "jump": "≥ 3 km",
        "loiter": "≥ 3 h",
        "rendezvous": "≥ 1.5 h",
        "deviation": "≥ 1.5 km for ≥ 20 min",
    }
    criteria = []
    for kind in TYPES:
        p_t, r_t, _ = TARGETS[kind]
        pr, rc, n = _pr([r for r in results if r["type"] == kind and r["in_target_bucket"]])
        met = bool(n and pr >= p_t and rc >= r_t)
        criteria.append({"type": kind, "precision": pr, "recall": rc, "n": n, "met": met})
        lines.append(
            f"| {kind} | {bucket_txt[kind]} | {n} | {pr:.2f} (≥ {p_t:.2f}) "
            f"| {rc:.2f} (≥ {r_t:.2f}) | {'✅' if met else '❌'} |"
        )
    lines += [
        "",
        "## Base alarm rate on unmodified test data (alarms per 1,000 vessel-hours)",
        "",
        "Some of these may be real anomalies; they are reported, not treated as errors.",
        "",
        "| type | vessel group | alarms | vessel-hours | per 1,000 h |",
        "|---|---|---:|---:|---:|",
    ]
    for r in alarms.iter_rows(named=True):
        lines.append(
            f"| {r['type']} | {r['vessel_group']} | {r['len']} | {r['vessel_hours']:,.0f} "
            f"| {r['per_1000h']:.2f} |"
        )
    lines += ["", f"Total events detected across the season: {events.height:,}.", ""]
    return "\n".join(lines), criteria


def stage(cfg: Config) -> None:
    """CLI stage: learn context, detect on all data, run the injection study, write reports."""
    base = Path(cfg.paths.processed)
    points = pl.read_parquet(base / "points.parquet").filter(pl.col("voyage_id").is_not_null())
    voyages = pl.read_parquet(base / "voyages.parquet")
    ref = region_centre(dict(cfg.region))
    train_pts = points.filter(pl.col("split") == "train")
    ctx = learn_context(train_pts, *ref, **dict(cfg.anomaly["context"]))
    homes = learn_vessel_homes(train_pts, ctx)
    det = Detectors(cfg, ctx, homes)
    pts = points.select(
        "mmsi", "t", "lat", "lon", "sog_kn", "cog_deg", "vessel_group", "voyage_id", "split"
    )
    events = det.all(pts.drop("voyage_id", "split"))
    events.write_parquet(base / "anomalies.parquet")
    test_pts = pts.filter(pl.col("split") == "test").drop("voyage_id", "split")
    alarms = base_alarm_rate(det.all(test_pts), test_pts)
    results = injection_study(
        det, pts, voyages, int(cfg.anomaly["eval"]["n_per_type"]), int(cfg.seed)
    )
    md, criteria = report_markdown(results, alarms, events)
    write_text(Path(cfg.paths.reports) / "anomaly.md", md)
    write_text(
        Path(cfg.paths.reports) / "anomaly_eval.json",
        json.dumps(
            {"criteria": criteria, "injections": results, "alarms": alarms.to_dicts()},
            indent=1,
            default=str,
        ),
    )
    log.info(
        "anomaly: %d events; criteria %s", events.height, [(c["type"], c["met"]) for c in criteria]
    )
