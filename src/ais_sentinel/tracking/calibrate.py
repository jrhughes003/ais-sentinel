"""Calibrate the tracker and its simulator to real AIS (training split only).

Tracking attempts 1–2 tuned the filters inside a simulator whose manoeuvres were much
harsher than real ships' (turn rates uniform on 0.05–1.0 °/s). The result: filters that are
underconfident on real data (DECISIONS D19). This stage fixes that in two steps, both using
the **training split** only.

1. **Fit the simulator to real behaviour.** It measures speeds, turn rates, accelerations,
   the share of time spent turning, reporting intervals and outlier rates from the attempt-1
   smoothed training tracks, and maps them to a :class:`SimConfig`. Turn rates and
   accelerations are heavy-tailed (mostly gentle, occasionally sharp), so they are drawn
   log-uniformly between bounds taken from their quantiles. ``p_turn`` is chosen by a small
   search so that simulated ships turn for the same share of time as real ones.

2. **Tune both filters by one-step predictive likelihood on real data.** This is the
   standard *prediction-error* (innovations) method for estimating noise parameters when
   there is no ground truth. A filter whose noise model is right makes each next fix as
   probable as possible. Each fix's log-likelihood is mixed with the same uniform outlier
   density used by the IMM, ``log(L + λ)``, so a rare gross outlier cannot dominate the
   score. Fixes that exactly repeat the previous position are skipped: they are a reporting
   artefact (mostly moored vessels), not new measurements. Both filters get the same
   objective, sample and grid budget.

The chosen parameters are then evaluated *unchanged* on fresh seeds of the calibrated
simulator (``sim-study`` in ``fixed`` mode) and against the real-data NIS criterion.
"""

from __future__ import annotations

import itertools
import json
import logging
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from ais_sentinel.config import Config
from ais_sentinel.data.vessel_types import COMMERCIAL
from ais_sentinel.geo import KNOT_MS
from ais_sentinel.io import write_text
from ais_sentinel.tracking.imm import run_imm
from ais_sentinel.tracking.kf import run_cv_filter
from ais_sentinel.tracking.pipeline import _enu, cv_params_from_config, imm_params_from_config
from ais_sentinel.tracking.simulate import SimConfig, simulate_track

log = logging.getLogger(__name__)
TURN_THRESHOLD_DEG_S = 0.05

# Grid history (DECISIONS D20): the first grid's optimum sat on its low edges, and so did an
# extended one. That exposed a loophole in the objective (restart fixes were not scored),
# not a real preference for stiffer filters. With restarts scored, the grid spans both
# directions. Both filters picked gap_q = 0 on real data.
CV_GRID: dict[str, list[Any]] = {
    "q_accel": [3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2],
    "r_pos_m": [1.0, 1.5, 3.0, 5.0],
    "gap_q": [0.0, 0.05],
}
IMM_GRID: dict[str, list[Any]] = {
    "q_cruise": [1e-5, 3e-5, 1e-4, 5e-4, 2e-3],
    "q_turn": [0.005, 0.02, 0.05],
    "q_omega": [2e-8, 2e-7],
    "r_pos_m": [0.75, 1.5, 3.0],
    "gap_q": [0.0],
}


# --------------------------------------------------------------------------- statistics


def manoeuvre_stats(tracks: pl.DataFrame, voyages: pl.DataFrame) -> dict[str, float]:
    """Real manoeuvre and reporting statistics of moving commercial training vessels."""
    ids = voyages.filter(
        (pl.col("split") == "train") & pl.col("vessel_group").is_in(sorted(COMMERCIAL))
    )["voyage_id"].implode()
    t = tracks.filter(pl.col("voyage_id").is_in(ids)).sort("voyage_id", "t")
    ts = t["t"].dt.epoch("us").to_numpy() / 1e6
    vid = t["voyage_id"].to_numpy()
    ve, vn = t["s_ve"].to_numpy(), t["s_vn"].to_numpy()
    spd = np.hypot(ve, vn)
    hdg = np.arctan2(ve, vn)
    same = np.r_[False, vid[1:] == vid[:-1]]
    dt = np.r_[np.nan, np.diff(ts)]
    moving = spd > 1.5
    ok = same & (dt > 20) & (dt <= 130) & moving & np.r_[False, moving[:-1]]
    dh = np.angle(np.exp(1j * (hdg - np.r_[0.0, hdg[:-1]])))
    w = np.degrees(np.abs(dh[ok]) / dt[ok])
    acc = np.abs((spd - np.r_[0.0, spd[:-1]])[ok] / dt[ok])
    d_moving = dt[same & moving]
    rejected = 1 - float(t["accepted"].mean())  # type: ignore[arg-type]
    return {
        "steps": int(ok.sum()),
        "speed_p5_kn": float(np.percentile(spd[moving], 5) / KNOT_MS),
        "speed_p95_kn": float(np.percentile(spd[moving], 95) / KNOT_MS),
        "turn_share": float(np.mean(w > TURN_THRESHOLD_DEG_S)),
        "turn_p99_deg_s": float(np.percentile(w, 99)),
        "accel_p90": float(np.percentile(acc, 90)),
        "accel_p99": float(np.percentile(acc, 99)),
        "report_median_s": float(np.median(d_moving)),
        "gap_share_180s": float(np.mean(d_moving > 180)),
        "rejected_share": rejected,
    }


def _sim_turn_share(cfg: SimConfig, seeds: range) -> float:
    shares = []
    for s in seeds:
        tr = simulate_track(s, cfg)
        mv = tr.segment != "stopped"
        shares.append(np.mean(np.abs(np.degrees(tr.omega[mv])) > TURN_THRESHOLD_DEG_S))
    return float(np.mean(shares))


def fit_sim_config(st: dict[str, float], pos_sigma_m: float) -> tuple[SimConfig, float]:
    """Map real statistics to a simulator config (returns it and the achieved turn share)."""
    base = SimConfig(
        speed_kn=(round(st["speed_p5_kn"], 1), round(st["speed_p95_kn"], 1)),
        turn_rate_deg_s=(TURN_THRESHOLD_DEG_S, round(max(st["turn_p99_deg_s"], 0.1), 3)),
        turn_angle_deg=(15.0, 90.0),
        accel_ms2=(round(st["accel_p90"], 4), round(max(st["accel_p99"], st["accel_p90"] * 2), 4)),
        report_s=round(st["report_median_s"], 0),
        p_drop=0.02,
        p_burst=round(max(st["gap_share_180s"], 1e-4), 4),
        pos_sigma_m=pos_sigma_m,
        p_outlier=round(max(st["rejected_share"], 5e-4), 4),
        log_uniform=True,
    )
    best, best_err = base, np.inf
    achieved = float("nan")
    for p_turn in (0.1, 0.15, 0.2, 0.3, 0.45, 0.6):
        c = replace(base, p_turn=p_turn)
        share = _sim_turn_share(c, range(500, 520))
        if abs(share - st["turn_share"]) < best_err:
            best, best_err, achieved = c, abs(share - st["turn_share"]), share
    return best, achieved


# --------------------------------------------------------------------------- likelihood


def _voyage_arrays(points: pl.DataFrame) -> list[tuple[np.ndarray, ...]]:
    out: list[tuple[np.ndarray, ...]] = []
    for v in points.partition_by("voyage_id", maintain_order=True):
        z, vel, _, _, _ = _enu(v)
        t = v["t"].dt.epoch("us").to_numpy() / 1e6
        rep = np.r_[False, np.all(z[1:] == z[:-1], axis=1)]
        out.append((t, z, vel, rep))
    return out


def _score(args: tuple[str, Any, list[tuple[np.ndarray, ...]], float]) -> float:
    """Mean clutter-robust one-step log-likelihood over non-repeated fixes."""
    kind, params, voyages, clutter = args
    total, n = 0.0, 0
    log_c = np.log(clutter)
    for t, z, vel, rep in voyages:
        res = run_cv_filter(t, z, params, vel) if kind == "cv" else run_imm(t, z, params, vel)
        ll = res.loglik
        m = np.isfinite(ll) & ~rep
        total += float(np.logaddexp(ll[m], log_c).sum())
        n += int(m.sum())
    return total / max(n, 1)


def tune_on_real(
    kind: str,
    base: Any,
    grid: dict[str, list[Any]],
    voyages: list[tuple[np.ndarray, ...]],
    clutter: float,
    workers: int,
) -> tuple[Any, list[dict[str, Any]]]:
    """Grid search maximising predictive log-likelihood (both filters, same objective)."""
    keys = list(grid)
    cands = [
        replace(base, **dict(zip(keys, vals, strict=True)))
        for vals in itertools.product(*grid.values())
    ]
    jobs = [(kind, c, voyages, clutter) for c in cands]
    if workers <= 1:
        scores = [_score(j) for j in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool:
            scores = list(pool.map(_score, jobs))
    history = [
        {"params": {k: getattr(c, k) for k in keys}, "loglik": s}
        for c, s in zip(cands, scores, strict=True)
    ]
    best = cands[int(np.argmax(scores))]
    log.info(
        "calibrate[%s]: best mean loglik %.3f with %s",
        kind,
        max(scores),
        {k: getattr(best, k) for k in keys},
    )
    return best, history


# --------------------------------------------------------------------------- stage


def _comparison(st: dict[str, float], sc: SimConfig, achieved: float) -> list[tuple[str, str, str]]:
    """Rows of the real-vs-simulator table."""
    return [
        (
            "speed 5–95% (kn)",
            f"{st['speed_p5_kn']:.1f}–{st['speed_p95_kn']:.1f}",
            f"{sc.speed_kn[0]}–{sc.speed_kn[1]}",
        ),
        (
            "share of time turning (> 0.05 °/s)",
            f"{100 * st['turn_share']:.1f}%",
            f"{100 * achieved:.1f}%",
        ),
        (
            "turn rate upper bound (99th pct, °/s)",
            f"{st['turn_p99_deg_s']:.2f}",
            f"{sc.turn_rate_deg_s[1]} (log-uniform)",
        ),
        (
            "acceleration 90–99% (m/s²)",
            f"{st['accel_p90']:.4f}–{st['accel_p99']:.4f}",
            f"{sc.accel_ms2[0]}–{sc.accel_ms2[1]} (log-uniform)",
        ),
        ("median report interval (s)", f"{st['report_median_s']:.0f}", f"{sc.report_s:.0f}"),
        ("share of gaps > 3 min", f"{100 * st['gap_share_180s']:.2f}%", f"p_burst {sc.p_burst}"),
        ("outlier share", f"{100 * st['rejected_share']:.2f}%", f"{100 * sc.p_outlier:.2f}%"),
        ("position noise σ (m)", "–", f"{sc.pos_sigma_m} (IMM's likelihood-chosen R)"),
    ]


def _jsonable(p: Any) -> dict[str, Any]:
    d = asdict(p)
    return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in d.items()}


def stage(cfg: Config) -> None:
    """CLI stage: calibrate the simulator and both filters to real training data."""
    base = Path(cfg.paths.processed)
    tracks = pl.read_parquet(base / "tracks.parquet")
    voyages = pl.read_parquet(base / "voyages.parquet")
    points = pl.read_parquet(base / "points.parquet")
    cc = cfg.tracking.get("calibration", {})
    st = manoeuvre_stats(tracks, voyages)
    log.info("calibrate: real stats %s", st)

    ids = voyages.filter((pl.col("split") == "train") & pl.col("usable"))["voyage_id"].to_list()
    rng = np.random.default_rng(int(cfg.seed))
    pick = sorted(
        rng.choice(
            len(ids), size=min(int(cc.get("n_voyages", 150)), len(ids)), replace=False
        ).tolist()
    )
    sample = points.filter(pl.col("voyage_id").is_in([ids[i] for i in pick])).sort("voyage_id", "t")
    arrays = _voyage_arrays(sample)
    imm0 = imm_params_from_config(cfg)
    clutter = float(imm0.clutter_density)
    workers = int(cfg.tracking.get("workers", 4))
    cv_best, cv_hist = tune_on_real(
        "cv", cv_params_from_config(cfg), CV_GRID, arrays, clutter, workers
    )
    imm_best, imm_hist = tune_on_real("imm", imm0, IMM_GRID, arrays, clutter, workers)
    sim_cfg, achieved = fit_sim_config(st, pos_sigma_m=float(imm_best.r_pos_m))

    out: dict[str, Any] = {
        "real_stats": st,
        "n_voyages": len(arrays),
        "n_fixes": int(sum(len(a[0]) for a in arrays)),
        "sim_config": _jsonable(sim_cfg),
        "sim_turn_share": achieved,
        "cv": _jsonable(cv_best),
        "imm": _jsonable(imm_best),
        "history": {"cv": cv_hist, "imm": imm_hist},
    }
    rep = Path(cfg.paths.reports)
    write_text(rep / "tracking_calibration.json", json.dumps(out, indent=1))
    lines = [
        "# Tracker calibration on real AIS (training split)",
        "",
        f"Real manoeuvre statistics from {st['steps']:,} one-minute steps of moving commercial "
        "training vessels, mapped to the simulator; filter noise chosen by one-step predictive "
        f"log-likelihood on {len(arrays)} seeded training voyages ({out['n_fixes']:,} fixes).",
        "",
        "## Real behaviour vs simulator",
        "",
        "| statistic | real | calibrated simulator |",
        "|---|---:|---:|",
        *[f"| {a} | {b} | {c} |" for a, b, c in _comparison(st, sim_cfg, achieved)],
        "",
        "## Chosen filter noise (max predictive log-likelihood)",
        "",
        f"- CV-KF: `{json.dumps({k: out['cv'][k] for k in CV_GRID})}`",
        f"- IMM: `{json.dumps({k: out['imm'][k] for k in IMM_GRID})}`",
        "",
    ]
    write_text(rep / "tracking_calibration.md", "\n".join(lines))
    log.info("calibrate: wrote reports/tracking_calibration.{md,json}")
