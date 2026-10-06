"""Simulation study: tune and validate the CV-KF and IMM against known ground truth.

Protocol (PLAN §6.1):

* **Tune** each filter family by grid search on seeds ``tune_seeds``. Both families
  minimise the *same* objective: position RMSE over all moving fixes (straight and
  manoeuvre). So neither is tuned specially for manoeuvres, and "best-tuned CV-KF" means
  best on the shared objective.
* **Evaluate** the chosen configurations once on the disjoint seeds ``eval_seeds``, with
  RMSE (plus median and p95, since RMSE is tail-sensitive) by segment type, NEES
  consistency and outlier-rejection precision/recall.

**NEES** (normalised estimation error squared) is ``e' P⁻¹ e``, where e is the true
estimation error. For a consistent filter it averages to the state dimension. We use the
4-D sub-state [e, n, ve, vn], so the expected value is 4. Averaged over R Monte Carlo runs
at fix index k, R·ANEES_k follows χ²(4R). The 95% band is therefore
[χ²(4R, 0.025), χ²(4R, 0.975)] / R. We report the fraction of fix indices that fall
inside the band.
"""

from __future__ import annotations

import itertools
import json
import logging
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.stats import chi2

from ais_sentinel.config import Config
from ais_sentinel.io import write_text
from ais_sentinel.tracking.imm import IMMParams, run_imm
from ais_sentinel.tracking.kf import CVParams, FilterResult, run_cv_filter
from ais_sentinel.tracking.simulate import SimTrack, simulate_track

log = logging.getLogger(__name__)
Array = NDArray[np.float64]
WARMUP = 5  # fixes skipped at track start (initial transient)


def _masks(tr: SimTrack) -> dict[str, NDArray[np.bool_]]:
    seg = tr.segment
    keep = np.ones(len(seg), dtype=bool)
    keep[:WARMUP] = False
    return {
        "straight": keep & (seg == "straight"),
        "manoeuvre": keep & np.isin(seg, ["turn", "speed"]),
        "stopped": keep & (seg == "stopped"),
        "moving": keep & (seg != "stopped"),
    }


def _run(kind: str, params: CVParams | IMMParams, tr: SimTrack) -> FilterResult:
    if kind == "cv":
        assert isinstance(params, CVParams)
        return run_cv_filter(tr.t, tr.z, params)
    assert isinstance(params, IMMParams)
    return run_imm(tr.t, tr.z, params)


def evaluate(kind: str, params: CVParams | IMMParams, tracks: list[SimTrack]) -> dict[str, Any]:
    """Error, consistency and outlier-rejection metrics for one configuration."""
    sq: dict[str, list[Array]] = {k: [] for k in ("straight", "manoeuvre", "stopped", "moving")}
    nees_rows: list[Array] = []
    tp = fp = fn = 0
    for tr in tracks:
        res = _run(kind, params, tr)
        err = res.x[:, :2] - tr.truth[:, :2]
        d = np.hypot(err[:, 0], err[:, 1])
        for k, m in _masks(tr).items():
            sq[k].append(d[m])
        e4 = res.x[:, :4] - tr.truth
        P4 = res.P[:, :4, :4]
        nees_rows.append(np.einsum("ni,nij,nj->n", e4, np.linalg.inv(P4), e4))
        rejected = ~res.accepted
        tp += int(np.sum(rejected & tr.outlier))
        fp += int(np.sum(rejected & ~tr.outlier))
        fn += int(np.sum(~rejected & tr.outlier))
    out: dict[str, Any] = {}
    for k, parts in sq.items():
        v = np.concatenate(parts)
        out[k] = {
            "n": int(v.size),
            "rmse_m": float(np.sqrt(np.mean(v**2))) if v.size else float("nan"),
            "median_m": float(np.median(v)) if v.size else float("nan"),
            "p95_m": float(np.percentile(v, 95)) if v.size else float("nan"),
        }
    k_max = min(len(r) for r in nees_rows)
    stack = np.vstack([r[:k_max] for r in nees_rows])[:, WARMUP:]
    runs = stack.shape[0]
    anees = stack.mean(axis=0)
    lo, hi = chi2.ppf([0.025, 0.975], df=4 * runs) / runs
    out["nees"] = {
        "runs": runs,
        "steps": int(anees.size),
        "mean_anees": float(anees.mean()),
        "median_anees": float(np.median(anees)),
        "band": [float(lo), float(hi)],
        "frac_in_band": float(np.mean((anees >= lo) & (anees <= hi))),
        "anees_trace": [round(float(a), 3) for a in anees],
    }
    out["outlier_rejection"] = {
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
        "rejected_non_outliers": fp,
    }
    return out


# Attempt 2 (DECISIONS D19): both filters get the same two new options, measurement noise
# r_pos_m and gap-aware process noise gap_q. The IMM's sojourn times and clutter density
# are fixed at the values attempt 1 chose on the same tuning seeds, to keep the grid small.
CV_GRID: dict[str, list[float]] = {
    "q_accel": [0.001, 0.003, 0.01, 0.03, 0.1, 0.3],
    "gate_prob": [0.999, 0.9999, 0.99999],
    "r_pos_m": [5.0, 10.0],
    "gap_q": [0.0, 0.05, 0.3],
}
IMM_GRID: dict[str, list[Any]] = {
    "q_cruise": [0.0005, 0.002],
    "q_turn": [0.01, 0.05, 0.2],
    "q_omega": [2e-7, 2e-6],
    "r_pos_m": [5.0, 10.0],
    "gap_q": [0.0, 0.05, 0.3],
    "sojourn_s": [(1800.0, 600.0, 300.0)],
    "clutter_density": [5e-11],
}


def _grid(base: Any, grid: dict[str, list[Any]]) -> list[Any]:
    keys = list(grid)
    return [
        replace(base, **dict(zip(keys, vals, strict=True)))
        for vals in itertools.product(*grid.values())
    ]


def tune(kind: str, tracks: list[SimTrack]) -> tuple[CVParams | IMMParams, list[dict[str, Any]]]:
    """Grid search minimising moving-fix position RMSE. Returns the best params and a log."""
    base: CVParams | IMMParams = CVParams() if kind == "cv" else IMMParams()
    candidates = _grid(base, CV_GRID if kind == "cv" else IMM_GRID)
    history = []
    best, best_score = candidates[0], np.inf
    for c in candidates:
        score = evaluate(kind, c, tracks)["moving"]["rmse_m"]
        history.append({"params": _params_dict(c), "moving_rmse_m": score})
        if score < best_score:
            best, best_score = c, score
    log.info("tuned %s: moving RMSE %.1f m with %s", kind, best_score, _params_dict(best))
    return best, history


def _params_dict(p: CVParams | IMMParams) -> dict[str, Any]:
    d = asdict(p)
    return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in d.items()}


def report_markdown(res: dict[str, Any]) -> str:
    """Render the simulation study as markdown."""
    cv, imm = res["eval"]["cv"], res["eval"]["imm"]

    def row(name: str, seg: str) -> str:
        a, b = cv[seg], imm[seg]
        gain = 100 * (1 - b["rmse_m"] / a["rmse_m"])
        return (
            f"| {name} | {a['n']:,} | {a['rmse_m']:.1f} | {b['rmse_m']:.1f} | {gain:+.1f}% | "
            f"{a['median_m']:.1f} / {b['median_m']:.1f} | {a['p95_m']:.1f} / {b['p95_m']:.1f} |"
        )

    crit = res["criteria"]
    lines = [
        "# Tracking: simulation study",
        "",
        f"Tuning seeds: {res['tune_seeds'][0]}–{res['tune_seeds'][-1]} "
        f"({len(res['tune_seeds'])} voyages). Evaluation seeds: {res['eval_seeds'][0]}–"
        f"{res['eval_seeds'][-1]} ({len(res['eval_seeds'])} voyages, disjoint). Each voyage is "
        "4 h with 60 s AIS-like reporting, dropouts, bursts of missing reports, 5 m GPS noise "
        "and 0.5% gross outliers.",
        "",
        "## Position error (m): CV-KF vs IMM",
        "",
        "| segment | fixes | CV RMSE | IMM RMSE | IMM gain | median CV / IMM | p95 CV / IMM |",
        "|---|---:|---:|---:|---:|---:|---:|",
        row("straight", "straight"),
        row("manoeuvre (turns, speed changes)", "manoeuvre"),
        row("stopped", "stopped"),
        row("all moving", "moving"),
        "",
        "## Consistency (NEES, 4-D [e, n, ve, vn], expected value 4)",
        "",
        "| filter | mean ANEES | median ANEES | 95% band | steps in band |",
        "|---|---:|---:|---|---:|",
    ]
    for name, r in (("CV-KF", cv), ("IMM", imm)):
        nz = r["nees"]
        lines.append(
            f"| {name} | {nz['mean_anees']:.2f} | {nz['median_anees']:.2f} | "
            f"[{nz['band'][0]:.2f}, {nz['band'][1]:.2f}] | {100 * nz['frac_in_band']:.0f}% |"
        )
    lines += [
        "",
        "## Outlier rejection (injected gross outliers)",
        "",
        "| filter | precision | recall | good fixes rejected |",
        "|---|---:|---:|---:|",
    ]
    for name, r in (("CV-KF", cv), ("IMM", imm)):
        o = r["outlier_rejection"]
        lines.append(
            f"| {name} | {o['precision']:.2f} | {o['recall']:.2f} | {o['rejected_non_outliers']} |"
        )
    lines += [
        "",
        "## Success criteria (PLAN §9.2, set before results)",
        "",
        "| criterion | target | result | met? |",
        "|---|---|---|---|",
    ]
    for c in crit:
        lines.append(
            f"| {c['name']} | {c['target']} | {c['result']} | {'✅' if c['met'] else '❌'} |"
        )
    lines += [
        "",
        "## Tuned parameters",
        "",
        f"- CV-KF: `{json.dumps(res['tuned']['cv'])}`",
        f"- IMM: `{json.dumps({k: v for k, v in res['tuned']['imm'].items() if k != 'switch'})}`",
        "",
    ]
    return "\n".join(lines)


def run_study(cfg: Config) -> dict[str, Any]:
    """Tune both filters, evaluate them and check the PLAN §9.2 criteria."""
    sc = cfg.tracking["sim"]
    tune_seeds = list(range(int(sc["tune_seeds"][0]), int(sc["tune_seeds"][1])))
    eval_seeds = list(range(int(sc["eval_seeds"][0]), int(sc["eval_seeds"][1])))
    tune_tracks = [simulate_track(s) for s in tune_seeds]
    eval_tracks = [simulate_track(s) for s in eval_seeds]
    cv_best, cv_hist = tune("cv", tune_tracks)
    imm_best, imm_hist = tune("imm", tune_tracks)
    ev = {"cv": evaluate("cv", cv_best, eval_tracks), "imm": evaluate("imm", imm_best, eval_tracks)}
    gain_m = 1 - ev["imm"]["manoeuvre"]["rmse_m"] / ev["cv"]["manoeuvre"]["rmse_m"]
    worse_s = ev["imm"]["straight"]["rmse_m"] / ev["cv"]["straight"]["rmse_m"] - 1
    in_band = ev["imm"]["nees"]["frac_in_band"]
    criteria = [
        {
            "name": "IMM manoeuvre RMSE vs best-tuned CV-KF",
            "target": ">= 20% lower",
            "result": f"{100 * gain_m:.1f}% lower",
            "met": bool(gain_m >= 0.20),
        },
        {
            "name": "IMM straight-segment RMSE vs CV-KF",
            "target": "<= 10% worse",
            "result": f"{100 * worse_s:+.1f}%",
            "met": bool(worse_s <= 0.10),
        },
        {
            "name": "IMM NEES steps inside 95% band",
            "target": ">= 80%",
            "result": f"{100 * in_band:.0f}%",
            "met": bool(in_band >= 0.80),
        },
    ]
    return {
        "tune_seeds": tune_seeds,
        "eval_seeds": eval_seeds,
        "tuned": {"cv": _params_dict(cv_best), "imm": _params_dict(imm_best)},
        "tuning_history": {"cv": cv_hist, "imm": imm_hist},
        "eval": ev,
        "criteria": criteria,
    }


def stage(cfg: Config) -> None:
    """CLI stage: run the simulation study and write reports/tracking.{md,json}."""
    res = run_study(cfg)
    out = Path(cfg.paths.reports)
    out.mkdir(parents=True, exist_ok=True)
    write_text(out / "tracking_sim.json", json.dumps(res, indent=1))
    write_text(out / "tracking.md", report_markdown(res))
    for c in res["criteria"]:
        log.info(
            "criterion %s: %s (target %s) met=%s", c["name"], c["result"], c["target"], c["met"]
        )
