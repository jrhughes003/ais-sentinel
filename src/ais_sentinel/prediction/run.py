"""Prediction experiment: baselines vs ML, calibrated on validation, scored on locked test.

Stages:

* ``evaluate`` (validation split): build samples, run every baseline, tune the kNN and
  choose the IMM forecast variant on validation, train the ML ensembles (early-stopping on
  validation), fit per-horizon covariance calibration on validation, and write
  ``reports/prediction_val.md``. Models and calibration are saved to ``models/``.
* ``holdout`` (locked test split, October 2023): apply the saved models and calibration
  unchanged, then write ``reports/prediction_test.md``. A dated row with the git commit and
  config hash is appended to ``reports/holdout_runs.md``. Run it once per completed model
  version.

Fairness rules (PLAN §6.3):

* Every model is scored on the *same* samples, horizons and metrics.
* The ML winner (M1 vs M2) is chosen on validation.
* The comparison baseline is the **best baseline on the test set at each horizon**. Picking
  it on test is conservative, because it can only make the ML comparison harder.
"""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch

from ais_sentinel.config import Config
from ais_sentinel.evaluation.metrics import (
    fit_cov_scale,
    paired_difference_ci,
    per_sample_scores,
    summarise,
)
from ais_sentinel.io import write_text
from ais_sentinel.prediction.baselines import Forecast, dead_reckoning
from ais_sentinel.prediction.features import build_features
from ais_sentinel.prediction.filter_forecast import cv_forecast, imm_forecast
from ais_sentinel.prediction.knn import KNNParams, build_library, knn_forecast, region_centre
from ais_sentinel.prediction.ml import (
    MLConfig,
    SeqModel,
    TrainedModel,
    ensemble_forecast,
    state_summary,
    train_model,
)
from ais_sentinel.prediction.samples import build_samples
from ais_sentinel.tracking.pipeline import cv_params_from_config, imm_params_from_config

log = logging.getLogger(__name__)
BASELINES = ("dead_reckoning", "kf_cv", "imm", "knn_route")
LABELS = {
    "dead_reckoning": "B0 dead reckoning",
    "kf_cv": "B1 CV Kalman",
    "imm": "B2 IMM",
    "imm_switching": "B2' IMM (switching)",
    "knn_route": "B3 kNN route",
    "gru": "M1 GRU (Gaussian ensemble)",
    "gru_mdn": "M2 GRU (mixture ensemble)",
}


# ----------------------------------------------------------------------------- inputs


def _inputs(cfg: Config) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    base = Path(cfg.paths.processed)
    return (
        pl.read_parquet(base / "tracks.parquet"),
        pl.read_parquet(base / "voyages.parquet"),
        pl.read_parquet(base / "filter_states.parquet"),
    )


def _split_samples(
    tracks: pl.DataFrame, voyages: pl.DataFrame, cfg: Config, split: str, every_min: float
) -> pl.DataFrame:
    pc = cfg.prediction
    ids = voyages.filter(pl.col("split") == split)["voyage_id"]
    tr = tracks.filter(pl.col("voyage_id").is_in(ids.implode()))
    return build_samples(
        tr,
        voyages,
        list(pc.horizons_min),
        float(pc.history_min),
        every_min,
        min_sog_kn=float(pc.min_sog_kn),
    )


# ----------------------------------------------------------------------------- models


def baseline_forecasts(
    samples: pl.DataFrame,
    states: pl.DataFrame,
    lib_params: KNNParams,
    lib: Any,
    cfg: Config,
    imm_switching: bool | None,
) -> tuple[dict[str, Forecast], dict[str, float]]:
    """Run B0–B3. ``imm_switching=None`` runs both IMM variants (for selection on val)."""
    H = list(cfg.prediction.horizons_min)
    out = {
        "dead_reckoning": dead_reckoning(samples, H),
        "kf_cv": cv_forecast(samples, states, H, cv_params_from_config(cfg).q_accel),
    }
    immp = imm_params_from_config(cfg)
    if imm_switching is None or not imm_switching:
        out["imm"] = imm_forecast(samples, states, H, immp, switching=False)
    if imm_switching is None or imm_switching:
        fc = imm_forecast(samples, states, H, immp, switching=True)
        out["imm_switching" if imm_switching is None else "imm"] = replace(
            fc, model="imm_switching" if imm_switching is None else "imm"
        )
    knn_fc, fallback = knn_forecast(samples, lib, H)
    out["knn_route"] = knn_fc
    return out, {"knn_fallback": fallback}


def _ml_cfgs(cfg: Config) -> dict[str, tuple[MLConfig, list[int]]]:
    mc = dict(cfg.prediction["ml"])
    base = MLConfig(**{k: v for k, v in mc.items() if k in MLConfig.__dataclass_fields__})
    return {
        "gru": (replace(base, components=1), list(mc["seeds_gaussian"])),
        "gru_mdn": (replace(base, components=int(mc["mdn_components"])), list(mc["seeds_mdn"])),
    }


def _model_dir(cfg: Config) -> Path:
    return Path(cfg.paths.get("models", "models"))


def save_models(cfg: Config, name: str, models: list[TrainedModel]) -> None:
    """Persist ensemble members (state dict + scale + config + history)."""
    d = _model_dir(cfg)
    d.mkdir(parents=True, exist_ok=True)
    torch.save(
        [
            {
                "state": m.model.state_dict(),
                "scale": m.scale,
                "cfg": asdict(m.cfg),
                "history": m.history,
            }
            for m in models
        ],
        d / f"{name}.pt",
    )


def load_models(
    cfg: Config, name: str, n_seq: int, n_ctx: int, horizons: int
) -> list[TrainedModel]:
    """Load ensemble members saved by :func:`save_models`."""
    blobs = torch.load(_model_dir(cfg) / f"{name}.pt", weights_only=False)
    out = []
    for b in blobs:
        mcfg = MLConfig(**b["cfg"])
        model = SeqModel(n_seq, n_ctx, horizons, mcfg)
        model.load_state_dict(b["state"])
        out.append(TrainedModel(model, b["scale"], mcfg, b["history"]))
    return out


# ----------------------------------------------------------------------------- reporting


def _fmt_table(summary: pl.DataFrame) -> list[str]:
    lines = [
        "| model | horizon | n | voyages | mean km [95% CI] | median km | p90 km "
        "| cov50 | cov90 | NLL |",
        "|---|---:|---:|---:|---|---:|---:|---:|---:|---:|",
    ]
    for r in summary.iter_rows(named=True):
        lines.append(
            f"| {LABELS.get(r['model'], r['model'])} | {r['horizon_min']} | {r['n']:,} "
            f"| {r['voyages']:,} | "
            f"{r['mean_km']:.2f} [{r['mean_lo']:.2f}, {r['mean_hi']:.2f}] | {r['median_km']:.2f} | "
            f"{r['p90_km']:.2f} | {r['cov50']:.2f} | {r['cov90']:.2f} | {r['nll']:.2f} |"
        )
    return lines


def _git_hash() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _score(fcs: dict[str, Forecast], samples: pl.DataFrame) -> pl.DataFrame:
    return pl.concat([per_sample_scores(fc, samples) for fc in fcs.values()])


# ----------------------------------------------------------------------------- stages


def evaluate_stage(cfg: Config) -> None:
    """Validation experiment: tune, train, calibrate, and report on the validation split."""
    pc = cfg.prediction
    H = list(pc.horizons_min)
    tracks, voyages, states = _inputs(cfg)
    train_ids = set(voyages.filter(pl.col("split") == "train")["voyage_id"].to_list())
    val = _split_samples(tracks, voyages, cfg, "val", float(pc.anchor_every_min))
    log.info("validation samples: %d", val.height)

    # kNN: tune a small grid on validation.
    ref = region_centre(dict(cfg.region))
    best_knn: tuple[float, KNNParams, Any] | None = None
    knn_log = []
    for pos in pc.knn["pos_scale_m"]:
        for course in pc.knn["course_scale_deg"]:
            # The library depends on the feature scales only, so build it once per pair and
            # vary k at query time.
            base_p = KNNParams(pos_scale_m=float(pos), course_scale_deg=float(course))
            base_lib = build_library(tracks, train_ids, *ref, base_p)
            for k in pc.knn["k"]:
                kp = replace(base_p, k=int(k))
                lib = replace(base_lib, params=kp)
                fc, fb = knn_forecast(val, lib, H)
                err = (
                    per_sample_scores(fc, val)
                    .group_by("horizon_min")
                    .agg(pl.col("error_km").mean())
                )
                score = float(err["error_km"].to_numpy().mean())  # mean over horizons
                knn_log.append(
                    {
                        "pos_scale_m": pos,
                        "course_scale_deg": course,
                        "k": k,
                        "score_km": score,
                        "fallback": fb,
                    }
                )
                if best_knn is None or score < best_knn[0]:
                    best_knn = (score, kp, lib)
    assert best_knn is not None
    _, knn_p, lib = best_knn
    log.info("kNN chosen: %s", knn_p)

    fcs, extra = baseline_forecasts(val, states, knn_p, lib, cfg, imm_switching=None)

    # ML: train on the train split (denser anchors), early-stop on validation.
    train_s = _split_samples(tracks, voyages, cfg, "train", float(pc.ml["train_anchor_every_min"]))
    log.info("ML training samples: %d", train_s.height)
    hist = int(pc.history_min)
    f_tr = build_features(tracks, train_s, H, hist, *ref)
    f_va = build_features(tracks, val, H, hist, *ref)
    ml_summary: dict[str, Any] = {}
    for name, (mcfg, seeds) in _ml_cfgs(cfg).items():
        members = [train_model(f_tr, f_va, replace(mcfg, seed=int(s))) for s in seeds]
        save_models(cfg, name, members)
        fcs[name] = ensemble_forecast(
            members, f_va, val["lat0"].to_numpy(), val["lon0"].to_numpy(), H, name
        )
        ml_summary[name] = [state_summary(m) for m in members]

    raw = _score(fcs, val)
    scales = {m: fit_cov_scale(raw.filter(pl.col("model") == m)) for m in fcs}
    cal = _score({m: fc.with_cov_scale(scales[m]) for m, fc in fcs.items()}, val)
    table = summarise(cal, ["model", "horizon_min"], n_boot=int(pc.n_boot), seed=int(cfg.seed))

    imm_choice = _choose(table, ["imm", "imm_switching"])
    ml_choice = _choose(table, ["gru", "gru_mdn"])
    choices = {
        "imm_switching": imm_choice == "imm_switching",
        "best_ml": ml_choice,
        "knn": asdict(knn_p),
        "cov_scales": scales,
        "knn_grid": knn_log,
        "knn_fallback_val": extra["knn_fallback"],
        "ml": ml_summary,
        "n_train_samples": train_s.height,
        "n_val_samples": val.height,
    }
    d = _model_dir(cfg)
    write_text(d / "choices.json", json.dumps(choices, indent=1, default=str))
    lines = [
        f"# Prediction: validation results ({cfg.splits['val'][0]} – {cfg.splits['val'][1]})",
        "",
        f"Validation samples: {val.height:,} anchors (underway, SOG ≥ {pc.min_sog_kn} kn). "
        f"ML training samples: {train_s.height:,}. Covariances are calibrated per model and "
        "horizon on this split (so coverage here is ~0.90 by construction; the test split "
        "is the honest check).",
        "",
        f"Chosen on validation: IMM variant = **{imm_choice}**, ML model = **{ml_choice}**, "
        f"kNN = `{json.dumps(asdict(knn_p))}` (DR fallback {100 * extra['knn_fallback']:.1f}%).",
        "",
        *_fmt_table(table),
        "",
    ]
    write_text(Path(cfg.paths.reports) / "prediction_val.md", "\n".join(lines))
    _save_forecasts(cfg, "val", {m: fc.with_cov_scale(scales[m]) for m, fc in fcs.items()}, val)
    log.info("evaluate: wrote reports/prediction_val.md")


def _choose(table: pl.DataFrame, names: list[str]) -> str:
    """Model with the lowest mean error averaged over horizons."""
    t = table.filter(pl.col("model").is_in(names)).group_by("model").agg(pl.col("mean_km").mean())
    return str(t.sort("mean_km")["model"][0])


def _save_forecasts(
    cfg: Config, split: str, fcs: dict[str, Forecast], samples: pl.DataFrame | None = None
) -> None:
    """Persist calibrated forecasts (and the samples they score) for the site export."""
    if samples is not None:
        samples.write_parquet(Path(cfg.paths.processed) / f"samples_{split}.parquet")
    rows = []
    for name, fc in fcs.items():
        lat, lon = fc.latlon()
        rows.append(
            pl.DataFrame(
                {
                    "model": name,
                    "sample_id": np.repeat(fc.sample_id, len(fc.horizons_min)),
                    "horizon_min": np.tile(fc.horizons_min, len(fc.sample_id)),
                    "lat": lat.ravel(),
                    "lon": lon.ravel(),
                    "cee": fc.cov[..., 0, 0].ravel(),
                    "cnn": fc.cov[..., 1, 1].ravel(),
                    "cen": fc.cov[..., 0, 1].ravel(),
                }
            )
        )
    pl.concat(rows).write_parquet(Path(cfg.paths.processed) / f"forecasts_{split}.parquet")


def holdout_stage(cfg: Config) -> None:
    """Locked-test evaluation. Uses only artifacts fixed on train/validation."""
    pc = cfg.prediction
    H = list(pc.horizons_min)
    choices = json.loads((_model_dir(cfg) / "choices.json").read_text(encoding="utf-8"))
    tracks, voyages, states = _inputs(cfg)
    train_ids = set(voyages.filter(pl.col("split") == "train")["voyage_id"].to_list())
    test = _split_samples(tracks, voyages, cfg, "test", float(pc.anchor_every_min))
    ref = region_centre(dict(cfg.region))
    kp_dict: dict[str, Any] = {
        k: (tuple(v) if isinstance(v, list) else v) for k, v in choices["knn"].items()
    }
    kp = KNNParams(**kp_dict)
    lib = build_library(tracks, train_ids, *ref, kp)
    fcs, extra = baseline_forecasts(
        test, states, kp, lib, cfg, imm_switching=bool(choices["imm_switching"])
    )
    f_te = build_features(tracks, test, H, int(pc.history_min), *ref)
    for name in ("gru", "gru_mdn"):
        members = load_models(cfg, name, f_te.seq.shape[-1], f_te.ctx.shape[-1], len(H))
        fcs[name] = ensemble_forecast(
            members, f_te, test["lat0"].to_numpy(), test["lon0"].to_numpy(), H, name
        )
    scales = {m: {int(k): float(v) for k, v in s.items()} for m, s in choices["cov_scales"].items()}
    scales["imm"] = scales["imm_switching"] if choices["imm_switching"] else scales["imm"]
    fcs = {m: fc.with_cov_scale(scales[m]) for m, fc in fcs.items()}
    scores = _score(fcs, test)
    table = summarise(scores, ["model", "horizon_min"], n_boot=int(pc.n_boot), seed=int(cfg.seed))

    # Vessel-unseen subset: test voyages whose MMSI never appears in training.
    train_mmsi = set(voyages.filter(pl.col("split") == "train")["mmsi"].to_list())
    unseen_ids = test.filter(~pl.col("mmsi").is_in(sorted(train_mmsi)))["sample_id"]
    unseen = summarise(
        scores.filter(pl.col("sample_id").is_in(unseen_ids.implode())),
        ["model", "horizon_min"],
        n_boot=int(pc.n_boot),
    )
    by_group = summarise(scores, ["model", "vessel_group", "horizon_min"], n_boot=0)

    best_ml = str(choices["best_ml"])
    verdict = []
    for h in H:
        tb = table.filter((pl.col("horizon_min") == h) & pl.col("model").is_in(list(BASELINES)))
        if tb.is_empty():
            continue  # no test truth at this horizon
        best_b = str(tb.sort("mean_km")["model"][0])
        d, lo, hi = paired_difference_ci(
            scores.filter(pl.col("model") == best_ml),
            scores.filter(pl.col("model") == best_b),
            h,
            int(pc.n_boot),
        )
        base_mean = float(tb.sort("mean_km")["mean_km"][0])
        verdict.append(
            {
                "horizon": h,
                "best_baseline": best_b,
                "diff_km": d,
                "lo": lo,
                "hi": hi,
                "rel": d / base_mean,
            }
        )
    cov_ml = table.filter(pl.col("model") == best_ml)["cov90"].to_list()
    na = {"diff_km": float("nan"), "lo": float("nan"), "hi": float("nan"), "rel": float("nan")}
    v60 = next((v for v in verdict if v["horizon"] == 60), na)
    v120 = next((v for v in verdict if v["horizon"] == 120), na)
    met_point = bool(v60["hi"] < 0 and v120["hi"] < 0 and v120["rel"] <= -0.15)
    met_cov = all(0.85 <= c <= 0.95 for c in cov_ml)

    lines = [
        f"# Prediction: locked test results ({cfg.splits['test'][0]} – {cfg.splits['test'][1]})",
        "",
        f"Run at {datetime.now(UTC).isoformat(timespec='seconds')} (commit `{_git_hash()}`, config "
        f"`{cfg.hash()}`). Test samples: {test.height:,}. Models, kNN settings, IMM variant and "
        "covariance calibration were all fixed on train/validation before this run.",
        "",
        "## All models",
        "",
        *_fmt_table(table),
        "",
        f"## Best ML model ({LABELS[best_ml]}) vs the best baseline at each horizon",
        "",
        "Difference in mean error (ML − baseline); negative = ML better. 95% CI from a paired "
        "voyage-cluster bootstrap.",
        "",
        "| horizon | best baseline | diff km [95% CI] | relative |",
        "|---:|---|---|---:|",
        *[
            f"| {v['horizon']} | {LABELS[v['best_baseline']]} | {v['diff_km']:+.2f} "
            f"[{v['lo']:+.2f}, {v['hi']:+.2f}] | "
            f"{100 * v['rel']:+.1f}% |"
            for v in verdict
        ],
        "",
        "## Success criteria (PLAN §9.3, set before results)",
        "",
        "| criterion | target | result | met? |",
        "|---|---|---|---|",
        "| ML beats best baseline at 60 & 120 min, ≥15% at 120 "
        "| CI excludes 0 at both; ≤ −15% at 120 | "
        f"60: {100 * v60['rel']:+.1f}% [{v60['lo']:+.2f}, {v60['hi']:+.2f}] km; "
        f"120: {100 * v120['rel']:+.1f}% "
        f"[{v120['lo']:+.2f}, {v120['hi']:+.2f}] km | {'✅' if met_point else '❌'} |",
        f"| 90% ellipse coverage of best ML model | within [0.85, 0.95] at every horizon | "
        f"{', '.join(f'{c:.2f}' for c in cov_ml)} | {'✅' if met_cov else '❌'} |",
        "",
        "## Vessel-unseen subset (MMSI never seen in training)",
        "",
        *_fmt_table(unseen),
        "",
        "## By vessel group (no CIs)",
        "",
        *_fmt_table(
            by_group.drop("vessel_group").with_columns(
                model=by_group["model"] + " / " + by_group["vessel_group"]
            )
        ),
        "",
    ]
    write_text(Path(cfg.paths.reports) / "prediction_test.md", "\n".join(lines))
    write_text(
        Path(cfg.paths.reports) / "prediction_test.json",
        json.dumps(
            {
                "table": table.to_dicts(),
                "verdict": verdict,
                "met_point": met_point,
                "met_cov": met_cov,
                "knn_fallback": extra["knn_fallback"],
            },
            indent=1,
            default=str,
        ),
    )
    _save_forecasts(cfg, "test", fcs, test)
    _append_holdout_log(cfg, best_ml, verdict, cov_ml, met_point, met_cov, test.height)
    log.info("holdout: wrote reports/prediction_test.md")


def _append_holdout_log(
    cfg: Config,
    best_ml: str,
    verdict: list[dict[str, Any]],
    cov: list[float],
    met_p: bool,
    met_c: bool,
    n: int,
) -> None:
    # One shared log for every locked test (v1: Oct 2023, v2: Oct 2024).
    path = Path(str(cfg.paths.get("holdout_log", Path(cfg.paths.reports) / "holdout_runs.md")))
    header = (
        "# Locked test runs\n\nEach row is one evaluation of a locked test split. "
        "Rows are appended, never edited.\n\n"
        "| date (UTC) | commit | config | samples | best ML | Δ60 vs best baseline | Δ120 "
        "| cov90 (ML) | "
        "point target | coverage target |\n|---|---|---|---:|---|---:|---:|---|---|---|\n"
    )
    text = path.read_text(encoding="utf-8") if path.exists() else header
    na = {"rel": float("nan")}
    v = {60: na, 120: na, **{x["horizon"]: x for x in verdict}}
    text += (
        f"| {datetime.now(UTC).isoformat(timespec='minutes')} | {_git_hash()} | {cfg.hash()} "
        f"| {n:,} (test {cfg.splits['test'][0]}–{cfg.splits['test'][1]}) | {best_ml} "
        f"| {100 * v[60]['rel']:+.1f}% | {100 * v[120]['rel']:+.1f}% "
        f"| {', '.join(f'{c:.2f}' for c in cov)} | "
        f"{'met' if met_p else 'not met'} | {'met' if met_c else 'not met'} |\n"
    )
    write_text(path, text)
