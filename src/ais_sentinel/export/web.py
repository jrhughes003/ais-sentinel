"""Export small, licence-compliant JSON extracts for the static website.

The site runs entirely in the browser, so everything it shows is precomputed here and kept
small:

* positions rounded to 5 decimals (about 1 m) in compact column arrays;
* only **commercial** vessels in showcase material (privacy: no pleasure craft);
* curated subsets selected by a **seeded random draw**, never by which model looks good.

Files written to ``paths.site_data``:

* ``tracks.json``: showcase voyages, with raw fixes and the IMM-smoothed track.
* ``predictions.json``: forecast samples (history, truth and every model's forecast).
* ``anomalies.json``: flagged events, with explanations and track context.
* ``results.json``: headline numbers, metric tables and the success-criteria table.

Every file carries the data attribution.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import yaml

from ais_sentinel.config import Config
from ais_sentinel.data.vessel_types import COMMERCIAL
from ais_sentinel.geo import haversine_m
from ais_sentinel.io import write_text

log = logging.getLogger(__name__)

ATTRIBUTION = (
    "AIS data: U.S. Coast Guard Navigation Center via NOAA Office for Coastal Management / "
    "BOEM, MarineCadastre.gov (CC0 1.0). Not for navigation."
)
MODELS = [
    ("dead_reckoning", "B0 Dead reckoning", "Dead reckoning", "baseline"),
    ("kf_cv", "B1 Kalman (CV) extrapolation", "Kalman CV", "baseline"),
    ("imm", "B2 IMM extrapolation", "IMM", "baseline"),
    ("knn_route", "B3 Route analogs (kNN)", "Route kNN", "baseline"),
    ("gru", "M1 GRU, Gaussian (ensemble)", "GRU", "ml"),
    ("gru_mdn", "M2 GRU, mixture (ensemble)", "GRU-MDN", "ml"),
]
ANOMALY_TYPES = ("gap", "jump", "loiter", "deviation", "rendezvous")


def _round(a: Any, nd: int = 5) -> list[float | None]:
    return [None if not np.isfinite(x) else round(float(x), nd) for x in np.asarray(a, dtype=float)]


def write_json(path: Path, obj: Any) -> int:
    """Write compact JSON (NaN/inf become null). Returns the size in bytes."""

    def clean(o: Any) -> Any:
        if isinstance(o, float) and not np.isfinite(o):
            return None
        if isinstance(o, list | tuple):
            return [clean(x) for x in o]
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, np.generic):
            return clean(o.item())
        return o

    text = json.dumps(clean(obj), separators=(",", ":"), ensure_ascii=False, default=str)
    return write_text(path, text)


def _meta(cfg: Config) -> dict[str, Any]:
    return {
        "generated": datetime.now(UTC).isoformat(timespec="seconds"),
        "attribution": ATTRIBUTION,
        "region": dict(cfg.region),
        "note": cfg.export.get("note"),  # e.g. "development preview", shown as a banner
    }


def _split_label(cfg: Config, split: str) -> str:
    """Human label for a split, from the configured dates (never hard-coded)."""
    d0, d1 = (str(x) for x in cfg.splits[split])
    return f"{'locked test' if split == 'test' else 'validation'} period {d0} – {d1}"


def _models() -> list[dict[str, str]]:
    return [{"id": i, "label": lab, "short": sh, "kind": k} for i, lab, sh, k in MODELS]


# --------------------------------------------------------------------------- tracks


def pick_showcase_voyages(
    voyages: pl.DataFrame, n: int, min_path_km: float = 15.0, seed: int = 0
) -> pl.DataFrame:
    """Choose ``n`` long, moving, named commercial voyages: one per vessel, spread over groups.

    Within each group, the candidates are drawn at random (seeded) from the longest half, so
    the showcase is representative rather than hand-picked.
    """
    cand = (
        voyages.filter(
            pl.col("usable")
            & pl.col("vessel_group").is_in(sorted(COMMERCIAL))
            & (pl.col("path_km") >= min_path_km)
            & (pl.col("moving_frac") >= 0.5)
            & pl.col("vessel_name").is_not_null()
        )
        .sort("path_km", descending=True)
        .unique("mmsi", keep="first", maintain_order=True)
    )
    if cand.is_empty():
        return cand
    rng = np.random.default_rng(seed)
    groups = cand["vessel_group"].unique().sort().to_list()
    per = max(1, n // len(groups))
    picked = []
    for g in groups:
        sub = cand.filter(pl.col("vessel_group") == g)
        top = sub.head(max(per, sub.height // 2))
        idx = rng.choice(top.height, size=min(per, top.height), replace=False)
        picked.append(top[sorted(idx.tolist())])
    out = pl.concat(picked)
    if out.height < n:
        rest = cand.join(out.select("voyage_id"), on="voyage_id", how="anti")
        out = pl.concat([out, rest.head(n - out.height)])
    return out.head(n).sort("t_start")


def voyage_payload(track: pl.DataFrame, meta: dict[str, Any]) -> dict[str, Any]:
    """Compact JSON object for one voyage's raw and smoothed track."""
    t = track["t"].dt.epoch("s").to_numpy().astype(np.int64)
    pee, pnn, pen = (track[c].to_numpy() for c in ("s_pee", "s_pnn", "s_pen"))
    tr, det = pee + pnn, pee * pnn - pen**2
    lam_max = tr / 2 + np.sqrt(np.maximum(tr**2 / 4 - det, 0.0))  # larger eigenvalue (m²)
    if "i_mu_stationary" in track.columns:
        mu = np.column_stack(
            [track[c].to_numpy() for c in ("i_mu_stationary", "i_mu_cruising", "i_mu_turning")]
        )
        mode = np.argmax(mu, axis=1).tolist()
    else:
        mode = [1] * track.height
    length = meta.get("length_m")
    return {
        "id": meta["voyage_id"],
        "mmsi": int(meta["mmsi"]),
        "name": meta["vessel_name"],
        "group": meta["vessel_group"],
        "length_m": None if not length else round(float(length)),
        "t0": int(t[0]),
        "dt": np.diff(t, prepend=t[0]).tolist(),
        "lat": _round(track["lat"].to_numpy()),
        "lon": _round(track["lon"].to_numpy()),
        "s_lat": _round(track["s_lat"].to_numpy()),
        "s_lon": _round(track["s_lon"].to_numpy()),
        "s_sd_m": _round(np.sqrt(lam_max), 1),
        "mode": mode,
        "rejected": np.flatnonzero(~track["accepted"].to_numpy()).tolist(),
        "sog_kn": _round(track["sog_kn"].cast(pl.Float64).fill_null(np.nan).to_numpy(), 1),
        "cog_deg": _round(track["cog_deg"].cast(pl.Float64).fill_null(np.nan).to_numpy(), 1),
    }


# --------------------------------------------------------------------------- predictions


def prediction_payload(
    samples: pl.DataFrame,
    forecasts: pl.DataFrame,
    tracks: pl.DataFrame,
    names: dict[int, str],
    horizons: list[int],
    n: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Seeded random draw of commercial samples with complete futures (no cherry-picking)."""
    full = samples.filter(
        pl.col("vessel_group").is_in(sorted(COMMERCIAL))
        & pl.all_horizontal([pl.col(f"lat_{h}").is_not_null() for h in horizons])
    )
    if full.is_empty():
        return []
    rng = np.random.default_rng(seed)
    pick = full[sorted(rng.choice(full.height, size=min(n, full.height), replace=False).tolist())]
    by_voyage = {
        v["voyage_id"][0]: v.sort("t")
        for v in tracks.filter(pl.col("voyage_id").is_in(pick["voyage_id"].implode())).partition_by(
            "voyage_id"
        )
    }
    fc = forecasts.filter(pl.col("sample_id").is_in(pick["sample_id"].implode()))
    out = []
    for s in pick.iter_rows(named=True):
        v = by_voyage[s["voyage_id"]]
        t0 = s["t0"]
        hist = v.filter((pl.col("t") <= t0) & (pl.col("t") >= t0 - pl.duration(minutes=60)))
        fut = v.filter(
            (pl.col("t") > t0) & (pl.col("t") <= t0 + pl.duration(minutes=max(horizons)))
        )
        preds: dict[str, Any] = {}
        errs: dict[str, Any] = {}
        for m in fc.filter(pl.col("sample_id") == s["sample_id"]).partition_by("model"):
            m = m.sort("horizon_min")
            mid = m["model"][0]
            preds[mid] = {
                "lat": _round(m["lat"].to_numpy()),
                "lon": _round(m["lon"].to_numpy()),
                "cee": _round(m["cee"].to_numpy(), 0),
                "cnn": _round(m["cnn"].to_numpy(), 0),
                "cen": _round(m["cen"].to_numpy(), 0),
            }
            hs = m["horizon_min"].to_list()
            tl = np.array([s[f"lat_{h}"] for h in hs], dtype=float)
            tn = np.array([s[f"lon_{h}"] for h in hs], dtype=float)
            dist = haversine_m(m["lat"].to_numpy(), m["lon"].to_numpy(), tl, tn) / 1000
            errs[mid] = _round(dist, 3)
        out.append(
            {
                "id": s["sample_id"],
                "mmsi": int(s["mmsi"]),
                "name": names.get(int(s["mmsi"])),
                "group": s["vessel_group"],
                "t0": int(t0.timestamp()),
                "lat0": round(float(s["lat0"]), 5),
                "lon0": round(float(s["lon0"]), 5),
                "hist": {"lat": _round(hist["lat"]), "lon": _round(hist["lon"])},
                "future": {
                    "t": [round((x - t0).total_seconds() / 60, 1) for x in fut["t"].to_list()],
                    "lat": _round(fut["lat"]),
                    "lon": _round(fut["lon"]),
                },
                "preds": preds,
                "errors_km": errs,
            }
        )
    return out


# --------------------------------------------------------------------------- anomalies


def with_event_ids(events: pl.DataFrame) -> pl.DataFrame:
    """Add a stable ``id`` (``type-mmsi-epoch``), used in URLs and case-study notes."""
    return events.with_columns(
        id=pl.concat_str(
            [
                pl.col("type"),
                pl.col("mmsi").cast(pl.String),
                pl.col("t_start").dt.epoch("s").cast(pl.String),
            ],
            separator="-",
        )
    )


def anomaly_payload(
    events: pl.DataFrame,
    points: pl.DataFrame,
    names: dict[int, str],
    per_type: int,
    case_studies: dict[str, str],
) -> list[dict[str, Any]]:
    """Top-scoring commercial events per type (plus every case study), with track context."""
    ev = with_event_ids(events.filter(pl.col("vessel_group").is_in(sorted(COMMERCIAL))))
    parts = [
        ev.filter(pl.col("type") == t).sort("score", descending=True).head(per_type)
        for t in ANOMALY_TYPES
    ]
    parts.append(ev.filter(pl.col("id").is_in(list(case_studies))))
    chosen = pl.concat(parts).unique("id", keep="first")
    pts = points.select("mmsi", "t", "lat", "lon").sort("mmsi", "t")

    def context(mmsi: int | None, t0: datetime, t1: datetime) -> dict[str, list[Any]] | None:
        if mmsi is None:
            return None
        c = pts.filter(
            (pl.col("mmsi") == mmsi)
            & (pl.col("t") >= t0 - pl.duration(minutes=60))
            & (pl.col("t") <= t1 + pl.duration(minutes=60))
        )
        if c.height > 400:
            c = c.gather_every(int(np.ceil(c.height / 400)))
        return {
            "t": c["t"].dt.epoch("s").to_list(),
            "lat": _round(c["lat"]),
            "lon": _round(c["lon"]),
        }

    out = []
    for e in chosen.sort("t_start").iter_rows(named=True):
        out.append(
            {
                "id": e["id"],
                "type": e["type"],
                "mmsi": int(e["mmsi"]),
                "mmsi2": None if e["mmsi2"] is None else int(e["mmsi2"]),
                "name": names.get(int(e["mmsi"])),
                "group": e["vessel_group"],
                "t_start": int(e["t_start"].timestamp()),
                "t_end": int(e["t_end"].timestamp()),
                "lat": round(float(e["lat"]), 5),
                "lon": round(float(e["lon"]), 5),
                "lat_end": round(float(e["lat_end"]), 5),
                "lon_end": round(float(e["lon_end"]), 5),
                "duration_min": round(float(e["duration_min"]), 1),
                "score": round(float(e["score"]), 3),
                "explanation": e["explanation"],
                "case_study": case_studies.get(e["id"]),
                "track": context(int(e["mmsi"]), e["t_start"], e["t_end"]),
                "track2": context(e["mmsi2"], e["t_start"], e["t_end"]),
            }
        )
    return out


# --------------------------------------------------------------------------- results


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _criteria_tracking(sim: dict[str, Any]) -> list[dict[str, Any]]:
    keys = ("name", "target", "result", "met")
    return [{"area": "Tracking", **{k: c[k] for k in keys}} for c in sim["criteria"]]


def results_payload(
    cfg: Config, voyages: pl.DataFrame, n_points: int, n_days: int
) -> dict[str, Any]:
    """Headline numbers, tables and the success-criteria list from the generated reports."""
    rep = Path(cfg.paths.reports)
    criteria: list[dict[str, Any]] = []
    tracking_rows: list[dict[str, Any]] = []
    sim = _read_json(rep / "tracking_sim.json")
    if sim:
        criteria += _criteria_tracking(sim)
        cv, imm = sim["eval"]["cv"], sim["eval"]["imm"]
        for seg in ("straight", "manoeuvre", "stopped", "moving"):
            tracking_rows.append(
                {
                    "segment": seg,
                    "cv_rmse": cv[seg]["rmse_m"],
                    "imm_rmse": imm[seg]["rmse_m"],
                    "cv_median": cv[seg]["median_m"],
                    "imm_median": imm[seg]["median_m"],
                    "cv_p95": cv[seg]["p95_m"],
                    "imm_p95": imm[seg]["p95_m"],
                }
            )
    nis = (
        _read_json(rep / "tracking_nis_attempt3.json")
        or _read_json(rep / "tracking_nis_attempt2.json")
        or _read_json(rep / "tracking_nis.json")
    )
    if nis:
        criteria.append({"area": "Tracking", **nis["criterion"]})
    test = _read_json(rep / "prediction_test.json")
    prediction = None
    verdict: list[dict[str, Any]] = []
    if test:
        prediction = {"split": _split_label(cfg, "test"), "rows": test["table"], "by_group": []}
        verdict = test["verdict"]
        criteria.append(
            {
                "area": "Prediction",
                "name": "Best ML model beats the best baseline at 60 and 120 min (≥ 15% at 120)",
                "target": "paired 95% CI excludes 0 at both horizons",
                "result": ", ".join(f"{v['horizon']} min: {100 * v['rel']:+.1f}%" for v in verdict),
                "met": bool(test["met_point"]),
            }
        )
        criteria.append(
            {
                "area": "Prediction",
                "name": "90% ellipse coverage of the best ML model",
                "target": "within 85–95% at every horizon",
                "result": "see the coverage chart",
                "met": bool(test["met_cov"]),
            }
        )
    an = _read_json(rep / "anomaly_eval.json")
    anomaly = None
    if an:
        rows = pl.DataFrame(an["injections"])
        by_mag = (
            rows.group_by("type", "magnitude")
            .agg(
                n=pl.len(),
                recall=pl.col("detected").mean(),
                matched=pl.col("matched_events").sum(),
                new=pl.col("new_events").sum(),
            )
            .with_columns(
                precision=pl.when(pl.col("new") > 0).then(pl.col("matched") / pl.col("new"))
            )
            .sort("type", "magnitude")
        )
        anomaly = {
            "by_magnitude": by_mag.select(
                "type", "magnitude", "n", "precision", "recall"
            ).to_dicts()
        }
        for c in an["criteria"]:
            criteria.append(
                {
                    "area": "Anomaly",
                    "name": f"{c['type']}: precision / recall in the target bucket",
                    "target": c.get("target", "see PLAN §9.4"),
                    "result": f"P {c['precision']:.2f} / R {c['recall']:.2f} (n={c['n']})",
                    "met": bool(c["met"]),
                }
            )
    t0, t1 = (str(x) for x in cfg.splits["test"])
    return {
        **_meta(cfg),
        "period": {
            "start": str(cfg.download.start),
            "end": str(cfg.download.end),
            "test_start": t0,
            "test_end": t1,
        },
        "counts": {
            "points": n_points,
            "vessels": int(voyages["mmsi"].n_unique()),
            "voyages": int(voyages.filter(pl.col("usable")).height),
            "days": n_days,
        },
        "models": _models(),
        "prediction": prediction,
        "verdict": verdict,
        "tracking": tracking_rows,
        "tracking_note": cfg.export.get("tracking_note"),
        "anomaly": anomaly,
        "criteria": criteria,
    }


# --------------------------------------------------------------------------- stage


def load_case_studies(path: Path) -> dict[str, str]:
    """Case-study notes keyed by event id (``configs/case_studies.yaml``), if present."""
    if not path.exists():
        return {}
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return {str(k): str(v).strip() for k, v in raw.items()}


def stage(cfg: Config) -> None:
    """CLI stage: write the site's data files to ``paths.site_data``."""
    base = Path(cfg.paths.processed)
    out = Path(cfg.paths.site_data)
    ex = cfg.export
    voyages = pl.read_parquet(base / "voyages.parquet")
    tracks = pl.read_parquet(base / "tracks.parquet")
    points = pl.read_parquet(base / "points.parquet")
    named = voyages.filter(pl.col("vessel_name").is_not_null()).unique("mmsi", keep="first")
    names = dict(zip(named["mmsi"].to_list(), named["vessel_name"].to_list(), strict=True))
    size = 0

    picked = pick_showcase_voyages(voyages, int(ex["n_showcase"]), seed=int(cfg.seed))
    payload = [
        voyage_payload(tracks.filter(pl.col("voyage_id") == m["voyage_id"]).sort("t"), m)
        for m in picked.iter_rows(named=True)
    ]
    size += write_json(out / "tracks.json", {**_meta(cfg), "voyages": payload})

    split = "test" if (base / "forecasts_test.parquet").exists() else "val"
    if (base / f"forecasts_{split}.parquet").exists():
        samples = pl.read_parquet(base / f"samples_{split}.parquet")
        forecasts = pl.read_parquet(base / f"forecasts_{split}.parquet").filter(
            pl.col("model").is_in([m[0] for m in MODELS])
        )
        horizons = list(cfg.prediction.horizons_min)
        preds = prediction_payload(
            samples, forecasts, tracks, names, horizons, int(ex["n_predictions"]), int(cfg.seed)
        )
        size += write_json(
            out / "predictions.json",
            {
                **_meta(cfg),
                "split": _split_label(cfg, split),
                "horizons": horizons,
                "models": _models(),
                "samples": preds,
            },
        )

    if (base / "anomalies.parquet").exists():
        events = pl.read_parquet(base / "anomalies.parquet")
        cases = load_case_studies(Path(str(ex.get("case_studies", "configs/case_studies.yaml"))))
        ev = anomaly_payload(events, points, names, int(ex["n_events_per_type"]), cases)
        counts = {str(k): int(v) for k, v in events.group_by("type").len().iter_rows()}
        size += write_json(out / "anomalies.json", {**_meta(cfg), "events": ev, "counts": counts})

    n_days = len(list((Path(cfg.paths.interim) / "days").glob("aoi_*.parquet")))
    size += write_json(out / "results.json", results_payload(cfg, voyages, points.height, n_days))
    if ex.get("onnx", True):
        from ais_sentinel.export.onnx_export import export_onnx

        size += export_onnx(cfg)
    log.info("export: %d voyages; %.1f KB total", len(payload), size / 1024)
