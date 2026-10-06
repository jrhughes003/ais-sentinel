"""Run both trackers (CV-KF and IMM) over every usable voyage.

Outputs in ``paths.processed``:

* ``tracks.parquet``: one row per fix, with the raw lat/lon plus:
  * ``f_*``: **CV-KF filtered** (causal) estimates;
  * ``i_*``: **IMM filtered** (causal) estimates, mode probabilities and NIS;
  * ``s_*``: **IMM smoothed** estimates (RTS on the moment-matched IMM output). These use
    *future* fixes, so they are for display and anomaly analysis only, never for forecasts.
* ``filter_states.parquet``: full filter states at forecast **anchor** fixes (see
  :mod:`ais_sentinel.prediction.samples`). That is the CV state and covariance, and for
  each IMM mode its posterior state, covariance and probability. The filter-extrapolation
  baselines propagate these forward. States are in the voyage's ENU frame, anchored at
  its first fix (``ref_lat``, ``ref_lon``).

Voyages are independent, so they are processed in parallel worker processes.
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from ais_sentinel.config import Config
from ais_sentinel.geo import enu_to_latlon, latlon_to_enu, sog_cog_to_enu_velocity
from ais_sentinel.prediction.samples import anchor_indices
from ais_sentinel.tracking.imm import IMMParams, run_imm
from ais_sentinel.tracking.kf import CVParams, rts_smooth, run_cv_filter

log = logging.getLogger(__name__)


def cv_params_from_config(cfg: Config) -> CVParams:
    """CV filter parameters from ``cfg.tracking.cv`` (values chosen by the simulation study)."""
    tr = cfg.tracking
    return CVParams(
        q_accel=float(tr.cv["q_accel"]),
        r_pos_m=float(tr.cv["r_pos_m"]),
        gate_prob=float(tr.cv.get("gate_prob", tr.gate_prob)),
        max_consecutive_rejects=int(tr.max_consecutive_rejects),
    )


def imm_params_from_config(cfg: Config) -> IMMParams:
    """IMM parameters from ``cfg.tracking.imm``."""
    d: dict[str, Any] = {}
    for k, v in dict(cfg.tracking.get("imm", {})).items():
        # Cast explicitly: YAML 1.1 reads "2e-07" (no decimal point) as a string.
        d[k] = tuple(float(x) for x in v) if isinstance(v, list) else float(v)
    d.setdefault("max_consecutive_rejects", int(cfg.tracking.max_consecutive_rejects))
    return IMMParams(**d)


def _enu(v: pl.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    lat, lon = v["lat"].to_numpy(), v["lon"].to_numpy()
    lat0, lon0 = float(lat[0]), float(lon[0])
    e, n = latlon_to_enu(lat, lon, lat0, lon0)
    sog = v["sog_kn"].cast(pl.Float64).fill_null(np.nan).to_numpy()
    cog = v["cog_deg"].cast(pl.Float64).fill_null(np.nan).to_numpy()
    ve, vn = sog_cog_to_enu_velocity(sog, cog)
    return np.column_stack([e, n]), np.column_stack([ve, vn]), lat, lat0, lon0


def track_voyage(
    v: pl.DataFrame, cvp: CVParams, immp: IMMParams, history_min: float, every_min: float
) -> tuple[pl.DataFrame, pl.DataFrame | None]:
    """Run both filters on one voyage (sorted by time).

    Returns (per-fix table, anchor snapshot table or None).
    """
    z, vel, lat, lat0, lon0 = _enu(v)
    t = v["t"].dt.epoch("us").to_numpy() / 1e6
    idx = anchor_indices(t, history_min * 60, every_min * 60)
    cv = run_cv_filter(t, z, cvp, vel)
    imm = run_imm(t, z, immp, vel, snapshot_idx=idx)
    xs, Ps = rts_smooth(imm)
    f_lat, f_lon = enu_to_latlon(cv.x[:, 0], cv.x[:, 1], lat0, lon0)
    i_lat, i_lon = enu_to_latlon(imm.x[:, 0], imm.x[:, 1], lat0, lon0)
    s_lat, s_lon = enu_to_latlon(xs[:, 0], xs[:, 1], lat0, lon0)
    fixes = pl.DataFrame(
        {
            "voyage_id": v["voyage_id"],
            "mmsi": v["mmsi"],
            "t": v["t"],
            "lat": lat,
            "lon": v["lon"].to_numpy(),
            "sog_kn": v["sog_kn"],
            "cog_deg": v["cog_deg"],
            "f_lat": f_lat,
            "f_lon": f_lon,
            "f_ve": cv.x[:, 2],
            "f_vn": cv.x[:, 3],
            "f_pee": cv.P[:, 0, 0],
            "f_pnn": cv.P[:, 1, 1],
            "f_pen": cv.P[:, 0, 1],
            "f_nis": cv.nis,
            "f_accepted": cv.accepted,
            "i_lat": i_lat,
            "i_lon": i_lon,
            "i_ve": imm.x[:, 2],
            "i_vn": imm.x[:, 3],
            "i_omega": imm.x[:, 4],
            "i_pee": imm.P[:, 0, 0],
            "i_pnn": imm.P[:, 1, 1],
            "i_pen": imm.P[:, 0, 1],
            "i_mu_stationary": imm.mu[:, 0],
            "i_mu_cruising": imm.mu[:, 1],
            "i_mu_turning": imm.mu[:, 2],
            "nis": imm.nis,
            "accepted": imm.accepted,
            "reinit": imm.reinit,
            "s_lat": s_lat,
            "s_lon": s_lon,
            "s_ve": xs[:, 2],
            "s_vn": xs[:, 3],
            "s_pee": Ps[:, 0, 0],
            "s_pnn": Ps[:, 1, 1],
            "s_pen": Ps[:, 0, 1],
        }
    )
    if len(idx) == 0:
        return fixes, None
    k = len(idx)
    snaps = pl.DataFrame(
        {
            "voyage_id": v["voyage_id"].gather(idx),
            "t0": v["t"].gather(idx),
            "ref_lat": np.full(k, lat0),
            "ref_lon": np.full(k, lon0),
            "cv_x": cv.x[idx].tolist(),
            "cv_P": cv.P[idx].reshape(k, -1).tolist(),
            "imm_mu": imm.mu[idx].tolist(),
            "imm_mode_x": imm.snap_x.reshape(k, -1).tolist(),
            "imm_mode_P": imm.snap_P.reshape(k, -1).tolist(),
        }
    )
    return fixes, snaps


def _worker(args: tuple[list[pl.DataFrame], CVParams, IMMParams, float, float]) -> Any:
    voyages, cvp, immp, hist, every = args
    fixes, snaps = [], []
    for v in voyages:
        f, s = track_voyage(v, cvp, immp, hist, every)
        fixes.append(f)
        if s is not None:
            snaps.append(s)
    return pl.concat(fixes), (pl.concat(snaps) if snaps else None)


def track_all(
    points: pl.DataFrame,
    cvp: CVParams,
    immp: IMMParams,
    history_min: float,
    every_min: float,
    workers: int | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Track every usable voyage, in parallel. Returns (tracks, filter_states)."""
    usable = points.filter(pl.col("usable")).sort("voyage_id", "t")
    voyages = usable.select(
        "voyage_id", "mmsi", "t", "lat", "lon", "sog_kn", "cog_deg"
    ).partition_by("voyage_id", maintain_order=True)
    workers = workers or max(1, min(4, (os.cpu_count() or 2)))
    # Balance work: deal voyages round-robin into many chunks.
    n_chunks = max(1, workers * 8)
    chunks = [voyages[i::n_chunks] for i in range(n_chunks) if voyages[i::n_chunks]]
    jobs = [(c, cvp, immp, history_min, every_min) for c in chunks]
    if workers == 1:
        results = [_worker(j) for j in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_worker, jobs))
    tracks = pl.concat([r[0] for r in results]).sort("voyage_id", "t")
    snaps = pl.concat([r[1] for r in results if r[1] is not None]).sort("voyage_id", "t0")
    return tracks, snaps


def stage(cfg: Config) -> None:
    """CLI stage: track all usable voyages; write tracks and anchor filter states."""
    base = Path(cfg.paths.processed)
    points = pl.read_parquet(base / "points.parquet")
    pc = cfg.prediction
    tracks, snaps = track_all(
        points,
        cv_params_from_config(cfg),
        imm_params_from_config(cfg),
        float(pc.history_min),
        float(pc.anchor_every_min),
        workers=int(cfg.tracking.get("workers", 4)),
    )
    tracks.write_parquet(base / "tracks.parquet", compression="zstd")
    snaps.write_parquet(base / "filter_states.parquet", compression="zstd")
    for name, nis_col, acc_col in (("cv", "f_nis", "f_accepted"), ("imm", "nis", "accepted")):
        nis = tracks[nis_col].drop_nans().drop_nulls()
        log.info(
            "track[%s]: %d fixes; rejected %.2f%%; NIS mean %.2f; P(NIS>5.99)=%.3f",
            name,
            tracks.height,
            100 * (1 - float(tracks[acc_col].mean())),  # type: ignore[arg-type]
            float(nis.mean()),  # type: ignore[arg-type]
            float((nis > 5.991).mean()),  # type: ignore[arg-type]
        )
