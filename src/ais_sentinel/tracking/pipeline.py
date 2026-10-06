"""Run the tracker over every usable voyage and store filtered and smoothed tracks.

Output ``tracks.parquet`` has one row per fix. Alongside the raw lat/lon it holds:

* ``f_*``: **filtered (causal)** estimates. They use only data up to that fix, so they
  are safe inputs for forecasting.
* ``s_*``: **smoothed** estimates (RTS). They use future fixes, so they are for display
  and anomaly analysis only.
* Position covariance in metres² (``*_pee``, ``*_pnn``, ``*_pen``), the step's NIS, and
  accepted/reinit flags.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import polars as pl

from ais_sentinel.config import Config
from ais_sentinel.geo import enu_to_latlon, latlon_to_enu, sog_cog_to_enu_velocity
from ais_sentinel.tracking.kf import CVParams, rts_smooth, run_cv_filter

log = logging.getLogger(__name__)


def cv_params_from_config(cfg: Config) -> CVParams:
    """Build CV filter parameters from ``cfg.tracking``."""
    tr = cfg.tracking
    return CVParams(
        q_accel=float(tr.cv["q_accel"]),
        r_pos_m=float(tr.cv["r_pos_m"]),
        gate_prob=float(tr.gate_prob),
        max_consecutive_rejects=int(tr.max_consecutive_rejects),
    )


def track_voyage(v: pl.DataFrame, params: CVParams) -> pl.DataFrame:
    """Filter and smooth one voyage's points (sorted by time)."""
    lat = v["lat"].to_numpy()
    lon = v["lon"].to_numpy()
    lat0, lon0 = float(lat[0]), float(lon[0])
    e, n = latlon_to_enu(lat, lon, lat0, lon0)
    t = (v["t"].dt.epoch("us").to_numpy() / 1e6).astype(float)
    sog = v["sog_kn"].cast(pl.Float64).fill_null(np.nan).to_numpy()
    cog = v["cog_deg"].cast(pl.Float64).fill_null(np.nan).to_numpy()
    ve, vn = sog_cog_to_enu_velocity(sog, cog)
    res = run_cv_filter(t, np.column_stack([e, n]), params, np.column_stack([ve, vn]))
    xs, Ps = rts_smooth(res)
    f_lat, f_lon = enu_to_latlon(res.x[:, 0], res.x[:, 1], lat0, lon0)
    s_lat, s_lon = enu_to_latlon(xs[:, 0], xs[:, 1], lat0, lon0)
    return pl.DataFrame(
        {
            "voyage_id": v["voyage_id"],
            "mmsi": v["mmsi"],
            "t": v["t"],
            "lat": lat,
            "lon": lon,
            "f_lat": f_lat,
            "f_lon": f_lon,
            "f_ve": res.x[:, 2],
            "f_vn": res.x[:, 3],
            "f_pee": res.P[:, 0, 0],
            "f_pnn": res.P[:, 1, 1],
            "f_pen": res.P[:, 0, 1],
            "s_lat": s_lat,
            "s_lon": s_lon,
            "s_ve": xs[:, 2],
            "s_vn": xs[:, 3],
            "s_pee": Ps[:, 0, 0],
            "s_pnn": Ps[:, 1, 1],
            "s_pen": Ps[:, 0, 1],
            "nis": res.nis,
            "accepted": res.accepted,
            "reinit": res.reinit,
        }
    )


def track_all(points: pl.DataFrame, params: CVParams) -> pl.DataFrame:
    """Track every usable voyage in ``points``."""
    usable = points.filter(pl.col("usable")).sort("voyage_id", "t")
    out = [track_voyage(v, params) for v in usable.partition_by("voyage_id", maintain_order=True)]
    return pl.concat(out) if out else pl.DataFrame()


def stage(cfg: Config) -> None:
    """CLI stage: track all usable voyages and write ``tracks.parquet``."""
    base = Path(cfg.paths.processed)
    points = pl.read_parquet(base / "points.parquet")
    tracks = track_all(points, cv_params_from_config(cfg))
    tracks.write_parquet(base / "tracks.parquet", compression="zstd")
    nis = tracks["nis"].drop_nans().drop_nulls()
    log.info(
        "track: %d fixes, %d voyages; rejected %.2f%%, reinit %d; NIS mean %.2f, P(NIS>5.99)=%.3f",
        tracks.height,
        tracks["voyage_id"].n_unique(),
        100 * (1 - tracks["accepted"].mean()),  # type: ignore[operator]
        int(tracks["reinit"].sum()),
        float(nis.mean()),  # type: ignore[arg-type]
        float((nis > 5.991).mean()),  # type: ignore[arg-type]
    )
