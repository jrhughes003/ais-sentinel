"""B3: historical nearest-neighbour ("route analog") trajectory prediction.

Idea: ships in this region follow dredged channels and recurring routes. To predict where
a vessel goes next, find past vessels (from the **training split only**) that were at the
same place, heading the same way, and see where *they* went.

* **Library:** dense anchors from training voyages, each storing position (in a fixed
  regional ENU frame), course, speed and its true future displacement every
  ``grid_min`` minutes up to ``max_horizon_min``.
* **Query:** a KD-tree search in a scaled feature space. One unit corresponds to
  ``pos_scale_m`` of position difference or ``course_scale_deg`` of course difference
  (course is embedded as a scaled unit vector, so 359° and 1° are close).
* **Speed adjustment:** a neighbour moving at speed v_n reaches a point on the route
  ``h`` minutes ahead at time h. A query vessel at speed v_q reaches the same point at
  ``h · v_n / v_q``. So each neighbour's future is read off at time
  ``h · v_q / v_n``, clipped to a sane ratio range.
* **Prediction:** the inverse-distance-weighted mean of the neighbours' displacements.
  The weighted covariance of the neighbours, plus a small floor, is the uncertainty.
  That covariance is later recalibrated on validation.
* **Fallback:** if fewer than ``min_neighbours`` lie within ``max_dist`` units, the model
  falls back to dead reckoning for that sample. The fallback fraction is reported.

Displacements are in the regional frame. Between it and each anchor's tangent frame, the
axes differ by under 1° within the AOI, which is negligible at these scales.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from ais_sentinel.geo import KNOT_MS, latlon_to_enu
from ais_sentinel.prediction.baselines import Forecast, dead_reckoning
from ais_sentinel.prediction.samples import anchor_indices, interp_track

Array = NDArray[np.float64]


@dataclass(frozen=True)
class KNNParams:
    """Hyperparameters of the route-analog predictor (tuned on validation)."""

    k: int = 20
    pos_scale_m: float = 500.0
    course_scale_deg: float = 20.0
    max_dist: float = 3.0
    min_neighbours: int = 3
    grid_min: int = 5
    max_horizon_min: int = 180
    speed_ratio: tuple[float, float] = (0.6, 1.6)
    library_every_min: float = 5.0
    min_sog_kn: float = 2.0
    cov_floor_m: float = 50.0


@dataclass
class KNNLibrary:
    """Training-split analog library."""

    ref_lat: float
    ref_lon: float
    params: KNNParams
    tree: cKDTree
    speed_ms: Array  # (L,)
    future: Array  # (L, G, 2) displacement in the regional frame; NaN if unavailable
    grid_s: Array  # (G,)


def _features(e: Array, n: Array, cog_deg: Array, p: KNNParams) -> Array:
    c = np.radians(cog_deg)
    k = 1.0 / np.radians(p.course_scale_deg)  # chord ≈ angle for small differences
    return np.column_stack([e / p.pos_scale_m, n / p.pos_scale_m, k * np.sin(c), k * np.cos(c)])


def build_library(
    tracks: pl.DataFrame, train_voyage_ids: set[str], ref_lat: float, ref_lon: float, p: KNNParams
) -> KNNLibrary:
    """Build the analog library from training voyages only."""
    grid_s = (
        np.arange(p.grid_min, p.max_horizon_min + p.grid_min, p.grid_min, dtype=np.float64) * 60.0
    )
    feats, speeds, futures = [], [], []
    tr = tracks.filter(pl.col("voyage_id").is_in(sorted(train_voyage_ids))).sort("voyage_id", "t")
    for v in tr.partition_by("voyage_id", maintain_order=True):
        t = v["t"].dt.epoch("us").to_numpy() / 1e6
        if len(t) < 3:
            continue
        idx = anchor_indices(t, 0.0, p.library_every_min * 60)
        sog = v["sog_kn"].cast(pl.Float64).fill_null(np.nan).to_numpy()
        cog = v["cog_deg"].cast(pl.Float64).fill_null(np.nan).to_numpy()
        ok = (sog[idx] >= p.min_sog_kn) & np.isfinite(cog[idx])
        idx = idx[ok]
        if len(idx) == 0:
            continue
        e, n = latlon_to_enu(v["lat"].to_numpy(), v["lon"].to_numpy(), ref_lat, ref_lon)
        xy = np.column_stack([e, n])
        fut = np.stack(
            [interp_track(t, xy, t[idx] + g, 180.0) - xy[idx] for g in grid_s], axis=1
        )  # (A, G, 2)
        feats.append(_features(e[idx], n[idx], cog[idx], p))
        speeds.append(sog[idx] * KNOT_MS)
        futures.append(fut)
    if not feats:
        raise ValueError("empty kNN library")
    return KNNLibrary(
        ref_lat,
        ref_lon,
        p,
        cKDTree(np.vstack(feats)),
        np.concatenate(speeds),
        np.concatenate(futures),
        grid_s,
    )


def _read_future(fut: Array, grid_s: Array, t_query: Array) -> Array:
    """Interpolate neighbour futures (K, G, 2) at per-neighbour times (K,), returning (K, 2)."""
    g = np.concatenate([[0.0], grid_s])
    f = np.concatenate([np.zeros((fut.shape[0], 1, 2)), fut], axis=1)
    j = np.clip(np.searchsorted(g, t_query), 1, len(g) - 1)
    w = ((t_query - g[j - 1]) / (g[j] - g[j - 1]))[:, None]
    rows = np.arange(len(t_query))
    out = f[rows, j - 1] + w * (f[rows, j] - f[rows, j - 1])
    out[t_query > g[-1]] = np.nan
    return out


def knn_forecast(
    samples: pl.DataFrame, lib: KNNLibrary, horizons_min: list[int]
) -> tuple[Forecast, float]:
    """Predict with route analogs. Returns (forecast, fallback fraction)."""
    p = lib.params
    m = samples.height
    lat0 = samples["lat0"].to_numpy().astype(float)
    lon0 = samples["lon0"].to_numpy().astype(float)
    sog = samples["sog_kn"].cast(pl.Float64).fill_null(0.0).to_numpy()
    cog = samples["cog_deg"].cast(pl.Float64).fill_null(0.0).to_numpy()
    e, n = latlon_to_enu(lat0, lon0, lib.ref_lat, lib.ref_lon)
    dist, nb = lib.tree.query(_features(e, n, cog, p), k=p.k, distance_upper_bound=p.max_dist)
    dr = dead_reckoning(samples, horizons_min)
    H = len(horizons_min)
    mean_reg = np.full((m, H, 2), np.nan)
    cov = dr.cov.copy()
    v_q = np.maximum(sog * KNOT_MS, 0.1)
    for i in range(m):
        ok = np.isfinite(dist[i])
        if ok.sum() < p.min_neighbours:
            continue
        idx, d = nb[i][ok], dist[i][ok]
        w = 1.0 / (d + 0.1)
        ratio = np.clip(v_q[i] / lib.speed_ms[idx], *p.speed_ratio)
        for k, h in enumerate(horizons_min):
            pts = _read_future(lib.future[idx], lib.grid_s, h * 60.0 * ratio)
            good = np.all(np.isfinite(pts), axis=1)
            if good.sum() < p.min_neighbours:
                continue
            ww = w[good] / w[good].sum()
            mu = ww @ pts[good]
            dev = pts[good] - mu
            mean_reg[i, k] = mu
            cov[i, k] = np.einsum("k,ki,kj->ij", ww, dev, dev) + np.eye(2) * p.cov_floor_m**2
    have = np.isfinite(mean_reg[..., 0])
    mean = np.where(have[..., None], mean_reg, dr.mean_en)
    cov = np.where(have[..., None, None], cov, dr.cov)
    fc = Forecast(
        "knn_route", samples["sample_id"].to_list(), list(horizons_min), lat0, lon0, mean, cov
    )
    return fc, float(1.0 - have.mean())


def region_centre(region: dict[str, float]) -> tuple[float, float]:
    """Centre of the AOI box (used as the kNN library's frame origin)."""
    return (region["lat_min"] + region["lat_max"]) / 2, (region["lon_min"] + region["lon_max"]) / 2
