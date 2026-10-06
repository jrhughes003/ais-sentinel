"""Forecast baselines B1 (CV-KF) and B2 (IMM), extrapolating filter states from anchors.

Both start from the *filtered* (causal) state at the anchor fix, stored by the tracking
stage in ``filter_states.parquet``, so a forecast uses no data after its anchor time.

* **B1, CV-KF:** exact linear-Gaussian prediction, ``x_h = F(h) x`` and
  ``P_h = F P Fᵀ + Q(h)``.
* **B2, IMM predictor:** each mode's posterior is propagated with its own motion model,
  and the forecast is the moment-matched mixture weighted by the anchor's mode
  probabilities. Its covariance widens when the modes disagree, a cheap way to express
  "it might be turning".

  By default the modes are **frozen** over the horizon (``switching=False``). The
  filter's Markov prior (mean cruising time of about 15 min) is right for tracking,
  where each new fix corrects the mode, but for a 2 h forecast with no new fixes it
  pushes almost all probability into the stationary and turning modes. The averaged
  mean then lands far short of a vessel that simply keeps going (a 50% error on a
  straight line at 120 min). With ``switching=True``, interaction and mode switching
  are applied in sub-steps of ``substep_s``. Both variants are compared on validation.

States live in each voyage's ENU frame (anchored at the voyage's first fix). Forecast means
are converted to lat/lon and then into the anchor's own frame, the frame shared by all
models. The covariance is not rotated between the two tangent planes: their axes differ by
the meridian convergence (under 1° across the AOI), a negligible effect.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from numpy.typing import NDArray

from ais_sentinel.geo import enu_to_latlon, latlon_to_enu
from ais_sentinel.prediction.baselines import Forecast
from ais_sentinel.tracking.imm import IMMParams, mixture_moments, predict_mode, transition_matrix
from ais_sentinel.tracking.models import cv_process_noise, cv_transition

Array = NDArray[np.float64]


def _join_states(samples: pl.DataFrame, states: pl.DataFrame) -> pl.DataFrame:
    j = samples.select("sample_id", "voyage_id", "t0", "lat0", "lon0").join(
        states, on=["voyage_id", "t0"], how="left"
    )
    missing = j["ref_lat"].null_count()
    if missing:
        raise ValueError(f"{missing} samples have no stored filter state")
    return j


def _to_anchor_frame(
    e: Array, n: Array, ref_lat: Array, ref_lon: Array, lat0: Array, lon0: Array
) -> Array:
    """Voyage-frame ENU (M, H) -> anchor-frame ENU (M, H, 2)."""
    lat, lon = enu_to_latlon(e, n, ref_lat[:, None], ref_lon[:, None])
    ea, na = latlon_to_enu(lat, lon, lat0[:, None], lon0[:, None])
    return np.stack([ea, na], axis=-1)


def cv_forecast(
    samples: pl.DataFrame, states: pl.DataFrame, horizons_min: list[int], q_accel: float
) -> Forecast:
    """B1: CV-KF extrapolation with its full predicted covariance."""
    j = _join_states(samples, states)
    x0 = np.asarray(j["cv_x"].to_list(), dtype=float)  # (M, 4)
    P0 = np.asarray(j["cv_P"].to_list(), dtype=float).reshape(-1, 4, 4)
    means_e, means_n, covs = [], [], []
    for h in horizons_min:
        dt = h * 60.0
        F = cv_transition(dt)
        x = x0 @ F.T
        means_e.append(x[:, 0])
        means_n.append(x[:, 1])
        covs.append((F @ P0 @ F.T + cv_process_noise(dt, q_accel))[:, :2, :2])
    lat0 = j["lat0"].to_numpy().astype(float)
    lon0 = j["lon0"].to_numpy().astype(float)
    mean = _to_anchor_frame(
        np.column_stack(means_e),
        np.column_stack(means_n),
        j["ref_lat"].to_numpy(),
        j["ref_lon"].to_numpy(),
        lat0,
        lon0,
    )
    return Forecast(
        "kf_cv", j["sample_id"].to_list(), list(horizons_min), lat0, lon0, mean, np.stack(covs, 1)
    )


def imm_predict(
    mode_x: Array,
    mode_P: Array,
    mu: Array,
    horizons_s: list[float],
    p: IMMParams,
    substep_s: float,
    switching: bool = False,
) -> tuple[Array, Array]:
    """IMM predictor for one anchor.

    Args:
        mode_x: (3, 5) per-mode posterior states; mode_P: (3, 5, 5); mu: (3,).
        horizons_s: increasing horizons in seconds.

    Returns:
        (means (H, 5), covariances (H, 5, 5)) of the moment-matched mixture.
    """
    xs, Ps, w = mode_x.copy(), mode_P.copy(), mu.copy()
    t = 0.0
    out_x, out_P = [], []
    for h in horizons_s:
        while t < h - 1e-9:
            dt = min(substep_s, h - t)
            pi = transition_matrix(dt, p) if switching else np.eye(3)
            c = np.maximum(w @ pi, 1e-300)
            mix = (pi * w[:, None]) / c[None, :]
            nx, nP = np.zeros_like(xs), np.zeros_like(Ps)
            for m in range(3):
                x0, P0 = mixture_moments(mix[:, m], xs, Ps)
                nx[m], nP[m], _ = predict_mode(m, x0, P0, dt, p)
            xs, Ps, w = nx, nP, c / c.sum()
            t += dt
        mx, mP = mixture_moments(w, xs, Ps)
        out_x.append(mx)
        out_P.append(mP)
    return np.asarray(out_x), np.asarray(out_P)


def imm_forecast(
    samples: pl.DataFrame,
    states: pl.DataFrame,
    horizons_min: list[int],
    params: IMMParams,
    substep_s: float = 300.0,
    switching: bool = False,
) -> Forecast:
    """B2: IMM-predictor extrapolation (mode mixture over the horizon)."""
    j = _join_states(samples, states)
    mx = np.asarray(j["imm_mode_x"].to_list(), dtype=float).reshape(-1, 3, 5)
    mP = np.asarray(j["imm_mode_P"].to_list(), dtype=float).reshape(-1, 3, 5, 5)
    mu = np.asarray(j["imm_mu"].to_list(), dtype=float)
    hs = [h * 60.0 for h in horizons_min]
    m = len(mu)
    me = np.zeros((m, len(hs)))
    mn = np.zeros_like(me)
    cov = np.zeros((m, len(hs), 2, 2))
    for i in range(m):
        xh, Ph = imm_predict(mx[i], mP[i], mu[i], hs, params, substep_s, switching)
        me[i], mn[i] = xh[:, 0], xh[:, 1]
        cov[i] = Ph[:, :2, :2]
    lat0 = j["lat0"].to_numpy().astype(float)
    lon0 = j["lon0"].to_numpy().astype(float)
    mean = _to_anchor_frame(me, mn, j["ref_lat"].to_numpy(), j["ref_lon"].to_numpy(), lat0, lon0)
    name = "imm_switching" if switching else "imm"
    return Forecast(name, j["sample_id"].to_list(), list(horizons_min), lat0, lon0, mean, cov)
