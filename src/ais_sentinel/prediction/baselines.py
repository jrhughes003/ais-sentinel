"""Physics baselines for multi-horizon position prediction.

Every predictor returns a :class:`Forecast`. For each sample and horizon it holds a
predicted mean displacement and a 2×2 covariance, both in the local ENU frame anchored at
that sample's last observed fix (metres). This common interface scores every model
identically, including on the probabilistic metrics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from numpy.typing import NDArray

from ais_sentinel.geo import enu_to_latlon, latlon_to_enu, sog_cog_to_enu_velocity
from ais_sentinel.tracking.models import cv_process_noise, cv_transition

Array = NDArray[np.float64]


@dataclass
class Forecast:
    """Predictions for M samples at H horizons."""

    model: str
    sample_id: list[str]
    horizons_min: list[int]
    lat0: Array  # (M,) anchor latitude
    lon0: Array  # (M,) anchor longitude
    mean_en: Array  # (M, H, 2) predicted displacement from the anchor (m)
    cov: Array  # (M, H, 2, 2) covariance (m²)

    def latlon(self) -> tuple[Array, Array]:
        """Predicted positions as (lat, lon), each (M, H)."""
        return enu_to_latlon(
            self.mean_en[..., 0], self.mean_en[..., 1], self.lat0[:, None], self.lon0[:, None]
        )

    def with_cov_scale(self, scale: dict[int, float]) -> Forecast:
        """Return a copy whose covariance is multiplied by a per-horizon scalar."""
        s = np.array([scale[h] for h in self.horizons_min])[None, :, None, None]
        return Forecast(
            self.model,
            self.sample_id,
            self.horizons_min,
            self.lat0,
            self.lon0,
            self.mean_en,
            self.cov * s,
        )


def _anchors(samples: pl.DataFrame) -> tuple[Array, Array]:
    return samples["lat0"].to_numpy().astype(float), samples["lon0"].to_numpy().astype(float)


def dead_reckoning(samples: pl.DataFrame, horizons_min: list[int]) -> Forecast:
    """B0: extrapolate the last *reported* SOG/COG in a straight line.

    The covariance is a placeholder: isotropic, with standard deviation 10% of the distance
    travelled (floor 50 m). It is recalibrated on validation data by
    :func:`ais_sentinel.evaluation.metrics.fit_cov_scale`.
    """
    lat0, lon0 = _anchors(samples)
    sog = samples["sog_kn"].cast(pl.Float64).fill_null(0.0).to_numpy()
    cog = samples["cog_deg"].cast(pl.Float64).fill_null(0.0).to_numpy()
    ve, vn = sog_cog_to_enu_velocity(sog, cog)
    h_s = np.asarray(horizons_min, dtype=float) * 60
    mean = np.stack([np.outer(ve, h_s), np.outer(vn, h_s)], axis=-1)
    sigma = np.maximum(0.1 * np.linalg.norm(mean, axis=-1), 50.0)
    cov = np.zeros((*mean.shape[:2], 2, 2))
    cov[..., 0, 0] = cov[..., 1, 1] = sigma**2
    return Forecast(
        "dead_reckoning", samples["sample_id"].to_list(), list(horizons_min), lat0, lon0, mean, cov
    )


def kf_extrapolation(
    samples: pl.DataFrame, horizons_min: list[int], q_accel: float, vel_var: float = 0.25
) -> Forecast:
    """B1: propagate the CV Kalman filter's *filtered* state forward in time.

    The filtered state uses only fixes up to the anchor, so the forecast is causal. The
    covariance is the filter's own ``F P Fᵀ + Q``. Per-fix velocity variance is not stored
    by the tracking stage, so a nominal ``vel_var`` (m²/s²) is used.
    """
    lat0, lon0 = _anchors(samples)
    fe, fn = latlon_to_enu(samples["f_lat"].to_numpy(), samples["f_lon"].to_numpy(), lat0, lon0)
    x0 = np.column_stack([fe, fn, samples["f_ve"].to_numpy(), samples["f_vn"].to_numpy()])
    P0 = np.zeros((len(lat0), 4, 4))
    P0[:, 0, 0] = samples["f_pee"].to_numpy()
    P0[:, 1, 1] = samples["f_pnn"].to_numpy()
    P0[:, 0, 1] = P0[:, 1, 0] = samples["f_pen"].to_numpy()
    P0[:, 2, 2] = P0[:, 3, 3] = vel_var
    means, covs = [], []
    for h in horizons_min:
        dt = h * 60.0
        F = cv_transition(dt)
        means.append((x0 @ F.T)[:, :2])
        covs.append((F @ P0 @ F.T + cv_process_noise(dt, q_accel))[:, :2, :2])
    return Forecast(
        "kf_cv",
        samples["sample_id"].to_list(),
        list(horizons_min),
        lat0,
        lon0,
        np.stack(means, axis=1),
        np.stack(covs, axis=1),
    )
