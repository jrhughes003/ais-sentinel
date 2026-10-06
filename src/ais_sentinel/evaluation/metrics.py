"""Prediction metrics with voyage-cluster bootstrap confidence intervals.

**Point error:** the haversine distance between the predicted and true position, in km.

**Probabilistic metrics:** each forecast carries a 2-D Gaussian (mean, covariance). For
truth y and prediction (μ, Σ), the squared Mahalanobis distance d² = (y−μ)ᵀ Σ⁻¹ (y−μ)
follows χ²₂ when the forecast is calibrated. Hence:

* **Coverage at level p** is the fraction of samples with d² ≤ χ²₂(p), i.e. truth falls
  inside the p-probability ellipse. A calibrated model gets about p.
* **NLL** is the Gaussian negative log-likelihood. Lower is better, and it rewards being both
  accurate and appropriately confident.
* **Ellipse area** at 90% is π · χ²₂(0.9) · √det Σ. It measures sharpness: narrower is
  better, *if* coverage holds.

**Cluster bootstrap:** samples from the same voyage are strongly correlated; ten anchors on
one straight run are not ten independent tests. Resampling individual samples would
understate uncertainty. We therefore resample *whole voyages* with replacement, recompute
the metric, and take the 2.5/97.5 percentiles over 1,000 resamples.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from numpy.typing import NDArray
from scipy.stats import chi2

from ais_sentinel.geo import haversine_m, latlon_to_enu
from ais_sentinel.prediction.baselines import Forecast

Array = NDArray[np.float64]


def truth_arrays(samples: pl.DataFrame, horizons_min: list[int]) -> tuple[Array, Array]:
    """Truth lat/lon as (M, H) arrays, NaN where the horizon is unavailable."""
    lat = np.column_stack(
        [samples[f"lat_{h}"].cast(pl.Float64).fill_null(np.nan).to_numpy() for h in horizons_min]
    )
    lon = np.column_stack(
        [samples[f"lon_{h}"].cast(pl.Float64).fill_null(np.nan).to_numpy() for h in horizons_min]
    )
    return lat, lon


def per_sample_scores(fc: Forecast, samples: pl.DataFrame) -> pl.DataFrame:
    """Long table: one row per (sample, horizon) with error_km, d2 and nll."""
    t_lat, t_lon = truth_arrays(samples, fc.horizons_min)
    p_lat, p_lon = fc.latlon()
    err_km = haversine_m(p_lat, p_lon, t_lat, t_lon) / 1000.0
    te, tn = latlon_to_enu(t_lat, t_lon, fc.lat0[:, None], fc.lon0[:, None])
    resid = np.stack([te, tn], axis=-1) - fc.mean_en
    cov = fc.cov + np.eye(2) * 1.0  # 1 m² jitter guards against singular covariances
    inv = np.linalg.inv(cov)
    d2 = np.einsum("mhi,mhij,mhj->mh", resid, inv, resid)
    logdet = np.log(np.linalg.det(cov))
    nll = 0.5 * (d2 + logdet + 2 * np.log(2 * np.pi))
    area90 = np.pi * chi2.ppf(0.9, 2) * np.sqrt(np.linalg.det(cov)) / 1e6  # km²
    m, h = err_km.shape
    return pl.DataFrame(
        {
            "model": [fc.model] * (m * h),
            "sample_id": np.repeat(fc.sample_id, h),
            "voyage_id": np.repeat(samples["voyage_id"].to_numpy(), h),
            "vessel_group": np.repeat(samples["vessel_group"].to_numpy(), h),
            "horizon_min": np.tile(fc.horizons_min, m),
            "error_km": err_km.ravel(),
            "d2": d2.ravel(),
            "nll": nll.ravel(),
            "area90_km2": area90.ravel(),
        }
    ).filter(pl.col("error_km").is_not_nan())


def cluster_bootstrap_ci(
    values: Array,
    clusters: NDArray[np.int64],
    stat: str = "mean",
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Point estimate and (lo, hi) percentile CI, resampling whole clusters.

    Args:
        values: per-sample values.
        clusters: integer cluster label per sample (e.g. voyage index).
        stat: ``"mean"`` or ``"median"``.
    """
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(clusters, return_inverse=True)
    k = len(uniq)
    point = float(np.mean(values) if stat == "mean" else np.median(values))
    if k < 2 or n_boot == 0:
        return point, float("nan"), float("nan")
    if stat == "mean":
        # Fast path: the mean of a resample = Σ cluster sums / Σ cluster counts.
        sums = np.bincount(inv, weights=values, minlength=k)
        counts = np.bincount(inv, minlength=k).astype(float)
        draws = rng.integers(0, k, size=(n_boot, k))
        boot = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    else:
        order = np.argsort(inv)
        groups = np.split(values[order], np.cumsum(np.bincount(inv, minlength=k))[:-1])
        boot = np.array(
            [
                np.median(np.concatenate([groups[j] for j in rng.integers(0, k, k)]))
                for _ in range(n_boot)
            ]
        )
    lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    return point, float(lo), float(hi)


def summarise(
    scores: pl.DataFrame, by: list[str], n_boot: int = 1000, seed: int = 0
) -> pl.DataFrame:
    """Aggregate per-sample scores into a metrics table with cluster-bootstrap CIs."""
    q50, q90 = chi2.ppf(0.5, 2), chi2.ppf(0.9, 2)
    rows = []
    for key, g in scores.group_by(by, maintain_order=True):
        clusters = g["voyage_id"].cast(pl.Categorical).to_physical().to_numpy().astype(np.int64)
        err = g["error_km"].to_numpy()
        mean, lo, hi = cluster_bootstrap_ci(err, clusters, "mean", n_boot, seed)
        d2 = g["d2"].to_numpy()
        cov50, _, _ = cluster_bootstrap_ci((d2 <= q50).astype(float), clusters, "mean", 0)
        cov90, c90lo, c90hi = cluster_bootstrap_ci(
            (d2 <= q90).astype(float), clusters, "mean", n_boot, seed
        )
        rows.append(
            {
                **dict(zip(by, key if isinstance(key, tuple) else (key,), strict=True)),
                "n": g.height,
                "voyages": len(np.unique(clusters)),
                "mean_km": mean,
                "mean_lo": lo,
                "mean_hi": hi,
                "median_km": float(np.median(err)),
                "p90_km": float(np.quantile(err, 0.9)),
                "cov50": cov50,
                "cov90": cov90,
                "cov90_lo": c90lo,
                "cov90_hi": c90hi,
                "nll": float(np.mean(g["nll"].to_numpy())),
                "area90_km2": float(np.median(g["area90_km2"].to_numpy())),
            }
        )
    return pl.DataFrame(rows).sort(by)


def paired_difference_ci(
    a: pl.DataFrame, b: pl.DataFrame, horizon: int, n_boot: int = 1000, seed: int = 0
) -> tuple[float, float, float]:
    """Mean of (error_a − error_b) on the samples both models scored, with a cluster CI.

    A negative value means model *a* is better. The CI excludes 0 when the difference is
    significant at the 5% level, accounting for within-voyage correlation.
    """
    ja = a.filter(pl.col("horizon_min") == horizon).select("sample_id", "voyage_id", ea="error_km")
    jb = b.filter(pl.col("horizon_min") == horizon).select("sample_id", eb="error_km")
    j = ja.join(jb, on="sample_id", how="inner")
    diff = (j["ea"] - j["eb"]).to_numpy()
    clusters = j["voyage_id"].cast(pl.Categorical).to_physical().to_numpy().astype(np.int64)
    return cluster_bootstrap_ci(diff, clusters, "mean", n_boot, seed)


def fit_cov_scale(scores: pl.DataFrame, target: float = 0.9) -> dict[int, float]:
    """Per-horizon covariance scale making the ``target`` ellipse coverage exact.

    Scaling Σ by s divides d² by s, so the scale is s = quantile_target(d²) / χ²₂(target).
    This is split-conformal calibration on the d² score: fit it on validation, then apply
    it unchanged to test.
    """
    q = chi2.ppf(target, 2)
    out: dict[int, float] = {}
    for (h,), g in scores.group_by(["horizon_min"]):
        out[int(h)] = float(np.quantile(g["d2"].to_numpy(), target) / q)
    return out
