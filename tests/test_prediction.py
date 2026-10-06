"""Prediction samples, baselines and metrics."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest

from ais_sentinel.evaluation.metrics import (
    cluster_bootstrap_ci,
    fit_cov_scale,
    paired_difference_ci,
    per_sample_scores,
    summarise,
)
from ais_sentinel.geo import enu_to_latlon, sog_cog_to_enu_velocity
from ais_sentinel.prediction.baselines import Forecast, dead_reckoning
from ais_sentinel.prediction.samples import anchor_indices, build_samples, interp_track

H = [15, 30, 60, 120]
T0 = datetime(2023, 7, 1, tzinfo=UTC)


def test_interp_track_exact_bracketed_and_missing() -> None:
    t = np.array([0.0, 60.0, 120.0, 600.0])
    xy = np.array([[0.0, 0.0], [60.0, 0.0], [120.0, 0.0], [600.0, 0.0]])
    q = np.array([60.0, 90.0, 300.0, 700.0, -10.0])
    out = interp_track(t, xy, q, max_bracket_s=180)
    assert out[0, 0] == 60.0  # exact
    assert out[1, 0] == pytest.approx(90.0)  # bracketed within 3 min
    assert np.isnan(out[2, 0])  # bracket 120..600 too wide
    assert np.isnan(out[3, 0])  # beyond the track
    assert np.isnan(out[4, 0])  # before the track


def test_anchor_indices_respects_history_and_spacing() -> None:
    t = np.arange(0, 7200, 60.0)
    idx = anchor_indices(t, history_s=3600, every_s=600)
    assert t[idx[0]] == 3600
    assert np.all(np.diff(t[idx]) >= 600)


def straight_voyage(minutes: int, sog_kn: float = 10.0, cog_deg: float = 60.0) -> pl.DataFrame:
    """A constant-course, constant-speed voyage, with exact SOG/COG and filter state.

    Built by stepping one minute at a time in the local frame of the previous fix, so the
    course over ground stays constant, as for a ship holding a steady heading.
    """
    ve, vn = sog_cog_to_enu_velocity(sog_kn, cog_deg)
    lat, lon = [42.0], [-83.0]
    for _ in range(minutes - 1):
        la, lo = enu_to_latlon(ve * 60.0, vn * 60.0, lat[-1], lon[-1])
        lat.append(float(la))
        lon.append(float(lo))
    n = minutes
    return pl.DataFrame(
        {
            "voyage_id": ["v1"] * n,
            "mmsi": [316000001] * n,
            "t": [T0 + timedelta(minutes=i) for i in range(n)],
            "lat": lat,
            "lon": lon,
            "sog_kn": [sog_kn] * n,
            "cog_deg": [cog_deg] * n,
            "f_lat": lat,
            "f_lon": lon,
            "f_ve": [float(ve)] * n,
            "f_vn": [float(vn)] * n,
            "f_pee": [25.0] * n,
            "f_pnn": [25.0] * n,
            "f_pen": [0.0] * n,
        }
    ).with_columns(pl.col("t").cast(pl.Datetime("us", "UTC")))


VOYAGES = pl.DataFrame(
    {"voyage_id": ["v1"], "split": ["train"], "vessel_group": ["cargo"], "length_m": [180.0]}
)


def test_samples_stop_at_voyage_end_no_future_leakage() -> None:
    tr = straight_voyage(150)  # 150 min long: history 60 + up to 89 min of future
    s = build_samples(tr, VOYAGES, H, history_min=60, anchor_every_min=10)
    assert s.height > 0
    t_end = tr["t"].max()
    for h in H:
        valid = s.filter(pl.col(f"lat_{h}").is_not_null())
        assert (valid["t0"] + timedelta(minutes=h) <= t_end).all()
    assert s["lat_120"].null_count() == s.height  # never enough future for 120 min


def test_dead_reckoning_and_kf_are_exact_on_straight_line() -> None:
    tr = straight_voyage(240)
    s = build_samples(tr, VOYAGES, H, 60, 10)
    # 10 kn for h minutes = 0.3087 km per minute travelled.
    travelled_km = pl.col("horizon_min") * 10.0 * 1.852 / 60
    for fc in (dead_reckoning(s, H),):
        sc = per_sample_scores(fc, s).with_columns(rel=pl.col("error_km") / travelled_km)
        # Dead reckoning extrapolates in the anchor's tangent plane, while a constant course
        # is a rhumb line. They diverge laterally by ~ d^2 sin(cog) tan(lat) / (2R): 84 m
        # after 37 km at 42 N (0.23%), negligible next to km-scale forecast errors.
        assert sc["rel"].max() < 0.003, fc.model


def _gaussian_forecast(
    rng: np.random.Generator, m: int, true_sd: float, model_sd: float
) -> tuple[Forecast, pl.DataFrame]:
    """Truth = mean + N(0, true_sd² I); forecast claims model_sd. All at one anchor."""
    lat0 = np.full(m, 42.0)
    lon0 = np.full(m, -83.0)
    mean = rng.normal(0, 3000, (m, len(H), 2))
    truth = mean + rng.normal(0, true_sd, mean.shape)
    t_lat, t_lon = enu_to_latlon(truth[..., 0], truth[..., 1], lat0[:, None], lon0[:, None])
    cov = np.zeros((m, len(H), 2, 2))
    cov[..., 0, 0] = cov[..., 1, 1] = model_sd**2
    ids = [f"s{i}" for i in range(m)]
    samples = pl.DataFrame(
        {
            "sample_id": ids,
            "voyage_id": [f"v{i // 5}" for i in range(m)],
            "vessel_group": ["cargo"] * m,
            **{f"lat_{h}": t_lat[:, k] for k, h in enumerate(H)},
            **{f"lon_{h}": t_lon[:, k] for k, h in enumerate(H)},
        }
    )
    return Forecast("g", ids, H, lat0, lon0, mean, cov), samples


def test_coverage_is_nominal_for_calibrated_gaussian() -> None:
    fc, s = _gaussian_forecast(np.random.default_rng(0), 4000, 200.0, 200.0)
    table = summarise(per_sample_scores(fc, s), ["horizon_min"], n_boot=100)
    assert np.allclose(table["cov90"].to_numpy(), 0.9, atol=0.02)
    assert np.allclose(table["cov50"].to_numpy(), 0.5, atol=0.03)


def test_fit_cov_scale_recovers_variance_ratio_and_fixes_coverage() -> None:
    rng = np.random.default_rng(1)
    fc, s = _gaussian_forecast(rng, 4000, 400.0, 200.0)  # model 2x too confident
    scale = fit_cov_scale(per_sample_scores(fc, s))
    assert all(3.6 < v < 4.4 for v in scale.values())  # (400/200)^2 = 4
    fixed = summarise(per_sample_scores(fc.with_cov_scale(scale), s), ["horizon_min"], n_boot=0)
    assert np.allclose(fixed["cov90"].to_numpy(), 0.9, atol=0.02)


def test_cluster_bootstrap_wider_than_iid_for_correlated_samples() -> None:
    rng = np.random.default_rng(2)
    cluster_means = rng.normal(0, 1, 50)
    clusters = np.repeat(np.arange(50), 20)
    values = cluster_means[clusters] + rng.normal(0, 0.1, clusters.size)
    _, lo_c, hi_c = cluster_bootstrap_ci(values, clusters, n_boot=500)
    _, lo_i, hi_i = cluster_bootstrap_ci(values, np.arange(values.size), n_boot=500)
    assert (hi_c - lo_c) > 2 * (hi_i - lo_i)
    assert lo_c < 0 < hi_c  # true mean 0 lies inside


def test_paired_difference_detects_better_model() -> None:
    rng = np.random.default_rng(3)
    n = 600
    base = pl.DataFrame(
        {
            "sample_id": [f"s{i}" for i in range(n)],
            "voyage_id": [f"v{i // 6}" for i in range(n)],
            "horizon_min": [60] * n,
            "error_km": rng.gamma(2.0, 2.0, n),
        }
    )
    better = base.with_columns(error_km=pl.col("error_km") * 0.7)
    d, _lo, hi = paired_difference_ci(better, base, 60, n_boot=300)
    assert d < 0
    assert hi < 0


def test_filter_forecasts_from_track_stage_snapshots_are_accurate_on_straight_line() -> None:
    from ais_sentinel.prediction.filter_forecast import cv_forecast, imm_forecast
    from ais_sentinel.tracking.imm import IMMParams
    from ais_sentinel.tracking.kf import CVParams
    from ais_sentinel.tracking.pipeline import track_voyage

    pts = straight_voyage(300).select("voyage_id", "mmsi", "t", "lat", "lon", "sog_kn", "cog_deg")
    fixes, snaps = track_voyage(pts, CVParams(q_accel=0.01), IMMParams(), 60, 10)
    assert snaps is not None
    s = build_samples(fixes, VOYAGES, H, 60, 10)
    travelled_km = pl.col("horizon_min") * 10.0 * 1.852 / 60
    for fc in (cv_forecast(s, snaps, H, 0.01), imm_forecast(s, snaps, H, IMMParams())):
        sc = per_sample_scores(fc, s).with_columns(rel=pl.col("error_km") / travelled_km)
        assert sc["rel"].max() < 0.01, fc.model  # within 1% of distance travelled
        # Uncertainty grows with horizon.
        area = sc.group_by("horizon_min").agg(pl.col("area90_km2").median()).sort("horizon_min")
        assert area["area90_km2"].is_sorted()
