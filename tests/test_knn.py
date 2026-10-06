"""Route-analog (kNN) baseline tests on a synthetic L-shaped shipping lane."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl

from ais_sentinel.evaluation.metrics import per_sample_scores
from ais_sentinel.geo import KNOT_MS, enu_to_latlon
from ais_sentinel.prediction.baselines import dead_reckoning
from ais_sentinel.prediction.knn import KNNParams, build_library, knn_forecast
from ais_sentinel.prediction.samples import build_samples

REF = (42.25, -82.95)
H = [15, 30, 60]


def l_route_voyage(
    vid: str, start: datetime, speed_kn: float, rng: np.random.Generator
) -> pl.DataFrame:
    """East for 15 km, then a sharp turn north for 20 km; one fix per minute."""
    v = speed_kn * KNOT_MS * 60  # metres per minute
    pts, cogs = [], []
    e = n = 0.0
    while e < 15_000:
        pts.append((e, n))
        cogs.append(90.0)
        e += v
    while n < 20_000:
        pts.append((e, n))
        cogs.append(0.0)
        n += v
    xy = np.asarray(pts) + rng.normal(0, 5, (len(pts), 2))
    lat, lon = enu_to_latlon(xy[:, 0] - 10_000, xy[:, 1] - 5_000, *REF)
    k = len(pts)
    return pl.DataFrame(
        {
            "voyage_id": [vid] * k,
            "mmsi": [316000000] * k,
            "t": [start + timedelta(minutes=i) for i in range(k)],
            "lat": lat,
            "lon": lon,
            "sog_kn": [speed_kn] * k,
            "cog_deg": cogs,
        }
    ).with_columns(pl.col("t").cast(pl.Datetime("us", "UTC")))


def test_knn_follows_the_route_around_the_corner_where_dead_reckoning_cannot() -> None:
    rng = np.random.default_rng(0)
    t0 = datetime(2023, 6, 1, tzinfo=UTC)
    train = [
        l_route_voyage(f"tr{i}", t0 + timedelta(hours=3 * i), rng.uniform(8, 12), rng)
        for i in range(12)
    ]
    lib = build_library(
        pl.concat(train), {f"tr{i}" for i in range(12)}, *REF, KNNParams(library_every_min=1.0)
    )
    test = l_route_voyage("te", t0 + timedelta(days=30), 10.0, rng)
    meta = pl.DataFrame(
        {"voyage_id": ["te"], "split": ["test"], "vessel_group": ["cargo"], "length_m": [150.0]}
    )
    s = build_samples(test, meta, H, history_min=20, anchor_every_min=5)
    # Anchors heading east within ~5 km of the corner: the future turns north.
    s = s.filter(pl.col("cog_deg") == 90.0)
    fc, fallback = knn_forecast(s, lib, H)
    assert fallback < 0.05
    knn = per_sample_scores(fc, s).filter(pl.col("horizon_min") == 30)
    dr = per_sample_scores(dead_reckoning(s, H), s).filter(pl.col("horizon_min") == 30)
    assert knn["error_km"].mean() < 0.3 * dr["error_km"].mean()  # type: ignore[operator]


def test_knn_falls_back_to_dead_reckoning_far_from_any_route() -> None:
    rng = np.random.default_rng(1)
    t0 = datetime(2023, 6, 1, tzinfo=UTC)
    lib = build_library(l_route_voyage("tr", t0, 10.0, rng), {"tr"}, *REF, KNNParams())
    far = l_route_voyage("far", t0 + timedelta(days=1), 10.0, rng).with_columns(
        lat=pl.col("lat") + 0.3  # ~33 km north of the lane
    )
    meta = pl.DataFrame(
        {"voyage_id": ["far"], "split": ["test"], "vessel_group": ["cargo"], "length_m": [150.0]}
    )
    s = build_samples(far, meta, H, 20, 5)
    fc, fallback = knn_forecast(s, lib, H)
    assert fallback == 1.0
    np.testing.assert_allclose(fc.mean_en, dead_reckoning(s, H).mean_en)
