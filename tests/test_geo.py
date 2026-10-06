"""Geodesy tests, including property-based round trips."""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from ais_sentinel.geo import (
    KNOT_MS,
    ecef_to_geodetic,
    enu_to_latlon,
    enu_velocity_to_sog_cog,
    geodetic_to_ecef,
    haversine_m,
    latlon_to_enu,
    sog_cog_to_enu_velocity,
)

LAT0, LON0 = 42.25, -82.95  # AOI centre

lats = st.floats(41.0, 43.5)
lons = st.floats(-84.0, -82.0)


@settings(max_examples=200, deadline=None)
@given(lats, lons)
def test_enu_round_trip_submetre(lat: float, lon: float) -> None:
    e, n = latlon_to_enu(lat, lon, LAT0, LON0)
    lat2, lon2 = enu_to_latlon(e, n, LAT0, LON0)
    assert haversine_m(lat, lon, lat2, lon2) < 0.5


@settings(max_examples=200, deadline=None)
@given(lats, lons)
def test_enu_distance_matches_haversine(lat: float, lon: float) -> None:
    e, n = latlon_to_enu(lat, lon, LAT0, LON0)
    d_enu = float(np.hypot(e, n))
    d_h = float(haversine_m(LAT0, LON0, lat, lon))
    # Sphere vs ellipsoid differ by up to ~0.5%; tangent-plane shortening is negligible here.
    assert abs(d_enu - d_h) <= 0.006 * d_h + 0.5


def test_enu_axes_point_east_and_north() -> None:
    e, n = latlon_to_enu(LAT0 + 0.01, LON0, LAT0, LON0)
    assert abs(e) < 1e-6
    assert 1110 < n < 1112  # 0.01 deg latitude ~ 1111 m
    e, n = latlon_to_enu(LAT0, LON0 + 0.01, LAT0, LON0)
    assert 820 < e < 830  # 0.01 deg longitude ~ 111.3 km * cos(42.25 deg) / 100
    # A parallel of latitude curves poleward of the tangent plane's east axis by
    # e^2 tan(lat) / (2R) ~ 0.049 m at 826 m: real geometry, not an error.
    assert np.isclose(n, e**2 * np.tan(np.radians(LAT0)) / (2 * 6_378_137.0), rtol=0.02)


@given(st.floats(-89.0, 89.0), st.floats(-179.0, 179.0), st.floats(-100.0, 1000.0))
def test_ecef_round_trip(lat: float, lon: float, h: float) -> None:
    lat2, lon2, h2 = ecef_to_geodetic(geodetic_to_ecef(lat, lon, h))
    assert np.isclose(lat2, lat, atol=1e-9)
    assert np.isclose(lon2, lon, atol=1e-9)
    assert np.isclose(h2, h, atol=1e-3)


def test_haversine_known_value() -> None:
    # One minute of latitude is ~1 nautical mile on the mean sphere.
    assert np.isclose(haversine_m(42.0, -83.0, 42.0 + 1 / 60, -83.0), 1853.2, atol=1.0)


@given(st.floats(0.1, 40.0), st.floats(0.0, 359.9))
def test_velocity_round_trip(sog: float, cog: float) -> None:
    ve, vn = sog_cog_to_enu_velocity(sog, cog)
    assert np.isclose(np.hypot(ve, vn), sog * KNOT_MS)
    s2, c2 = enu_velocity_to_sog_cog(ve, vn)
    assert np.isclose(s2, sog)
    assert min(abs(c2 - cog), 360 - abs(c2 - cog)) < 1e-6


def test_cog_convention_clockwise_from_north() -> None:
    ve, vn = sog_cog_to_enu_velocity(10.0, 90.0)  # due east
    assert ve > 5
    assert abs(vn) < 1e-9


def test_vectorised_anchors_match_scalar_calls() -> None:
    rng = np.random.default_rng(0)
    lat0 = rng.uniform(41.5, 43.0, 5)
    lon0 = rng.uniform(-83.5, -82.4, 5)
    lat = lat0[:, None] + rng.uniform(-0.2, 0.2, (5, 4))
    lon = lon0[:, None] + rng.uniform(-0.2, 0.2, (5, 4))
    e, n = latlon_to_enu(lat, lon, lat0[:, None], lon0[:, None])
    for i in range(5):
        es, ns = latlon_to_enu(lat[i], lon[i], lat0[i], lon0[i])
        assert np.allclose(e[i], es)
        assert np.allclose(n[i], ns)
    lat2, lon2 = enu_to_latlon(e, n, lat0[:, None], lon0[:, None])
    assert np.allclose(lat2, lat, atol=1e-8)
    assert np.allclose(lon2, lon, atol=1e-8)
