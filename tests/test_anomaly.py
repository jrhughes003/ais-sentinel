"""Anomaly detectors, learned context and synthetic injection."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest

from ais_sentinel.anomaly.context import learn_context
from ais_sentinel.anomaly.detectors import (
    VesselHome,
    detect_jumps,
    detect_loitering,
    detect_rendezvous,
    detect_route_deviation,
)
from ais_sentinel.anomaly.inject import (
    inject_deviation,
    inject_jump,
    inject_loiter,
    inject_rendezvous,
)
from ais_sentinel.geo import enu_to_latlon
from tests.test_knn import REF, l_route_voyage

T0 = datetime(2023, 6, 1, tzinfo=UTC)


def _pts(v: pl.DataFrame, mmsi: int = 316000001, group: str = "cargo") -> pl.DataFrame:
    return v.with_columns(
        mmsi=pl.lit(mmsi, dtype=pl.Int64),
        vessel_group=pl.lit(group),
        sog_kn=pl.col("sog_kn").cast(pl.Float32),  # as in the real cleaned data
        cog_deg=pl.col("cog_deg").cast(pl.Float32),
    ).select("mmsi", "t", "lat", "lon", "sog_kn", "cog_deg", "vessel_group", "voyage_id")


def _stationary(
    mmsi: int, lat: float, lon: float, hours: float, start: datetime, rng: np.random.Generator
) -> pl.DataFrame:
    n = int(hours * 20)
    la, lo = enu_to_latlon(rng.normal(0, 8, n), rng.normal(0, 8, n), lat, lon)
    return pl.DataFrame(
        {
            "mmsi": [mmsi] * n,
            "t": [start + timedelta(minutes=3 * i) for i in range(n)],
            "lat": la,
            "lon": lo,
            "sog_kn": rng.uniform(0, 0.5, n),
            "cog_deg": rng.uniform(0, 360, n),
            "vessel_group": ["tug_tow"] * n,
            "voyage_id": [f"s{mmsi}"] * n,
        },
        schema_overrides={
            "t": pl.Datetime("us", "UTC"),
            "sog_kn": pl.Float32,
            "cog_deg": pl.Float32,
        },
    )


@pytest.fixture(scope="module")
def train() -> pl.DataFrame:
    """Twelve laker transits on the L lane plus a busy port berth used by 4 tugs."""
    rng = np.random.default_rng(0)
    frames = [
        _pts(
            l_route_voyage(f"tr{i}", T0 + timedelta(hours=3 * i), rng.uniform(8, 12), rng),
            316000100 + i,
        )
        for i in range(12)
    ]
    port_lat, port_lon = enu_to_latlon(-14000.0, -9000.0, *REF)
    frames += [
        _stationary(366000000 + k, float(port_lat), float(port_lon), 20, T0, rng) for k in range(4)
    ]
    return pl.concat(frames).with_columns(
        pl.col("sog_kn").cast(pl.Float32), pl.col("cog_deg").cast(pl.Float32)
    )


@pytest.fixture(scope="module")
def ctx(train: pl.DataFrame):  # type: ignore[no-untyped-def]
    return learn_context(train, *REF)


def test_context_learns_port_and_lane(ctx, train: pl.DataFrame) -> None:  # type: ignore[no-untyped-def]
    port_lat, port_lon = enu_to_latlon(-14000.0, -9000.0, *REF)
    assert ctx.in_port(np.array([port_lat]), np.array([port_lon]))[0]
    lane = train.filter(pl.col("mmsi") == 316000100)
    assert not ctx.in_port(lane["lat"].to_numpy(), lane["lon"].to_numpy()).any()
    dens, prob = ctx.traffic_at(
        lane["lat"].to_numpy(), lane["lon"].to_numpy(), lane["cog_deg"].to_numpy()
    )
    assert np.median(dens) >= 10
    assert np.nanmedian(prob) > 0.5


def test_jump_detected_and_jitter_ignored() -> None:
    rng = np.random.default_rng(1)
    v = _pts(l_route_voyage("a", T0, 10.0, rng))
    assert detect_jumps(v).height == 0
    mod, _ = inject_jump(v, T0 + timedelta(minutes=40), 5.0, 1, rng)
    ev = detect_jumps(mod)
    assert ev.height == 1  # out-and-back spike merged into one event
    assert "implausible" in ev["explanation"][0]
    small, _ = inject_jump(v, T0 + timedelta(minutes=40), 0.3, 1, rng)
    assert detect_jumps(small).height == 0


def test_loiter_detected_away_from_port_not_in_port(ctx) -> None:  # type: ignore[no-untyped-def]
    rng = np.random.default_rng(2)
    v = _pts(l_route_voyage("b", T0 + timedelta(days=30), 10.0, rng))
    mod, _ = inject_loiter(v, T0 + timedelta(days=30, minutes=30), 3.0, rng)
    ev = detect_loitering(mod, ctx, VesselHome())
    assert ev.height == 1
    assert abs(ev["duration_min"][0] - 180) < 10
    port_lat, port_lon = enu_to_latlon(-14000.0, -9000.0, *REF)
    moored = _stationary(
        366000099, float(port_lat), float(port_lon), 6, T0 + timedelta(days=30), rng
    )
    assert detect_loitering(moored, ctx, VesselHome()).height == 0


def test_route_deviation_detected_for_large_offset_only(ctx) -> None:  # type: ignore[no-untyped-def]
    rng = np.random.default_rng(3)
    v = _pts(l_route_voyage("c", T0 + timedelta(days=30), 10.0, rng))
    assert detect_route_deviation(v, ctx).height == 0
    mod, _ = inject_deviation(v, T0 + timedelta(days=30, minutes=20), 3.0, 40, rng)
    assert detect_route_deviation(mod, ctx).height >= 1


def test_rendezvous_detected(ctx) -> None:  # type: ignore[no-untyped-def]
    rng = np.random.default_rng(4)
    v = _pts(l_route_voyage("d", T0 + timedelta(days=30), 10.0, rng))
    mod, _ = inject_rendezvous(v, T0 + timedelta(days=30, minutes=30), 2.0, 999000001, rng)
    ev = detect_rendezvous(mod, ctx, VesselHome())
    assert ev.height == 1
    assert {ev["mmsi"][0], ev["mmsi2"][0]} == {316000001, 999000001}
    assert ev["duration_min"][0] >= 100


def test_rendezvous_in_port_is_ignored(ctx) -> None:  # type: ignore[no-untyped-def]
    rng = np.random.default_rng(5)
    port_lat, port_lon = enu_to_latlon(-14000.0, -9000.0, *REF)
    a = _stationary(366000001, float(port_lat), float(port_lon), 4, T0 + timedelta(days=30), rng)
    b = _stationary(366000002, float(port_lat), float(port_lon), 4, T0 + timedelta(days=30), rng)
    assert detect_rendezvous(pl.concat([a, b]), ctx, VesselHome()).height == 0
