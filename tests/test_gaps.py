"""Gap detector rules."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import polars as pl

from ais_sentinel.anomaly.gaps import detect_gaps

REGION = {"lat_min": 41.4, "lat_max": 43.1, "lon_min": -83.6, "lon_max": -82.3}
T0 = datetime(2023, 7, 1, 10, tzinfo=UTC)


def track(
    mmsi: int, minutes: list[int], lat: float = 42.3, lon0: float = -83.0, sog: float = 10.0
) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "mmsi": [mmsi] * len(minutes),
            "t": [T0 + timedelta(minutes=m) for m in minutes],
            "lat": [lat] * len(minutes),
            "lon": [lon0 + 0.004 * m for m in minutes],
            "sog_kn": [sog] * len(minutes),
            "vessel_group": ["cargo"] * len(minutes),
        }
    ).with_columns(pl.col("t").cast(pl.Datetime("us", "UTC")), pl.col("sog_kn").cast(pl.Float32))


def test_moving_gap_inside_aoi_is_detected_with_explanation() -> None:
    pts = track(316000001, [*range(0, 30), *range(75, 100)])  # 46-minute silence
    ev = detect_gaps(pts, REGION, min_gap_min=30)
    assert ev.height == 1
    row = ev.row(0, named=True)
    assert round(row["duration_min"]) == 46
    assert "Silent for 46 min" in row["explanation"]


def test_short_gap_moored_vessel_and_edge_gaps_are_ignored() -> None:
    short = track(316000001, [*range(0, 30), *range(50, 60)])  # 21 min
    moored = track(316000002, [*range(0, 30), *range(90, 100)], sog=0.0)
    edge = track(316000003, [*range(0, 30), *range(90, 100)], lat=41.41)  # 1 km from edge
    ev = detect_gaps(pl.concat([short, moored, edge]), REGION, min_gap_min=30, edge_buffer_km=3)
    assert ev.height == 0


def test_gap_spanning_missing_data_day_is_ignored() -> None:
    pts = track(316000001, [0, 1, 2, 60 * 24 * 2, 60 * 24 * 2 + 1]).with_columns(
        lon=pl.lit(-83.0)  # keep it inside the AOI; only the timing matters here
    )  # resumes 2 days later
    ok_days = {date(2023, 7, 1), date(2023, 7, 3)}  # 2023-07-02 missing from our download
    assert detect_gaps(pts, REGION, available_days=ok_days).height == 0
    all_days = ok_days | {date(2023, 7, 2)}
    assert detect_gaps(pts, REGION, available_days=all_days).height == 1
