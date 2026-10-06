"""Cleaning, segmentation and split-assignment tests (including leakage guarantees)."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import polars as pl
import pytest

from ais_sentinel.data.clean import clean_points
from ais_sentinel.data.segment import assign_voyages, split_bounds, summarise_voyages
from ais_sentinel.data.vessel_types import vessel_group, vessel_group_expr

T0 = datetime(2023, 7, 1, 12, 0, tzinfo=UTC)
Row = tuple[int, int, float, float, float]


def make_points(rows: list[Row]) -> pl.DataFrame:
    """rows: (mmsi, minutes after T0, lat, lon, sog)."""
    return pl.DataFrame(
        {
            "mmsi": [r[0] for r in rows],
            "t": [T0 + timedelta(minutes=r[1]) for r in rows],
            "lat": [r[2] for r in rows],
            "lon": [r[3] for r in rows],
            "sog_kn": [r[4] for r in rows],
            "cog_deg": [90.0] * len(rows),
            "heading_deg": [511.0] * len(rows),
            "vessel_type": [70] * len(rows),
            "vessel_name": ["SHIP"] * len(rows),
            "length_m": [180.0] * len(rows),
            "transceiver": ["A"] * len(rows),
        },
        schema_overrides={"t": pl.Datetime("us", "UTC"), "sog_kn": pl.Float32},
    )


def straight(mmsi: int, n: int, start_min: int = 0) -> list[Row]:
    """About 10 kn eastward: 0.004 deg lon/min at 42 N is ~330 m/min."""
    return [(mmsi, start_min + i, 42.0, -83.0 + 0.004 * (i + start_min), 10.0) for i in range(n)]


def test_clean_drops_invalid_and_duplicates_and_nulls_sentinels() -> None:
    rows = straight(316000001, 5)
    rows += [rows[2]]  # duplicate (mmsi, t)
    rows += [(3160001, 1, 42.0, -83.0, 1.0)]  # invalid MMSI (7 digits)
    rows += [(993160001, 1, 42.0, -83.0, 1.0)]  # aid to navigation
    df = make_points(rows).with_columns(
        sog_kn=pl.when(pl.int_range(pl.len()) == 0).then(102.3).otherwise(pl.col("sog_kn"))
    )
    df = pl.concat([df, make_points(straight(316000009, 1)).with_columns(transceiver=pl.lit("52"))])
    out, rep = clean_points(df, max_sog_kn=102.2, max_implied_speed_kn=60)
    assert rep.dropped_invalid_mmsi == 2
    assert rep.dropped_malformed == 1
    assert rep.dropped_duplicates == 1
    assert out.height == 5
    assert out["heading_deg"].null_count() == 5  # 511 -> null
    assert out["sog_kn"].null_count() == 1  # 102.3 -> null
    assert out.select(pl.col("t").is_sorted()).item()


def test_isolated_spike_flagged_not_dropped() -> None:
    rows = straight(316000001, 10)
    m, i, lat, lon, s = rows[5]
    rows[5] = (m, i, lat + 0.1, lon, s)  # 11 km jump in 1 minute, then back
    out, rep = clean_points(make_points(rows), 102.2, 60)
    assert rep.flagged_spikes == 1
    assert out.height == 10
    assert out.filter(pl.col("spike"))["t"].to_list() == [T0 + timedelta(minutes=5)]


def test_voyage_split_on_gap() -> None:
    rows = straight(316000001, 20) + straight(316000001, 20, start_min=60)  # 41-min gap
    df, _ = clean_points(make_points(rows), 102.2, 60)
    df = assign_voyages(df, max_gap_min=30)
    assert sorted(df["voyage_id"].unique()) == [
        "316000001-20230701T1200",
        "316000001-20230701T1300",
    ]


SPLITS = {
    "train": [date(2023, 5, 1), date(2023, 8, 31)],
    "val": [date(2023, 9, 2), date(2023, 9, 30)],
    "test": [date(2023, 10, 2), date(2023, 10, 31)],
}


def _voyages_from_starts(starts: list[datetime], minutes: int = 120) -> pl.DataFrame:
    frames = []
    for k, s in enumerate(starts):
        pts = make_points(
            [(316000000 + k, i, 42.0, -83.0 + 0.004 * i, 10.0) for i in range(minutes)]
        )
        frames.append(pts.with_columns(t=pl.lit(s) + pl.duration(minutes=pl.int_range(pl.len()))))
    df, _ = clean_points(pl.concat(frames), 102.2, 60)
    df = assign_voyages(df, 30)
    return summarise_voyages(df, split_bounds(SPLITS), min_points=10, min_duration_min=30)


def test_split_assignment_and_boundary_voyages_excluded() -> None:
    v = _voyages_from_starts(
        [
            datetime(2023, 7, 1, 0, tzinfo=UTC),  # train
            datetime(2023, 8, 31, 23, tzinfo=UTC),  # runs into buffer day: none
            datetime(2023, 9, 1, 6, tzinfo=UTC),  # buffer day: none
            datetime(2023, 9, 15, 6, tzinfo=UTC),  # val
            datetime(2023, 10, 31, 21, tzinfo=UTC),  # test (ends 22:59 on Oct 31)
            datetime(2023, 10, 31, 23, tzinfo=UTC),  # runs past test end: none
        ]
    )
    assert v.sort("t_start")["split"].to_list() == ["train", "none", "none", "val", "test", "none"]


def test_leakage_no_voyage_in_two_splits_and_split_time_ranges_disjoint() -> None:
    starts = [datetime(2023, 5, 1, tzinfo=UTC) + timedelta(hours=7 * k) for k in range(600)]
    v = _voyages_from_starts(starts, minutes=180)
    assert v["voyage_id"].n_unique() == v.height
    used = v.filter(pl.col("split") != "none")
    spans = used.group_by("split").agg(pl.col("t_start").min(), pl.col("t_end").max())
    d = {r["split"]: (r["t_start"], r["t_end"]) for r in spans.iter_rows(named=True)}
    assert d["train"][1] < d["val"][0]
    assert d["val"][1] < d["test"][0]
    # Separation is at least the 1-day buffer minus one voyage length.
    assert d["val"][0] - d["train"][1] >= timedelta(hours=21)


@pytest.mark.parametrize(
    ("code", "group"),
    [
        (70, "cargo"),
        (89, "tanker"),
        (60, "passenger"),
        (31, "tug_tow"),
        (30, "fishing"),
        (37, "pleasure"),
        (35, "other"),
        (None, "unknown"),
        (1004, "cargo"),
        (1025, "tug_tow"),
    ],
)
def test_vessel_group_python_and_polars_agree(code: int | None, group: str) -> None:
    assert vessel_group(code) == group
    df = pl.DataFrame({"vessel_type": [code]}, schema={"vessel_type": pl.Int32})
    assert df.select(vessel_group_expr()).item() == group
