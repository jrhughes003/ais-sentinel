"""Voyage segmentation and time-based split assignment.

A **voyage** is a maximal run of one vessel's reports with no gap longer than
``max_gap_min``. Splits are assigned per voyage. A voyage belongs to a split only if its
*entire* time span lies inside that split's dates. Voyages that straddle a boundary, or that
fall in a buffer day, get ``split = "none"`` and are never used for training or evaluation.
This guarantees that no voyage contributes to two splits.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, time, timedelta

import polars as pl

from ais_sentinel.data.vessel_types import vessel_group_expr

SPLITS = ("train", "val", "test")


def assign_voyages(df: pl.DataFrame, max_gap_min: float) -> pl.DataFrame:
    """Add ``voyage_id`` (``"<mmsi>-<YYYYmmddTHHMM>"`` of the first fix) to sorted points."""
    t_s = pl.col("t").dt.epoch("us") // 1_000_000
    gap = (t_s - t_s.shift(1).over("mmsi")) > max_gap_min * 60
    df = df.with_columns(_seg=gap.fill_null(True).cum_sum().over("mmsi"))
    start = pl.col("t").min().over("mmsi", "_seg")
    return df.with_columns(
        voyage_id=pl.concat_str(
            [pl.col("mmsi").cast(pl.String), start.dt.strftime("%Y%m%dT%H%M")], separator="-"
        )
    ).drop("_seg")


def split_bounds(splits_cfg: dict[str, list[date]]) -> dict[str, tuple[datetime, datetime]]:
    """Convert inclusive split dates to half-open UTC datetime intervals ``[start, end)``."""
    out: dict[str, tuple[datetime, datetime]] = {}
    for name in SPLITS:
        d0, d1 = (date.fromisoformat(str(d)) for d in splits_cfg[name])
        out[name] = (
            datetime.combine(d0, time(0), tzinfo=UTC),
            datetime.combine(d1 + timedelta(days=1), time(0), tzinfo=UTC),
        )
    return out


def summarise_voyages(
    df: pl.DataFrame,
    bounds: dict[str, tuple[datetime, datetime]],
    min_points: int,
    min_duration_min: float,
) -> pl.DataFrame:
    """One row per voyage, with static attributes, kinematic summaries and the split label."""
    v = (
        df.group_by("voyage_id")
        .agg(
            mmsi=pl.col("mmsi").first(),
            t_start=pl.col("t").min(),
            t_end=pl.col("t").max(),
            n_points=pl.len(),
            vessel_type=pl.col("vessel_type").drop_nulls().mode().first(),
            vessel_name=pl.col("vessel_name").drop_nulls().mode().first(),
            length_m=pl.col("length_m").median(),
            transceiver=pl.col("transceiver").drop_nulls().mode().first(),
            moving_frac=(pl.col("sog_kn") > 1.0).mean(),
            path_km=pl.col("dist_prev_m").fill_null(0).sum() / 1000,
            n_spikes=pl.col("spike").sum(),
        )
        .with_columns(
            duration_min=(pl.col("t_end") - pl.col("t_start")).dt.total_seconds() / 60,
        )
        .with_columns(vessel_group_expr())
    )
    split = pl.lit("none")
    for name, (lo, hi) in bounds.items():
        inside = (pl.col("t_start") >= lo) & (pl.col("t_end") < hi)
        split = pl.when(inside).then(pl.lit(name)).otherwise(split)
    usable = (pl.col("n_points") >= min_points) & (pl.col("duration_min") >= min_duration_min)
    return (
        v.with_columns(split=split, usable=usable)
        .sort("mmsi", "t_start")
        .select(
            "voyage_id",
            "mmsi",
            "vessel_name",
            "vessel_type",
            "vessel_group",
            "length_m",
            "transceiver",
            "t_start",
            "t_end",
            "duration_min",
            "n_points",
            "moving_frac",
            "path_km",
            "n_spikes",
            "split",
            "usable",
        )
    )
