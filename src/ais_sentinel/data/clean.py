"""Clean AOI points: invalid identifiers, sentinel values, duplicates and kinematic spikes.

Policy:

* **Drop** rows that cannot be used at all: null keys, MMSIs that are not ship stations,
  rows misaligned by a stray delimiter (transceiver not A/B), and exact duplicate
  (MMSI, time) reports.
* **Null out** the AIS "not available" sentinels: SOG 102.3, COG 360, heading 511.
* **Flag, but keep,** kinematically implausible isolated points (``spike``). The tracker
  rejects them by gating, and the anomaly detector reports them, so deleting them here would
  hide exactly the behaviour we want to detect.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import polars as pl

from ais_sentinel.geo import EARTH_RADIUS_M, KNOT_MS

log = logging.getLogger(__name__)

# Ship-station MMSIs are 9 digits whose first three digits (the Maritime Identification
# Digits) lie in 201–775. Coast stations (00MID), SAR aircraft (111MID), aids to navigation
# (99MID) and handheld/test identities fall outside this range.
MMSI_MIN = 201_000_000
MMSI_MAX = 775_999_999


@dataclass
class CleanReport:
    """Row counts removed or modified by each cleaning step."""

    rows_in: int = 0
    dropped_null_keys: int = 0
    dropped_invalid_mmsi: int = 0
    dropped_malformed: int = 0
    dropped_duplicates: int = 0
    nulled_sog: int = 0
    nulled_cog: int = 0
    nulled_heading: int = 0
    flagged_spikes: int = 0
    rows_out: int = 0
    notes: list[str] = field(default_factory=list)


def haversine_expr(lat1: pl.Expr, lon1: pl.Expr, lat2: pl.Expr, lon2: pl.Expr) -> pl.Expr:
    """Great-circle distance in metres, as a polars expression."""
    p1, p2 = lat1.radians(), lat2.radians()
    dp = p2 - p1
    dl = (lon2 - lon1).radians()
    a = (dp / 2).sin() ** 2 + p1.cos() * p2.cos() * (dl / 2).sin() ** 2
    return 2 * EARTH_RADIUS_M * a.clip(0.0, 1.0).sqrt().arcsin()


def add_step_kinematics(df: pl.DataFrame) -> pl.DataFrame:
    """Add per-vessel distance/time to the previous and next fix, and implied speeds (kn).

    ``df`` must be sorted by (mmsi, t).
    """
    prev_lat, prev_lon = pl.col("lat").shift(1).over("mmsi"), pl.col("lon").shift(1).over("mmsi")
    next_lat, next_lon = pl.col("lat").shift(-1).over("mmsi"), pl.col("lon").shift(-1).over("mmsi")
    t_s = pl.col("t").dt.epoch("us").cast(pl.Float64) / 1e6
    return df.with_columns(
        dist_prev_m=haversine_expr(prev_lat, prev_lon, pl.col("lat"), pl.col("lon")),
        dist_next_m=haversine_expr(pl.col("lat"), pl.col("lon"), next_lat, next_lon),
        dt_prev_s=t_s - t_s.shift(1).over("mmsi"),
        dt_next_s=t_s.shift(-1).over("mmsi") - t_s,
    ).with_columns(
        v_in_kn=pl.col("dist_prev_m") / pl.col("dt_prev_s").clip(lower_bound=1.0) / KNOT_MS,
        v_out_kn=pl.col("dist_next_m") / pl.col("dt_next_s").clip(lower_bound=1.0) / KNOT_MS,
    )


def clean_points(
    df: pl.DataFrame, max_sog_kn: float, max_implied_speed_kn: float
) -> tuple[pl.DataFrame, CleanReport]:
    """Apply the cleaning policy described in the module docstring.

    Args:
        df: canonical-schema points (any order).
        max_sog_kn: SOG values above this are treated as "not available" (102.3 sentinel).
        max_implied_speed_kn: a point whose implied speed both *into* and *out of* it
            exceeds this is flagged as an isolated spike.

    Returns:
        The cleaned frame sorted by (mmsi, t), with a boolean ``spike`` column, and a report.
    """
    rep = CleanReport(rows_in=df.height)

    keyed = df.drop_nulls(["mmsi", "t", "lat", "lon"])
    rep.dropped_null_keys = df.height - keyed.height

    valid = keyed.filter(pl.col("mmsi").is_between(MMSI_MIN, MMSI_MAX))
    rep.dropped_invalid_mmsi = keyed.height - valid.height

    # A row shifted by a stray comma has a non-A/B value in its last (transceiver) field.
    well_formed = valid.filter(
        pl.col("transceiver").is_null() | pl.col("transceiver").is_in(["A", "B"])
    )
    rep.dropped_malformed = valid.height - well_formed.height
    valid = well_formed

    dedup = valid.sort("mmsi", "t", maintain_order=True).unique(
        ["mmsi", "t"], keep="first", maintain_order=True
    )
    rep.dropped_duplicates = valid.height - dedup.height

    rep.nulled_sog = int(dedup.select((pl.col("sog_kn") > max_sog_kn).sum()).item())
    rep.nulled_cog = int(dedup.select((pl.col("cog_deg") >= 360).sum()).item())
    rep.nulled_heading = int(dedup.select((pl.col("heading_deg") >= 360).sum()).item())
    out = dedup.with_columns(
        sog_kn=pl.when(pl.col("sog_kn") > max_sog_kn).then(None).otherwise(pl.col("sog_kn")),
        cog_deg=pl.when(pl.col("cog_deg") >= 360).then(None).otherwise(pl.col("cog_deg")),
        heading_deg=pl.when(pl.col("heading_deg") >= 360)
        .then(None)
        .otherwise(pl.col("heading_deg")),
    )

    out = add_step_kinematics(out).with_columns(
        spike=(
            (pl.col("v_in_kn") > max_implied_speed_kn) & (pl.col("v_out_kn") > max_implied_speed_kn)
        ).fill_null(False)
    )
    rep.flagged_spikes = int(out["spike"].sum())
    rep.rows_out = out.height
    log.info("clean: %s", rep)
    return out, rep
