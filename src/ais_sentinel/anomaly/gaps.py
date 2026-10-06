"""AIS gap ("going dark") detection.

A **gap** is a silence of at least ``min_gap_min`` in one vessel's reports, while the
vessel was moving before and after (SOG ≥ ``min_sog_kn``), and with both ends well inside
the area of interest. Silences that start or end near the AOI edge are usually vessels
leaving or entering coverage, not going dark. Moored vessels are excluded because Class B
units in particular often stop transmitting at the dock.

Each event carries a plain-language explanation for analysts. A later refinement weights
events by local receiver coverage (PLAN §5.3).
"""

from __future__ import annotations

from collections.abc import Collection
from datetime import date, timedelta

import polars as pl

from ais_sentinel.geo import KNOT_MS

KM_PER_DEG_LAT = 111.32


def edge_distance_km(
    lat: pl.Expr, lon: pl.Expr, lat_min: float, lat_max: float, lon_min: float, lon_max: float
) -> pl.Expr:
    """Approximate distance (km) from a point to the nearest edge of a lat/lon box."""
    km_per_deg_lon = KM_PER_DEG_LAT * pl.lit((lat_min + lat_max) / 2).radians().cos()
    return pl.min_horizontal(
        (lat - lat_min) * KM_PER_DEG_LAT,
        (lat_max - lat) * KM_PER_DEG_LAT,
        (lon - lon_min) * km_per_deg_lon,
        (lon_max - lon) * km_per_deg_lon,
    )


def detect_gaps(
    points: pl.DataFrame,
    region: dict[str, float],
    min_gap_min: float = 30.0,
    min_sog_kn: float = 1.0,
    edge_buffer_km: float = 3.0,
    available_days: Collection[date] | None = None,
) -> pl.DataFrame:
    """Return one row per detected gap event.

    ``points`` are cleaned points (any voyage assignment). They must hold mmsi, t, lat,
    lon, sog_kn and vessel_group, and are re-sorted by (mmsi, t) internally.

    ``available_days`` is the set of UTC days actually present in the dataset. A silence
    spanning a day with no data is a hole in *our* data, not the vessel going dark, so such
    candidates are discarded. If None, every day is assumed available.
    """
    p = points.sort("mmsi", "t").select("mmsi", "t", "lat", "lon", "sog_kn", "vessel_group")
    nxt = {c: pl.col(c).shift(-1).over("mmsi") for c in ("t", "lat", "lon", "sog_kn")}
    edge = {k: float(region[k]) for k in ("lat_min", "lat_max", "lon_min", "lon_max")}
    cand = (
        p.with_columns(
            t_end=nxt["t"],
            lat_end=nxt["lat"],
            lon_end=nxt["lon"],
            sog_end=nxt["sog_kn"],
        )
        .rename({"t": "t_start", "lat": "lat_start", "lon": "lon_start", "sog_kn": "sog_start"})
        .drop_nulls("t_end")
        .with_columns(gap_min=(pl.col("t_end") - pl.col("t_start")).dt.total_seconds() / 60)
        .filter(pl.col("gap_min") >= min_gap_min)
    )
    from ais_sentinel.data.clean import haversine_expr

    cand = cand.with_columns(
        edge_start_km=edge_distance_km(pl.col("lat_start"), pl.col("lon_start"), **edge),
        edge_end_km=edge_distance_km(pl.col("lat_end"), pl.col("lon_end"), **edge),
        gap_dist_km=haversine_expr(
            pl.col("lat_start"), pl.col("lon_start"), pl.col("lat_end"), pl.col("lon_end")
        )
        / 1000,
    ).with_columns(
        implied_kn=pl.col("gap_dist_km") * 1000 / (pl.col("gap_min") * 60) / KNOT_MS,
    )
    events = cand.filter(
        (pl.col("sog_start").fill_null(0) >= min_sog_kn)
        & (pl.col("sog_end").fill_null(0) >= min_sog_kn)
        & (pl.col("edge_start_km") >= edge_buffer_km)
        & (pl.col("edge_end_km") >= edge_buffer_km)
    )
    if available_days is not None:
        events = events.filter(_span_is_covered(events, set(available_days)))
    return events.with_columns(
        type=pl.lit("gap"),
        score=pl.col("gap_min") / min_gap_min,
        explanation=pl.format(
            "Silent for {} min while moving ({} kn before, {} kn after), {} km inside the "
            "coverage area; it covered {} km during the silence.",
            pl.col("gap_min").round(0).cast(pl.Int64),
            pl.col("sog_start").round(1),
            pl.col("sog_end").round(1),
            pl.min_horizontal("edge_start_km", "edge_end_km").round(1),
            pl.col("gap_dist_km").round(1),
        ),
    ).select(
        "type",
        "mmsi",
        "vessel_group",
        "t_start",
        "t_end",
        "gap_min",
        "lat_start",
        "lon_start",
        "lat_end",
        "lon_end",
        "sog_start",
        "sog_end",
        "gap_dist_km",
        "implied_kn",
        "score",
        "explanation",
    )


def _span_is_covered(events: pl.DataFrame, days: set[date]) -> pl.Series:
    """Boolean mask: every UTC day touched by each event is in ``days``."""
    out = []
    for t0, t1 in zip(events["t_start"].to_list(), events["t_end"].to_list(), strict=True):
        d, last = t0.date(), t1.date()
        ok = True
        while d <= last:
            if d not in days:
                ok = False
                break
            d += timedelta(days=1)
        out.append(ok)
    return pl.Series(out, dtype=pl.Boolean)
