"""Rule-based anomaly detectors: kinematic jumps, loitering, route deviation, rendezvous.

Every detector returns events in one schema (:data:`EVENT_SCHEMA`). Each event has a
plain-language ``explanation`` stating the measured values against the thresholds, so an
analyst can judge it in seconds. Thresholds live in ``configs/default.yaml``
(``anomaly.*``) and are grounded in PLAN §5.3.

Inputs are cleaned points (``mmsi, t, lat, lon, sog_kn, cog_deg, vessel_group``),
independent of the tracker, so the detectors can be evaluated on their own.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from ais_sentinel.anomaly.context import Context
from ais_sentinel.geo import KNOT_MS, haversine_m, latlon_to_enu

EVENT_SCHEMA: dict[str, pl.DataType] = {
    "type": pl.String(),
    "mmsi": pl.Int64(),
    "mmsi2": pl.Int64(),
    "vessel_group": pl.String(),
    "t_start": pl.Datetime("us", "UTC"),
    "t_end": pl.Datetime("us", "UTC"),
    "lat": pl.Float64(),
    "lon": pl.Float64(),
    "lat_end": pl.Float64(),
    "lon_end": pl.Float64(),
    "duration_min": pl.Float64(),
    "score": pl.Float64(),
    "explanation": pl.String(),
}


def empty_events() -> pl.DataFrame:
    """An empty frame with the event schema."""
    return pl.DataFrame(schema=EVENT_SCHEMA)


def _events(rows: list[dict[str, object]]) -> pl.DataFrame:
    if not rows:
        return empty_events()
    return pl.DataFrame(rows, schema=EVENT_SCHEMA)


@dataclass
class VesselHome:
    """Cells where each vessel habitually stays (learned from training), per MMSI."""

    cells: dict[int, set[int]] = field(default_factory=dict)


def learn_vessel_homes(
    train_points: pl.DataFrame, ctx: Context, min_hours: float = 6.0
) -> VesselHome:
    """Cells where a vessel itself spent at least ``min_hours`` stationary in training."""
    p = train_points.sort("mmsi", "t")
    t = p["t"].dt.epoch("us").to_numpy() / 1e6
    m = p["mmsi"].to_numpy()
    sog = p["sog_kn"].cast(pl.Float64).fill_null(0.0).to_numpy()
    dt_next = np.concatenate([np.diff(t), [np.nan]])
    still = np.concatenate([m[1:] == m[:-1], [False]]) & (sog < 1.0) & (dt_next <= 3600)
    cells = ctx.port_grid.cells(p["lat"].to_numpy()[still], p["lon"].to_numpy()[still])
    df = (
        pl.DataFrame({"mmsi": m[still], "cell": cells, "h": dt_next[still] / 3600})
        .group_by("mmsi", "cell")
        .agg(pl.col("h").sum())
        .filter(pl.col("h") >= min_hours)
    )
    out: dict[int, set[int]] = {}
    for mm, c in zip(df["mmsi"].to_list(), df["cell"].to_list(), strict=True):
        out.setdefault(int(mm), set()).update(
            {c + dx * 100_000 + dy for dx in (-1, 0, 1) for dy in (-1, 0, 1)}
        )
    return VesselHome(out)


def _arrays(p: pl.DataFrame) -> dict[str, Any]:
    return {
        "mmsi": p["mmsi"].to_numpy(),
        "t": p["t"].dt.epoch("us").to_numpy() / 1e6,
        "lat": p["lat"].to_numpy(),
        "lon": p["lon"].to_numpy(),
        "sog": p["sog_kn"].cast(pl.Float64).fill_null(np.nan).to_numpy(),
        "cog": p["cog_deg"].cast(pl.Float64).fill_null(np.nan).to_numpy(),
        "group": p["vessel_group"].fill_null("unknown").to_numpy(),
    }


def _ts(sec: float) -> object:
    import datetime as _dt

    return _dt.datetime.fromtimestamp(float(sec), tz=_dt.UTC)


def _runs(
    flag: NDArray[np.bool_],
    same_vessel: NDArray[np.bool_],
    t: NDArray[np.float64],
    max_gap_s: float,
) -> list[tuple[int, int]]:
    """Index runs [i0, i1] of consecutive flagged fixes of one vessel, bridging gaps up to
    ``max_gap_s``."""
    runs: list[tuple[int, int]] = []
    start = -1
    for i in range(len(flag)):
        if not flag[i]:
            if start >= 0:
                runs.append((start, i - 1))
                start = -1
            continue
        if start >= 0 and not (same_vessel[i] and t[i] - t[i - 1] <= max_gap_s):
            runs.append((start, i - 1))
            start = -1
        if start < 0:
            start = i
    if start >= 0:
        runs.append((start, len(flag) - 1))
    return runs


# --------------------------------------------------------------------------- jumps


def detect_jumps(
    points: pl.DataFrame,
    min_jump_m: float = 1000.0,
    abs_kn: float = 40.0,
    sog_factor: float = 2.0,
    sog_margin_kn: float = 10.0,
) -> pl.DataFrame:
    """Kinematically impossible position jumps.

    A step from fix i−1 to fix i is flagged when it covers at least ``min_jump_m``, at an
    implied speed above ``max(abs_kn, sog_factor · reported SOG + sog_margin_kn)``. The
    distance floor stops GPS jitter over a few seconds from producing absurd speeds.
    Flagged steps within 10 minutes are merged, so an out-and-back spike is one event.
    """
    p = points.sort("mmsi", "t")
    a = _arrays(p)
    same = np.concatenate([[False], a["mmsi"][1:] == a["mmsi"][:-1]])
    d = np.concatenate(
        [[0.0], haversine_m(a["lat"][:-1], a["lon"][:-1], a["lat"][1:], a["lon"][1:])]
    )
    dt = np.concatenate([[np.inf], np.diff(a["t"])])
    v = d / np.maximum(dt, 1.0) / KNOT_MS
    sog_ref = np.fmax(
        np.nan_to_num(a["sog"], nan=0.0),
        np.concatenate([[0.0], np.nan_to_num(a["sog"][:-1], nan=0.0)]),
    )
    thr = np.maximum(abs_kn, sog_factor * sog_ref + sog_margin_kn)
    flag = same & (d >= min_jump_m) & (v > thr)
    rows: list[dict[str, object]] = []
    idx = np.flatnonzero(flag)
    k = 0
    while k < len(idx):
        i0 = i1 = idx[k]
        while (
            k + 1 < len(idx)
            and a["mmsi"][idx[k + 1]] == a["mmsi"][i0]
            and a["t"][idx[k + 1]] - a["t"][i1] <= 600
        ):
            k += 1
            i1 = idx[k]
        worst = idx[(idx >= i0) & (idx <= i1)]
        j = worst[np.argmax(d[worst])]
        rows.append(
            {
                "type": "jump",
                "mmsi": int(a["mmsi"][j]),
                "mmsi2": None,
                "vessel_group": str(a["group"][j]),
                "t_start": _ts(a["t"][i0 - 1]),
                "t_end": _ts(a["t"][i1]),
                "lat": float(a["lat"][j - 1]),
                "lon": float(a["lon"][j - 1]),
                "lat_end": float(a["lat"][j]),
                "lon_end": float(a["lon"][j]),
                "duration_min": float((a["t"][i1] - a["t"][i0 - 1]) / 60),
                "score": float(v[j] / thr[j]),
                "explanation": (
                    f"Position jumped {d[j] / 1000:.1f} km in {dt[j]:.0f} s "
                    f"(implied {v[j]:.0f} kn) "
                    f"while reporting {sog_ref[j]:.1f} kn; limit {thr[j]:.0f} kn. Physically "
                    "implausible: possible spoofing, GPS fault, or two transmitters "
                    "sharing one MMSI."
                ),
            }
        )
        k += 1
    return _events(rows)


# --------------------------------------------------------------------------- loitering


def detect_loitering(
    points: pl.DataFrame,
    ctx: Context,
    homes: VesselHome,
    max_sog_kn: float = 2.0,
    min_hours: float = 2.0,
    max_port_frac: float = 0.2,
    max_gap_min: float = 30.0,
) -> pl.DataFrame:
    """Long slow periods away from ports, anchorages and the vessel's own usual moorings."""
    p = points.sort("mmsi", "t")
    a = _arrays(p)
    same = np.concatenate([[False], a["mmsi"][1:] == a["mmsi"][:-1]])
    slow = np.nan_to_num(a["sog"], nan=99.0) < max_sog_kn
    rows: list[dict[str, object]] = []
    for i0, i1 in _runs(slow, same, a["t"], max_gap_min * 60):
        dur_h = (a["t"][i1] - a["t"][i0]) / 3600
        if dur_h < min_hours:
            continue
        lat, lon = a["lat"][i0 : i1 + 1], a["lon"][i0 : i1 + 1]
        mm = int(a["mmsi"][i0])
        if ctx.in_port(lat, lon).mean() > max_port_frac:
            continue
        home = homes.cells.get(mm, set())
        if (
            home
            and np.mean([int(c) in home for c in ctx.port_grid.cells(lat, lon)]) > max_port_frac
        ):
            continue
        clat, clon = float(np.mean(lat)), float(np.mean(lon))
        radius = float(np.max(haversine_m(clat, clon, lat, lon)))
        rows.append(
            {
                "type": "loiter",
                "mmsi": mm,
                "mmsi2": None,
                "vessel_group": str(a["group"][i0]),
                "t_start": _ts(a["t"][i0]),
                "t_end": _ts(a["t"][i1]),
                "lat": clat,
                "lon": clon,
                "lat_end": clat,
                "lon_end": clon,
                "duration_min": dur_h * 60,
                "score": dur_h / min_hours,
                "explanation": (
                    f"Stayed below {max_sog_kn:g} kn for {dur_h:.1f} h within {radius:.0f} m, "
                    "outside "
                    "known port/anchorage areas and this vessel's usual moorings."
                ),
            }
        )
    return _events(rows)


# --------------------------------------------------------------------------- route deviation


def detect_route_deviation(
    points: pl.DataFrame,
    ctx: Context,
    min_sog_kn: float = 3.0,
    max_cell_voyages: int = 2,
    min_course_prob: float = 0.02,
    min_cell_voyages_for_course: int = 10,
    min_minutes: float = 10.0,
    max_gap_min: float = 5.0,
) -> pl.DataFrame:
    """Sustained travel through rarely-used water, or against the usual direction of traffic."""
    p = points.sort("mmsi", "t")
    a = _arrays(p)
    same = np.concatenate([[False], a["mmsi"][1:] == a["mmsi"][:-1]])
    moving = np.nan_to_num(a["sog"], nan=0.0) >= min_sog_kn
    dens, prob = ctx.traffic_at(a["lat"], a["lon"], a["cog"])
    off_lane = moving & (dens <= max_cell_voyages)
    wrong_way = (
        moving
        & (dens >= min_cell_voyages_for_course)
        & (np.nan_to_num(prob, nan=1.0) < min_course_prob)
    )
    flag = off_lane | wrong_way
    rows: list[dict[str, object]] = []
    for i0, i1 in _runs(flag, same, a["t"], max_gap_min * 60):
        minutes = (a["t"][i1] - a["t"][i0]) / 60
        if minutes < min_minutes:
            continue
        sl = slice(i0, i1 + 1)
        kind = "off-lane" if off_lane[sl].mean() >= wrong_way[sl].mean() else "wrong-way"
        dist_km = float(
            np.sum(
                haversine_m(
                    a["lat"][i0:i1],
                    a["lon"][i0:i1],
                    a["lat"][i0 + 1 : i1 + 1],
                    a["lon"][i0 + 1 : i1 + 1],
                )
            )
            / 1000
        )
        if kind == "off-lane":
            why = (
                f"Travelled {minutes:.0f} min ({dist_km:.1f} km) off the usual lanes: "
                f"{100 * float(off_lane[sl].mean()):.0f}% of its fixes were in 500 m cells that at "
                f"most {max_cell_voyages} training voyages ever crossed (median "
                f"{float(np.median(dens[sl])):.0f} voyages per cell along this stretch)."
            )
        else:
            why = (
                f"Travelled {minutes:.0f} min ({dist_km:.1f} km) on a heading that only "
                f"{100 * float(np.nanmean(prob[sl])):.1f}% of training traffic used in these cells "
                "(wrong-way or unusual crossing)."
            )
        rows.append(
            {
                "type": "deviation",
                "mmsi": int(a["mmsi"][i0]),
                "mmsi2": None,
                "vessel_group": str(a["group"][i0]),
                "t_start": _ts(a["t"][i0]),
                "t_end": _ts(a["t"][i1]),
                "lat": float(a["lat"][i0]),
                "lon": float(a["lon"][i0]),
                "lat_end": float(a["lat"][i1]),
                "lon_end": float(a["lon"][i1]),
                "duration_min": float(minutes),
                "score": float(minutes / min_minutes),
                "explanation": why,
            }
        )
    return _events(rows)


# --------------------------------------------------------------------------- rendezvous


def detect_rendezvous(
    points: pl.DataFrame,
    ctx: Context,
    homes: VesselHome,
    max_dist_m: float = 500.0,
    max_sog_kn: float = 2.0,
    min_hours: float = 1.0,
    bin_min: float = 5.0,
) -> pl.DataFrame:
    """Two vessels close together and slow for a long time, away from ports and moorings.

    Fixes are binned into ``bin_min``-minute bins, with one mean position per vessel per bin.
    Pairs within ``max_dist_m`` are found per bin with a KD-tree, and a pair's consecutive
    bins (one missing bin allowed) form an encounter.
    """
    a = _arrays(points)
    slow = np.nan_to_num(a["sog"], nan=99.0) < max_sog_kn
    keep = slow & ~ctx.in_port(a["lat"], a["lon"])
    if not keep.any():
        return empty_events()
    cells = ctx.port_grid.cells(a["lat"], a["lon"])
    for i in np.flatnonzero(keep):
        h = homes.cells.get(int(a["mmsi"][i]))
        if h and int(cells[i]) in h:
            keep[i] = False
    b = np.floor(a["t"][keep] / (bin_min * 60)).astype(np.int64)
    df = (
        pl.DataFrame(
            {
                "mmsi": a["mmsi"][keep],
                "bin": b,
                "lat": a["lat"][keep],
                "lon": a["lon"][keep],
                "group": a["group"][keep],
            }
        )
        .group_by("mmsi", "bin")
        .agg(pl.col("lat").mean(), pl.col("lon").mean(), pl.col("group").first())
        .sort("bin")
    )
    ref_lat, ref_lon = ctx.port_grid.ref_lat, ctx.port_grid.ref_lon
    e, n = latlon_to_enu(df["lat"].to_numpy(), df["lon"].to_numpy(), ref_lat, ref_lon)
    xy = np.column_stack([e, n])
    mm = df["mmsi"].to_numpy()
    bins = df["bin"].to_numpy()
    dlat, dlon = df["lat"].to_numpy(), df["lon"].to_numpy()
    pairs: dict[tuple[int, int], list[tuple[int, float, float]]] = {}
    for bval in np.unique(bins):
        rows_b = np.flatnonzero(bins == bval)
        if len(rows_b) < 2:
            continue
        tree = cKDTree(xy[rows_b])
        for i, j in tree.query_pairs(max_dist_m):
            r1, r2 = rows_b[i], rows_b[j]
            key = (int(min(mm[r1], mm[r2])), int(max(mm[r1], mm[r2])))
            pairs.setdefault(key, []).append(
                (
                    int(bval),
                    float((dlat[r1] + dlat[r2]) / 2),
                    float((dlon[r1] + dlon[r2]) / 2),
                )
            )
    group = dict(zip(df["mmsi"].to_list(), df["group"].to_list(), strict=False))
    out: list[dict[str, object]] = []
    for (m1, m2), obs in pairs.items():
        obs.sort()
        run = [obs[0]]
        for o in [*obs[1:], (10**12, 0.0, 0.0)]:
            if o[0] - run[-1][0] <= 2:
                run.append(o)
                continue
            hours = (run[-1][0] - run[0][0] + 1) * bin_min / 60
            if hours >= min_hours:
                lat = float(np.mean([r[1] for r in run]))
                lon = float(np.mean([r[2] for r in run]))
                out.append(
                    {
                        "type": "rendezvous",
                        "mmsi": m1,
                        "mmsi2": m2,
                        "vessel_group": str(group.get(m1, "unknown")),
                        "t_start": _ts(run[0][0] * bin_min * 60),
                        "t_end": _ts((run[-1][0] + 1) * bin_min * 60),
                        "lat": lat,
                        "lon": lon,
                        "lat_end": lat,
                        "lon_end": lon,
                        "duration_min": hours * 60,
                        "score": hours / min_hours,
                        "explanation": (
                            f"MMSI {m1} and MMSI {m2} stayed within {max_dist_m:.0f} m of each "
                            f"other for {hours:.1f} h, both below {max_sog_kn:g} kn, "
                            "away from known "
                            "ports/anchorages and their usual moorings."
                        ),
                    }
                )
            run = [o]
    return _events(out)
