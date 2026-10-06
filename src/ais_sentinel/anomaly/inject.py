"""Inject synthetic anomalies with known ground truth into real vessel tracks.

There are no labelled anomalies in public AIS. So detection is measured by taking real,
test-period voyages, injecting one anomaly with a known type, time span and magnitude, and
checking whether the detector finds it (PLAN §6.4). Each injector takes one vessel's
points (cleaned schema, sorted by time) and returns the modified points plus an
:class:`Injection` describing the ground truth.

Injected fixes look like the surrounding data: one-minute reporting while moving, three
minutes while stationary, and a few metres of GPS noise.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import polars as pl

from ais_sentinel.geo import KNOT_MS, enu_to_latlon, latlon_to_enu, sog_cog_to_enu_velocity


@dataclass(frozen=True)
class Injection:
    """Ground truth for one injected anomaly."""

    type: str
    mmsi: int
    t_start: datetime
    t_end: datetime
    magnitude: float  # minutes (gap), km (jump), hours (loiter, rendezvous), km (deviation)
    mmsi2: int | None = None
    duration_min: float | None = None


def _cols(v: pl.DataFrame) -> pl.DataFrame:
    return v.select("mmsi", "t", "lat", "lon", "sog_kn", "cog_deg", "vessel_group")


def _new_points(
    template: pl.DataFrame,
    t: list[datetime],
    lat: np.ndarray,
    lon: np.ndarray,
    sog: np.ndarray,
    cog: np.ndarray,
) -> pl.DataFrame:
    """Build fixes with the template's MMSI and group."""
    n = len(t)
    return pl.DataFrame(
        {
            "mmsi": [int(template["mmsi"][0])] * n,
            "t": t,
            "lat": np.asarray(lat, dtype=float),
            "lon": np.asarray(lon, dtype=float),
            "sog_kn": np.asarray(sog, dtype=float),
            "cog_deg": np.asarray(cog, dtype=float),
            "vessel_group": [template["vessel_group"][0]] * n,
        }
    ).cast(dict(_cols(template).schema))  # type: ignore[arg-type]


def inject_gap(v: pl.DataFrame, t0: datetime, minutes: float) -> tuple[pl.DataFrame, Injection]:
    """Delete all reports in (t0, t0 + minutes)."""
    t1 = t0 + timedelta(minutes=minutes)
    out = v.filter(~((pl.col("t") > t0) & (pl.col("t") < t1)))
    return _cols(out), Injection("gap", int(v["mmsi"][0]), t0, t1, minutes, duration_min=minutes)


def inject_jump(
    v: pl.DataFrame, t0: datetime, km: float, n_fixes: int, rng: np.random.Generator
) -> tuple[pl.DataFrame, Injection]:
    """Displace ``n_fixes`` consecutive fixes from t0 by ``km`` in a random direction."""
    idx = np.flatnonzero(v["t"].to_numpy() >= np.datetime64(t0.replace(tzinfo=None), "us"))[
        :n_fixes
    ]
    ang = rng.uniform(0, 2 * np.pi)
    lat, lon = v["lat"].to_numpy().copy(), v["lon"].to_numpy().copy()
    for i in idx:
        lat[i], lon[i] = enu_to_latlon(
            km * 1000 * np.cos(ang), km * 1000 * np.sin(ang), lat[i], lon[i]
        )
    out = v.with_columns(lat=pl.Series(lat), lon=pl.Series(lon))
    ts = v["t"].gather(idx).to_list()
    return _cols(out), Injection(
        "jump",
        int(v["mmsi"][0]),
        ts[0],
        ts[-1],
        km,
        duration_min=(ts[-1] - ts[0]).total_seconds() / 60,
    )


def _drift(
    lat0: float,
    lon0: float,
    start: datetime,
    hours: float,
    rng: np.random.Generator,
    step_s: float = 180.0,
) -> tuple[list[datetime], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Slow random-walk drift around a point (stationary AIS cadence)."""
    n = max(2, int(hours * 3600 / step_s))
    e = np.cumsum(rng.normal(0, 15, n))
    nn = np.cumsum(rng.normal(0, 15, n))
    e, nn = e - e[0], nn - nn[0]
    lat, lon = enu_to_latlon(e + rng.normal(0, 5, n), nn + rng.normal(0, 5, n), lat0, lon0)
    sog = rng.uniform(0.0, 1.2, n)
    cog = rng.uniform(0, 360, n)
    t = [start + timedelta(seconds=step_s * k) for k in range(n)]
    return t, np.asarray(lat), np.asarray(lon), sog, cog


def inject_loiter(
    v: pl.DataFrame, t0: datetime, hours: float, rng: np.random.Generator
) -> tuple[pl.DataFrame, Injection]:
    """Stop at the t0 position, drift slowly for ``hours``, then resume the rest of the voyage
    (shifted later in time)."""
    before = v.filter(pl.col("t") <= t0)
    after = v.filter(pl.col("t") > t0)
    lat0, lon0 = float(before["lat"][-1]), float(before["lon"][-1])
    t, lat, lon, sog, cog = _drift(lat0, lon0, t0 + timedelta(minutes=3), hours, rng)
    shift = timedelta(hours=hours) + timedelta(minutes=3)
    stay = _new_points(v, t, lat, lon, sog, cog)
    out = pl.concat([_cols(before), stay, _cols(after).with_columns(t=pl.col("t") + shift)])
    return out, Injection("loiter", int(v["mmsi"][0]), t[0], t[-1], hours, duration_min=hours * 60)


def inject_deviation(
    v: pl.DataFrame, t0: datetime, offset_km: float, minutes: float, rng: np.random.Generator
) -> tuple[pl.DataFrame, Injection]:
    """Add a smooth sideways excursion (sin² bump, peak ``offset_km``) lasting ``minutes``."""
    t1 = t0 + timedelta(minutes=minutes)
    ts = v["t"].dt.epoch("us").to_numpy() / 1e6
    s0, s1 = t0.timestamp(), t1.timestamp()
    m = (ts >= s0) & (ts <= s1)
    lat, lon = v["lat"].to_numpy().copy(), v["lon"].to_numpy().copy()
    cog = v["cog_deg"].cast(pl.Float64).fill_null(0.0).to_numpy()
    side = rng.choice([-1.0, 1.0])
    for i in np.flatnonzero(m):
        frac = np.sin(np.pi * (ts[i] - s0) / (s1 - s0)) ** 2
        c = np.radians(cog[i] + 90 * side)  # perpendicular to the course
        lat[i], lon[i] = enu_to_latlon(
            offset_km * 1000 * frac * np.sin(c), offset_km * 1000 * frac * np.cos(c), lat[i], lon[i]
        )
    out = v.with_columns(lat=pl.Series(lat), lon=pl.Series(lon))
    return _cols(out), Injection(
        "deviation", int(v["mmsi"][0]), t0, t1, offset_km, duration_min=minutes
    )


def inject_rendezvous(
    v: pl.DataFrame, t0: datetime, hours: float, partner_mmsi: int, rng: np.random.Generator
) -> tuple[pl.DataFrame, Injection]:
    """Host loiters at t0 while a synthetic partner approaches, stays within ~300 m, and leaves.

    Returns points for **both** vessels (host and partner).
    """
    host, inj = inject_loiter(v, t0, hours, rng)
    stay = host.filter((pl.col("t") >= inj.t_start) & (pl.col("t") <= inj.t_end))
    lat0, lon0 = float(stay["lat"][0]), float(stay["lon"][0])
    ang = rng.uniform(0, 2 * np.pi)
    approach_min = 30
    speed = 10 * KNOT_MS
    # Approach: from 5 km away along a straight line, at one fix per minute.
    k = np.arange(approach_min)
    dist = 5000 * (1 - k / approach_min) + 150
    ea, na = dist * np.cos(ang), dist * np.sin(ang)
    la, lo = enu_to_latlon(ea, na, lat0, lon0)
    cog_in = (np.degrees(np.arctan2(-np.cos(ang), -np.sin(ang))) % 360) * np.ones(approach_min)
    t_app = [inj.t_start - timedelta(minutes=approach_min - int(i)) for i in k]
    # Alongside: the host's positions plus a 100–250 m offset with small jitter.
    off = rng.uniform(100, 250)
    hs_e, hs_n = latlon_to_enu(stay["lat"].to_numpy(), stay["lon"].to_numpy(), lat0, lon0)
    pe = hs_e + off * np.cos(ang) + rng.normal(0, 10, stay.height)
    pn = hs_n + off * np.sin(ang) + rng.normal(0, 10, stay.height)
    lb, lob = enu_to_latlon(pe, pn, lat0, lon0)
    # Depart the way it came.
    ve, vn = sog_cog_to_enu_velocity(10.0, float(cog_in[0] + 180) % 360)
    kd = np.arange(1, approach_min + 1) * 60.0
    lc, loc = enu_to_latlon(pe[-1] + ve * kd, pn[-1] + vn * kd, lat0, lon0)
    t_dep = [inj.t_end + timedelta(minutes=int(i)) for i in range(1, approach_min + 1)]
    partner_tpl = v.head(1).with_columns(
        mmsi=pl.lit(partner_mmsi, dtype=pl.Int64), vessel_group=pl.lit("other")
    )
    partner = pl.concat(
        [
            _new_points(
                partner_tpl,
                t_app,
                np.asarray(la),
                np.asarray(lo),
                np.full(approach_min, speed / KNOT_MS),
                cog_in,
            ),
            _new_points(
                partner_tpl,
                stay["t"].to_list(),
                np.asarray(lb),
                np.asarray(lob),
                stay["sog_kn"].cast(pl.Float64).to_numpy(),
                stay["cog_deg"].cast(pl.Float64).to_numpy(),
            ),
            _new_points(
                partner_tpl,
                t_dep,
                np.asarray(lc),
                np.asarray(loc),
                np.full(approach_min, 10.0),
                np.full(approach_min, (cog_in[0] + 180) % 360),
            ),
        ]
    )
    return pl.concat([host, partner]).sort("mmsi", "t"), Injection(
        "rendezvous",
        int(v["mmsi"][0]),
        inj.t_start,
        inj.t_end,
        hours,
        mmsi2=partner_mmsi,
        duration_min=hours * 60,
    )
