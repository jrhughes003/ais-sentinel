"""Build prediction samples: anchor fixes with ground-truth future positions.

Sample definition (PLAN §5.2):

* **Anchors** are fixes inside a usable voyage, spaced at least ``anchor_every_min`` apart,
  with at least ``history_min`` of the same voyage before them.
* The **target** at horizon h is the vessel's true position at ``t0 + h``. Because AIS
  reports never land exactly on ``t0 + h``, the target is linearly interpolated between the
  two bracketing fixes of the same voyage. Both fixes must lie within ``max_bracket_s`` of
  the target time; otherwise that horizon is missing (null).
* Only anchors where the vessel is **underway** (reported SOG ≥ ``min_sog_kn``) are kept.
  Moored vessels are trivially predictable and would dilute every metric.

Leakage guard: targets never extend beyond their own voyage, and voyages are split-pure, so
no sample's future crosses a split boundary.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from ais_sentinel.geo import enu_to_latlon, latlon_to_enu

ANCHOR_COLS = (
    "mmsi",
    "lat",
    "lon",
    "sog_kn",
    "cog_deg",
    "f_lat",
    "f_lon",
    "f_ve",
    "f_vn",
    "f_pee",
    "f_pnn",
    "f_pen",
)


def interp_track(
    t: np.ndarray, xy: np.ndarray, t_query: np.ndarray, max_bracket_s: float
) -> np.ndarray:
    """Linearly interpolate positions ``xy`` (N, 2) at ``t_query``.

    Returns (M, 2) with NaN rows wherever the query is outside the track, or is not
    bracketed by fixes within ``max_bracket_s`` on both sides. An exact time match is
    always valid.
    """
    j = np.searchsorted(t, t_query, side="right")  # t[j-1] <= q < t[j]
    i0 = np.clip(j - 1, 0, len(t) - 1)
    i1 = np.clip(j, 0, len(t) - 1)
    exact = (j >= 1) & (t[i0] == t_query)
    bracketed = (
        (j >= 1)
        & (j < len(t))
        & (t_query - t[i0] <= max_bracket_s)
        & (t[i1] - t_query <= max_bracket_s)
    )
    span = np.where(t[i1] > t[i0], t[i1] - t[i0], 1.0)
    w = np.clip((t_query - t[i0]) / span, 0.0, 1.0)[:, None]
    out = xy[i0] + w * (xy[i1] - xy[i0])
    out[~(exact | bracketed)] = np.nan
    return out


def anchor_indices(t: np.ndarray, history_s: float, every_s: float) -> np.ndarray:
    """Greedy anchor selection: the first fix with enough history, then every ``every_s``."""
    out: list[int] = []
    next_ok = t[0] + history_s
    for i, ti in enumerate(t):
        if ti >= next_ok:
            out.append(i)
            next_ok = ti + every_s
    return np.asarray(out, dtype=int)


def voyage_samples(
    v: pl.DataFrame,
    horizons_min: list[int],
    history_min: float,
    anchor_every_min: float,
    max_bracket_s: float,
    min_sog_kn: float,
) -> pl.DataFrame | None:
    """Samples for one voyage (``v`` sorted by time), or None if it yields none."""
    t = v["t"].dt.epoch("us").to_numpy() / 1e6
    idx = anchor_indices(t, history_min * 60, anchor_every_min * 60)
    if len(idx) == 0:
        return None
    sog = v["sog_kn"].cast(pl.Float64).fill_null(0.0).to_numpy()
    idx = idx[sog[idx] >= min_sog_kn]
    if len(idx) == 0:
        return None
    lat, lon = v["lat"].to_numpy(), v["lon"].to_numpy()
    lat_ref, lon_ref = float(lat[0]), float(lon[0])
    e, n = latlon_to_enu(lat, lon, lat_ref, lon_ref)
    xy = np.column_stack([e, n])
    cols: dict[str, object] = {"anchor_idx": idx, "t0": v["t"].gather(idx)}
    for c in ANCHOR_COLS:
        if c in v.columns:  # filter-state columns are optional (absent for raw fixes)
            cols[c if c not in ("lat", "lon") else f"{c}0"] = v[c].gather(idx)
    for h in horizons_min:
        target = interp_track(t, xy, t[idx] + h * 60.0, max_bracket_s)
        la, lo = enu_to_latlon(target[:, 0], target[:, 1], lat_ref, lon_ref)
        missing = np.isnan(target[:, 0])
        cols[f"lat_{h}"] = pl.Series(np.where(missing, np.nan, la)).fill_nan(None)
        cols[f"lon_{h}"] = pl.Series(np.where(missing, np.nan, lo)).fill_nan(None)
    return pl.DataFrame(cols).with_columns(voyage_id=pl.lit(v["voyage_id"][0]))


def build_samples(
    tracks: pl.DataFrame,
    voyages: pl.DataFrame,
    horizons_min: list[int],
    history_min: float,
    anchor_every_min: float,
    max_bracket_s: float = 180.0,
    min_sog_kn: float = 2.0,
) -> pl.DataFrame:
    """Create one row per anchor fix, with the true lat/lon at each horizon.

    ``tracks`` is the output of the tracking stage joined with ``sog_kn`` and ``cog_deg``.
    """
    parts = [
        voyage_samples(v, horizons_min, history_min, anchor_every_min, max_bracket_s, min_sog_kn)
        for v in tracks.sort("voyage_id", "t").partition_by("voyage_id", maintain_order=True)
    ]
    samples = pl.concat([p for p in parts if p is not None])
    meta = voyages.select("voyage_id", "split", "vessel_group", "length_m")
    return samples.join(meta, on="voyage_id", how="left").with_columns(
        sample_id=pl.concat_str(
            [pl.col("voyage_id"), pl.col("t0").dt.strftime("%Y%m%dT%H%M%S")], separator="@"
        )
    )
