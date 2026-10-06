"""Feature tensors for the ML sequence models.

For each prediction sample (anchor fix), this builds:

* ``seq`` (M, T, F): the last ``history_min`` minutes resampled to a 1-minute grid
  ending at the anchor. Channels:
  * east and north offset from the anchor (km, anchor frame);
  * SOG / 20 kn;
  * sin and cos of COG;
  * a gap flag, set to 1 where the nearest real fix is more than 3 minutes away
    (linearly interpolated positions there are a guess).
* ``ctx`` (M, C): static context. That is the anchor position in the regional frame
  (/50 km, so the model can learn *where* routes bend), SOG, COG (sin, cos), log length,
  and a one-hot vessel group.
* ``dr`` (M, H, 2): the dead-reckoning displacement (km) per horizon.
* ``y`` (M, H, 2): true displacement **minus** dead reckoning (km). The network predicts
  corrections to the physics baseline instead of re-learning straight-line motion, which
  is data-efficient. With a zero correction, the model already equals dead reckoning.
* ``ymask`` (M, H): True where the truth at that horizon exists.

Only fixes at or before the anchor time enter ``seq`` and ``ctx``, so the features are
causal (tested).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from numpy.typing import NDArray

from ais_sentinel.data.vessel_types import GROUPS
from ais_sentinel.geo import latlon_to_enu, sog_cog_to_enu_velocity

Array = NDArray[np.float32]
SEQ_CHANNELS = ("de_km", "dn_km", "sog", "sin_cog", "cos_cog", "gap")


@dataclass
class FeatureSet:
    """Model inputs and targets for M samples."""

    sample_id: list[str]
    seq: Array
    ctx: Array
    dr: Array
    y: Array
    ymask: NDArray[np.bool_]


def _voyage_arrays(v: pl.DataFrame) -> tuple[np.ndarray, ...]:
    t = v["t"].dt.epoch("us").to_numpy() / 1e6
    lat, lon = v["lat"].to_numpy(), v["lon"].to_numpy()
    sog = v["sog_kn"].cast(pl.Float64).fill_null(np.nan).to_numpy()
    cog = v["cog_deg"].cast(pl.Float64).fill_null(np.nan).to_numpy()
    return t, lat, lon, sog, cog


def _ffill(a: np.ndarray) -> np.ndarray:
    """Forward-fill NaNs (leading NaNs become 0)."""
    out = a.copy()
    idx = np.where(np.isfinite(out), np.arange(len(out)), 0)
    np.maximum.accumulate(idx, out=idx)
    out = out[idx]
    out[~np.isfinite(out)] = 0.0
    return out


def build_features(
    tracks: pl.DataFrame,
    samples: pl.DataFrame,
    horizons_min: list[int],
    history_min: int,
    ref_lat: float,
    ref_lon: float,
) -> FeatureSet:
    """Build model tensors for ``samples`` (rows from :func:`build_samples`)."""
    T = history_min + 1
    grid = np.arange(-history_min, 1, dtype=float) * 60.0  # seconds relative to the anchor
    H = len(horizons_min)
    m = samples.height
    seq = np.zeros((m, T, len(SEQ_CHANNELS)), dtype=np.float32)
    by_voyage = {
        v["voyage_id"][0]: _voyage_arrays(v)
        for v in tracks.sort("voyage_id", "t").partition_by("voyage_id", maintain_order=True)
    }
    s_vid = samples["voyage_id"].to_list()
    s_t0 = samples["t0"].dt.epoch("us").to_numpy() / 1e6
    lat0 = samples["lat0"].to_numpy().astype(float)
    lon0 = samples["lon0"].to_numpy().astype(float)
    for i in range(m):
        t, lat, lon, sog, cog = by_voyage[s_vid[i]]
        past = t <= s_t0[i] + 1e-6  # causal: nothing after the anchor
        tp, latp, lonp, sogp, cogp = t[past], lat[past], lon[past], sog[past], cog[past]
        e, n = latlon_to_enu(latp, lonp, lat0[i], lon0[i])
        tq = s_t0[i] + grid
        seq[i, :, 0] = np.interp(tq, tp, e) / 1000.0
        seq[i, :, 1] = np.interp(tq, tp, n) / 1000.0
        j = np.clip(np.searchsorted(tp, tq), 0, len(tp) - 1)
        jm = np.clip(j - 1, 0, len(tp) - 1)
        nearest = np.where(np.abs(tp[j] - tq) < np.abs(tp[jm] - tq), j, jm)
        seq[i, :, 2] = _ffill(sogp)[nearest] / 20.0
        c = np.radians(_ffill(cogp)[nearest])
        seq[i, :, 3] = np.sin(c)
        seq[i, :, 4] = np.cos(c)
        seq[i, :, 5] = (np.abs(tp[nearest] - tq) > 180.0).astype(np.float32)
        before = tq < tp[0]
        seq[i, before, :5] = 0.0  # no history before the voyage start
        seq[i, before, 5] = 1.0

    sog0 = samples["sog_kn"].cast(pl.Float64).fill_null(0.0).to_numpy()
    cog0 = samples["cog_deg"].cast(pl.Float64).fill_null(0.0).to_numpy()
    er, nr = latlon_to_enu(lat0, lon0, ref_lat, ref_lon)
    length = samples["length_m"].cast(pl.Float64).fill_null(0.0).to_numpy()
    length = np.where(length > 0, length, 100.0)
    groups = samples["vessel_group"].fill_null("unknown").to_list()
    onehot = np.zeros((m, len(GROUPS)), dtype=np.float32)
    for i, g in enumerate(groups):
        onehot[i, GROUPS.index(g) if g in GROUPS else GROUPS.index("unknown")] = 1.0
    c0 = np.radians(cog0)
    ctx = np.column_stack(
        [
            er / 50_000,
            nr / 50_000,
            sog0 / 20.0,
            np.sin(c0),
            np.cos(c0),
            np.log(length) / 6.0,
            onehot,
        ]
    ).astype(np.float32)

    ve, vn = sog_cog_to_enu_velocity(sog0, cog0)
    hs = np.asarray(horizons_min, dtype=float) * 60.0
    dr = np.stack([np.outer(ve, hs), np.outer(vn, hs)], axis=-1) / 1000.0
    y = np.full((m, H, 2), np.nan)
    for k, h in enumerate(horizons_min):
        lat_h = samples[f"lat_{h}"].cast(pl.Float64).fill_null(np.nan).to_numpy()
        lon_h = samples[f"lon_{h}"].cast(pl.Float64).fill_null(np.nan).to_numpy()
        te, tn = latlon_to_enu(lat_h, lon_h, lat0, lon0)
        y[:, k, 0] = te / 1000.0
        y[:, k, 1] = tn / 1000.0
    ymask = np.isfinite(y[..., 0])
    y = np.where(ymask[..., None], y - dr, 0.0)
    return FeatureSet(
        samples["sample_id"].to_list(),
        seq,
        ctx,
        dr.astype(np.float32),
        y.astype(np.float32),
        ymask,
    )
