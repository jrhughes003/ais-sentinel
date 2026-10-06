"""Export small, licence-compliant JSON extracts for the static website.

The site runs entirely in the browser, so everything it shows must be precomputed and kept
small: a curated set of **commercial** voyages (privacy: no pleasure craft), positions
rounded to 5 decimals (about 1 m), and compact column-oriented arrays.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from ais_sentinel.config import Config
from ais_sentinel.data.download import load_manifest
from ais_sentinel.data.vessel_types import COMMERCIAL
from ais_sentinel.io import write_text

log = logging.getLogger(__name__)

ATTRIBUTION = (
    "AIS data: U.S. Coast Guard Navigation Center via NOAA Office for Coastal Management / "
    "BOEM, MarineCadastre.gov (CC0 1.0). Not for navigation."
)


def _round(a: np.ndarray, nd: int = 5) -> list[float]:
    return [round(float(x), nd) for x in a]


def pick_showcase_voyages(
    voyages: pl.DataFrame, n: int, min_path_km: float = 15.0, seed: int = 0
) -> pl.DataFrame:
    """Choose ``n`` long, moving, commercial voyages, at most one per vessel."""
    cand = (
        voyages.filter(
            pl.col("usable")
            & pl.col("vessel_group").is_in(sorted(COMMERCIAL))
            & (pl.col("path_km") >= min_path_km)
            & (pl.col("moving_frac") >= 0.5)
            & pl.col("vessel_name").is_not_null()
        )
        .sort("path_km", descending=True)
        .unique("mmsi", keep="first", maintain_order=True)
    )
    # Spread the selection across vessel groups rather than taking the longest of one kind.
    per_group = max(1, n // max(1, cand["vessel_group"].n_unique()))
    picked = cand.group_by("vessel_group", maintain_order=True).head(per_group).select(cand.columns)
    if picked.height < n:
        rest = cand.join(picked.select("voyage_id"), on="voyage_id", how="anti")
        picked = pl.concat([picked, rest.head(n - picked.height)])
    return picked.head(n).sort("t_start")


def voyage_payload(track: pl.DataFrame, meta: dict[str, Any]) -> dict[str, Any]:
    """Compact JSON object for one voyage's raw and smoothed track."""
    t = (track["t"].dt.epoch("s").to_numpy()).astype(int)
    # One-sigma position uncertainty: the square root of the larger covariance eigenvalue.
    pee, pnn, pen = (track[c].to_numpy() for c in ("s_pee", "s_pnn", "s_pen"))
    tr, det = pee + pnn, pee * pnn - pen**2
    lam_max = tr / 2 + np.sqrt(np.maximum(tr**2 / 4 - det, 0.0))
    return {
        "id": meta["voyage_id"],
        "mmsi": int(meta["mmsi"]),
        "name": meta["vessel_name"],
        "group": meta["vessel_group"],
        "length_m": None if meta["length_m"] is None else round(float(meta["length_m"])),
        "t0": int(t[0]),
        "dt": np.diff(t, prepend=t[0]).tolist(),
        "lat": _round(track["lat"].to_numpy()),
        "lon": _round(track["lon"].to_numpy()),
        "s_lat": _round(track["s_lat"].to_numpy()),
        "s_lon": _round(track["s_lon"].to_numpy()),
        "s_sd_m": _round(np.sqrt(lam_max), 1),
        "rejected": np.flatnonzero(~track["accepted"].to_numpy()).tolist(),
        "sog_kn": _round(track["sog_kn"].cast(pl.Float64).fill_null(np.nan).to_numpy(), 1),
    }


def write_json(path: Path, obj: Any) -> int:
    """Write compact JSON (NaN becomes null). Returns the size in bytes."""
    path.parent.mkdir(parents=True, exist_ok=True)

    def clean(o: Any) -> Any:
        if isinstance(o, float) and not np.isfinite(o):
            return None
        if isinstance(o, list):
            return [clean(x) for x in o]
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        return o

    text = json.dumps(clean(obj), separators=(",", ":"), ensure_ascii=False, default=str)
    return write_text(path, text)


def stage(cfg: Config) -> None:
    """CLI stage: write the site's data files to ``paths.site_data``."""
    from ais_sentinel.anomaly.gaps import detect_gaps

    base = Path(cfg.paths.processed)
    out = Path(cfg.paths.site_data)
    voyages = pl.read_parquet(base / "voyages.parquet")
    tracks = pl.read_parquet(base / "tracks.parquet")
    picked = pick_showcase_voyages(voyages, int(cfg.export["n_showcase"]))
    payload = []
    for meta in picked.iter_rows(named=True):
        tr = tracks.filter(pl.col("voyage_id") == meta["voyage_id"]).sort("t")
        payload.append(voyage_payload(tr, meta))
    region = dict(cfg.region)
    meta_obj = {
        "generated": datetime.now(UTC).isoformat(timespec="seconds"),
        "attribution": ATTRIBUTION,
        "region": region,
    }
    size = write_json(out / "tracks.json", {**meta_obj, "voyages": payload})

    points = pl.read_parquet(base / "points.parquet")
    days = {date.fromisoformat(d) for d in load_manifest(cfg)}
    g = dict(cfg.anomaly["gap"])
    g.pop("min_reception", None)
    gaps = detect_gaps(points, region, available_days=days, **g)
    gaps = gaps.filter(pl.col("vessel_group").is_in(sorted(COMMERCIAL)))
    events = [
        {
            **{k: v for k, v in r.items() if k not in ("t_start", "t_end")},
            "t_start": int(r["t_start"].timestamp()),
            "t_end": int(r["t_end"].timestamp()),
        }
        for r in gaps.sort("t_start").iter_rows(named=True)
    ]
    size += write_json(out / "anomalies.json", {**meta_obj, "events": events})
    log.info("export: %d voyages, %d events, %.1f KB", len(payload), len(events), size / 1024)
