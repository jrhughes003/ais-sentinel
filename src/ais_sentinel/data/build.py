"""Build stage: per-day AOI files → cleaned, segmented points and a voyage table.

Outputs (in ``paths.processed``):

* ``points.parquet``: every cleaned report, with ``voyage_id``, ``split`` and step
  kinematics.
* ``voyages.parquet``: one row per voyage (see :func:`summarise_voyages`).

It also writes ``reports/data_summary.md``.
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from pathlib import Path

import polars as pl

from ais_sentinel.config import Config
from ais_sentinel.data.clean import CleanReport, clean_points
from ais_sentinel.data.segment import assign_voyages, split_bounds, summarise_voyages
from ais_sentinel.io import write_text

log = logging.getLogger(__name__)


def processed_paths(cfg: Config) -> tuple[Path, Path]:
    """Return the (points, voyages) Parquet paths."""
    base = Path(cfg.paths.processed)
    return base / "points.parquet", base / "voyages.parquet"


def build_dataset(cfg: Config) -> tuple[pl.DataFrame, pl.DataFrame, CleanReport]:
    """Run cleaning and segmentation over every per-day AOI file present."""
    day_files = sorted((Path(cfg.paths.interim) / "days").glob("aoi_*.parquet"))
    if not day_files:
        raise FileNotFoundError("No per-day AOI files; run the download stage first")
    raw = pl.concat([pl.read_parquet(f) for f in day_files], how="vertical")
    log.info("build: %d files, %d raw AOI rows", len(day_files), raw.height)

    points, rep = clean_points(
        raw, float(cfg.clean.max_sog_kn), float(cfg.clean.max_implied_speed_kn)
    )
    points = assign_voyages(points, float(cfg.segment.max_gap_min))
    voyages = summarise_voyages(
        points,
        split_bounds(cfg.splits),
        int(cfg.segment.min_points),
        float(cfg.segment.min_duration_min),
    )
    points = points.join(
        voyages.select("voyage_id", "split", "usable", "vessel_group"), on="voyage_id", how="left"
    )
    return points, voyages, rep


def summary_markdown(
    points: pl.DataFrame, voyages: pl.DataFrame, rep: CleanReport, n_days: int
) -> str:
    """Render a human-readable data summary."""
    lines = [
        "# Data summary",
        "",
        f"Days processed: **{n_days}**. Clean points: **{points.height:,}**. "
        f"Distinct vessels: **{points['mmsi'].n_unique():,}**. Voyages: **{voyages.height:,}**.",
        "",
        "## Cleaning",
        "",
        "| step | rows |",
        "|---|---:|",
    ]
    for k, v in asdict(rep).items():
        if k != "notes":
            lines.append(f"| {k} | {v:,} |")
    usable = voyages.filter(pl.col("usable"))
    by_split = (
        usable.group_by("split")
        .agg(
            voyages=pl.len(),
            vessels=pl.col("mmsi").n_unique(),
            points=pl.col("n_points").sum(),
            hours=(pl.col("duration_min").sum() / 60).round(0),
        )
        .sort("split")
    )
    lines += [
        "",
        "## Usable voyages by split",
        "",
        "| split | voyages | vessels | points | hours |",
    ]
    lines.append("|---|---:|---:|---:|---:|")
    for r in by_split.iter_rows(named=True):
        lines.append(
            f"| {r['split']} | {r['voyages']:,} | {r['vessels']:,} | {r['points']:,} | "
            f"{r['hours']:,.0f} |"
        )
    by_group = (
        usable.group_by("vessel_group")
        .agg(voyages=pl.len(), vessels=pl.col("mmsi").n_unique(), points=pl.col("n_points").sum())
        .sort("points", descending=True)
    )
    lines += ["", "## Usable voyages by vessel group", "", "| group | voyages | vessels | points |"]
    lines.append("|---|---:|---:|---:|")
    for r in by_group.iter_rows(named=True):
        lines.append(
            f"| {r['vessel_group']} | {r['voyages']:,} | {r['vessels']:,} | {r['points']:,} |"
        )
    return "\n".join(lines) + "\n"


def stage(cfg: Config) -> None:
    """CLI stage: build processed points and voyages, plus the data summary report."""
    points, voyages, rep = build_dataset(cfg)
    p_path, v_path = processed_paths(cfg)
    p_path.parent.mkdir(parents=True, exist_ok=True)
    points.write_parquet(p_path, compression="zstd")
    voyages.write_parquet(v_path, compression="zstd")
    n_days = len(list((Path(cfg.paths.interim) / "days").glob("aoi_*.parquet")))
    report = Path(cfg.paths.reports) / "data_summary.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    write_text(report, summary_markdown(points, voyages, rep, n_days))
    log.info("build: wrote %s, %s, %s", p_path, v_path, report)
