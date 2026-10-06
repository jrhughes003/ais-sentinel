"""Download MarineCadastre daily files and filter them to the area of interest (AOI).

National daily files are about 330 MB zipped / 0.9 GB unzipped, and the AOI is under 1% of
the rows. Each file is therefore downloaded, filtered to the AOI in streaming mode with
polars, written as a small per-day Parquet file in ``data/interim/``, and (by default)
deleted. A JSON manifest records what was processed so re-runs skip completed days.

Downloads are strictly sequential with a configurable pause between files, to stay polite
to NOAA's servers.
"""

from __future__ import annotations

import json
import logging
import tempfile
import time
import zipfile
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import polars as pl
import requests

from ais_sentinel.config import Config, as_date
from ais_sentinel.data.schema import detect_rename, to_canonical

log = logging.getLogger(__name__)

CHUNK = 1 << 20  # 1 MiB


@dataclass(frozen=True)
class BBox:
    """A lat/lon bounding box in degrees (WGS84)."""

    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float

    @classmethod
    def from_config(cls, cfg: Config) -> BBox:
        """Build the AOI box from ``cfg.region``."""
        r = cfg.region
        return cls(r.lat_min, r.lat_max, r.lon_min, r.lon_max)

    def expr(self) -> pl.Expr:
        """Polars predicate selecting rows inside the box."""
        return pl.col("lat").is_between(self.lat_min, self.lat_max) & pl.col("lon").is_between(
            self.lon_min, self.lon_max
        )


@dataclass
class DayRecord:
    """Manifest entry for one processed day."""

    day: str
    url: str
    bytes_downloaded: int
    rows_national: int
    rows_aoi: int
    processed_at: str


def daterange(start: date, end: date) -> list[date]:
    """Inclusive list of dates from ``start`` to ``end``."""
    return [start + timedelta(days=i) for i in range((end - start).days + 1)]


def interim_path(cfg: Config, day: date) -> Path:
    """Path of the per-day AOI Parquet file."""
    return Path(cfg.paths.interim) / "days" / f"aoi_{day.isoformat()}.parquet"


def manifest_path(cfg: Config) -> Path:
    """Path of the JSON download manifest."""
    return Path(cfg.paths.interim) / "manifest.json"


def load_manifest(cfg: Config) -> dict[str, DayRecord]:
    """Load the manifest (empty if it does not exist yet)."""
    path = manifest_path(cfg)
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {k: DayRecord(**v) for k, v in raw.items()}


def save_manifest(cfg: Config, manifest: dict[str, DayRecord]) -> None:
    """Atomically write the manifest."""
    path = manifest_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(
        json.dumps({k: asdict(v) for k, v in sorted(manifest.items())}, indent=1),
        encoding="utf-8",
    )
    tmp.replace(path)


def url_for(cfg: Config, day: date) -> str:
    """Fill the URL template for a given day."""
    return str(cfg.download.url_template).format(year=day.year, month=day.month, day=day.day)


def fetch(url: str, dest: Path, timeout: float) -> int:
    """Stream ``url`` to ``dest`` (via a ``.part`` file). Returns bytes written."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    n = 0
    with requests.get(url, stream=True, timeout=timeout) as resp:
        resp.raise_for_status()
        with part.open("wb") as fh:
            for chunk in resp.iter_content(CHUNK):
                fh.write(chunk)
                n += len(chunk)
    part.replace(dest)
    return n


def filter_csv_to_aoi(csv_path: Path, out_path: Path, bbox: BBox) -> tuple[int, int]:
    """Filter one national CSV to the AOI and write canonical Parquet.

    Returns:
        (national row count, AOI row count).
    """
    header = csv_path.open(encoding="utf-8").readline().strip().split(",")
    rename = detect_rename(header)
    # MarineCadastre files never use CSV quoting (every line has exactly 16 commas), but
    # vessel names contain literal quotes such as ``RAY "CHIEF" TONEY``. Treating ``"`` as a
    # quote character would corrupt parsing, so quoting is disabled.
    # Rare malformed lines carry an extra field (e.g. a leading comma, seen once in 8.2 M
    # rows on 2023-05-05). ``truncate_ragged_lines`` keeps the parse going; such a line
    # shifts right, so its MMSI or timestamp fails to parse and cleaning drops it.
    raw = pl.scan_csv(
        csv_path, infer_schema=False, quote_char=None, truncate_ragged_lines=True
    )  # all strings; cast explicitly in to_canonical
    canon = to_canonical(raw, rename)
    n_total = int(raw.select(pl.len()).collect(engine="streaming").item())
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp.parquet")
    canon.filter(bbox.expr()).sink_parquet(tmp, compression="zstd")
    tmp.replace(out_path)
    n_aoi = int(pl.scan_parquet(out_path).select(pl.len()).collect().item())
    return n_total, n_aoi


def raw_zip_path(cfg: Config, day: date) -> Path:
    """Where a day's national zip is (transiently) stored."""
    return Path(cfg.paths.raw) / Path(url_for(cfg, day)).name


def ensure_zip(cfg: Config, day: date) -> int:
    """Download a day's zip unless it is already present. Returns its size in bytes."""
    zip_path = raw_zip_path(cfg, day)
    if zip_path.exists():
        return zip_path.stat().st_size
    return fetch(url_for(cfg, day), zip_path, float(cfg.download.timeout_seconds))


def process_day(cfg: Config, day: date, bbox: BBox) -> DayRecord:
    """Filter one day's (downloaded) zip to the AOI and clean up the raw file."""
    nbytes = ensure_zip(cfg, day)
    zip_path = raw_zip_path(cfg, day)
    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory(dir=zip_path.parent) as tmpdir:
        with zipfile.ZipFile(zip_path) as zf:
            members = [m for m in zf.namelist() if m.lower().endswith(".csv")]
            if len(members) != 1:
                raise ValueError(f"{zip_path}: expected one CSV member, found {members}")
            csv_path = Path(zf.extract(members[0], tmpdir))
        n_total, n_aoi = filter_csv_to_aoi(csv_path, interim_path(cfg, day), bbox)
    if not cfg.download.keep_raw:
        zip_path.unlink(missing_ok=True)
    log.info(
        "%s: %d national rows -> %d AOI rows (filter %.0f s)",
        day,
        n_total,
        n_aoi,
        time.perf_counter() - t0,
    )
    return DayRecord(
        day=day.isoformat(),
        url=url_for(cfg, day),
        bytes_downloaded=nbytes,
        rows_national=n_total,
        rows_aoi=n_aoi,
        processed_at=datetime.now(UTC).isoformat(timespec="seconds"),
    )


def _polite_fetch(cfg: Config, day: date) -> int:
    """Pause, then download: used by the single prefetch worker."""
    time.sleep(float(cfg.download.pause_seconds))
    return ensure_zip(cfg, day)


def run_download(cfg: Config, days: list[date] | None = None) -> dict[str, DayRecord]:
    """Process every configured day that is not already in the manifest.

    Day N+1's zip is downloaded on one background thread while day N is being filtered.
    The executor has a single worker, so at most one HTTP download is ever in flight, and
    each request is still preceded by the politeness pause. Peak disk use grows by one
    zip (about 330 MB).
    """
    bbox = BBox.from_config(cfg)
    if days is None:
        days = daterange(as_date(cfg.download.start), as_date(cfg.download.end))
    manifest = load_manifest(cfg)
    todo = [d for d in days if d.isoformat() not in manifest or not interim_path(cfg, d).exists()]
    log.info("%d of %d days to process", len(todo), len(days))
    if not todo:
        return manifest
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending: Future[int] = pool.submit(ensure_zip, cfg, todo[0])
        for i, day in enumerate(todo):
            current = pending
            try:
                current.result()  # wait for this day's download
            except Exception as exc:
                log.error("%s download failed: %s: %s (will retry)", day, type(exc).__name__, exc)
                if i + 1 < len(todo):
                    pending = pool.submit(_polite_fetch, cfg, todo[i + 1])
                continue
            if i + 1 < len(todo):
                pending = pool.submit(_polite_fetch, cfg, todo[i + 1])
            try:
                manifest[day.isoformat()] = process_day(cfg, day, bbox)
            except Exception as exc:  # one bad day must not stop a multi-hour run
                log.error(
                    "%s failed: %s: %s (will retry on next run)", day, type(exc).__name__, exc
                )
                continue
            save_manifest(cfg, manifest)
    return manifest


def stage(cfg: Config) -> None:
    """CLI stage: download and AOI-filter every configured day."""
    manifest = run_download(cfg)
    rows = sum(r.rows_aoi for r in manifest.values())
    log.info("manifest: %d days, %d AOI rows", len(manifest), rows)
