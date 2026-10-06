"""Ingest tests: schema normalisation and AOI filtering, using tiny synthetic files."""

from __future__ import annotations

import zipfile
from datetime import date
from pathlib import Path

import polars as pl
import pytest

from ais_sentinel.config import Config
from ais_sentinel.data.download import (
    BBox,
    daterange,
    filter_csv_to_aoi,
    interim_path,
    load_manifest,
    run_download,
    url_for,
)
from ais_sentinel.data.schema import CANONICAL_SCHEMA, CSV2_RENAME, LEGACY_RENAME, detect_rename

LEGACY = (
    "MMSI,BaseDateTime,LAT,LON,SOG,COG,Heading,VesselName,IMO,CallSign,VesselType,Status,"
    "Length,Width,Draft,Cargo,TransceiverClass\n"
    "245299000,2023-07-12T00:00:03,41.87042,-82.59744,12.1,285.0,285.0,ATLANTICBORG,"
    "IMO9466350,PCEC,70,0,142,21,9.7,79,A\n"
    "316023959,2023-07-12T00:00:05,42.98361,-82.40972,0.0,16.8,511.0,PRIDE,,CFN6705,31,0,,,,52,A\n"
    '367704390,2023-07-12T00:00:04,42.3634,-83.0119,0.0,302.2,511.0,RAY "CHIEF" TONEY,,WDI5024,'
    "31,12,25,,,57,A\n"
    "111111111,2023-07-12T00:01:00,30.0,-90.0,5.0,10.0,10.0,FAR AWAY,,,70,0,,,,,A\n"
)
CSV2 = (
    "mmsi,base_date_time,longitude,latitude,sog,cog,heading,vessel_name,imo,call_sign,"
    "vessel_type,status,length,width,draft,cargo,transceiver\n"
    "245299000,2024-07-12 00:00:03,-82.59744,41.87042,12.1,285.0,285,ATLANTICBORG,IMO9466350,"
    "PCEC,70,0,142,21,9.7,79,A\n"
)
AOI = BBox(41.4, 43.1, -83.6, -82.3)


def test_detect_rename_both_formats() -> None:
    assert detect_rename(LEGACY.splitlines()[0].split(",")) is LEGACY_RENAME
    assert detect_rename(CSV2.splitlines()[0].split(",")) is CSV2_RENAME
    with pytest.raises(ValueError, match="Unrecognised"):
        detect_rename(["foo", "bar"])


@pytest.mark.parametrize("text", [LEGACY, CSV2])
def test_filter_produces_canonical_utc_schema(tmp_path: Path, text: str) -> None:
    src = tmp_path / "in.csv"
    src.write_text(text, encoding="utf-8")
    out = tmp_path / "out.parquet"
    n_total, n_aoi = filter_csv_to_aoi(src, out, AOI)
    df = pl.read_parquet(out)
    assert n_total == text.count("\n") - 1
    assert n_aoi == df.height
    assert dict(df.schema) == CANONICAL_SCHEMA
    row = df.filter(pl.col("mmsi") == 245299000).row(0, named=True)
    # lat/lon must not be swapped between formats.
    assert row["lat"] == pytest.approx(41.87042)
    assert row["lon"] == pytest.approx(-82.59744)
    # Timestamp is labelled UTC without shifting the wall-clock value.
    assert row["t"].hour == 0
    assert row["t"].second == 3
    assert str(row["t"].tzinfo) == "UTC"


def test_filter_drops_points_outside_aoi_and_keeps_blank_as_null(tmp_path: Path) -> None:
    src = tmp_path / "in.csv"
    src.write_text(LEGACY, encoding="utf-8")
    out = tmp_path / "out.parquet"
    _, n_aoi = filter_csv_to_aoi(src, out, AOI)
    df = pl.read_parquet(out)
    assert n_aoi == 3
    assert 111111111 not in df["mmsi"].to_list()
    pride = df.filter(pl.col("mmsi") == 316023959).row(0, named=True)
    assert pride["imo"] is None
    assert pride["length_m"] is None
    quoted = df.filter(pl.col("mmsi") == 367704390).row(0, named=True)
    assert quoted["vessel_name"] == 'RAY "CHIEF" TONEY'
    assert quoted["call_sign"] == "WDI5024"


def test_run_download_uses_existing_zip_and_skips_done_days(tmp_cfg: Config) -> None:
    day = date(2023, 7, 12)
    raw = Path(tmp_cfg.paths.raw)
    raw.mkdir(parents=True)
    zip_path = raw / Path(url_for(tmp_cfg, day)).name
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("AIS_2023_07_12.csv", LEGACY)
    manifest = run_download(tmp_cfg, [day])
    assert manifest["2023-07-12"].rows_aoi == 3
    assert interim_path(tmp_cfg, day).exists()
    assert not zip_path.exists(), "raw national file should be deleted after filtering"
    # Second run: nothing to do, and no network access is attempted (zip is gone).
    assert run_download(tmp_cfg, [day]) == load_manifest(tmp_cfg)


def test_daterange_inclusive() -> None:
    days = daterange(date(2023, 5, 30), date(2023, 6, 2))
    assert days[0] == date(2023, 5, 30)
    assert days[-1] == date(2023, 6, 2)
    assert len(days) == 4


def test_url_template() -> None:
    from ais_sentinel.config import load_config

    assert url_for(load_config(), date(2023, 5, 1)).endswith("/2023/AIS_2023_05_01.zip")
