"""Canonical point schema and normalisers for the MarineCadastre file formats.

MarineCadastre has published the same data under different column names over the years:

* legacy CSV in zip (2015–2023): ``MMSI,BaseDateTime,LAT,LON,SOG,COG,Heading,...``
* ``csv2`` Zstandard CSV (2024+): ``mmsi,base_date_time,longitude,latitude,sog,...``

Everything downstream works on the *canonical* schema defined here, so supporting a new
format only requires a new rename map.
"""

from __future__ import annotations

import polars as pl

#: Canonical columns and dtypes. Units are in the column names; times are UTC.
CANONICAL_SCHEMA: dict[str, pl.DataType] = {
    "mmsi": pl.Int64(),
    "t": pl.Datetime("us", "UTC"),
    "lat": pl.Float64(),
    "lon": pl.Float64(),
    "sog_kn": pl.Float32(),
    "cog_deg": pl.Float32(),
    "heading_deg": pl.Float32(),
    "vessel_name": pl.String(),
    "imo": pl.String(),
    "call_sign": pl.String(),
    "vessel_type": pl.Int32(),
    "status": pl.Int32(),
    "length_m": pl.Float32(),
    "width_m": pl.Float32(),
    "draft_m": pl.Float32(),
    "cargo": pl.Int32(),
    "transceiver": pl.String(),
}

LEGACY_RENAME: dict[str, str] = {
    "MMSI": "mmsi",
    "BaseDateTime": "t",
    "LAT": "lat",
    "LON": "lon",
    "SOG": "sog_kn",
    "COG": "cog_deg",
    "Heading": "heading_deg",
    "VesselName": "vessel_name",
    "IMO": "imo",
    "CallSign": "call_sign",
    "VesselType": "vessel_type",
    "Status": "status",
    "Length": "length_m",
    "Width": "width_m",
    "Draft": "draft_m",
    "Cargo": "cargo",
    "TransceiverClass": "transceiver",
}

CSV2_RENAME: dict[str, str] = {
    "mmsi": "mmsi",
    "base_date_time": "t",
    "latitude": "lat",
    "longitude": "lon",
    "sog": "sog_kn",
    "cog": "cog_deg",
    "heading": "heading_deg",
    "vessel_name": "vessel_name",
    "imo": "imo",
    "call_sign": "call_sign",
    "vessel_type": "vessel_type",
    "status": "status",
    "length": "length_m",
    "width": "width_m",
    "draft": "draft_m",
    "cargo": "cargo",
    "transceiver": "transceiver",
}


def detect_rename(columns: list[str]) -> dict[str, str]:
    """Pick the rename map matching a file header.

    Raises:
        ValueError: if the header matches no known MarineCadastre format.
    """
    cols = set(columns)
    for mapping in (LEGACY_RENAME, CSV2_RENAME):
        if set(mapping) <= cols:
            return mapping
    raise ValueError(f"Unrecognised AIS header: {sorted(cols)}")


def to_canonical(lf: pl.LazyFrame, rename: dict[str, str]) -> pl.LazyFrame:
    """Rename, parse and cast a raw (all-string) frame to the canonical schema.

    Unparseable numbers become null rather than raising (``strict=False``); the cleaning
    stage then decides what to drop. Timestamps carry no zone suffix in the source files but
    are documented as UTC, so they are parsed as naive and then *labelled* UTC (not converted).
    """
    lf = lf.select([pl.col(src).alias(dst) for src, dst in rename.items()])
    exprs: list[pl.Expr] = []
    for name, dtype in CANONICAL_SCHEMA.items():
        col = pl.col(name)
        if name == "t":
            exprs.append(
                col.str.replace("T", " ")
                .str.to_datetime("%Y-%m-%d %H:%M:%S", time_unit="us", strict=False)
                .dt.replace_time_zone("UTC")
                .alias(name)
            )
        elif dtype == pl.String():
            exprs.append(col.cast(pl.String).str.strip_chars().replace("", None).alias(name))
        elif dtype in (pl.Int32(), pl.Int64()):
            # Some integer fields are written as "57.0"; go via float.
            exprs.append(col.cast(pl.Float64, strict=False).cast(dtype, strict=False).alias(name))
        else:
            exprs.append(col.cast(dtype, strict=False).alias(name))
    return lf.select(exprs)
