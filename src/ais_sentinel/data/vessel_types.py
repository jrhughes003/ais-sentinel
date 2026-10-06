"""Map AIS ship-type codes to the coarse vessel groups used for reporting.

Codes follow ITU-R M.1371 (0–99) plus MarineCadastre's AVIS codes (1001–1025); see
MarineCadastre's ``vessel-type-codes-2018.pdf``.
"""

from __future__ import annotations

import polars as pl

GROUPS = (
    "cargo",
    "tanker",
    "passenger",
    "tug_tow",
    "fishing",
    "pleasure",
    "other",
    "unknown",
)

#: Groups considered "commercial" for the showcase (privacy: avoid pleasure craft).
COMMERCIAL = frozenset({"cargo", "tanker", "passenger", "tug_tow", "fishing"})


def vessel_group(code: int | None) -> str:
    """Return the vessel group name for an AIS / AVIS vessel-type code."""
    if code is None or code == 0:
        return "unknown"
    if 70 <= code <= 79 or code in (1003, 1004, 1016):
        return "cargo"
    if 80 <= code <= 89 or code in (1017, 1023, 1024):
        return "tanker"
    if 60 <= code <= 69 or 1012 <= code <= 1015:
        return "passenger"
    if code in (21, 22, 31, 32, 52, 1025):
        return "tug_tow"
    if code in (30, 1001, 1002):
        return "fishing"
    if code in (36, 37, 1019):
        return "pleasure"
    return "other"


def vessel_group_expr(col: str = "vessel_type") -> pl.Expr:
    """Polars expression that maps a vessel-type column to its group name."""
    c = pl.col(col)
    return (
        pl.when(c.is_null() | (c == 0))
        .then(pl.lit("unknown"))
        .when(c.is_between(70, 79) | c.is_in([1003, 1004, 1016]))
        .then(pl.lit("cargo"))
        .when(c.is_between(80, 89) | c.is_in([1017, 1023, 1024]))
        .then(pl.lit("tanker"))
        .when(c.is_between(60, 69) | c.is_between(1012, 1015))
        .then(pl.lit("passenger"))
        .when(c.is_in([21, 22, 31, 32, 52, 1025]))
        .then(pl.lit("tug_tow"))
        .when(c.is_in([30, 1001, 1002]))
        .then(pl.lit("fishing"))
        .when(c.is_in([36, 37, 1019]))
        .then(pl.lit("pleasure"))
        .otherwise(pl.lit("other"))
        .alias("vessel_group")
    )
