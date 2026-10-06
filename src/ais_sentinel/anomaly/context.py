"""Context models learned from the **training split**, used by the anomaly detectors.

All three are simple grids in the regional ENU frame, which makes them explainable ("only
2 training voyages ever passed through this 500 m cell"):

* **Reception grid** (``reception_cell_m``): the fraction of expected one-minute reports
  actually received from moving vessels. MarineCadastre keeps at most one report per vessel
  per minute, so a moving vessel in good coverage appears about every 60 s. A step of dt
  seconds between consecutive fixes therefore implies ≈ dt/60 − 1 missed reports, which
  are attributed to the cell at the step's midpoint. Low-reception cells explain silences
  without anyone going dark.
* **Port/anchorage zones** (``port_cell_m``): cells where many stationary vessel-hours
  were logged by several distinct vessels, dilated by one cell. Stopping there is normal.
* **Traffic grid** (``traffic_cell_m``): the number of distinct training voyages through
  each cell, plus a 12-bin course histogram per cell. Being where almost nobody goes, or
  heading against the usual flow, is unusual.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl
from numpy.typing import NDArray

from ais_sentinel.geo import latlon_to_enu

COURSE_BINS = 12


@dataclass
class Grid:
    """Integer grid over the regional ENU frame."""

    ref_lat: float
    ref_lon: float
    cell_m: float

    def cells(self, lat: NDArray[np.float64], lon: NDArray[np.float64]) -> NDArray[np.int64]:
        """Cell key per point (ix * 100000 + iy, with offsets keeping keys positive)."""
        e, n = latlon_to_enu(lat, lon, self.ref_lat, self.ref_lon)
        ix = np.floor(np.asarray(e) / self.cell_m).astype(np.int64) + 50_000
        iy = np.floor(np.asarray(n) / self.cell_m).astype(np.int64) + 50_000
        return ix * 100_000 + iy


@dataclass
class Context:
    """Learned context for the detectors."""

    reception_grid: Grid
    reception: dict[int, float]  # cell -> received / expected
    port_grid: Grid
    port_cells: set[int]
    traffic_grid: Grid
    traffic_voyages: dict[int, int]  # cell -> distinct voyages
    course_hist: dict[int, NDArray[np.float64]] = field(default_factory=dict)  # cell -> probs

    def reception_at(
        self, lat: NDArray[np.float64], lon: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Reception rate per point (NaN where the cell was never observed)."""
        return np.array(
            [self.reception.get(int(c), np.nan) for c in self.reception_grid.cells(lat, lon)]
        )

    def in_port(self, lat: NDArray[np.float64], lon: NDArray[np.float64]) -> NDArray[np.bool_]:
        """True where the point lies in a learned port/anchorage zone."""
        return np.array(
            [int(c) in self.port_cells for c in self.port_grid.cells(lat, lon)], dtype=bool
        )

    def traffic_at(
        self, lat: NDArray[np.float64], lon: NDArray[np.float64], cog_deg: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """(distinct training voyages in the cell, probability of this course bin in the cell).

        The course probability is NaN where the cell has no course history.
        """
        cells = self.traffic_grid.cells(lat, lon)
        dens = np.array([self.traffic_voyages.get(int(c), 0) for c in cells], dtype=float)
        bins = (np.nan_to_num(np.asarray(cog_deg), nan=0.0) // (360 / COURSE_BINS)).astype(
            int
        ) % COURSE_BINS
        prob = np.array(
            [
                self.course_hist[int(c)][b] if int(c) in self.course_hist else np.nan
                for c, b in zip(cells, bins, strict=True)
            ]
        )
        return dens, prob


def _dilate(cells: set[int]) -> set[int]:
    out = set(cells)
    for c in cells:
        ix, iy = divmod(c, 100_000)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                out.add((ix + dx) * 100_000 + (iy + dy))
    return out


def learn_context(
    train_points: pl.DataFrame,
    ref_lat: float,
    ref_lon: float,
    reception_cell_m: float = 2000.0,
    port_cell_m: float = 500.0,
    traffic_cell_m: float = 500.0,
    port_min_vessel_hours: float = 12.0,
    port_min_vessels: int = 3,
    course_min_fixes: int = 20,
) -> Context:
    """Learn reception, port and traffic grids from training-split points."""
    p = train_points.sort("mmsi", "t")
    lat, lon = p["lat"].to_numpy(), p["lon"].to_numpy()
    t = p["t"].dt.epoch("us").to_numpy() / 1e6
    mmsi = p["mmsi"].to_numpy()
    sog = p["sog_kn"].cast(pl.Float64).fill_null(0.0).to_numpy()
    cog = p["cog_deg"].cast(pl.Float64).fill_null(np.nan).to_numpy()

    # Reception: steps between consecutive fixes of the same moving vessel (dt <= 30 min).
    same = np.concatenate([[False], mmsi[1:] == mmsi[:-1]])
    dt = np.concatenate([[np.nan], np.diff(t)])
    step = same & (dt <= 1800) & (sog >= 2.0) & np.concatenate([[False], sog[:-1] >= 2.0])
    rg = Grid(ref_lat, ref_lon, reception_cell_m)
    idx = np.flatnonzero(step)
    mid = rg.cells((lat[idx] + lat[idx - 1]) / 2, (lon[idx] + lon[idx - 1]) / 2)
    missed = np.maximum(np.round(dt[idx] / 60.0) - 1, 0)
    rec_df = (
        pl.DataFrame({"cell": mid, "missed": missed})
        .group_by("cell")
        .agg(received=pl.len(), missed=pl.col("missed").sum())
        .with_columns(rate=pl.col("received") / (pl.col("received") + pl.col("missed")))
    )
    reception = dict(zip(rec_df["cell"].to_list(), rec_df["rate"].to_list(), strict=True))

    # Ports/anchorages: stationary vessel-hours by cell (time until the vessel's next fix).
    pg = Grid(ref_lat, ref_lon, port_cell_m)
    dt_next = np.concatenate([np.diff(t), [np.nan]])
    same_next = np.concatenate([mmsi[1:] == mmsi[:-1], [False]])
    still = same_next & (sog < 1.0) & (dt_next <= 3600)
    sdf = (
        pl.DataFrame(
            {
                "cell": pg.cells(lat[still], lon[still]),
                "mmsi": mmsi[still],
                "h": dt_next[still] / 3600,
            }
        )
        .group_by("cell")
        .agg(hours=pl.col("h").sum(), vessels=pl.col("mmsi").n_unique())
        .filter(
            (pl.col("hours") >= port_min_vessel_hours) & (pl.col("vessels") >= port_min_vessels)
        )
    )
    port_cells = _dilate(set(sdf["cell"].to_list()))

    # Traffic: distinct voyages and course histogram of moving fixes per cell.
    tg = Grid(ref_lat, ref_lon, traffic_cell_m)
    moving = sog >= 2.0
    cells_m = tg.cells(lat[moving], lon[moving])
    vids = p["voyage_id"].to_numpy()[moving] if "voyage_id" in p.columns else mmsi[moving]
    tdf = (
        pl.DataFrame({"cell": cells_m, "vid": vids})
        .group_by("cell")
        .agg(n=pl.col("vid").n_unique())
    )
    traffic = dict(zip(tdf["cell"].to_list(), tdf["n"].to_list(), strict=True))
    ok = np.isfinite(cog[moving])
    bins = (cog[moving][ok] // (360 / COURSE_BINS)).astype(int) % COURSE_BINS
    hdf = pl.DataFrame({"cell": cells_m[ok], "bin": bins}).group_by("cell", "bin").len()
    course_hist: dict[int, NDArray[np.float64]] = {}
    for (cell,), g in hdf.group_by(["cell"]):
        counts = np.zeros(COURSE_BINS)
        counts[g["bin"].to_numpy()] = g["len"].to_numpy()
        if counts.sum() >= course_min_fixes:
            # Laplace smoothing so unseen bins get a small, nonzero probability.
            course_hist[int(cell)] = (counts + 0.5) / (counts.sum() + 0.5 * COURSE_BINS)
    return Context(rg, reception, pg, port_cells, tg, traffic, course_hist)
