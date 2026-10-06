"""Geodesy helpers: WGS84 ↔ local East-North-Up (ENU) frames, distances and unit conversions.



Why a local ENU frame? Kalman filters and neural networks want Cartesian coordinates in

metres. Treating degrees as distances is wrong, because one degree of longitude is only

about 82 km at 42 °N versus 111 km for latitude. We convert each geodetic point to Earth-

Centred Earth-Fixed (ECEF) coordinates, then rotate into a tangent plane anchored at a

reference point. This is exact; there is no map projection. For points within about 150 km

of the anchor, the tangent plane's horizontal distances match true ground distances to

better than 0.05%.



All functions are vectorised over numpy arrays.

"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

# WGS84 ellipsoid

WGS84_A = 6_378_137.0

WGS84_F = 1 / 298.257223563

WGS84_E2 = WGS84_F * (2 - WGS84_F)

#: Mean Earth radius (IUGG), used for haversine distances.

EARTH_RADIUS_M = 6_371_008.8


KNOT_MS = 1852.0 / 3600.0  # 1 knot in m/s

NM_M = 1852.0  # 1 nautical mile in metres


FloatArray = NDArray[np.float64]


def geodetic_to_ecef(lat_deg: ArrayLike, lon_deg: ArrayLike, h_m: ArrayLike = 0.0) -> FloatArray:
    """Convert geodetic coordinates to ECEF. Returns an array of shape ``(..., 3)``."""

    lat = np.radians(np.asarray(lat_deg, dtype=float))

    lon = np.radians(np.asarray(lon_deg, dtype=float))

    h = np.asarray(h_m, dtype=float)

    n = WGS84_A / np.sqrt(1 - WGS84_E2 * np.sin(lat) ** 2)

    x = (n + h) * np.cos(lat) * np.cos(lon)

    y = (n + h) * np.cos(lat) * np.sin(lon)

    z = (n * (1 - WGS84_E2) + h) * np.sin(lat)

    return np.stack(np.broadcast_arrays(x, y, z), axis=-1)


def ecef_to_geodetic(xyz: ArrayLike) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Convert ECEF ``(..., 3)`` to (lat_deg, lon_deg, h_m).



    Uses Bowring's initial guess followed by a few fixed-point iterations. This converges

    to sub-millimetre accuracy for near-surface points.

    """

    p = np.asarray(xyz, dtype=float)

    x, y, z = p[..., 0], p[..., 1], p[..., 2]

    lon = np.arctan2(y, x)

    r = np.hypot(x, y)

    lat = np.arctan2(z, r * (1 - WGS84_E2))

    h = np.zeros_like(lat)

    for _ in range(5):
        n = WGS84_A / np.sqrt(1 - WGS84_E2 * np.sin(lat) ** 2)

        h = r / np.cos(lat) - n

        lat = np.arctan2(z, r * (1 - WGS84_E2 * n / (n + h)))

    return np.degrees(lat), np.degrees(lon), h


def _enu_basis(
    lat0_deg: ArrayLike, lon0_deg: ArrayLike
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """East, north and up unit vectors in ECEF at the reference point(s).



    Each has shape ``(..., 3)``. Anchors may be arrays, so many local frames are handled at

    once (for example, one per prediction sample).

    """

    lat0 = np.radians(np.asarray(lat0_deg, dtype=float))

    lon0 = np.radians(np.asarray(lon0_deg, dtype=float))

    sl, cl = np.sin(lat0), np.cos(lat0)

    so, co = np.sin(lon0), np.cos(lon0)

    east = np.stack(np.broadcast_arrays(-so, co, np.zeros_like(so)), axis=-1)

    north = np.stack(np.broadcast_arrays(-sl * co, -sl * so, cl), axis=-1)

    up = np.stack(np.broadcast_arrays(cl * co, cl * so, sl), axis=-1)

    return east, north, up


def latlon_to_enu(
    lat_deg: ArrayLike, lon_deg: ArrayLike, lat0_deg: ArrayLike, lon0_deg: ArrayLike
) -> tuple[FloatArray, FloatArray]:
    """Project points into the ENU tangent plane anchored at (lat0, lon0).



    Points and anchors broadcast against each other, so ``lat0`` of shape (M, 1) with

    ``lat`` of shape (M, H) gives one frame per row.



    Returns:

        (east_m, north_m). The "up" component, which is a few hundred metres below the

        plane at 100 km, is dropped: vessels move on the surface.

    """

    d = geodetic_to_ecef(lat_deg, lon_deg) - geodetic_to_ecef(lat0_deg, lon0_deg)

    east, north, _ = _enu_basis(lat0_deg, lon0_deg)

    return np.sum(d * east, axis=-1), np.sum(d * north, axis=-1)


def enu_to_latlon(
    east_m: ArrayLike, north_m: ArrayLike, lat0_deg: ArrayLike, lon0_deg: ArrayLike
) -> tuple[FloatArray, FloatArray]:
    """Inverse of :func:`latlon_to_enu`, projecting back onto the ellipsoid surface.



    The tangent-plane point is lifted to the surface along the local vertical. That

    vertical is approximated by the anchor's "up" axis, which is accurate to well under

    1 m within 150 km. Broadcasting rules are the same as for :func:`latlon_to_enu`.

    """

    e = np.asarray(east_m, dtype=float)

    n = np.asarray(north_m, dtype=float)

    east, north, up = _enu_basis(lat0_deg, lon0_deg)

    ref = geodetic_to_ecef(lat0_deg, lon0_deg)

    # Height of the tangent plane above the ellipsoid is about (e² + n²) / (2R); remove it.

    u = -(e**2 + n**2) / (2 * WGS84_A)

    xyz = ref + e[..., None] * east + n[..., None] * north + u[..., None] * up

    lat, lon, _ = ecef_to_geodetic(xyz)

    return lat, lon


def haversine_m(lat1: ArrayLike, lon1: ArrayLike, lat2: ArrayLike, lon2: ArrayLike) -> FloatArray:
    """Great-circle distance in metres on a sphere of mean Earth radius.



    The spherical approximation differs from the ellipsoidal geodesic by up to 0.5%. That

    is ample for error metrics, and it is the convention in the AIS prediction literature.

    """

    p1, p2 = np.radians(np.asarray(lat1, dtype=float)), np.radians(np.asarray(lat2, dtype=float))

    dp = p2 - p1

    dl = np.radians(np.asarray(lon2, dtype=float) - np.asarray(lon1, dtype=float))

    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2

    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def sog_cog_to_enu_velocity(sog_kn: ArrayLike, cog_deg: ArrayLike) -> tuple[FloatArray, FloatArray]:
    """Convert speed over ground (knots) and course over ground (degrees clockwise from true

    north) into east/north velocity in m/s."""

    v = np.asarray(sog_kn, dtype=float) * KNOT_MS

    c = np.radians(np.asarray(cog_deg, dtype=float))

    return v * np.sin(c), v * np.cos(c)


def enu_velocity_to_sog_cog(ve: ArrayLike, vn: ArrayLike) -> tuple[FloatArray, FloatArray]:
    """Inverse of :func:`sog_cog_to_enu_velocity`. Returns (sog_kn, cog_deg in [0, 360))."""

    ve_a, vn_a = np.asarray(ve, dtype=float), np.asarray(vn, dtype=float)

    sog = np.hypot(ve_a, vn_a) / KNOT_MS

    cog = np.degrees(np.arctan2(ve_a, vn_a)) % 360.0

    return sog, cog
