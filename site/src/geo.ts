// Small geometry helpers for drawing on the map (display only; analysis happens in Python).

const M_PER_DEG_LAT = 111_320;

/** Offset a lat/lon by east/north metres (local flat approximation, fine at display scale). */
export function offset(lat: number, lon: number, e: number, n: number): [number, number] {
  const dLat = n / M_PER_DEG_LAT;
  const dLon = e / (M_PER_DEG_LAT * Math.cos((lat * Math.PI) / 180));
  return [lon + dLon, lat + dLat];
}

/**
 * Polygon ring for the p-probability ellipse of a 2-D Gaussian with ENU covariance
 * [[cee, cen], [cen, cnn]] (m^2) centred at (lat, lon).
 * radius^2 = chi2_2(p) = -2 ln(1 - p).
 */
export function ellipse(
  lat: number,
  lon: number,
  cee: number,
  cnn: number,
  cen: number,
  p = 0.9,
  steps = 48,
): [number, number][] {
  const k = Math.sqrt(-2 * Math.log(1 - p));
  // Eigen-decomposition of the symmetric 2x2 covariance.
  const tr = cee + cnn;
  const det = cee * cnn - cen * cen;
  const disc = Math.sqrt(Math.max((tr * tr) / 4 - det, 0));
  const l1 = Math.max(tr / 2 + disc, 0);
  const l2 = Math.max(tr / 2 - disc, 0);
  const theta = Math.abs(cen) < 1e-12 && cee >= cnn ? 0 : Math.atan2(l1 - cee, cen);
  const a = k * Math.sqrt(l1);
  const b = k * Math.sqrt(l2);
  const ring: [number, number][] = [];
  for (let i = 0; i <= steps; i++) {
    const t = (2 * Math.PI * i) / steps;
    const x = a * Math.cos(t);
    const y = b * Math.sin(t);
    const e = x * Math.cos(theta) - y * Math.sin(theta);
    const n = x * Math.sin(theta) + y * Math.cos(theta);
    ring.push(offset(lat, lon, e, n));
  }
  return ring;
}

/** Circle of radius r metres as a polygon ring. */
export function circle(lat: number, lon: number, r: number, steps = 24): [number, number][] {
  const ring: [number, number][] = [];
  for (let i = 0; i <= steps; i++) {
    const t = (2 * Math.PI * i) / steps;
    ring.push(offset(lat, lon, r * Math.cos(t), r * Math.sin(t)));
  }
  return ring;
}

export function bounds(lons: number[], lats: number[], pad = 0): [[number, number], [number, number]] {
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (let i = 0; i < lons.length; i++) {
    if (!Number.isFinite(lons[i]) || !Number.isFinite(lats[i])) continue;
    x0 = Math.min(x0, lons[i]);
    x1 = Math.max(x1, lons[i]);
    y0 = Math.min(y0, lats[i]);
    y1 = Math.max(y1, lats[i]);
  }
  return [
    [x0 - pad, y0 - pad],
    [x1 + pad, y1 + pad],
  ];
}
