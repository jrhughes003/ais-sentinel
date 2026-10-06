// Browser port of ais_sentinel.prediction.features.build_features (Python). Keep in sync:
// the site test "in-browser model matches Python" compares both on exported fixtures.

const A = 6_378_137.0;
const F = 1 / 298.257223563;
const E2 = F * (2 - F);
const KNOT_MS = 1852 / 3600;
const rad = (d: number) => (d * Math.PI) / 180;

function ecef(latDeg: number, lonDeg: number): [number, number, number] {
  const lat = rad(latDeg);
  const lon = rad(lonDeg);
  const n = A / Math.sqrt(1 - E2 * Math.sin(lat) ** 2);
  return [n * Math.cos(lat) * Math.cos(lon), n * Math.cos(lat) * Math.sin(lon), n * (1 - E2) * Math.sin(lat)];
}

/** Exact WGS84 -> local ENU (east, north) in metres, as geo.latlon_to_enu. */
export function latlonToEnu(lat: number, lon: number, lat0: number, lon0: number): [number, number] {
  const [x, y, z] = ecef(lat, lon);
  const [x0, y0, z0] = ecef(lat0, lon0);
  const dx = x - x0, dy = y - y0, dz = z - z0;
  const sl = Math.sin(rad(lat0)), cl = Math.cos(rad(lat0));
  const so = Math.sin(rad(lon0)), co = Math.cos(rad(lon0));
  return [-so * dx + co * dy, -sl * co * dx - sl * so * dy + cl * dz];
}

/** Inverse (as geo.enu_to_latlon): tangent-plane point lifted to the ellipsoid. */
export function enuToLatlon(e: number, n: number, lat0: number, lon0: number): [number, number] {
  const sl = Math.sin(rad(lat0)), cl = Math.cos(rad(lat0));
  const so = Math.sin(rad(lon0)), co = Math.cos(rad(lon0));
  const u = -(e * e + n * n) / (2 * A);
  const [x0, y0, z0] = ecef(lat0, lon0);
  const x = x0 - so * e - sl * co * n + cl * co * u;
  const y = y0 + co * e - sl * so * n + cl * so * u;
  const z = z0 + cl * n + sl * u;
  const lon = Math.atan2(y, x);
  const r = Math.hypot(x, y);
  let lat = Math.atan2(z, r * (1 - E2));
  for (let i = 0; i < 5; i++) {
    const nn = A / Math.sqrt(1 - E2 * Math.sin(lat) ** 2);
    const h = r / Math.cos(lat) - nn;
    lat = Math.atan2(z, r * (1 - (E2 * nn) / (nn + h)));
  }
  return [(lat * 180) / Math.PI, (lon * 180) / Math.PI];
}

export interface RawWindow {
  t: number[]; // epoch seconds, increasing, all <= t0
  lat: number[];
  lon: number[];
  sog: (number | null)[];
  cog: (number | null)[];
}

export interface AnchorInfo {
  t0: number;
  lat0: number;
  lon0: number;
  sog0: number | null;
  cog0: number | null;
  length_m: number | null;
  group: string;
}

function interp(x: number, xs: number[], ys: number[]): number {
  // numpy.interp semantics: clamp outside the range.
  if (x <= xs[0]) return ys[0];
  if (x >= xs[xs.length - 1]) return ys[ys.length - 1];
  let lo = 0, hi = xs.length - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (xs[mid] <= x) lo = mid;
    else hi = mid;
  }
  const w = (x - xs[lo]) / (xs[hi] - xs[lo]);
  return ys[lo] + w * (ys[hi] - ys[lo]);
}

function searchsortedLeft(xs: number[], x: number): number {
  let lo = 0, hi = xs.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (xs[mid] < x) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

function ffill(a: (number | null)[]): number[] {
  const out: number[] = [];
  let last = NaN;
  for (const v of a) {
    if (v != null && Number.isFinite(v)) last = v;
    out.push(last);
  }
  return out.map((v) => (Number.isFinite(v) ? v : 0));
}

/** Sequence (T x 6) and context features for one anchor; mirrors build_features exactly. */
export function buildFeatures(
  raw: RawWindow,
  a: AnchorInfo,
  historyMin: number,
  refLat: number,
  refLon: number,
  groups: string[],
): { seq: Float32Array; ctx: Float32Array; T: number } {
  const T = historyMin + 1;
  const idx = raw.t.map((t, i) => [t, i] as const).filter(([t]) => t <= a.t0 + 1e-6).map(([, i]) => i);
  const tp = idx.map((i) => raw.t[i]);
  const en = idx.map((i) => latlonToEnu(raw.lat[i], raw.lon[i], a.lat0, a.lon0));
  const e = en.map((p) => p[0]);
  const n = en.map((p) => p[1]);
  const sog = ffill(idx.map((i) => raw.sog[i]));
  const cog = ffill(idx.map((i) => raw.cog[i]));
  const seq = new Float32Array(T * 6);
  for (let k = 0; k < T; k++) {
    const tq = a.t0 + (k - historyMin) * 60;
    const o = k * 6;
    if (tq < tp[0]) {
      seq[o + 5] = 1; // before the voyage start
      continue;
    }
    seq[o] = interp(tq, tp, e) / 1000;
    seq[o + 1] = interp(tq, tp, n) / 1000;
    const j = Math.min(Math.max(searchsortedLeft(tp, tq), 0), tp.length - 1);
    const jm = Math.min(Math.max(j - 1, 0), tp.length - 1);
    const near = Math.abs(tp[j] - tq) < Math.abs(tp[jm] - tq) ? j : jm;
    seq[o + 2] = sog[near] / 20;
    const c = rad(cog[near]);
    seq[o + 3] = Math.sin(c);
    seq[o + 4] = Math.cos(c);
    seq[o + 5] = Math.abs(tp[near] - tq) > 180 ? 1 : 0;
  }
  const [er, nr] = latlonToEnu(a.lat0, a.lon0, refLat, refLon);
  const sog0 = a.sog0 ?? 0;
  const c0 = rad(a.cog0 ?? 0);
  const length = a.length_m && a.length_m > 0 ? a.length_m : 100;
  const ctx = new Float32Array(6 + groups.length);
  ctx.set([er / 50_000, nr / 50_000, sog0 / 20, Math.sin(c0), Math.cos(c0), Math.log(length) / 6]);
  const g = groups.indexOf(a.group);
  ctx[6 + (g >= 0 ? g : groups.indexOf("unknown"))] = 1;
  return { seq, ctx, T };
}

/** Dead-reckoning displacement (km) per horizon from the anchor's SOG/COG. */
export function deadReckoning(a: AnchorInfo, horizonsMin: number[]): [number, number][] {
  const v = (a.sog0 ?? 0) * KNOT_MS;
  const c = rad(a.cog0 ?? 0);
  return horizonsMin.map((h) => [(v * Math.sin(c) * h * 60) / 1000, (v * Math.cos(c) * h * 60) / 1000]);
}
