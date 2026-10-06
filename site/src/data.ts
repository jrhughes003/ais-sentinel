// Data contracts for the JSON files written by `ais-sentinel export` (src/ais_sentinel/export/web.py).
// Everything the site shows is precomputed; nothing is fetched from anywhere but this site.

export interface Region {
  name: string;
  lat_min: number;
  lat_max: number;
  lon_min: number;
  lon_max: number;
}

export interface Meta {
  generated: string;
  attribution: string;
  region: Region;
  note: string | null; // e.g. "development preview": shown prominently when present
}

export interface Voyage {
  id: string;
  mmsi: number;
  name: string | null;
  group: string;
  length_m: number | null;
  t0: number; // epoch seconds of first fix
  dt: number[]; // seconds since previous fix (first = 0)
  lat: number[];
  lon: number[];
  s_lat: number[];
  s_lon: number[];
  s_sd_m: number[]; // 1-sigma position uncertainty of the smoothed track (m)
  mode: number[]; // dominant IMM mode per fix: 0 stationary, 1 cruising, 2 turning
  rejected: number[]; // indices of fixes rejected as outliers
  sog_kn: (number | null)[];
  cog_deg: (number | null)[];
}

export interface TracksFile extends Meta {
  voyages: Voyage[];
}

export interface ModelInfo {
  id: string;
  label: string;
  short: string;
  kind: "baseline" | "ml";
}

export interface PredSample {
  id: string;
  mmsi: number;
  name: string | null;
  group: string;
  t0: number;
  lat0: number;
  lon0: number;
  hist: { lat: number[]; lon: number[] };
  future: { t: number[]; lat: number[]; lon: number[] }; // t = minutes after t0
  preds: Record<string, { lat: number[]; lon: number[]; cee: number[]; cnn: number[]; cen: number[] }>;
  errors_km: Record<string, (number | null)[]>;
}

export interface PredictionsFile extends Meta {
  split: string;
  horizons: number[];
  models: ModelInfo[];
  samples: PredSample[];
}

export interface AnomalyEvent {
  id: string;
  type: "gap" | "jump" | "loiter" | "deviation" | "rendezvous";
  mmsi: number;
  mmsi2: number | null;
  name: string | null;
  group: string;
  t_start: number;
  t_end: number;
  lat: number;
  lon: number;
  lat_end: number;
  lon_end: number;
  duration_min: number;
  score: number;
  explanation: string;
  case_study: string | null;
  track: { t: number[]; lat: number[]; lon: number[] };
  track2: { t: number[]; lat: number[]; lon: number[] } | null;
}

export interface AnomaliesFile extends Meta {
  events: AnomalyEvent[];
  counts: Record<string, number>;
}

export interface MetricRow {
  model: string;
  horizon_min: number;
  n: number;
  voyages: number;
  mean_km: number;
  mean_lo: number;
  mean_hi: number;
  median_km: number;
  p90_km: number;
  cov50: number;
  cov90: number;
  nll: number;
}

export interface Criterion {
  area: string;
  name: string;
  target: string;
  result: string;
  met: boolean;
}

export interface ResultsFile extends Meta {
  period: { start: string; end: string; test_start: string; test_end: string };
  counts: { points: number; vessels: number; voyages: number; days: number };
  models: ModelInfo[];
  prediction: { split: string; rows: MetricRow[]; by_group: (MetricRow & { vessel_group: string })[] } | null;
  verdict: { horizon: number; best_baseline: string; diff_km: number; lo: number; hi: number; rel: number }[];
  tracking: { segment: string; cv_rmse: number; imm_rmse: number; cv_median: number; imm_median: number; cv_p95: number; imm_p95: number }[];
  anomaly: { by_magnitude: { type: string; magnitude: number; n: number; precision: number; recall: number }[] } | null;
  criteria: Criterion[];
}

const BASE = import.meta.env.BASE_URL;
const cache = new Map<string, Promise<unknown>>();

/** Fetch a JSON data file once (cached for the session). */
export function getData<T>(name: string): Promise<T> {
  if (!cache.has(name)) {
    cache.set(
      name,
      fetch(`${BASE}data/${name}`).then((r) => {
        if (!r.ok) throw new Error(`${name}: HTTP ${r.status}`);
        return r.json();
      }),
    );
  }
  return cache.get(name) as Promise<T>;
}

export const GROUP_LABEL: Record<string, string> = {
  cargo: "Cargo",
  tanker: "Tanker",
  passenger: "Passenger / ferry",
  tug_tow: "Tug / tow",
  fishing: "Fishing",
  pleasure: "Pleasure craft",
  other: "Other",
  unknown: "Unknown",
};

export const ANOMALY_LABEL: Record<AnomalyEvent["type"], string> = {
  gap: "AIS gap (going dark)",
  jump: "Impossible jump",
  loiter: "Loitering",
  deviation: "Route deviation",
  rendezvous: "Rendezvous",
};

export function fmtTime(epoch: number): string {
  return new Date(epoch * 1000).toISOString().replace("T", " ").slice(0, 16) + " UTC";
}

export function fmtDuration(min: number): string {
  if (min < 90) return `${Math.round(min)} min`;
  return `${(min / 60).toFixed(1)} h`;
}

export function vesselLabel(name: string | null, mmsi: number): string {
  return name ? `${name} (MMSI ${mmsi})` : `MMSI ${mmsi}`;
}
