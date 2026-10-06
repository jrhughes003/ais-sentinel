// Run the exported GRU ensemble in the browser with ONNX Runtime Web (WASM, single thread:
// GitHub Pages cannot send the COOP/COEP headers that multi-threading needs).
import * as ort from "onnxruntime-web/wasm";
import wasmUrl from "onnxruntime-web/ort-wasm-simd-threaded.wasm?url";
import { type AnchorInfo, buildFeatures, deadReckoning, enuToLatlon, type RawWindow } from "./features";

ort.env.wasm.wasmPaths = { wasm: wasmUrl };
ort.env.wasm.numThreads = 1;

export interface ModelMeta {
  model: string;
  members: string[];
  horizons_min: number[];
  history_min: number;
  groups: string[];
  ref_lat: number;
  ref_lon: number;
  scales: number[][];
  cov_calibration: number[];
  fixtures: (AnchorInfo & {
    raw: RawWindow;
    seq: number[][];
    ctx: number[];
    pred_lat: number[];
    pred_lon: number[];
    pred_cov: number[][][];
  })[];
}

export interface LivePrediction {
  lat: number[];
  lon: number[];
  cov: [number, number, number][]; // (cee, cnn, cen) in m^2, calibrated
  seq: Float32Array;
  ctx: Float32Array;
}

const BASE = import.meta.env.BASE_URL;
let loaded: Promise<{ meta: ModelMeta; sessions: ort.InferenceSession[] }> | null = null;

export function loadModel(): Promise<{ meta: ModelMeta; sessions: ort.InferenceSession[] }> {
  loaded ??= (async () => {
    const meta = (await (await fetch(`${BASE}model/model.json`)).json()) as ModelMeta;
    const sessions = await Promise.all(
      meta.members.map((m) => ort.InferenceSession.create(`${BASE}model/${m}`, { executionProviders: ["wasm"] })),
    );
    return { meta, sessions };
  })();
  return loaded;
}

/** Forecast for one anchor: ensemble moment-matching, exactly as prediction/ml.py. */
export async function predict(raw: RawWindow, a: AnchorInfo): Promise<LivePrediction> {
  const { meta, sessions } = await loadModel();
  const H = meta.horizons_min.length;
  const { seq, ctx, T } = buildFeatures(raw, a, meta.history_min, meta.ref_lat, meta.ref_lon, meta.groups);
  const feeds = {
    seq: new ort.Tensor("float32", seq, [1, T, 6]),
    ctx: new ort.Tensor("float32", ctx, [1, ctx.length]),
  };
  const means: number[][][] = [];
  const covs: number[][][] = [];
  for (let k = 0; k < sessions.length; k++) {
    const out = await sessions[k].run(feeds);
    const mu = out.mu.data as Float32Array; // (1, H, K, 2)
    const ls = out.log_s.data as Float32Array;
    const rho = out.rho.data as Float32Array; // (1, H, K)
    const K = out.rho.dims[2];
    const logit = out.logit.data as Float32Array;
    const m: number[][] = [];
    const c: number[][] = [];
    for (let h = 0; h < H; h++) {
      const s = meta.scales[k][h];
      // Mixture moments over K components (K = 1 for the Gaussian model).
      const w = Array.from({ length: K }, (_, j) => Math.exp(logit[h * K + j]));
      const wsum = w.reduce((x, y) => x + y, 0);
      let me = 0, mn = 0;
      for (let j = 0; j < K; j++) {
        me += (w[j] / wsum) * mu[(h * K + j) * 2];
        mn += (w[j] / wsum) * mu[(h * K + j) * 2 + 1];
      }
      let cee = 0, cnn = 0, cen = 0;
      for (let j = 0; j < K; j++) {
        const se = Math.exp(ls[(h * K + j) * 2]);
        const sn = Math.exp(ls[(h * K + j) * 2 + 1]);
        const r = rho[h * K + j];
        const de = mu[(h * K + j) * 2] - me;
        const dn = mu[(h * K + j) * 2 + 1] - mn;
        const p = w[j] / wsum;
        cee += p * (se * se + de * de);
        cnn += p * (sn * sn + dn * dn);
        cen += p * (r * se * sn + de * dn);
      }
      m.push([me * s, mn * s]);
      c.push([cee * s * s, cnn * s * s, cen * s * s]);
    }
    means.push(m);
    covs.push(c);
  }
  const dr = deadReckoning(a, meta.horizons_min);
  const E = sessions.length;
  const lat: number[] = [];
  const lon: number[] = [];
  const cov: [number, number, number][] = [];
  for (let h = 0; h < H; h++) {
    const me = means.reduce((x, m) => x + m[h][0], 0) / E;
    const mn = means.reduce((x, m) => x + m[h][1], 0) / E;
    let cee = 0, cnn = 0, cen = 0;
    for (let k = 0; k < E; k++) {
      const de = means[k][h][0] - me;
      const dn = means[k][h][1] - mn;
      cee += (covs[k][h][0] + de * de) / E;
      cnn += (covs[k][h][1] + dn * dn) / E;
      cen += (covs[k][h][2] + de * dn) / E;
    }
    const scale = meta.cov_calibration[h] * 1e6; // km^2 -> m^2, then validation calibration
    const [la, lo] = enuToLatlon((dr[h][0] + me) * 1000, (dr[h][1] + mn) * 1000, a.lat0, a.lon0);
    lat.push(la);
    lon.push(lo);
    cov.push([cee * scale, cnn * scale, cen * scale]);
  }
  return { lat, lon, cov, seq, ctx };
}
