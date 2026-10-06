import { fmtTime, getData, GROUP_LABEL, type TracksFile, type Voyage, vesselLabel } from "../data";
import { bounds, ellipse } from "../geo";
import { createMap, fc, lineFeature, pointFeature, polygonFeature, setData, setVisible } from "../map";
import { modelColor } from "../models";
import { $, esc, query, setQuery } from "../util";

const MODE_COLOR = ["#8a94a3", "#0b63ce", "#e8590c"]; // stationary, cruising, turning
const MODE_NAME = ["stationary", "cruising", "turning"];

function times(v: Voyage): number[] {
  const out: number[] = [];
  let t = v.t0;
  for (const d of v.dt) out.push((t += d));
  return out;
}

function modeSegments(v: Voyage): GeoJSON.Feature[] {
  const feats: GeoJSON.Feature[] = [];
  let start = 0;
  for (let i = 1; i <= v.lat.length; i++) {
    if (i === v.lat.length || v.mode[i] !== v.mode[start]) {
      const j = Math.min(i, v.lat.length - 1);
      feats.push(lineFeature(v.s_lon.slice(start, j + 1), v.s_lat.slice(start, j + 1), { mode: v.mode[start] }));
      start = i;
    }
  }
  return feats;
}

export async function render(root: HTMLElement): Promise<void> {
  const data = await getData<TracksFile>("tracks.json");
  const groups = [...new Set(data.voyages.map((v) => v.group))].sort();
  root.innerHTML = `
  <div class="mapview">
    <aside class="panel" aria-label="Track controls">
      <h1>Vessel tracks</h1>
      <p class="small muted">Raw AIS reports vs the IMM-smoothed track with its 95% uncertainty.
      The colour shows the motion mode the filter believes the vessel is in.</p>
      <label for="voyage">Vessel &amp; voyage</label>
      <select id="voyage">
        ${groups
          .map(
            (g) => `<optgroup label="${esc(GROUP_LABEL[g] ?? g)}">${data.voyages
              .filter((v) => v.group === g)
              .map((v) => `<option value="${esc(v.id)}">${esc(v.name ?? v.mmsi)} · ${fmtTime(v.t0).slice(0, 10)}</option>`)
              .join("")}</optgroup>`,
          )
          .join("")}
      </select>
      <fieldset style="border:none;padding:0;margin:0.6rem 0 0">
        <legend class="visually-hidden">Layers</legend>
        <label class="check"><input type="checkbox" id="l-raw" checked /> Raw AIS fixes</label>
        <label class="check"><input type="checkbox" id="l-smooth" checked /> Smoothed track (IMM)</label>
        <label class="check"><input type="checkbox" id="l-unc" checked /> 95% uncertainty</label>
      </fieldset>
      <div class="legend" aria-hidden="true">
        <span><i class="swatch dot" style="background:var(--raw)"></i>raw report</span>
        <span><i class="swatch dot" style="background:var(--bad)"></i>rejected as outlier</span>
        ${MODE_NAME.map((m, i) => `<span><i class="swatch" style="background:${MODE_COLOR[i]}"></i>${m}</span>`).join("")}
        <span><i class="swatch area" style="background:var(--smooth)"></i>95% position uncertainty</span>
      </div>
      <div class="player">
        <button class="btn" id="play" type="button" aria-label="Play">▶</button>
        <input type="range" id="time" min="0" max="1" value="1" aria-label="Time along voyage" />
      </div>
      <p class="status" id="clock" aria-live="polite"></p>
      <div class="explain small" style="margin:0.6rem 0">
        <strong>Try the neural network.</strong> Pause anywhere with at least an hour of history,
        then run the GRU forecaster <em>in your browser</em> (ONNX Runtime, ~4 MB, loaded on demand).
        <div style="margin-top:0.5rem"><button class="btn primary" id="live" type="button" disabled>Predict from here</button></div>
        <p class="status" id="live-status" aria-live="polite" style="margin:0.4rem 0 0"></p>
      </div>
      <dl class="stat-grid" id="stats"></dl>
    </aside>
    <div class="map" id="map" role="region" aria-label="Map of the selected vessel track"></div>
  </div>`;

  const select = $(root, "#voyage") as HTMLSelectElement;
  const want = query().get("voyage");
  if (want && data.voyages.some((v) => v.id === want)) select.value = want;
  const map = await createMap($(root, "#map"), data.region, data.attribution);

  setData(map, "unc", fc([]));
  setData(map, "smooth", fc([]));
  setData(map, "raw", fc([]));
  setData(map, "trail", fc([]));
  setData(map, "now", fc([]));
  // Circle radius in metres: pixels = metres / (156543.03 * cos(lat) / 2^zoom).
  const cosLat = Math.cos((((data.region.lat_min + data.region.lat_max) / 2) * Math.PI) / 180);
  const k0 = 1 / (156543.03 * cosLat);
  map.addLayer({
    id: "unc",
    type: "circle",
    source: "unc",
    paint: {
      "circle-radius": ["interpolate", ["exponential", 2], ["zoom"], 0, ["*", ["get", "r"], k0], 22, ["*", ["get", "r"], k0 * 2 ** 22]],
      "circle-color": "#0b63ce",
      "circle-opacity": 0.12,
    },
  });
  map.addLayer({
    id: "smooth",
    type: "line",
    source: "smooth",
    layout: { "line-cap": "round", "line-join": "round" },
    paint: {
      "line-width": 3,
      "line-color": ["match", ["get", "mode"], 0, MODE_COLOR[0], 1, MODE_COLOR[1], MODE_COLOR[2]],
    },
  });
  map.addLayer({
    id: "raw",
    type: "circle",
    source: "raw",
    paint: {
      "circle-radius": ["case", ["get", "rej"], 5, 2.5],
      "circle-color": ["case", ["get", "rej"], "#c62828", "#5b6878"],
      "circle-stroke-width": ["case", ["get", "rej"], 1, 0],
      "circle-stroke-color": "#ffffff",
    },
  });
  for (const id of ["live-ell", "live-path", "live-pts", "live-dr", "live-truth"]) setData(map, id, fc([]));
  map.addLayer({ id: "live-ell", type: "fill", source: "live-ell", paint: { "fill-color": modelColor("gru"), "fill-opacity": 0.12 } });
  map.addLayer({ id: "live-ell-line", type: "line", source: "live-ell", paint: { "line-color": modelColor("gru"), "line-width": 1.4 } });
  map.addLayer({ id: "live-truth", type: "line", source: "live-truth", paint: { "line-color": "#16202c", "line-width": 3, "line-dasharray": [2, 1.5] } });
  map.addLayer({ id: "live-dr", type: "line", source: "live-dr", paint: { "line-color": modelColor("dead_reckoning"), "line-width": 2.5 } });
  map.addLayer({ id: "live-path", type: "line", source: "live-path", paint: { "line-color": modelColor("gru"), "line-width": 3 } });
  map.addLayer({
    id: "live-pts",
    type: "circle",
    source: "live-pts",
    paint: { "circle-radius": 5, "circle-color": modelColor("gru"), "circle-stroke-color": "#fff", "circle-stroke-width": 1.5 },
  });
  map.addLayer({ id: "trail", type: "line", source: "trail", paint: { "line-width": 5, "line-color": "#16202c", "line-opacity": 0.35 } });
  map.addLayer({
    id: "now",
    type: "circle",
    source: "now",
    paint: { "circle-radius": 7, "circle-color": "#ffb000", "circle-stroke-color": "#16202c", "circle-stroke-width": 2 },
  });

  let v: Voyage = data.voyages[0];
  let ts: number[] = [];
  const slider = $(root, "#time") as HTMLInputElement;
  const clock = $(root, "#clock");

  const liveBtn = $(root, "#live") as HTMLButtonElement;
  const liveStatus = $(root, "#live-status");
  const clearLive = (): void => {
    for (const id of ["live-ell", "live-path", "live-pts", "live-dr", "live-truth"]) setData(map, id, fc([]));
  };
  const showTime = (i: number): void => {
    const enough = ts.length > 0 && ts[i] - ts[0] >= 3600;
    liveBtn.disabled = !enough;
    liveBtn.title = enough ? "" : "Needs at least 60 minutes of history before this moment";
    setData(map, "now", fc([pointFeature(v.s_lon[i], v.s_lat[i])]));
    setData(map, "trail", fc([lineFeature(v.s_lon.slice(0, i + 1), v.s_lat.slice(0, i + 1))]));
    const sog = v.sog_kn[i];
    clock.textContent = `${fmtTime(ts[i])} · ${sog == null ? "SOG n/a" : `${sog.toFixed(1)} kn`} · ${MODE_NAME[v.mode[i]]} · ±${(2 * v.s_sd_m[i]).toFixed(0)} m (95%)`;
  };

  const show = (id: string): void => {
    v = data.voyages.find((x) => x.id === id) ?? data.voyages[0];
    ts = times(v);
    const rej = new Set(v.rejected);
    setData(map, "raw", fc(v.lon.map((x, i) => pointFeature(x, v.lat[i], { rej: rej.has(i) }))));
    setData(map, "smooth", fc(modeSegments(v)));
    setData(map, "unc", fc(v.s_lon.map((x, i) => pointFeature(x, v.s_lat[i], { r: 2 * v.s_sd_m[i] }))));
    slider.max = String(v.lat.length - 1);
    slider.value = slider.max;
    showTime(v.lat.length - 1);
    map.fitBounds(bounds(v.lon, v.lat), { padding: 50, duration: 500 });
    const hours = (ts[ts.length - 1] - ts[0]) / 3600;
    const counts = [0, 0, 0];
    for (const m of v.mode) counts[m]++;
    $(root, "#stats").innerHTML = `
      <dt>Vessel</dt><dd>${esc(vesselLabel(v.name, v.mmsi))}</dd>
      <dt>Type</dt><dd>${esc(GROUP_LABEL[v.group] ?? v.group)}${v.length_m ? `, ${v.length_m} m` : ""}</dd>
      <dt>Start</dt><dd>${fmtTime(v.t0)}</dd>
      <dt>Duration</dt><dd>${hours.toFixed(1)} h</dd>
      <dt>AIS reports</dt><dd>${v.lat.length.toLocaleString("en")}</dd>
      <dt>Rejected outliers</dt><dd>${v.rejected.length}</dd>
      <dt>Turning</dt><dd>${((100 * counts[2]) / v.mode.length).toFixed(0)}% of fixes</dd>
      <dt>Median ±95%</dt><dd>${(2 * [...v.s_sd_m].sort((a, b) => a - b)[Math.floor(v.s_sd_m.length / 2)]).toFixed(0)} m</dd>`;
    setQuery({ voyage: v.id });
    map.getContainer().setAttribute("aria-label", `Map of ${vesselLabel(v.name, v.mmsi)}: ${v.lat.length} AIS reports over ${hours.toFixed(1)} hours.`);
  };

  liveBtn.addEventListener("click", async () => {
    const i = Number(slider.value);
    liveBtn.disabled = true;
    liveStatus.textContent = "Loading the model…";
    try {
      const { predict, loadModel } = await import("../ml/infer");
      const { meta } = await loadModel();
      const t0 = performance.now();
      const anchor = {
        t0: ts[i], lat0: v.lat[i], lon0: v.lon[i], sog0: v.sog_kn[i], cog0: v.cog_deg[i],
        length_m: v.length_m, group: v.group,
      };
      const raw = { t: ts.slice(0, i + 1), lat: v.lat.slice(0, i + 1), lon: v.lon.slice(0, i + 1), sog: v.sog_kn.slice(0, i + 1), cog: v.cog_deg.slice(0, i + 1) };
      const p = await predict(raw, anchor);
      const ms = performance.now() - t0;
      const H = meta.horizons_min;
      // Truth: the vessel's actual positions over the next two hours (interpolated in time).
      const truthAt = (tq: number): [number, number] | null => {
        if (tq > ts[ts.length - 1]) return null;
        let j = i;
        while (j < ts.length - 1 && ts[j + 1] < tq) j++;
        const w = (tq - ts[j]) / Math.max(ts[j + 1] - ts[j], 1);
        return [v.lon[j] + w * (v.lon[j + 1] - v.lon[j]), v.lat[j] + w * (v.lat[j + 1] - v.lat[j])];
      };
      const fut = ts.map((t, k) => [t, k] as const).filter(([t]) => t > ts[i] && t <= ts[i] + 7200).map(([, k]) => k);
      setData(map, "live-truth", fc([lineFeature([v.lon[i], ...fut.map((k) => v.lon[k])], [v.lat[i], ...fut.map((k) => v.lat[k])])]));
      setData(map, "live-path", fc([lineFeature([v.lon[i], ...p.lon], [v.lat[i], ...p.lat])]));
      setData(map, "live-pts", fc(p.lon.map((x, k) => pointFeature(x, p.lat[k]))));
      setData(map, "live-ell", fc(p.lat.map((la, k) => polygonFeature(ellipse(la, p.lon[k], p.cov[k][0], p.cov[k][1], p.cov[k][2], 0.9)))));
      const { deadReckoning, enuToLatlon } = await import("../ml/features");
      const dr = deadReckoning(anchor, H).map(([e, n]) => enuToLatlon(e * 1000, n * 1000, anchor.lat0, anchor.lon0));
      setData(map, "live-dr", fc([lineFeature([v.lon[i], ...dr.map((d) => d[1])], [v.lat[i], ...dr.map((d) => d[0])])]));
      const errs = H.map((h, k) => {
        const tr = truthAt(ts[i] + h * 60);
        if (!tr) return `${h} min: –`;
        const dLat = ((p.lat[k] - tr[1]) * Math.PI) / 180;
        const dLon = ((p.lon[k] - tr[0]) * Math.PI) / 180;
        const a = Math.sin(dLat / 2) ** 2 + Math.cos((tr[1] * Math.PI) / 180) * Math.cos((p.lat[k] * Math.PI) / 180) * Math.sin(dLon / 2) ** 2;
        return `${h} min: ${((2 * 6371.0088 * Math.asin(Math.sqrt(a)))).toFixed(2)} km`;
      });
      liveStatus.innerHTML = `${esc(meta.model)} ran in your browser in ${ms.toFixed(0)} ms. Error vs what actually happened: ${esc(errs.join(" · "))}.
        <br/><span style="color:${modelColor("gru")}">■</span> GRU + 90% ellipses · <span style="color:${modelColor("dead_reckoning")}">■</span> dead reckoning · dashed: actual path.`;
    } catch (err) {
      liveStatus.textContent = `Could not run the model: ${String(err)}`;
    } finally {
      liveBtn.disabled = false;
    }
  });
  slider.addEventListener("input", clearLive);
  select.addEventListener("change", () => (clearLive(), (liveStatus.textContent = "")));
  select.addEventListener("change", () => show(select.value));
  slider.addEventListener("input", () => showTime(Number(slider.value)));
  for (const [box, layers] of [
    ["#l-raw", ["raw"]],
    ["#l-smooth", ["smooth"]],
    ["#l-unc", ["unc"]],
  ] as const) {
    $(root, box).addEventListener("change", (e) => setVisible(map, [...layers], (e.target as HTMLInputElement).checked));
  }

  let timer: number | null = null;
  const playBtn = $(root, "#play") as HTMLButtonElement;
  const stop = (): void => {
    if (timer != null) window.clearInterval(timer);
    timer = null;
    playBtn.textContent = "▶";
    playBtn.setAttribute("aria-label", "Play");
  };
  playBtn.addEventListener("click", () => {
    if (timer != null) return stop();
    if (Number(slider.value) >= Number(slider.max)) slider.value = "0";
    playBtn.textContent = "❚❚";
    playBtn.setAttribute("aria-label", "Pause");
    const step = Math.max(1, Math.round(v.lat.length / 300));
    timer = window.setInterval(() => {
      const i = Math.min(Number(slider.value) + step, Number(slider.max));
      slider.value = String(i);
      showTime(i);
      if (i >= Number(slider.max)) stop();
    }, 40);
  });
  window.addEventListener("hashchange", stop, { once: true });
  show(select.value);
}
