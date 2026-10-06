import { fmtTime, getData, GROUP_LABEL, type PredictionsFile, type PredSample, vesselLabel } from "../data";
import { bounds, ellipse } from "../geo";
import { modelColor } from "../models";
import { createMap, fc, lineFeature, pointFeature, polygonFeature, setData } from "../map";
import { $, esc, query, setQuery } from "../util";

export async function render(root: HTMLElement): Promise<void> {
  const data = await getData<PredictionsFile>("predictions.json");
  const H = data.horizons;
  const enabled = new Set(data.models.filter((m) => ["dead_reckoning", "knn_route", "gru"].includes(m.id)).map((m) => m.id));
  if (!enabled.size) data.models.slice(0, 3).forEach((m) => enabled.add(m.id));
  let horizonIdx = H.length - 1;

  root.innerHTML = `
  <div class="mapview">
    <aside class="panel" aria-label="Prediction controls">
      <h1>Predicted vs actual</h1>
      <p class="small muted">Each forecast starts at the anchor (white dot) using only data up to
      that moment. Ellipses are calibrated 90% regions: about 9 in 10 true positions should fall
      inside. Samples are from the <strong>${esc(data.split)}</strong> period.</p>
      <label for="sample">Vessel &amp; moment</label>
      <select id="sample">
        ${data.samples
          .map((s) => `<option value="${esc(s.id)}">${esc(s.name ?? s.mmsi)} · ${fmtTime(s.t0).slice(5, 16)}</option>`)
          .join("")}
      </select>
      <fieldset style="border:none;padding:0;margin:0.8rem 0 0">
        <legend style="font-weight:600;font-size:0.9rem">Models</legend>
        ${data.models
          .map(
            (m) => `<label class="check"><input type="checkbox" data-model="${esc(m.id)}" ${enabled.has(m.id) ? "checked" : ""}/>
              <i class="swatch" style="background:${modelColor(m.id)}"></i>${esc(m.label)}</label>`,
          )
          .join("")}
      </fieldset>
      <label for="horizon">Ellipse horizon</label>
      <select id="horizon">${H.map((h, i) => `<option value="${i}" ${i === horizonIdx ? "selected" : ""}>${h} min ahead</option>`).join("")}</select>
      <div class="legend" aria-hidden="true">
        <span><i class="swatch" style="background:var(--raw)"></i>last 60 min (history)</span>
        <span><i class="swatch" style="background:var(--truth)"></i>what actually happened</span>
      </div>
      <h2 style="font-size:1rem;margin-top:1rem">Error for this sample</h2>
      <div class="table-wrap"><table id="errors" aria-label="Forecast error by model and horizon"></table></div>
      <p class="small muted" style="margin-top:0.6rem">One sample is an anecdote, not a result:
      see <a href="#/results">Results</a> for errors over the whole test month with confidence intervals.</p>
    </aside>
    <div class="map" id="map" role="region" aria-label="Map comparing model forecasts with the actual path"></div>
  </div>`;

  const select = $(root, "#sample") as HTMLSelectElement;
  const want = query().get("sample");
  if (want && data.samples.some((s) => s.id === want)) select.value = want;
  const map = await createMap($(root, "#map"), data.region, data.attribution);
  for (const id of ["ell", "hist", "future", "pred", "predpts", "anchor"]) setData(map, id, fc([]));
  map.addLayer({ id: "ell", type: "fill", source: "ell", paint: { "fill-color": ["get", "color"], "fill-opacity": 0.07 } });
  map.addLayer({ id: "ell-line", type: "line", source: "ell", paint: { "line-color": ["get", "color"], "line-width": 1.6 } });
  map.addLayer({ id: "hist", type: "line", source: "hist", paint: { "line-color": "#5b6878", "line-width": 3 } });
  map.addLayer({ id: "future", type: "line", source: "future", paint: { "line-color": "#16202c", "line-width": 3, "line-dasharray": [2, 1.5] } });
  map.addLayer({ id: "pred", type: "line", source: "pred", paint: { "line-color": ["get", "color"], "line-width": 2.5 } });
  map.addLayer({
    id: "predpts",
    type: "circle",
    source: "predpts",
    paint: { "circle-radius": 4, "circle-color": ["get", "color"], "circle-stroke-color": "#fff", "circle-stroke-width": 1 },
  });
  map.addLayer({
    id: "anchor",
    type: "circle",
    source: "anchor",
    paint: { "circle-radius": 6, "circle-color": "#ffffff", "circle-stroke-color": "#16202c", "circle-stroke-width": 2 },
  });

  let s: PredSample = data.samples[0];
  const draw = (): void => {
    const ells: GeoJSON.Feature[] = [];
    const paths: GeoJSON.Feature[] = [];
    const pts: GeoJSON.Feature[] = [];
    for (const m of data.models) {
      const p = s.preds[m.id];
      if (!p || !enabled.has(m.id)) continue;
      paths.push(lineFeature([s.lon0, ...p.lon], [s.lat0, ...p.lat], { color: modelColor(m.id) }));
      p.lon.forEach((x, k) => pts.push(pointFeature(x, p.lat[k], { color: modelColor(m.id), h: H[k] })));
      const k = horizonIdx;
      ells.push(polygonFeature(ellipse(p.lat[k], p.lon[k], p.cee[k], p.cnn[k], p.cen[k], 0.9), { color: modelColor(m.id) }));
    }
    setData(map, "ell", fc(ells));
    setData(map, "pred", fc(paths));
    setData(map, "predpts", fc(pts));
  };

  const show = (id: string): void => {
    s = data.samples.find((x) => x.id === id) ?? data.samples[0];
    setData(map, "hist", fc([lineFeature(s.hist.lon, s.hist.lat)]));
    setData(map, "future", fc([lineFeature([s.lon0, ...s.future.lon], [s.lat0, ...s.future.lat])]));
    setData(map, "anchor", fc([pointFeature(s.lon0, s.lat0)]));
    draw();
    const lons = [...s.hist.lon, ...s.future.lon];
    const lats = [...s.hist.lat, ...s.future.lat];
    for (const m of data.models) {
      const p = s.preds[m.id];
      if (p && enabled.has(m.id)) (lons.push(...p.lon), lats.push(...p.lat));
    }
    map.fitBounds(bounds(lons, lats), { padding: 60, duration: 500 });
    const rows = data.models
      .map((m) => {
        const e = s.errors_km[m.id] ?? [];
        return `<tr><td><i class="swatch" style="background:${modelColor(m.id)}"></i> ${esc(m.short)}</td>${H.map((_, k) => `<td class="num">${e[k] == null ? "–" : e[k]!.toFixed(2)}</td>`).join("")}</tr>`;
      })
      .join("");
    $(root, "#errors").innerHTML = `<thead><tr><th>Model</th>${H.map((h) => `<th class="num">${h} min</th>`).join("")}</tr></thead><tbody>${rows}</tbody><caption class="visually-hidden">Error in km</caption>`;
    setQuery({ sample: s.id });
    map.getContainer().setAttribute(
      "aria-label",
      `Forecasts for ${vesselLabel(s.name, s.mmsi)} (${GROUP_LABEL[s.group] ?? s.group}) from ${fmtTime(s.t0)}.`,
    );
  };

  select.addEventListener("change", () => show(select.value));
  ($(root, "#horizon") as HTMLSelectElement).addEventListener("change", (e) => {
    horizonIdx = Number((e.target as HTMLSelectElement).value);
    draw();
  });
  root.querySelectorAll<HTMLInputElement>("input[data-model]").forEach((box) =>
    box.addEventListener("change", () => {
      if (box.checked) enabled.add(box.dataset.model!);
      else enabled.delete(box.dataset.model!);
      draw();
    }),
  );
  show(select.value);
}
