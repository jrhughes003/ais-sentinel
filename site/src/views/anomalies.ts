import { ANOMALY_LABEL, type AnomaliesFile, type AnomalyEvent, fmtDuration, fmtTime, getData, GROUP_LABEL, vesselLabel } from "../data";
import { bounds } from "../geo";
import { createMap, fc, lineFeature, pointFeature, setData } from "../map";
import { $, esc, query, setQuery } from "../util";

const TYPE_COLOR: Record<AnomalyEvent["type"], string> = {
  gap: "#9a6700",
  jump: "#c62828",
  loiter: "#6f42c1",
  deviation: "#0b7285",
  rendezvous: "#d6336c",
};

export async function render(root: HTMLElement): Promise<void> {
  const data = await getData<AnomaliesFile>("anomalies.json");
  const types = Object.keys(ANOMALY_LABEL) as AnomalyEvent["type"][];
  root.innerHTML = `
  <div class="mapview">
    <aside class="panel" aria-label="Anomaly list">
      <h1>Flagged behaviour</h1>
      <p class="small muted">Each flag comes with the measured values that triggered it. A flag is
      a prompt for an analyst, not an accusation: coverage holes, GPS faults and shared
      transponder IDs explain many of them.</p>
      <label for="type">Type</label>
      <select id="type">
        <option value="">All types (${data.events.length})</option>
        ${types.map((t) => `<option value="${t}">${esc(ANOMALY_LABEL[t])} (${data.counts[t] ?? 0} in season)</option>`).join("")}
      </select>
      <div class="legend" aria-hidden="true">
        ${types.map((t) => `<span><i class="swatch dot" style="background:${TYPE_COLOR[t]}"></i>${esc(ANOMALY_LABEL[t])}</span>`).join("")}
      </div>
      <div id="detail" class="explain" aria-live="polite" hidden></div>
      <ul class="event-list" id="list"></ul>
    </aside>
    <div class="map" id="map" role="region" aria-label="Map of flagged events"></div>
  </div>`;

  const map = await createMap($(root, "#map"), data.region, data.attribution);
  for (const id of ["events", "ctx", "ctx2", "ends"]) setData(map, id, fc([]));
  map.addLayer({ id: "ctx", type: "line", source: "ctx", paint: { "line-color": "#0b63ce", "line-width": 3 } });
  map.addLayer({ id: "ctx2", type: "line", source: "ctx2", paint: { "line-color": "#d6336c", "line-width": 3, "line-dasharray": [2, 1] } });
  map.addLayer({
    id: "events",
    type: "circle",
    source: "events",
    paint: {
      "circle-radius": ["case", ["get", "sel"], 9, 5],
      "circle-color": ["get", "color"],
      "circle-stroke-color": "#ffffff",
      "circle-stroke-width": ["case", ["get", "sel"], 3, 1],
    },
  });
  map.addLayer({
    id: "ends",
    type: "circle",
    source: "ends",
    paint: { "circle-radius": 5, "circle-color": "#ffffff", "circle-stroke-color": "#16202c", "circle-stroke-width": 2 },
  });

  const list = $(root, "#list");
  const typeSel = $(root, "#type") as HTMLSelectElement;
  let selected: string | null = query().get("event");

  const visible = (): AnomalyEvent[] =>
    data.events.filter((e) => !typeSel.value || e.type === typeSel.value).sort((a, b) => Number(!!b.case_study) - Number(!!a.case_study) || b.score - a.score);

  const paint = (): void => {
    const ev = visible();
    setData(
      map,
      "events",
      fc(ev.map((e) => pointFeature(e.lon, e.lat, { id: e.id, color: TYPE_COLOR[e.type], sel: e.id === selected }))),
    );
    list.innerHTML = ev
      .map(
        (e) => `<li><button type="button" data-id="${esc(e.id)}" aria-pressed="${e.id === selected}">
          <i class="swatch dot" style="background:${TYPE_COLOR[e.type]}"></i>
          <strong>${esc(ANOMALY_LABEL[e.type])}</strong>${e.case_study ? ' <span class="pill">case study</span>' : ""}<br/>
          ${esc(vesselLabel(e.name, e.mmsi))}<br/><span class="t">${fmtTime(e.t_start)} · ${fmtDuration(e.duration_min)}</span>
        </button></li>`,
      )
      .join("");
  };

  const select = (id: string): void => {
    const e = data.events.find((x) => x.id === id);
    if (!e) return;
    selected = id;
    setQuery({ event: id });
    const d = $(root, "#detail");
    d.hidden = false;
    d.innerHTML = `<strong>${esc(ANOMALY_LABEL[e.type])}</strong> · ${esc(vesselLabel(e.name, e.mmsi))}
      ${e.mmsi2 ? ` with MMSI ${e.mmsi2}` : ""}<br/>
      <span class="small muted">${esc(GROUP_LABEL[e.group] ?? e.group)} · ${fmtTime(e.t_start)} → ${fmtTime(e.t_end)}</span>
      <p style="margin:0.5rem 0 0">${esc(e.explanation)}</p>
      ${e.case_study ? `<p class="small" style="margin:0.5rem 0 0"><strong>Case study:</strong> ${esc(e.case_study)}</p>` : ""}`;
    setData(map, "ctx", fc([lineFeature(e.track.lon, e.track.lat)]));
    setData(map, "ctx2", fc(e.track2 ? [lineFeature(e.track2.lon, e.track2.lat)] : []));
    setData(map, "ends", fc([pointFeature(e.lon, e.lat), pointFeature(e.lon_end, e.lat_end)]));
    const lons = [...e.track.lon, ...(e.track2?.lon ?? []), e.lon, e.lon_end];
    const lats = [...e.track.lat, ...(e.track2?.lat ?? []), e.lat, e.lat_end];
    map.fitBounds(bounds(lons, lats, 0.01), { padding: 60, duration: 500, maxZoom: 14 });
    paint();
  };

  list.addEventListener("click", (ev) => {
    const btn = (ev.target as HTMLElement).closest<HTMLButtonElement>("button[data-id]");
    if (btn) select(btn.dataset.id!);
  });
  map.on("click", "events", (ev) => {
    const id = ev.features?.[0]?.properties?.id as string | undefined;
    if (id) select(id);
  });
  map.on("mouseenter", "events", () => (map.getCanvas().style.cursor = "pointer"));
  map.on("mouseleave", "events", () => (map.getCanvas().style.cursor = ""));
  typeSel.addEventListener("change", paint);
  paint();
  const first = selected && data.events.some((e) => e.id === selected) ? selected : visible()[0]?.id;
  if (first) select(first);
}
