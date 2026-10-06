import * as maplibregl from "maplibre-gl";
import "maplibre-gl/dist/maplibre-gl.css";
// MapLibre v6 parses tiles in a module Web Worker shipped as a separate file. Let Vite bundle
// it (with its shared chunk) and tell MapLibre where it ends up.
import workerUrl from "maplibre-gl/dist/maplibre-gl-worker.mjs?worker&url";
import "./style.css";

maplibregl.setWorkerUrl(workerUrl);

interface Voyage {
  id: string;
  mmsi: number;
  name: string | null;
  group: string;
  t0: number;
  dt: number[];
  lat: number[];
  lon: number[];
  s_lat: number[];
  s_lon: number[];
  s_sd_m: number[];
  rejected: number[];
}
interface GapEvent {
  type: string;
  mmsi: number;
  vessel_group: string;
  t_start: number;
  t_end: number;
  duration_min: number;
  lat: number;
  lon: number;
  lat_end: number;
  lon_end: number;
  explanation: string;
}

const BASE = import.meta.env.BASE_URL;
const STYLE = "https://tiles.openfreemap.org/styles/positron";

async function getJSON<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}data/${path}`);
  if (!res.ok) throw new Error(`${path}: HTTP ${res.status}`);
  return (await res.json()) as T;
}

function line(lon: number[], lat: number[]): GeoJSON.Feature {
  return {
    type: "Feature",
    properties: {},
    geometry: { type: "LineString", coordinates: lon.map((x, i) => [x, lat[i]]) },
  };
}

function points(v: Voyage): GeoJSON.FeatureCollection {
  const rejected = new Set(v.rejected);
  return {
    type: "FeatureCollection",
    features: v.lon.map((x, i) => ({
      type: "Feature",
      properties: { rejected: rejected.has(i) },
      geometry: { type: "Point", coordinates: [x, v.lat[i]] },
    })),
  };
}

async function init(): Promise<void> {
  const app = document.querySelector<HTMLDivElement>("#app")!;
  app.innerHTML = `
    <header class="bar">
      <h1>ais-sentinel</h1>
      <label>Vessel <select id="voyage"></select></label>
      <label><input type="checkbox" id="raw" checked /> Raw AIS fixes</label>
      <label><input type="checkbox" id="smooth" checked /> Kalman-smoothed track</label>
      <span class="legend"><i class="dot raw"></i>raw <i class="dot rej"></i>rejected outlier <i class="dot gap"></i>AIS gap</span>
    </header>
    <div id="map" role="region" aria-label="Map of vessel tracks"></div>
    <p id="status" class="status" aria-live="polite">Loading…</p>`;

  const [tracks, anomalies] = await Promise.all([
    getJSON<{ voyages: Voyage[]; attribution: string }>("tracks.json"),
    getJSON<{ events: GapEvent[] }>("anomalies.json"),
  ]);
  const select = document.querySelector<HTMLSelectElement>("#voyage")!;
  for (const v of tracks.voyages) {
    const opt = document.createElement("option");
    opt.value = v.id;
    const day = new Date(v.t0 * 1000).toISOString().slice(0, 10);
    opt.textContent = `${v.name ?? v.mmsi} (${v.group}, ${day})`;
    select.append(opt);
  }

  const map = new maplibregl.Map({
    container: "map",
    style: STYLE,
    bounds: [
      [-83.6, 41.4],
      [-82.3, 43.1],
    ],
    attributionControl: { customAttribution: tracks.attribution },
  });
  map.addControl(new maplibregl.NavigationControl(), "top-right");

  map.on("load", () => {
    map.addSource("smooth", { type: "geojson", data: line([], []) });
    map.addSource("raw", { type: "geojson", data: { type: "FeatureCollection", features: [] } });
    map.addLayer({
      id: "smooth",
      type: "line",
      source: "smooth",
      paint: { "line-color": "#1f6feb", "line-width": 3 },
    });
    map.addLayer({
      id: "raw",
      type: "circle",
      source: "raw",
      paint: {
        "circle-radius": ["case", ["get", "rejected"], 5, 2.5],
        "circle-color": ["case", ["get", "rejected"], "#d1242f", "#57606a"],
      },
    });
    for (const g of anomalies.events) {
      const el = document.createElement("button");
      el.className = "gap-marker";
      el.setAttribute("aria-label", `AIS gap, ${Math.round(g.duration_min)} minutes`);
      new maplibregl.Marker({ element: el })
        .setLngLat([g.lon, g.lat])
        .setPopup(new maplibregl.Popup().setText(`MMSI ${g.mmsi}: ${g.explanation}`))
        .addTo(map);
    }
    const show = (id: string): void => {
      const v = tracks.voyages.find((x) => x.id === id)!;
      (map.getSource("smooth") as maplibregl.GeoJSONSource).setData(line(v.s_lon, v.s_lat));
      (map.getSource("raw") as maplibregl.GeoJSONSource).setData(points(v));
      const lons = v.lon, lats = v.lat;
      map.fitBounds(
        [
          [Math.min(...lons), Math.min(...lats)],
          [Math.max(...lons), Math.max(...lats)],
        ],
        { padding: 40, duration: 600 },
      );
      document.querySelector("#status")!.textContent =
        `${v.lat.length} fixes, ${v.rejected.length} rejected by the filter's outlier gate.`;
    };
    select.addEventListener("change", () => show(select.value));
    const toggle = (box: string, layer: string): void => {
      document.querySelector<HTMLInputElement>(box)!.addEventListener("change", (e) => {
        const on = (e.target as HTMLInputElement).checked;
        map.setLayoutProperty(layer, "visibility", on ? "visible" : "none");
      });
    };
    toggle("#raw", "raw");
    toggle("#smooth", "smooth");
    show(select.value);
  });
}

init().catch((err: unknown) => {
  document.querySelector("#app")!.textContent = `Failed to load: ${String(err)}`;
});
