// MapLibre setup shared by the map views. Imported lazily so the landing page stays light.
import * as maplibregl from "maplibre-gl";
import "maplibre-gl/dist/maplibre-gl.css";
// MapLibre v6 parses tiles in a module Web Worker shipped as a separate file; let Vite bundle it.
import workerUrl from "maplibre-gl/dist/maplibre-gl-worker.mjs?worker&url";
import type { Region } from "./data";
import { prefersDark } from "./theme";

maplibregl.setWorkerUrl(workerUrl);

export { maplibregl };

const STYLES = {
  light: "https://tiles.openfreemap.org/styles/positron",
  dark: "https://tiles.openfreemap.org/styles/dark",
};


/** Create a map fitted to the study region, resolving once the style has loaded. */
export function createMap(container: HTMLElement, region: Region, attribution: string): Promise<maplibregl.Map> {
  const map = new maplibregl.Map({
    container,
    style: prefersDark() ? STYLES.dark : STYLES.light,
    bounds: [
      [region.lon_min, region.lat_min],
      [region.lon_max, region.lat_max],
    ],
    fitBoundsOptions: { padding: 20 },
    attributionControl: { compact: true, customAttribution: attribution },
    cooperativeGestures: false,
  });
  map.addControl(new maplibregl.NavigationControl({ showCompass: false }), "top-right");
  map.addControl(new maplibregl.ScaleControl({ unit: "metric" }), "bottom-left");
  container.setAttribute("tabindex", "0");
  return new Promise((resolve) => map.on("load", () => resolve(map)));
}

export function setData(map: maplibregl.Map, id: string, data: GeoJSON.GeoJSON): void {
  const src = map.getSource(id) as maplibregl.GeoJSONSource | undefined;
  if (src) src.setData(data);
  else map.addSource(id, { type: "geojson", data });
}

export const fc = (features: GeoJSON.Feature[]): GeoJSON.FeatureCollection => ({
  type: "FeatureCollection",
  features,
});

export const lineFeature = (lon: number[], lat: number[], props: Record<string, unknown> = {}): GeoJSON.Feature => ({
  type: "Feature",
  properties: props,
  geometry: { type: "LineString", coordinates: lon.map((x, i) => [x, lat[i]]) },
});

export const pointFeature = (lon: number, lat: number, props: Record<string, unknown> = {}): GeoJSON.Feature => ({
  type: "Feature",
  properties: props,
  geometry: { type: "Point", coordinates: [lon, lat] },
});

export const polygonFeature = (ring: [number, number][], props: Record<string, unknown> = {}): GeoJSON.Feature => ({
  type: "Feature",
  properties: props,
  geometry: { type: "Polygon", coordinates: [ring] },
});

/** Toggle a set of layers' visibility. */
export function setVisible(map: maplibregl.Map, layers: string[], on: boolean): void {
  for (const l of layers) if (map.getLayer(l)) map.setLayoutProperty(l, "visibility", on ? "visible" : "none");
}

/** A screen-reader-friendly summary that sits next to the (visual-only) map. */
export function describeMap(el: HTMLElement, text: string): void {
  el.setAttribute("aria-label", text);
}
