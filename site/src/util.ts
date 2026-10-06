// Tiny DOM helpers (no framework).

/** Escape text for safe interpolation into HTML. */
export function esc(s: unknown): string {
  return String(s ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

/** Query parameters of the current hash route (e.g. #/predict?sample=...). */
export function query(): URLSearchParams {
  const raw = location.hash.replace(/^#\/?/, "");
  return new URLSearchParams(raw.split("?")[1] ?? "");
}

/** Replace the query string of the current hash route without triggering a re-render. */
export function setQuery(params: Record<string, string>): void {
  const path = location.hash.replace(/^#\/?/, "").split("?")[0];
  const qs = new URLSearchParams(params).toString();
  history.replaceState(null, "", `#/${path}${qs ? `?${qs}` : ""}`);
}

export function $(root: ParentNode, sel: string): HTMLElement {
  const el = root.querySelector<HTMLElement>(sel);
  if (!el) throw new Error(`missing element ${sel}`);
  return el;
}

export function km(x: number | null | undefined, digits = 2): string {
  return x == null || !Number.isFinite(x) ? "–" : `${x.toFixed(digits)} km`;
}

export function pct(x: number | null | undefined, digits = 0): string {
  return x == null || !Number.isFinite(x) ? "–" : `${(100 * x).toFixed(digits)}%`;
}
