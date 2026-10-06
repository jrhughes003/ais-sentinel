// App shell: hash router, navigation state, theme toggle. Views are loaded on demand so the
// landing page does not pay for the map or chart libraries.
import "./style.css";

type View = (root: HTMLElement) => Promise<void> | void;

const routes: Record<string, { title: string; load: () => Promise<{ render: View }> }> = {
  "": { title: "Overview", load: () => import("./views/overview") },
  tracks: { title: "Tracks", load: () => import("./views/tracks") },
  predict: { title: "Predictions", load: () => import("./views/predict") },
  anomalies: { title: "Anomalies", load: () => import("./views/anomalies") },
  results: { title: "Results & methods", load: () => import("./views/results") },
  parity: { title: "Model parity check", load: () => import("./views/parity") }, // unlisted (tests)
};

const main = document.querySelector<HTMLElement>("#main")!;
const nav = document.querySelector<HTMLElement>("#nav")!;
const toggle = document.querySelector<HTMLButtonElement>(".nav-toggle")!;
let current: string | null = null;

function routeKey(): { key: string; query: URLSearchParams } {
  const raw = location.hash.replace(/^#\/?/, "");
  const [path, qs] = raw.split("?");
  const key = path in routes ? path : "";
  return { key, query: new URLSearchParams(qs ?? "") };
}

async function render(): Promise<void> {
  const { key } = routeKey();
  if (key === current && main.childElementCount) return; // query-only change: view handles it
  current = key;
  const route = routes[key];
  document.title = `${route.title} · ais-sentinel`;
  for (const a of nav.querySelectorAll<HTMLAnchorElement>("a")) {
    const k = a.getAttribute("href")!.replace(/^#\/?/, "");
    if (k === key) a.setAttribute("aria-current", "page");
    else a.removeAttribute("aria-current");
  }
  nav.classList.remove("open");
  toggle.setAttribute("aria-expanded", "false");
  main.innerHTML = `<div class="page"><p class="status" role="status">Loading…</p></div>`;
  try {
    const mod = await route.load();
    main.innerHTML = "";
    await mod.render(main);
    // Tables scroll horizontally on small screens: keyboard users must be able to scroll them.
    for (const w of main.querySelectorAll<HTMLElement>(".table-wrap")) {
      w.tabIndex = 0;
      w.setAttribute("role", "region");
      w.setAttribute("aria-label", "Data table");
    }
  } catch (err) {
    console.error(err);
    main.innerHTML = `<div class="page"><div class="error-box" role="alert">Could not load this view: ${String(err)}</div></div>`;
  }
  main.focus({ preventScroll: true });
  window.scrollTo(0, 0);
}

toggle.addEventListener("click", () => {
  const open = nav.classList.toggle("open");
  toggle.setAttribute("aria-expanded", String(open));
});

const themeBtn = document.querySelector<HTMLButtonElement>(".theme-toggle")!;
try {
  const saved = localStorage.getItem("theme");
  if (saved) document.documentElement.dataset.theme = saved;
} catch {
  /* storage unavailable: follow the OS setting */
}
themeBtn.addEventListener("click", () => {
  const dark =
    document.documentElement.dataset.theme === "dark" ||
    (!document.documentElement.dataset.theme && matchMedia("(prefers-color-scheme: dark)").matches);
  const next = dark ? "light" : "dark";
  document.documentElement.dataset.theme = next;
  try {
    localStorage.setItem("theme", next);
  } catch {
    /* ignore */
  }
  current = null; // re-render so maps pick the matching basemap
  void render();
});

window.addEventListener("hashchange", () => void render());
void render();
