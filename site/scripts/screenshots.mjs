// Capture README screenshots from a running preview server (npm run preview).
// Usage: node scripts/screenshots.mjs [baseUrl]   (writes ../docs/img/*.png)
import { chromium } from "@playwright/test";

const base = process.argv[2] ?? "http://localhost:4173/";
const shots = [
  { name: "overview", hash: "#/", ready: "h1" },
  { name: "tracks-live", hash: "#/tracks", ready: "#stats dd", live: true },
  { name: "predict", hash: "#/predict", ready: "#errors tbody tr" },
  { name: "anomalies", hash: "#/anomalies", ready: "#list li" },
  { name: "results", hash: "#/results", ready: "#chart-error svg" },
];
const browser = await chromium.launch();
for (const theme of ["light", "dark"]) {
  const page = await browser.newPage({ viewport: { width: 1280, height: 800 }, colorScheme: theme });
  for (const s of shots) {
    if (theme === "dark" && s.name !== "tracks-live") continue;
    await page.goto(base + s.hash);
    await page.waitForSelector(s.ready, { timeout: 60_000 });
    if (s.live) {
      await page.evaluate(() => {
        const r = document.querySelector("#time");
        r.value = String(Math.floor(Number(r.max) * 0.35));
        r.dispatchEvent(new Event("input"));
      });
      await page.click("#live");
      await page.waitForFunction(() => document.querySelector("#live-status")?.textContent?.includes("ran in"), null, { timeout: 60_000 });
    }
    await page.waitForTimeout(2500);
    await page.screenshot({ path: `../docs/img/${s.name}${theme === "dark" ? "-dark" : ""}.png` });
    console.log("saved", s.name, theme);
  }
  await page.close();
}
await browser.close();
