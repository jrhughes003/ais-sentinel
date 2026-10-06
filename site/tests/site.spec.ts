import AxeBuilder from "@axe-core/playwright";
import { expect, type Page, test } from "@playwright/test";

const VIEWS = [
  { hash: "#/", ready: "h1", text: "Where is that ship going" },
  { hash: "#/tracks", ready: "#stats dd", text: "Vessel tracks" },
  { hash: "#/predict", ready: "#errors tbody tr", text: "Predicted vs actual" },
  { hash: "#/anomalies", ready: "#list li", text: "Flagged behaviour" },
  { hash: "#/results", ready: "#criteria", text: "Results & methods" },
];

function collectErrors(page: Page): string[] {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  page.on("console", (m) => {
    // Basemap tiles are third-party; ignore network noise from them, keep our own errors.
    if (m.type() === "error" && !/tiles\.openfreemap|Failed to load resource/.test(m.text())) errors.push(m.text());
  });
  return errors;
}

for (const v of VIEWS) {
  test(`view ${v.hash} renders without errors`, async ({ page }, info) => {
    const errors = collectErrors(page);
    await page.goto(`./${v.hash}`);
    await expect(page.locator(v.ready).first()).toBeAttached({ timeout: 30_000 });
    await expect(page.locator("main")).toContainText(v.text);
    await page.waitForTimeout(1500);
    await page.screenshot({ path: `test-results/${info.project.name}-${v.hash.replace(/[#/]/g, "") || "home"}.png`, fullPage: false });
    expect(errors).toEqual([]);
  });

  test(`view ${v.hash} has no serious accessibility violations`, async ({ page }) => {
    await page.goto(`./${v.hash}`);
    await expect(page.locator(v.ready).first()).toBeAttached({ timeout: 30_000 });
    const results = await new AxeBuilder({ page })
      .withTags(["wcag2a", "wcag2aa"])
      .exclude(".maplibregl-canvas") // the WebGL canvas is described by its container's label
      .exclude(".maplibregl-ctrl-attrib") // third-party attribution control
      .analyze();
    const bad = results.violations.filter((x) => x.impact === "serious" || x.impact === "critical");
    expect(bad.map((x) => `${x.id}: ${x.nodes.map((n) => n.target.join(" ")).slice(0, 3).join(", ")}`)).toEqual([]);
  });
}

test("keyboard navigation reaches every page", async ({ page }, info) => {
  test.skip(info.project.name === "mobile", "menu is collapsed on mobile; covered by the menu test");
  await page.goto("./#/");
  for (const name of ["Tracks", "Predictions", "Anomalies", "Results & methods", "Overview"]) {
    await page.getByRole("link", { name, exact: true }).focus();
    await page.keyboard.press("Enter");
    await expect(page.getByRole("link", { name, exact: true })).toHaveAttribute("aria-current", "page");
  }
});

test("mobile menu opens and navigates", async ({ page }, info) => {
  test.skip(info.project.name !== "mobile", "mobile only");
  await page.goto("./#/");
  await page.getByRole("button", { name: "Menu" }).click();
  await page.getByRole("link", { name: "Anomalies" }).click();
  await expect(page.locator("main")).toContainText("Flagged behaviour");
});

test("initial page weight is small", async ({ page }) => {
  let bytes = 0;
  page.on("response", async (r) => {
    if (!r.url().startsWith("http://localhost")) return; // count our own assets only
    const len = Number(r.headers()["content-length"] ?? 0);
    bytes += len || (await r.body().catch(() => Buffer.alloc(0))).length;
  });
  await page.goto("./#/");
  await expect(page.locator("h1")).toBeVisible();
  await page.waitForLoadState("networkidle");
  expect(bytes).toBeLessThan(2 * 1024 * 1024);
});

test("in-browser model matches Python (features and predictions)", async ({ page }, info) => {
  test.skip(info.project.name === "mobile", "same code path as desktop");
  await page.goto("./#/parity");
  const pre = page.locator("#parity");
  await expect(pre).not.toHaveText("running…", { timeout: 60_000 });
  const r = JSON.parse((await pre.textContent())!);
  expect(r.n).toBeGreaterThan(0);
  expect(r.maxSeq).toBeLessThan(1e-3);
  expect(r.maxCtx).toBeLessThan(1e-4);
  expect(r.maxPosM).toBeLessThan(5); // metres
  expect(r.maxCovRel).toBeLessThan(1e-3);
});

test("live prediction runs from the tracks view", async ({ page }, info) => {
  test.skip(info.project.name === "mobile", "desktop is enough for the wiring check");
  await page.goto("./#/tracks");
  const btn = page.locator("#live");
  await expect(btn).toBeEnabled({ timeout: 30_000 });
  await btn.click();
  await expect(page.locator("#live-status")).toContainText("ran in your browser", { timeout: 60_000 });
});

test("fast navigation never shows a stale view", async ({ page }, info) => {
  test.skip(info.project.name === "mobile", "logic is viewport-independent");
  await page.goto("./#/tracks");
  await page.evaluate(() => (location.hash = "#/results")); // leave before tracks finishes
  await expect(page.locator("main")).toContainText("Results & methods", { timeout: 30_000 });
  await page.waitForTimeout(4000); // give the abandoned tracks view time to finish
  await expect(page.locator("main")).not.toContainText("Vessel tracks");
});
