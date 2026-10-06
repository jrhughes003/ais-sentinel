import { expect, test } from "@playwright/test";

test("map page loads tracks and anomalies", async ({ page }, info) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await page.goto("./");
  await expect(page.locator("#voyage option").first()).toBeAttached();
  await expect(page.locator("#status")).toContainText("fixes");
  await expect(page.locator(".maplibregl-canvas")).toBeVisible();
  await page.waitForTimeout(2500);
  await page.screenshot({ path: `test-results/smoke-${info.project.name}.png` });
  expect(errors).toEqual([]);
});
