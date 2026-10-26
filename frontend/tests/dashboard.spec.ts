import { expect, test } from "@playwright/test";

test.describe.configure({ mode: "serial" });

test("CPU dashboard labels its backend, units, and real metrics", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByLabel("Execution backend CPU")).toBeVisible();
  await expect(page.getByText("CPU development mode")).toBeVisible();
  await expect(page.getByText("CONNECTED", { exact: true })).toBeVisible();
  await expect(page.getByText(/Latency is per batch/)).toBeVisible();
  await expect(page.getByText(/LIVE WINDOW · 5s ROLLING/)).toBeVisible();
  await page.getByRole("button", { name: /find recommendations/i }).click();
  await expect(page.getByText("Synthetic Track", { exact: false }).first()).toBeVisible({ timeout: 10000 });
  await expect(page.getByText(/Ranked by similarity/)).toBeVisible();
  await page.getByRole("button", { name: /find recommendations/i }).click();
  // The hit-rate card shows a measured percentage from the API, not a placeholder.
  await expect(page.getByLabel("Occurrence hit rate").locator("strong")).toHaveText(/\d+\.\d%/, { timeout: 10000 });
  await page.getByRole("button", { name: /Synthetic Track/ }).first().click();
  await expect(page.getByLabel("Selected item metadata")).toBeVisible();
  await page.screenshot({ path: "../docs/dashboard.png", fullPage: true });
});

test("invalid history is rejected before submission", async ({ page }) => {
  await page.goto("/");
  await page.getByLabel(/Item IDs/).fill("1, 2, banana");
  await expect(page.getByRole("alert").filter({ hasText: "not a non-negative integer" })).toBeVisible();
  await expect(page.getByRole("button", { name: /find recommendations/i })).toBeDisabled();
  await page.getByLabel(/Item IDs/).fill("999999999");
  await expect(page.getByText(/outside the table/)).toBeVisible();
  await page.getByLabel(/Item IDs/).fill("");
  await expect(page.getByRole("button", { name: /find recommendations/i })).toBeEnabled();
});

test("scenario start locks policy controls until stopped", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByText("CONNECTED", { exact: true })).toBeVisible();
  await page.getByLabel("Duration s").fill("30");
  await page.getByRole("button", { name: /start 30 second scenario/i }).click();
  await expect(page.getByRole("progressbar", { name: "Scenario progress" })).toBeVisible({ timeout: 5000 });
  await expect(page.getByLabel(/Replacement policy/)).toBeDisabled();
  await expect(page.getByRole("button", { name: "Reset cache" })).toBeDisabled();
  await expect(page.getByText(/seed 42 · phase/)).toBeVisible();
  await page.getByRole("button", { name: /stop scenario/i }).click();
  await expect(page.getByText("No active scenario")).toBeVisible({ timeout: 10000 });
  await expect(page.getByLabel(/Replacement policy/)).toBeEnabled();
  await page.getByLabel(/Replacement policy/).selectOption("lru");
  await expect(page.getByText(/Policy changed to lru/)).toBeVisible();
});

test("comparison view loads a real saved run", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByLabel("Benchmark run")).toBeVisible({ timeout: 15000 });
  await expect(page.locator(".comparison-table-wrap tbody tr").first()).toBeVisible();
  await expect(page.getByText(/CPU simulator replay/)).toBeVisible();
});

test("disconnected API is reported, not hidden", async ({ page }) => {
  await page.route("**/api/v1/**", (route) => route.abort());
  await page.goto("/");
  await expect(page.getByText("DISCONNECTED", { exact: true })).toBeVisible({ timeout: 10000 });
  await expect(page.getByRole("alert").filter({ hasText: "Disconnected" })).toBeVisible();
});
