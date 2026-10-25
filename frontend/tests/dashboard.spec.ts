import { expect, test } from "@playwright/test";

test("CPU dashboard labels its backend and submits a history", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByText("CPU", { exact: true }).first()).toBeVisible();
  await expect(page.getByRole("heading", { name: "Try a listener history" })).toBeVisible();
  await page.getByRole("button", { name: /find recommendations/i }).click();
  await expect(page.getByText("Synthetic Track", { exact: false }).first()).toBeVisible({ timeout: 10000 });
  await page.getByRole("button", { name: /find recommendations/i }).click();
  await expect(page.getByText("50.0%", { exact: true })).toBeVisible({ timeout: 10000 });
  await page.screenshot({ path: "../docs/dashboard.png", fullPage: true });
});

