import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "./tests",
  use: { baseURL: "http://127.0.0.1:5173", headless: true },
  webServer: [
    {
      command: "cd .. && .venv/bin/python -m hotshift.cli data generate --config configs/smoke.yaml && .venv/bin/python -m hotshift.cli data catalog --config configs/smoke.yaml && .venv/bin/python -m hotshift.cli serve --config configs/smoke.yaml --backend cpu",
      url: "http://127.0.0.1:8000/health/live",
      reuseExistingServer: !process.env.CI,
      timeout: 120000,
    },
    {
      command: "npm run dev -- --host 127.0.0.1",
      url: "http://127.0.0.1:5173",
      reuseExistingServer: !process.env.CI,
      timeout: 120000,
    },
  ],
});
