import { defineConfig } from "@playwright/test";

// CPU-mode journey against the real API. A smoke CPU replay is generated first so the
// comparison view loads an actual artifact rather than mock data.
const python = process.env.HOTSHIFT_PYTHON ?? ".venv/bin/python";
const cli = `${python} -m hotshift.cli`;

export default defineConfig({
  testDir: "./tests",
  workers: 1,
  timeout: 60000,
  use: { baseURL: "http://127.0.0.1:5173", headless: true, viewport: { width: 1280, height: 900 } },
  webServer: [
    {
      command: `cd .. && ${cli} data generate --config configs/smoke.yaml --if-missing && ${cli} data catalog --config configs/smoke.yaml && ${cli} bench --matrix configs/smoke.yaml --smoke && ${cli} serve --config configs/smoke.yaml --backend cpu`,
      url: "http://127.0.0.1:8000/health/ready",
      reuseExistingServer: !process.env.CI,
      timeout: 180000,
    },
    {
      command: "npm run dev -- --host 127.0.0.1",
      url: "http://127.0.0.1:5173",
      reuseExistingServer: !process.env.CI,
      timeout: 120000,
    },
  ],
});
