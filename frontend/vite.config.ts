import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";

const apiTarget = process.env.HOTSHIFT_API_PROXY ?? "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    host: "127.0.0.1",
    port: 5173,
    strictPort: true,
    // Only the API and health routes are proxied; the dev server exposes nothing else.
    proxy: {
      "/api": apiTarget,
      "/health": apiTarget,
    },
  },
  build: {
    rollupOptions: {
      output: {
        manualChunks: { charts: ["recharts"], react: ["react", "react-dom"] },
      },
    },
  },
  test: { include: ["src/**/*.test.ts"] },
});
