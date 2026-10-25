import { afterEach, describe, expect, it, vi } from "vitest";
import { api } from "./client";

afterEach(() => vi.unstubAllGlobals());

describe("API client", () => {
  it("returns validated JSON payloads", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response('{"status":"ready"}', { status: 200 })));
    await expect(api<{ status: string }>("/health/ready")).resolves.toEqual({ status: "ready" });
  });

  it("surfaces backend validation errors", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response('{"detail":"invalid history"}', { status: 422 })));
    await expect(api("/api/v1/recommend")).rejects.toThrow("invalid history");
  });
});
