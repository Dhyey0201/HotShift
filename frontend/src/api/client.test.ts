import { afterEach, describe, expect, it, vi } from "vitest";
import { api, ApiError } from "./client";

afterEach(() => {
  vi.unstubAllGlobals();
  vi.useRealTimers();
});

describe("API client", () => {
  it("returns validated JSON payloads", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response('{"status":"ready"}', { status: 200 })));
    await expect(api<{ status: string }>("/health/ready")).resolves.toEqual({ status: "ready" });
  });

  it("surfaces backend validation errors with their status", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response('{"detail":"invalid history"}', { status: 422 })));
    const error = (await api("/api/v1/recommend").catch((caught: unknown) => caught)) as ApiError;
    expect(error).toBeInstanceOf(ApiError);
    expect(error.message).toBe("invalid history");
    expect(error.status).toBe(422);
  });

  it("joins FastAPI validation detail lists", async () => {
    const body = JSON.stringify({ detail: [{ msg: "too long" }, { msg: "bad type" }] });
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(body, { status: 422 })));
    await expect(api("/api/v1/recommend")).rejects.toThrow("too long; bad type");
  });

  it("reports timeouts distinctly from network failures", async () => {
    vi.stubGlobal("fetch", vi.fn((_: string, init: RequestInit) => new Promise((_, reject) => {
      init.signal?.addEventListener("abort", () => reject(new DOMException("aborted", "AbortError")));
    })));
    const error = (await api("/api/v1/metrics", { timeoutMs: 10 }).catch((caught: unknown) => caught)) as ApiError;
    expect(error.kind).toBe("timeout");
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("failed to fetch")));
    await expect(api("/api/v1/metrics")).rejects.toMatchObject({ kind: "network" });
  });
});
