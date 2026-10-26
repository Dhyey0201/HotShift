import { describe, expect, it } from "vitest";
import { compareRuns, errorCount, mib, ms, parseHistory, percent } from "./format";

describe("formatting", () => {
  it("renders null measurements as missing, never as zero", () => {
    expect(percent(null)).toBe("—");
    expect(ms(undefined)).toBe("—");
    expect(percent(0.5)).toBe("50.0%");
    expect(mib(1048576)).toBe("1.00 MiB");
  });

  it("sums only error counters", () => {
    expect(errorCount({ completed_requests: 9, errors_timeout: 2, errors_queue_full: 1 })).toBe(3);
  });
});

describe("history parsing", () => {
  it("accepts bounded comma or space separated IDs", () => {
    expect(parseHistory("1, 2 3", 10, 256)).toEqual({ ids: [1, 2, 3], error: null });
    expect(parseHistory("", 10, 256)).toEqual({ ids: [], error: null });
  });

  it("rejects invalid, out-of-range, and oversized histories", () => {
    expect(parseHistory("1, -2", 10, 256).error).toMatch(/not a non-negative integer/);
    expect(parseHistory("1.5", 10, 256).error).toMatch(/not a non-negative integer/);
    expect(parseHistory("10", 10, 256).error).toMatch(/outside the table/);
    expect(parseHistory("1 2 3", 10, 2).error).toMatch(/limit is 2/);
  });
});

describe("run compatibility", () => {
  const base = { run_id: "a", backend: "cuda", dataset_sha256: "d", rows: 10, dimensions: 4, cache_budget_bytes: 16, trace_sha256: { "stable-seed42": "t" } };

  it("accepts runs with identical data, budget, and shared traces", () => {
    expect(compareRuns(base, { ...base, run_id: "b" })).toEqual({ compatible: true, reasons: [] });
  });

  it("warns on any mismatch that invalidates a comparison", () => {
    const result = compareRuns(base, { ...base, run_id: "b", cache_budget_bytes: 32, trace_sha256: { "stable-seed42": "other" } });
    expect(result.compatible).toBe(false);
    expect(result.reasons).toEqual(["different cache budget", "trace stable-seed42 differs"]);
    expect(compareRuns(base, { ...base, trace_sha256: { "abrupt-seed42": "t" } }).reasons).toContain("no shared traces");
  });

  it("flags a backend difference without blocking", () => {
    const result = compareRuns(base, { ...base, backend: "cpu" });
    expect(result.compatible).toBe(true);
    expect(result.reasons[0]).toMatch(/different backends/);
  });
});
