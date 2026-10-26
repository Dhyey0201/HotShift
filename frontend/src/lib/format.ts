import type { RunManifest } from "../api/client";

export function percent(value: number | null | undefined) {
  return value == null ? "—" : `${(value * 100).toFixed(1)}%`;
}

export function ms(value: number | null | undefined) {
  return value == null ? "—" : `${value.toFixed(2)} ms`;
}

export function rate(value: number | null | undefined, unit: string) {
  return value == null ? "—" : `${value.toLocaleString("en-US", { maximumFractionDigits: 1 })} ${unit}`;
}

export function mib(value: number | null | undefined) {
  return value == null ? "—" : `${(value / 1048576).toFixed(2)} MiB`;
}

export function errorCount(counts: Record<string, number>) {
  return Object.entries(counts)
    .filter(([key]) => key.startsWith("errors_"))
    .reduce((total, [, count]) => total + count, 0);
}

export type ParsedHistory = { ids: number[]; error: string | null };

export function parseHistory(text: string, rows: number | undefined, maxIds: number): ParsedHistory {
  const tokens = text.split(/[\s,]+/).filter(Boolean);
  if (tokens.length > maxIds) return { ids: [], error: `History has ${tokens.length} IDs; the limit is ${maxIds}.` };
  const ids: number[] = [];
  for (const token of tokens) {
    if (!/^\d+$/.test(token)) return { ids: [], error: `“${token}” is not a non-negative integer item ID.` };
    const value = Number(token);
    if (rows != null && value >= rows) return { ids: [], error: `Item ${value} is outside the table (0–${rows - 1}).` };
    ids.push(value);
  }
  return { ids, error: null };
}

export type Compatibility = { compatible: boolean; reasons: string[] };

/** Runs are comparable only when data, budget, and every shared trace match exactly. */
export function compareRuns(left: RunManifest, right: RunManifest): Compatibility {
  const reasons: string[] = [];
  if (left.dataset_sha256 !== right.dataset_sha256) reasons.push("different dataset checksum");
  if (left.rows !== right.rows || left.dimensions !== right.dimensions) reasons.push("different table shape");
  if (left.cache_budget_bytes !== right.cache_budget_bytes) reasons.push("different cache budget");
  const leftTraces = left.trace_sha256 ?? {};
  const rightTraces = right.trace_sha256 ?? {};
  const shared = Object.keys(leftTraces).filter((key) => key in rightTraces);
  if (!shared.length) reasons.push("no shared traces");
  for (const key of shared) if (leftTraces[key] !== rightTraces[key]) reasons.push(`trace ${key} differs`);
  if (left.backend !== right.backend) reasons.push(`different backends (${left.backend} vs ${right.backend}); compare with care`);
  const blocking = reasons.filter((reason) => !reason.startsWith("different backends"));
  return { compatible: blocking.length === 0, reasons };
}
