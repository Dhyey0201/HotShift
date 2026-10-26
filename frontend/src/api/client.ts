export type Percentiles = { p50: number | null; p95: number | null; p99: number | null; sample_count: number };

export type Scenario = {
  run_id: string;
  scenario: string;
  trace_scenario: string;
  seed: number;
  duration_seconds: number;
  request_rate: number;
  status: string;
  progress: number;
  phase: string | null;
  phase_id: number | null;
  phase_changes: { phase_id: number; at_seconds: number; label: string }[];
  offered: number;
  completed: number;
  errors: number;
  dropped: number;
  initial_cache_occupancy: number;
  policy: string;
};

export type Status = {
  backend: string;
  device: string;
  healthy: boolean;
  dataset_version: string;
  rows: number;
  dimensions: number;
  cache_capacity: number;
  cache_occupancy: number;
  policy: string;
  policy_state?: { mode: string; shift_score: number | null };
  state_generation: number;
  scenario_id?: string | null;
  scenario?: Scenario | null;
  busy?: boolean;
  limits?: { max_history_ids: number; max_batch_requests: number; queue_capacity: number };
  memory?: Record<string, number | string | boolean | null>;
};

export type MetricsWindow = {
  seconds: number;
  batches: number;
  requests: number;
  occurrences: number;
  occurrence_hit_rate: number | null;
  unique_hit_rate: number | null;
  useful_lookup_ids_per_second: number;
  request_throughput_per_second: number;
  host_miss_bytes_per_second: number;
  h2d_bytes_per_second: number | null;
  admissions: number;
  evictions: number;
  batch_engine_ms: Percentiles;
  queue_delay_ms: Percentiles;
};

export type Metrics = {
  counts: Record<string, number>;
  occurrence_hit_rate: number | null;
  unique_hit_rate: number | null;
  occurrences: number;
  host_miss_bytes: number;
  h2d_bytes: number;
  useful_lookup_ids_per_second: number;
  request_throughput_per_second: number;
  host_miss_bytes_per_second: number;
  queue_occupancy: number;
  queue_capacity: number;
  batch_engine_ms: Percentiles;
  queue_delay_ms: Percentiles;
  window: MetricsWindow;
};

export type Recommendation = { item_id: number; title: string; artist: string; genre: string; region: string; score: number };
export type RecommendResponse = {
  items: Recommendation[];
  ranking: "similarity" | "popularity_fallback";
  returned: number;
  backend: string;
  policy: string;
  batch_size: number;
  timing_ms: { queue: number; batch_engine: number; scoring: number; server_total: number };
};

export type RunListItem = {
  run_id: string;
  status: string;
  kind?: string;
  backend?: string;
  dataset_version?: string;
  dataset_sha256?: string;
  rows?: number;
  dimensions?: number;
  cache_budget_bytes?: number;
  gpu?: string | null;
  smoke?: boolean;
  summary_count?: number;
  timestamp_utc?: string;
};

export type BenchmarkRow = {
  backend?: string;
  scenario: string;
  seed?: number;
  seeds?: number[];
  repeats?: number;
  policy: string;
  variant?: string;
  condition?: string;
  cache_fraction?: number;
  batch_size: number;
  occurrence_hit_rate?: number | null;
  occurrence_hit_rate_median?: number | null;
  request_throughput?: number | null;
  request_throughput_median?: number | null;
  request_throughput_min?: number | null;
  request_throughput_max?: number | null;
  batch_wall_ms_p99?: number | null;
  batch_wall_ms_p99_median?: number | null;
  h2d_bytes?: number | null;
  h2d_bytes_median?: number | null;
  host_miss_bytes?: number;
  recovery?: { status: string; recovery_requests?: number } | null;
  recovery_statuses?: string[] | null;
  recovery_requests_median?: number | null;
  memory?: Record<string, number | null>;
  output_check?: { passed: boolean };
  output_check_passed?: boolean;
};

export type RunManifest = {
  run_id: string;
  backend: string;
  kind?: string;
  dataset_sha256?: string;
  rows?: number;
  dimensions?: number;
  cache_budget_bytes?: number;
  trace_sha256?: Record<string, string>;
  environment?: { gpu?: string | null };
  smoke?: boolean;
};

export type RunDetail = { manifest: RunManifest; summary: BenchmarkRow[]; aggregate: BenchmarkRow[] | null };

export class ApiError extends Error {
  constructor(message: string, readonly status: number | null, readonly kind: "http" | "timeout" | "network" | "aborted") {
    super(message);
  }
}

export async function api<T>(path: string, init?: RequestInit & { timeoutMs?: number }): Promise<T> {
  const controller = new AbortController();
  const timeoutMs = init?.timeoutMs ?? 8000;
  let timedOut = false;
  const timer = setTimeout(() => {
    timedOut = true;
    controller.abort();
  }, timeoutMs);
  const outer = init?.signal;
  const forward = () => controller.abort();
  outer?.addEventListener("abort", forward);
  try {
    const response = await fetch(path, {
      ...init,
      signal: controller.signal,
      headers: { "content-type": "application/json", ...init?.headers },
    });
    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: response.statusText }));
      const detail = Array.isArray(error.detail) ? error.detail.map((item: { msg?: string }) => item.msg).join("; ") : error.detail;
      throw new ApiError(detail ?? `Request failed (${response.status})`, response.status, "http");
    }
    return (await response.json()) as T;
  } catch (error) {
    if (error instanceof ApiError) throw error;
    if (timedOut) throw new ApiError(`Request timed out after ${timeoutMs} ms`, null, "timeout");
    if (outer?.aborted) throw new ApiError("Request cancelled", null, "aborted");
    throw new ApiError("API unreachable", null, "network");
  } finally {
    clearTimeout(timer);
    outer?.removeEventListener("abort", forward);
  }
}
