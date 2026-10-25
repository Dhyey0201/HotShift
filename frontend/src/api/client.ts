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
  state_generation: number;
  scenario_id?: string | null;
};

export type Metrics = {
  counts: Record<string, number>;
  occurrence_hit_rate: number | null;
  unique_hit_rate: number | null;
  occurrences: number;
  host_miss_bytes: number;
  useful_lookup_ids_per_second: number;
  request_throughput_per_second: number;
  host_miss_bytes_per_second: number;
  queue_occupancy: number;
  batch_engine_ms: { p50: number | null; p95: number | null; p99: number | null; sample_count: number };
  queue_delay_ms: { p50: number | null; p95: number | null; p99: number | null; sample_count: number };
};

export type Recommendation = { item_id: number; title: string; artist: string; genre: string; score: number };
export type RunListItem = { run_id: string; status: string; backend?: string; dataset_version?: string; summary_count?: number; timestamp_utc?: string };
export type BenchmarkRow = { backend?: string; scenario: string; seed?: number; policy: string; batch_size: number; occurrence_hit_rate: number | null; unique_hit_rate: number | null; request_throughput: number | null; batch_engine_ms_p50?: number | null; batch_wall_ms_p50?: number | null; host_miss_bytes: number };

export async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    ...init,
    headers: { "content-type": "application/json", ...init?.headers },
  });
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(error.detail ?? `Request failed (${response.status})`);
  }
  return response.json() as Promise<T>;
}
