import type { Metrics, Status } from "../api/client";
import { errorCount, mib, ms, percent, rate } from "../lib/format";

export function MetricCards({ metrics, status }: { metrics: Metrics | null; status: Status | null }) {
  const window = metrics?.window;
  const span = window ? `last ${window.seconds}s` : "awaiting samples";
  const cuda = status?.backend === "cuda";
  const transfer = cuda ? window?.h2d_bytes_per_second : window?.host_miss_bytes_per_second;
  return (
    <section className="metrics-grid" aria-label="Live metrics">
      <Metric label="Occurrence hit rate" value={percent(window?.occurrence_hit_rate)} detail={`${span} · ${percent(metrics?.occurrence_hit_rate)} since start`} accent="lime" />
      <Metric label="Useful lookup rate" value={rate(window?.useful_lookup_ids_per_second, "IDs/s")} detail={`${span} · completed occurrences`} accent="blue" />
      <Metric label="Batch p95 / p99" value={`${ms(window?.batch_engine_ms.p95)} / ${ms(window?.batch_engine_ms.p99)}`} detail={`${window?.batch_engine_ms.sample_count ?? 0} batches · shared batch time`} accent="violet" />
      <Metric label="Queue delay p95" value={ms(window?.queue_delay_ms.p95)} detail={`${metrics?.queue_occupancy ?? 0} / ${metrics?.queue_capacity ?? "—"} queued now`} accent="orange" />
      <Metric
        label={cuda ? "Host→device copies" : "Miss row volume"}
        value={rate(transfer == null ? null : transfer / 1048576, "MiB/s")}
        detail={cuda ? `${span} · explicit copy payload` : `${mib(metrics?.host_miss_bytes)} total · CPU estimate, not PCIe`}
        accent="blue"
      />
      <Metric label="Completed / errors" value={`${metrics?.counts.completed_requests ?? 0} / ${errorCount(metrics?.counts ?? {})}`} detail={`${rate(window?.request_throughput_per_second, "req/s")} · ${span}`} accent="lime" />
    </section>
  );
}

function Metric({ label, value, detail, accent }: { label: string; value: string; detail: string; accent: string }) {
  return <article className={`metric-card ${accent}`} aria-label={label}><div className="metric-label">{label}</div><strong>{value}</strong><small>{detail}</small></article>;
}
