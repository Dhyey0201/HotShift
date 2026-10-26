import { useEffect, useMemo, useState } from "react";
import { api, type BenchmarkRow, type RunDetail, type RunListItem } from "../api/client";
import { compareRuns, ms, percent, rate } from "../lib/format";

type Props = { runs: RunListItem[] };

function rowsOf(detail: RunDetail | null): BenchmarkRow[] {
  if (!detail) return [];
  return detail.aggregate?.length ? detail.aggregate : detail.summary;
}

function pick(row: BenchmarkRow, base: "occurrence_hit_rate" | "request_throughput" | "batch_wall_ms_p99" | "h2d_bytes") {
  const median = row[`${base}_median` as keyof BenchmarkRow] as number | null | undefined;
  return median !== undefined ? median : (row[base] as number | null | undefined);
}

function recovery(row: BenchmarkRow) {
  if (row.recovery_statuses?.length) return row.recovery_requests_median != null ? `${row.recovery_requests_median.toFixed(0)} req` : row.recovery_statuses.map((status) => status.replaceAll("_", " ")).join(", ");
  if (row.recovery) return row.recovery.status === "recovered" ? `${row.recovery.recovery_requests} req` : row.recovery.status.replaceAll("_", " ");
  return "—";
}

function useRun(runId: string) {
  const [detail, setDetail] = useState<RunDetail | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    if (!runId) { setDetail(null); return; }
    const controller = new AbortController();
    setError(null);
    api<RunDetail>(`/api/v1/runs/${encodeURIComponent(runId)}`, { signal: controller.signal })
      .then(setDetail)
      .catch((caught) => { if (!controller.signal.aborted) { setDetail(null); setError(caught instanceof Error ? caught.message : "Run failed to load"); } });
    return () => controller.abort();
  }, [runId]);
  return { detail, error };
}

export function ComparisonView({ runs }: Props) {
  const completed = useMemo(() => runs.filter((run) => run.status === "complete" && (run.kind ?? "replay_matrix") === "replay_matrix"), [runs]);
  const [primary, setPrimary] = useState("");
  const [secondary, setSecondary] = useState("");
  useEffect(() => {
    if (!primary && completed.length) setPrimary(completed[completed.length - 1].run_id);
  }, [completed, primary]);
  const left = useRun(primary);
  const right = useRun(secondary);
  const compatibility = left.detail && right.detail ? compareRuns(left.detail.manifest, right.detail.manifest) : null;
  const rows = [
    ...rowsOf(left.detail).map((row) => ({ run: primary, row })),
    ...(compatibility?.compatible ? rowsOf(right.detail).map((row) => ({ run: secondary, row })) : []),
  ];
  const aggregated = Boolean(left.detail?.aggregate?.length);
  const backend = left.detail?.manifest.backend;

  return (
    <section className="panel comparison-panel">
      <div className="panel-head"><div><p className="eyebrow">SAVED BENCHMARK RUNS</p><h2>Policy comparison</h2></div><span className="synthetic-tag">DETERMINISTIC REPLAY</span></div>
      {completed.length ? <>
        <div className="compare-selectors">
          <label className="field-label" htmlFor="saved-run">Benchmark run</label>
          <select id="saved-run" value={primary} onChange={(event) => setPrimary(event.target.value)}>
            {completed.map((run) => <option key={run.run_id} value={run.run_id}>{run.run_id} · {run.backend}{run.gpu ? ` · ${run.gpu}` : ""}{run.smoke ? " · smoke" : ""}</option>)}
          </select>
          <label className="field-label" htmlFor="compare-run">Compare with <span>only runs sharing dataset, traces, and budget</span></label>
          <select id="compare-run" value={secondary} onChange={(event) => setSecondary(event.target.value)}>
            <option value="">None</option>
            {completed.filter((run) => run.run_id !== primary).map((run) => <option key={run.run_id} value={run.run_id}>{run.run_id} · {run.backend}</option>)}
          </select>
        </div>
        {(left.error || right.error) && <p className="field-error" role="alert">{left.error ?? right.error}</p>}
        {compatibility && !compatibility.compatible && <div className="warning-note" role="alert"><strong>Runs are not comparable.</strong> {compatibility.reasons.join("; ")}. Showing the first run only.</div>}
        {compatibility?.compatible && compatibility.reasons.length > 0 && <div className="warning-note" role="status">{compatibility.reasons.join("; ")}</div>}
        <div className="comparison-table-wrap"><table><caption className="sr-only">Benchmark results</caption><thead><tr>
          {secondary && compatibility?.compatible && <th>Run</th>}<th>Scenario</th><th>Policy</th><th>Variant</th><th>Batch</th><th>Cond.</th><th>Hit rate</th><th>Throughput</th><th>Batch p99</th><th>H2D bytes</th><th>Recovery</th><th>Outputs</th>
        </tr></thead>
          <tbody>{rows.map(({ run, row }, index) => {
            const passed = row.output_check_passed ?? row.output_check?.passed;
            const h2d = pick(row, "h2d_bytes");
            return <tr key={`${run}-${row.scenario}-${row.policy}-${row.variant}-${row.batch_size}-${row.condition}-${row.seed}-${index}`}>
              {secondary && compatibility?.compatible && <td>{run.slice(-8)}</td>}
              <td>{row.scenario}</td><td>{row.policy}</td><td>{row.variant ?? "—"}</td><td>{row.batch_size}</td><td>{row.condition ?? "warm"}</td>
              <td>{percent(pick(row, "occurrence_hit_rate"))}</td><td>{rate(pick(row, "request_throughput"), "req/s")}</td><td>{ms(pick(row, "batch_wall_ms_p99"))}</td>
              <td>{h2d == null ? "—" : `${(h2d / 1048576).toFixed(1)} MiB`}</td><td>{recovery(row)}</td><td>{passed == null ? "—" : passed ? "match" : "MISMATCH"}</td>
            </tr>;
          })}</tbody></table></div>
        <p className="comparison-note">
          {aggregated ? "Values are medians across repeated seeds; per-seed rows stay in summary.json. " : ""}
          {backend === "cuda"
            ? "CUDA batch wall time includes host planning, transfers, synchronization, kernels, and policy maintenance on the recorded GPU. torch_gpu_resident keeps the full table on the device: a higher-memory reference, not an equal-budget competitor."
            : "CPU simulator replay: hit rates describe policy semantics and timings are CPU execution; neither predicts GPU serving performance."}
          {" "}No speedup is shown without a named baseline; compare rows directly.
        </p>
      </> : <div className="empty-state"><p>No saved benchmark runs found.</p><small>Run <code>hotshift bench</code> for CPU replay or <code>hotshift bench-cuda</code> on an NVIDIA host.</small></div>}
    </section>
  );
}
