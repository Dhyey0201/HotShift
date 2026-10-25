import { useCallback, useEffect, useMemo, useState } from "react";
import { Area, AreaChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api, type BenchmarkRow, type Metrics, type Recommendation, type RunListItem, type Status } from "./api/client";

type Point = { time: string; hit: number | null; p95: number | null };
const profiles: Record<string, number[]> = {
  "Late-night listener": [4, 15, 27, 33, 49, 61, 88],
  "Analog explorer": [12, 18, 39, 54, 72, 101, 121],
  "Weekend mixer": [3, 23, 41, 66, 78, 93, 107],
};
const policies = ["no_cache", "static", "lru", "decayed_lfu", "adaptive"];

function percent(value: number | null | undefined) {
  return value == null ? "—" : `${(value * 100).toFixed(1)}%`;
}
function ms(value: number | null | undefined) {
  return value == null ? "—" : `${value.toFixed(2)} ms`;
}
function rate(value: number | null | undefined, unit: string) {
  return value == null ? "—" : `${value.toLocaleString(undefined, { maximumFractionDigits: 1 })} ${unit}`;
}
function errorCount(counts: Record<string, number>) {
  return Object.entries(counts).filter(([key]) => key.startsWith("errors_")).reduce((total, [, count]) => total + count, 0);
}
function bytes(value: number) {
  return `${(value / 1048576).toFixed(2)} MiB`;
}

export default function App() {
  const [status, setStatus] = useState<Status | null>(null);
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [history, setHistory] = useState(profiles["Late-night listener"].join(", "));
  const [items, setItems] = useState<Recommendation[]>([]);
  const [points, setPoints] = useState<Point[]>([]);
  const [scenario, setScenario] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("");
  const [connected, setConnected] = useState(false);
  const [runs, setRuns] = useState<RunListItem[]>([]);
  const [selectedRun, setSelectedRun] = useState("");
  const [comparison, setComparison] = useState<BenchmarkRow[]>([]);
  const selectedRunInfo = runs.find((run) => run.run_id === selectedRun);

  const refresh = useCallback(async () => {
    try {
      const [nextStatus, nextMetrics] = await Promise.all([
        api<Status>("/api/v1/status"),
        api<Metrics>("/api/v1/metrics"),
      ]);
      setStatus(nextStatus);
      setMetrics(nextMetrics);
      setScenario(nextStatus.scenario_id ?? null);
      setConnected(true);
      setPoints((current) => [
        ...current,
        {
          time: new Date().toLocaleTimeString([], { minute: "2-digit", second: "2-digit" }),
          hit: nextMetrics.occurrence_hit_rate,
          p95: nextMetrics.batch_engine_ms.p95,
        },
      ].slice(-600));
    } catch (error) {
      setConnected(false);
      setMessage(error instanceof Error ? error.message : "API unavailable");
    }
  }, []);

  useEffect(() => {
    void refresh();
    const timer = window.setInterval(() => void refresh(), 1000);
    return () => window.clearInterval(timer);
  }, [refresh]);

  useEffect(() => {
    const loadRuns = async () => {
      try {
        const result = await api<{ runs: RunListItem[] }>("/api/v1/runs");
        const completed = result.runs.filter((run) => run.status === "complete").reverse();
        setRuns(completed);
        if (!selectedRun && completed[0]) setSelectedRun(completed[0].run_id);
      } catch { /* The live dashboard remains usable without saved artifacts. */ }
    };
    void loadRuns();
    const timer = window.setInterval(() => void loadRuns(), 10000);
    return () => window.clearInterval(timer);
  }, [selectedRun]);

  useEffect(() => {
    if (!selectedRun) { setComparison([]); return; }
    void api<{ summary: BenchmarkRow[] }>(`/api/v1/runs/${encodeURIComponent(selectedRun)}`)
      .then((result) => setComparison(result.summary ?? []))
      .catch(() => setComparison([]));
  }, [selectedRun]);

  const ids = useMemo(() => history.split(/[\s,]+/).filter(Boolean).map(Number), [history]);
  async function recommend() {
    setBusy(true);
    setMessage("");
    try {
      const result = await api<{ items: Recommendation[] }>("/api/v1/recommend", {
        method: "POST",
        body: JSON.stringify({ history_ids: ids, top_k: 8, exclude_seen: true }),
      });
      setItems(result.items);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Recommendation request failed");
    } finally {
      setBusy(false);
      void refresh();
    }
  }
  async function changePolicy(policy: string) {
    setBusy(true);
    try {
      await api("/api/v1/policy", { method: "PUT", body: JSON.stringify({ policy }) });
      await refresh();
      setMessage(`Policy changed to ${policy}`);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Policy change failed");
    } finally {
      setBusy(false);
    }
  }
  async function resetCache() {
    setBusy(true);
    try {
      await api("/api/v1/cache/reset", { method: "POST" });
      setMessage("Cache reset");
      await refresh();
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Reset failed");
    } finally {
      setBusy(false);
    }
  }
  async function startScenario(name: string) {
    setBusy(true);
    try {
      const result = await api<{ run_id: string }>("/api/v1/scenarios/start", {
        method: "POST",
        body: JSON.stringify({ scenario: name, duration_seconds: 20, request_rate: 8, seed: 42 }),
      });
      setScenario(result.run_id);
      setMessage(`Scenario started · seed 42 · ${name}`);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Scenario failed to start");
    } finally {
      setBusy(false);
    }
  }
  async function stopScenario() {
    try {
      await api("/api/v1/scenarios/stop", { method: "POST" });
    } finally {
      setScenario(null);
      void refresh();
    }
  }

  return (
    <main className="shell">
      <header className="topbar">
        <a className="brand" href="#top"><span className="brand-mark">H</span><span>hotshift<span className="brand-dot">.</span></span></a>
        <div className="top-meta">
          <span className={`connection ${connected ? "online" : "offline"}`}><i />{connected ? "CONNECTED" : "DISCONNECTED"}</span>
          <span className="backend-badge">{status?.backend?.toUpperCase() ?? "BACKEND UNKNOWN"}</span>
          <span className="device-label">{status?.device ?? "Awaiting engine"}</span>
        </div>
      </header>

      <section className="hero" id="top">
        <div>
          <p className="eyebrow">EMBEDDING CACHE LAB <span>·</span> {status?.dataset_version ?? "synthetic data"}</p>
          <h1>Popularity moves.<br /><em>Your cache should too.</em></h1>
          <p className="hero-copy">Explore how a bounded embedding cache responds as listening patterns shift. Every metric below comes from the running HotShift API.</p>
        </div>
        <div className="system-card">
          <div className="system-heading"><span>ENGINE STATUS</span><span className={status?.healthy ? "health-dot" : "health-dot dim"} /></div>
          <div className="system-row"><span>Execution</span><strong>{status?.backend?.toUpperCase() ?? "—"}</strong></div>
          <div className="system-row"><span>Embedding table</span><strong>{status ? `${status.rows.toLocaleString()} × ${status.dimensions}` : "—"}</strong></div>
          <div className="system-row"><span>Cache budget</span><strong>{status ? `${status.cache_capacity.toLocaleString()} rows` : "—"}</strong></div>
          <div className="system-row"><span>Resident</span><strong>{status ? `${status.cache_occupancy.toLocaleString()} / ${status.cache_capacity.toLocaleString()}` : "—"}</strong></div>
        </div>
      </section>

      {status?.backend === "cpu" && <div className="cpu-note"><strong>CPU development mode</strong><span>CUDA kernels and GPU performance are not active on this backend.</span></div>}
      {message && <div className="notice" role="status">{message}<button onClick={() => setMessage("")} aria-label="Dismiss">×</button></div>}

      <section className="metrics-grid" aria-label="Live metrics">
        <Metric label="Occurrence hit rate" value={percent(metrics?.occurrence_hit_rate)} detail={`${metrics?.occurrences.toLocaleString() ?? "0"} looked up`} accent="lime" />
        <Metric label="Useful lookup rate" value={rate(metrics?.useful_lookup_ids_per_second, "IDs/s")} detail="Since service start" accent="blue" />
        <Metric label="Batch p95 / p99" value={`${ms(metrics?.batch_engine_ms.p95)} / ${ms(metrics?.batch_engine_ms.p99)}`} detail={`${metrics?.batch_engine_ms.sample_count ?? 0} batch samples`} accent="violet" />
        <Metric label="Queue p95" value={ms(metrics?.queue_delay_ms.p95)} detail={`${metrics?.queue_occupancy ?? 0} waiting`} accent="orange" />
        <Metric label="Host miss rate" value={rate(metrics?.host_miss_bytes_per_second, "B/s")} detail={`${bytes(metrics?.host_miss_bytes ?? 0)} source-row estimate`} accent="blue" />
        <Metric label="Completed / errors" value={`${metrics?.counts.completed_requests ?? 0} / ${errorCount(metrics?.counts ?? {})}`} detail={`${rate(metrics?.request_throughput_per_second, "req/s")} since service start`} accent="lime" />
      </section>

      <section className="panel chart-panel">
        <div className="panel-head"><div><p className="eyebrow">LIVE WINDOW</p><h2>Cache behavior</h2></div><span className="chart-legend"><i className="legend-hit" /> Hit rate <i className="legend-latency" /> Batch p95 ms</span></div>
        <div className="chart-wrap">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={points} margin={{ top: 12, right: 8, left: -18, bottom: 0 }}>
              <defs><linearGradient id="hitFill" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor="#b5ec66" stopOpacity={0.24} /><stop offset="100%" stopColor="#b5ec66" stopOpacity={0} /></linearGradient></defs>
              <CartesianGrid stroke="#252b33" vertical={false} />
              <XAxis dataKey="time" tick={{ fill: "#7e8793", fontSize: 11 }} axisLine={false} tickLine={false} minTickGap={32} />
              <YAxis yAxisId="hit" domain={[0, 1]} tickFormatter={(value) => `${Math.round(value * 100)}%`} tick={{ fill: "#7e8793", fontSize: 11 }} axisLine={false} tickLine={false} />
              <YAxis yAxisId="latency" orientation="right" tick={{ fill: "#7e8793", fontSize: 11 }} axisLine={false} tickLine={false} />
              <Tooltip contentStyle={{ background: "#171b21", border: "1px solid #303741", borderRadius: 10 }} labelStyle={{ color: "#cbd2db" }} />
              <Area yAxisId="hit" type="monotone" dataKey="hit" name="Occurrence hit rate" stroke="#b5ec66" fill="url(#hitFill)" connectNulls />
              <Area yAxisId="latency" type="monotone" dataKey="p95" name="Batch p95 (ms)" stroke="#7aa7ff" fill="none" connectNulls />
            </AreaChart>
          </ResponsiveContainer>
        </div>
        <div className="chart-foot"><span>Recent samples · bounded to 600 points</span><span>Latency is measured per batch, not per request</span></div>
      </section>

      <div className="content-grid">
        <section className="panel demo-panel">
          <div className="panel-head"><div><p className="eyebrow">MUSIC DISCOVERY</p><h2>Try a listener history</h2></div><span className="synthetic-tag">FICTIONAL CATALOG</span></div>
          <label className="field-label" htmlFor="listener">Listener profile</label>
          <select id="listener" onChange={(event) => setHistory(profiles[event.target.value].join(", "))}>
            {Object.keys(profiles).map((name) => <option key={name}>{name}</option>)}
          </select>
          <label className="field-label" htmlFor="history">Item IDs <span>comma or space separated · max 256</span></label>
          <textarea id="history" value={history} onChange={(event) => setHistory(event.target.value)} rows={3} />
          <button className="primary-button" disabled={busy || !connected} onClick={() => void recommend()}>{busy ? "Working…" : "Find recommendations"}<span>↗</span></button>
          <div className="recommendations">
            {items.length ? items.map((item) => <article className="song" key={item.item_id}><span className="song-index">{String(item.item_id).padStart(3, "0")}</span><div><strong>{item.title}</strong><small>{item.artist} · {item.genre}</small></div><span className="score">{item.score.toFixed(3)}</span></article>) : <div className="empty-state"><span className="empty-icon">♫</span><p>Your recommendations will appear here.</p><small>The CPU backend uses the same pooling contract; catalog ranking is synthetic.</small></div>}
          </div>
        </section>

        <section className="panel controls-panel">
          <div className="panel-head"><div><p className="eyebrow">EXPERIMENT CONTROLS</p><h2>Change the workload</h2></div></div>
          <label className="field-label" htmlFor="policy">Replacement policy</label>
          <div className="policy-row"><select id="policy" value={status?.policy ?? "adaptive"} disabled={busy || !!scenario} onChange={(event) => void changePolicy(event.target.value)}>{policies.map((policy) => <option key={policy} value={policy}>{policy.replace("_", " ")}</option>)}</select><button className="secondary-button" disabled={busy || !!scenario} onClick={() => void resetCache()}>Reset cache</button></div>
          <label className="field-label" htmlFor="scenario">Traffic scenario <span>seeded controls</span></label>
          <select id="scenario" defaultValue="viral_release" disabled={!!scenario || busy}>
            <option value="stable">Stable popularity</option><option value="viral_release">Viral release · abrupt shift</option><option value="gradual">Gradual drift</option><option value="brief_burst">Brief burst</option><option value="uniform">Uniform access</option>
          </select>
          <div className="scenario-actions">{scenario ? <button className="stop-button" onClick={() => void stopScenario()}>Stop scenario <span>■</span></button> : <button className="primary-button" disabled={busy || !connected} onClick={() => { const select = document.getElementById("scenario") as HTMLSelectElement; void startScenario(select.value); }}>{busy ? "Working…" : "Start 20 second scenario"}<span>▶</span></button>}</div>
          <div className="control-foot"><span className={scenario ? "pulse-dot active" : "pulse-dot"} />{scenario ? "Scenario active · policy controls locked" : "No active scenario"}</div>
          <div className="metric-explainer"><strong>Reading the dashboard</strong><p>Hit rate counts resident item occurrences at lookup. Host miss volume estimates source row bytes in CPU mode; it is not a measured transfer counter.</p></div>
        </section>
      </div>
      <section className="panel comparison-panel">
        <div className="panel-head"><div><p className="eyebrow">SAVED BENCHMARK RUNS</p><h2>Policy comparison</h2></div><span className="synthetic-tag">DETERMINISTIC WORKLOADS</span></div>
        {runs.length ? <>
          <label className="field-label" htmlFor="saved-run">Benchmark run</label>
          <select id="saved-run" value={selectedRun} onChange={(event) => setSelectedRun(event.target.value)}>
            {runs.map((run) => <option key={run.run_id} value={run.run_id}>{run.run_id} · {run.backend} · {run.dataset_version}</option>)}
          </select>
          <div className="comparison-table-wrap"><table><thead><tr><th>Scenario</th><th>Seed</th><th>Policy</th><th>Batch</th><th>Occurrence hit</th><th>Throughput</th><th>Batch p50 ms</th></tr></thead>
            <tbody>{comparison.map((row, index) => <tr key={`${row.scenario}-${row.seed}-${row.policy}-${row.batch_size}-${index}`}><td>{row.scenario}</td><td>{row.seed ?? "—"}</td><td>{row.policy}</td><td>{row.batch_size}</td><td>{percent(row.occurrence_hit_rate)}</td><td>{rate(row.request_throughput, "req/s")}</td><td>{ms(row.batch_wall_ms_p50 ?? row.batch_engine_ms_p50)}</td></tr>)}</tbody></table></div>
          <p className="comparison-note">{selectedRunInfo?.backend === "cuda" ? "CUDA batch wall time includes host planning, transfer, synchronization, kernels, and policy maintenance; it is specific to the recorded GPU runtime." : "CPU simulator replay hit rates describe simulated residency and CPU execution; they do not predict GPU serving performance."}</p>
        </> : <div className="empty-state"><p>No saved benchmark runs found.</p><small>Run <code>hotshift bench</code> for CPU replay or <code>hotshift bench-cuda</code> on an NVIDIA host.</small></div>}
      </section>
      <footer><span>HOTSHIFT <b>·</b> SYSTEMS DEMONSTRATION</span><span>Backend metrics only · synthetic data · no performance claims</span></footer>
    </main>
  );
}

function Metric({ label, value, detail, accent }: { label: string; value: string; detail: string; accent: string }) {
  return <article className={`metric-card ${accent}`}><div className="metric-label">{label}<span>↗</span></div><strong>{value}</strong><small>{detail}</small></article>;
}
