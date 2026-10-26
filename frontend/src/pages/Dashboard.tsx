import { useCallback, useEffect, useRef, useState } from "react";
import { api, ApiError, type Metrics, type RunListItem, type Status } from "../api/client";
import { ComparisonView } from "../components/ComparisonView";
import { LiveCharts, type PhaseMarker, type Point } from "../components/LiveCharts";
import { MetricCards } from "../components/MetricCards";
import { MusicDemo } from "../components/MusicDemo";
import { ScenarioControls } from "../components/ScenarioControls";
import { StatusHeader, SystemCard } from "../components/StatusHeader";

const MAX_POINTS = 600;

export default function Dashboard() {
  const [status, setStatus] = useState<Status | null>(null);
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [points, setPoints] = useState<Point[]>([]);
  const [markers, setMarkers] = useState<PhaseMarker[]>([]);
  const [connected, setConnected] = useState(false);
  const [loading, setLoading] = useState(true);
  const [message, setMessage] = useState("");
  const [runs, setRuns] = useState<RunListItem[]>([]);
  const lastPhase = useRef<string | null>(null);
  const inFlight = useRef<AbortController | null>(null);

  const refresh = useCallback(async () => {
    inFlight.current?.abort();
    const controller = new AbortController();
    inFlight.current = controller;
    try {
      const [nextStatus, nextMetrics] = await Promise.all([
        api<Status>("/api/v1/status", { signal: controller.signal, timeoutMs: 4000 }),
        api<Metrics>("/api/v1/metrics", { signal: controller.signal, timeoutMs: 4000 }),
      ]);
      const time = new Date().toLocaleTimeString([], { hour12: false, minute: "2-digit", second: "2-digit" });
      setStatus(nextStatus);
      setMetrics(nextMetrics);
      setConnected(true);
      const window = nextMetrics.window;
      setPoints((current) => [...current, {
        time,
        hit: window.occurrence_hit_rate,
        p95: window.batch_engine_ms.p95,
        p99: window.batch_engine_ms.p99,
        churn: window.admissions + window.evictions,
      }].slice(-MAX_POINTS));
      const phase = nextStatus.scenario ? `${nextStatus.scenario.run_id}:${nextStatus.scenario.phase}` : null;
      if (phase && phase !== lastPhase.current && nextStatus.scenario?.phase) {
        setMarkers((current) => [...current, { time, label: nextStatus.scenario!.phase! }].slice(-20));
      }
      lastPhase.current = phase;
    } catch (error) {
      if (error instanceof ApiError && error.kind === "aborted") return;
      setConnected(false);
      setMessage(error instanceof Error ? `API unavailable: ${error.message}` : "API unavailable");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void refresh();
    const timer = window.setInterval(() => void refresh(), 1000);
    return () => {
      window.clearInterval(timer);
      inFlight.current?.abort();
    };
  }, [refresh]);

  useEffect(() => {
    const controller = new AbortController();
    const loadRuns = async () => {
      try {
        const result = await api<{ runs: RunListItem[] }>("/api/v1/runs", { signal: controller.signal });
        setRuns(result.runs);
      } catch { /* The live dashboard remains usable without saved artifacts. */ }
    };
    void loadRuns();
    const timer = window.setInterval(() => void loadRuns(), 10000);
    return () => {
      window.clearInterval(timer);
      controller.abort();
    };
  }, []);

  return (
    <main className="shell">
      <StatusHeader status={status} connected={connected} />
      <section className="hero" id="top">
        <div>
          <p className="eyebrow">EMBEDDING CACHE LAB <span>·</span> {status?.dataset_version ?? "synthetic data"}</p>
          <h1>Popularity moves.<br /><em>Watch the cache respond.</em></h1>
          <p className="hero-copy">A bounded embedding cache serving pooled listener histories while traffic shifts. Every number below comes from the running HotShift API; nothing is simulated in the browser.</p>
        </div>
        <SystemCard status={status} />
      </section>

      {loading && !status && <div className="notice" role="status">Connecting to the HotShift API…</div>}
      {!loading && !connected && <div className="unhealthy-note" role="alert"><strong>Disconnected</strong><span>The API at /api is not responding. Start it with <code>hotshift serve</code>; metrics resume automatically.</span></div>}
      {status && !status.healthy && <div className="unhealthy-note" role="alert"><strong>Engine unhealthy</strong><span>A batch failed and the engine stopped serving rather than continue from uncertain state. Restart the service; it never falls back to another backend.</span></div>}
      {status?.backend === "cpu" && <div className="cpu-note"><strong>CPU development mode</strong><span>CUDA kernels are not active on this backend. Hit rates come from the CPU cache model; latencies are CPU timings, not GPU performance.</span></div>}
      {message && <div className="notice" role="status">{message}<button onClick={() => setMessage("")} aria-label="Dismiss message">×</button></div>}

      <MetricCards metrics={metrics} status={status} />
      <LiveCharts points={points} markers={markers} windowSeconds={metrics?.window.seconds ?? 5} />
      <div className="content-grid">
        <MusicDemo status={status} connected={connected} onDone={() => void refresh()} onMessage={setMessage} />
        <ScenarioControls status={status} connected={connected} onChange={() => void refresh()} onMessage={setMessage} />
      </div>
      <ComparisonView runs={runs} />
      <footer><span>HOTSHIFT <b>·</b> SYSTEMS DEMONSTRATION</span><span>Backend metrics only · synthetic data · no performance claims without recorded runs</span></footer>
    </main>
  );
}
