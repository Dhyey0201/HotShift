import { useState } from "react";
import { api, type Status } from "../api/client";

const policies = ["no_cache", "static", "lru", "decayed_lfu", "adaptive"];
const scenarios: Record<string, string> = {
  stable: "Stable popularity",
  viral_release: "Viral release · abrupt shift",
  gradual: "Gradual drift",
  brief_burst: "Brief burst",
  uniform: "Uniform access",
};

type Props = { status: Status | null; connected: boolean; onChange: () => void; onMessage: (message: string) => void };

function bounded(value: number, min: number, max: number) {
  return Number.isFinite(value) ? Math.min(max, Math.max(min, value)) : min;
}

export function ScenarioControls({ status, connected, onChange, onMessage }: Props) {
  const [busy, setBusy] = useState(false);
  const [scenario, setScenario] = useState("viral_release");
  const [seed, setSeed] = useState(42);
  const [duration, setDuration] = useState(20);
  const [requestRate, setRequestRate] = useState(20);
  const active = status?.scenario ?? null;
  const locked = busy || !!active || !connected || !!status?.busy;

  async function run(action: () => Promise<string>) {
    setBusy(true);
    try {
      onMessage(await action());
    } catch (error) {
      onMessage(error instanceof Error ? error.message : "Request failed");
    } finally {
      setBusy(false);
      onChange();
    }
  }

  const changePolicy = (policy: string) => run(async () => {
    await api("/api/v1/policy", { method: "PUT", body: JSON.stringify({ policy }) });
    return `Policy changed to ${policy}; the cache was emptied${policy === "static" ? " and the fixed warm set preloaded" : ""}.`;
  });
  const resetCache = () => run(async () => {
    await api("/api/v1/cache/reset", { method: "POST" });
    return "Cache reset to its empty (or static preloaded) state.";
  });
  const start = () => run(async () => {
    await api("/api/v1/scenarios/start", { method: "POST", body: JSON.stringify({ scenario, seed, duration_seconds: duration, request_rate: requestRate }) });
    return `Scenario started · ${scenarios[scenario]} · seed ${seed}. It begins from the current cache contents.`;
  });
  const stop = () => run(async () => {
    await api("/api/v1/scenarios/stop", { method: "POST", timeoutMs: 15000 });
    return "Scenario stopped; no new traffic is being generated.";
  });

  return (
    <section className="panel controls-panel">
      <div className="panel-head"><div><p className="eyebrow">EXPERIMENT CONTROLS</p><h2>Change the workload</h2></div></div>
      <label className="field-label" htmlFor="policy">Replacement policy <span>{active ? "locked while a scenario runs" : "switching drains work and empties the cache"}</span></label>
      <div className="policy-row">
        <select id="policy" value={status?.policy ?? "adaptive"} disabled={locked} onChange={(event) => void changePolicy(event.target.value)}>{policies.map((policy) => <option key={policy} value={policy}>{policy.replace("_", " ")}</option>)}</select>
        <button className="secondary-button" disabled={locked} onClick={() => void resetCache()}>Reset cache</button>
      </div>
      <label className="field-label" htmlFor="scenario">Traffic scenario <span>seeded and bounded</span></label>
      <select id="scenario" value={scenario} disabled={locked} onChange={(event) => setScenario(event.target.value)}>
        {Object.entries(scenarios).map(([value, label]) => <option key={value} value={value}>{label}</option>)}
      </select>
      <div className="scenario-grid">
        <label>Seed<input type="number" min={0} max={2147483647} value={seed} disabled={locked} onChange={(event) => setSeed(bounded(Math.trunc(Number(event.target.value)), 0, 2147483647))} /></label>
        <label>Duration s<input type="number" min={1} max={60} value={duration} disabled={locked} onChange={(event) => setDuration(bounded(Number(event.target.value), 1, 60))} /></label>
        <label>Rate req/s<input type="number" min={0.1} max={100} step={0.1} value={requestRate} disabled={locked} onChange={(event) => setRequestRate(bounded(Number(event.target.value), 0.1, 100))} /></label>
      </div>
      <div className="scenario-actions">
        {active
          ? <button className="stop-button" disabled={busy} onClick={() => void stop()}>Stop scenario <span>■</span></button>
          : <button className="primary-button" disabled={locked || !status?.healthy} onClick={() => void start()}>{busy ? "Working…" : `Start ${duration} second scenario`}<span>▶</span></button>}
      </div>
      {active && (
        <div className="scenario-progress" aria-live="polite">
          <div className="progress-track" role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(active.progress * 100)} aria-label="Scenario progress"><span style={{ width: `${Math.round(active.progress * 100)}%` }} /></div>
          <p>{scenarios[active.scenario] ?? active.scenario} · seed {active.seed} · phase <strong>{active.phase ?? "starting"}</strong> · {active.status}</p>
          <p>{active.offered} offered · {active.completed} completed · {active.errors} errors · {active.dropped} dropped by generator</p>
        </div>
      )}
      <div className="control-foot"><span className={active ? "pulse-dot active" : "pulse-dot"} />{active ? "Scenario active · policy and reset locked" : status?.busy ? "Maintenance command running" : "No active scenario"}</div>
      <div className="metric-explainer"><strong>Reading the dashboard</strong><p>Live runs are illustrative: they start from whatever is cached now, and timing depends on this machine. Formal policy comparisons come from offline replay of identical traces (see the comparison panel). Hit rate counts resident item occurrences at lookup time.</p></div>
    </section>
  );
}
