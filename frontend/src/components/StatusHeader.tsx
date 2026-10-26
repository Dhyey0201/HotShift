import type { Status } from "../api/client";

type Props = { status: Status | null; connected: boolean };

export function StatusHeader({ status, connected }: Props) {
  // The badge reflects only what the API reports; it never infers a GPU from the browser.
  const backend = status?.backend?.toUpperCase() ?? "BACKEND UNKNOWN";
  return (
    <header className="topbar">
      <a className="brand" href="#top"><span className="brand-mark">H</span><span>hotshift<span className="brand-dot">.</span></span></a>
      <div className="top-meta">
        <span className={`connection ${connected ? "online" : "offline"}`} role="status" aria-label={connected ? "API connected" : "API disconnected"}><i />{connected ? "CONNECTED" : "DISCONNECTED"}</span>
        <span className={`backend-badge ${status?.backend === "cuda" ? "gpu" : "cpu"}`} aria-label={`Execution backend ${backend}`}>{backend}</span>
        <span className="device-label">{status?.device ?? "Awaiting engine"}</span>
        <span className="device-label">{status ? `${status.dataset_version} · ${status.policy.replace("_", " ")}` : ""}</span>
      </div>
    </header>
  );
}

export function SystemCard({ status }: { status: Status | null }) {
  const memory = status?.memory;
  const deviceBytes = typeof memory?.device_tensor_bytes === "number" ? memory.device_tensor_bytes : null;
  return (
    <div className="system-card">
      <div className="system-heading"><span>ENGINE STATUS</span><span className={status?.healthy ? "health-dot" : "health-dot dim"} aria-label={status?.healthy ? "healthy" : "not healthy"} /></div>
      <div className="system-row"><span>Execution</span><strong>{status?.backend?.toUpperCase() ?? "—"}</strong></div>
      <div className="system-row"><span>Embedding table</span><strong>{status ? `${status.rows.toLocaleString("en-US")} × ${status.dimensions}` : "—"}</strong></div>
      <div className="system-row"><span>Cache budget (imposed)</span><strong>{status ? `${status.cache_capacity.toLocaleString("en-US")} rows` : "—"}</strong></div>
      <div className="system-row"><span>Resident</span><strong>{status ? `${status.cache_occupancy.toLocaleString("en-US")} / ${status.cache_capacity.toLocaleString("en-US")}` : "—"}</strong></div>
      <div className="system-row"><span>Policy mode</span><strong>{status?.policy_state?.mode ?? "—"}{status?.policy_state?.shift_score != null ? ` · TV ${status.policy_state.shift_score.toFixed(2)}` : ""}</strong></div>
      <div className="system-row"><span>Device tensors</span><strong>{deviceBytes == null ? "none (CPU)" : `${(deviceBytes / 1048576).toFixed(1)} MiB`}</strong></div>
    </div>
  );
}
