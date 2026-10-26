import { useMemo, useState } from "react";
import { api, type RecommendResponse, type Status } from "../api/client";
import { ms, parseHistory } from "../lib/format";

const profiles: Record<string, number[]> = {
  "Late-night listener": [4, 15, 27, 33, 49, 61, 88],
  "Analog explorer": [12, 18, 39, 54, 72, 101, 121],
  "Weekend mixer": [3, 23, 41, 66, 78, 93, 107],
};

type Props = { status: Status | null; connected: boolean; onDone: () => void; onMessage: (message: string) => void };

export function MusicDemo({ status, connected, onDone, onMessage }: Props) {
  const [history, setHistory] = useState(profiles["Late-night listener"].join(", "));
  const [result, setResult] = useState<RecommendResponse | null>(null);
  const [selected, setSelected] = useState<number | null>(null);
  const [busy, setBusy] = useState(false);
  const parsed = useMemo(() => parseHistory(history, status?.rows, status?.limits?.max_history_ids ?? 256), [history, status]);

  async function recommend() {
    if (parsed.error) return;
    setBusy(true);
    try {
      setResult(await api<RecommendResponse>("/api/v1/recommend", { method: "POST", body: JSON.stringify({ history_ids: parsed.ids, top_k: 8, exclude_seen: true }) }));
      setSelected(null);
    } catch (error) {
      onMessage(error instanceof Error ? error.message : "Recommendation request failed");
    } finally {
      setBusy(false);
      onDone();
    }
  }

  const detail = result?.items.find((item) => item.item_id === selected);
  return (
    <section className="panel demo-panel">
      <div className="panel-head"><div><p className="eyebrow">MUSIC DISCOVERY</p><h2>Try a listener history</h2></div><span className="synthetic-tag">FICTIONAL CATALOG</span></div>
      <label className="field-label" htmlFor="listener">Listener profile</label>
      <select id="listener" onChange={(event) => setHistory(profiles[event.target.value].join(", "))}>
        {Object.keys(profiles).map((name) => <option key={name}>{name}</option>)}
      </select>
      <label className="field-label" htmlFor="history">Item IDs <span>comma or space separated · max {status?.limits?.max_history_ids ?? 256}</span></label>
      <textarea id="history" value={history} onChange={(event) => setHistory(event.target.value)} rows={3} aria-invalid={parsed.error != null} aria-describedby="history-help" />
      <p id="history-help" className={parsed.error ? "field-error" : "field-help"} role={parsed.error ? "alert" : undefined}>
        {parsed.error ?? "Each ID is a row in the embedding table. The engine looks these rows up through the GPU cache (or the CPU model) and averages them into a listener vector."}
      </p>
      <button className="primary-button" disabled={busy || !connected || parsed.error != null || !status?.healthy} onClick={() => void recommend()}>{busy ? "Working…" : "Find recommendations"}<span>↗</span></button>
      <div className="recommendations">
        {result ? <>
          <p className="result-meta">{result.ranking === "popularity_fallback" ? "Empty history: deterministic popularity fallback" : "Ranked by similarity to the pooled history vector"} · {result.returned} items · queue {ms(result.timing_ms.queue)} · batch {ms(result.timing_ms.batch_engine)} · scoring {ms(result.timing_ms.scoring)}</p>
          {result.items.map((item) => (
            <button type="button" className={`song ${selected === item.item_id ? "selected" : ""}`} key={item.item_id} onClick={() => setSelected(item.item_id)} aria-pressed={selected === item.item_id}>
              <span className="song-index">{String(item.item_id).padStart(3, "0")}</span>
              <span className="song-text"><strong>{item.title}</strong><small>{item.artist} · {item.genre}</small></span>
              <span className="score">{item.score.toFixed(3)}</span>
            </button>
          ))}
          {detail && <dl className="item-detail" aria-label="Selected item metadata"><dt>Item</dt><dd>{detail.item_id}</dd><dt>Artist</dt><dd>{detail.artist}</dd><dt>Genre</dt><dd>{detail.genre}</dd><dt>Region</dt><dd>{detail.region}</dd><dt>Score</dt><dd>{detail.score.toFixed(4)}</dd></dl>}
        </> : <div className="empty-state"><span className="empty-icon">♫</span><p>Your recommendations will appear here.</p><small>Catalog, listeners, and genre similarity are synthetic; rankings show serving behavior, not recommendation quality.</small></div>}
      </div>
    </section>
  );
}
