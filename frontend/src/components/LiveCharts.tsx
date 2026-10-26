import { CartesianGrid, ComposedChart, Line, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis, Bar } from "recharts";

export type Point = { time: string; hit: number | null; p95: number | null; p99: number | null; churn: number };
export type PhaseMarker = { time: string; label: string };

const axis = { fill: "#8a939e", fontSize: 11 };
const tooltip = { contentStyle: { background: "#171b21", border: "1px solid #303741", borderRadius: 10 }, labelStyle: { color: "#cbd2db" } };

export function LiveCharts({ points, markers, windowSeconds }: { points: Point[]; markers: PhaseMarker[]; windowSeconds: number }) {
  return (
    <section className="panel chart-panel">
      <div className="panel-head">
        <div><p className="eyebrow">LIVE WINDOW · {windowSeconds}s ROLLING · 1 SAMPLE/s</p><h2>Cache behavior</h2></div>
        <span className="chart-legend"><i className="legend-hit" /> Hit rate <i className="legend-churn" /> Churn rows <i className="legend-latency" /> p95 ms <i className="legend-p99" /> p99 ms</span>
      </div>
      {points.length === 0 ? <div className="empty-state chart-empty"><p>No samples yet.</p><small>Send a request or start a scenario to populate the charts.</small></div> : <>
        <div className="chart-wrap" aria-label="Occurrence hit rate and cache churn over time">
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={points} syncId="live" margin={{ top: 12, right: 8, left: -10, bottom: 0 }}>
              <CartesianGrid stroke="#252b33" vertical={false} />
              <XAxis dataKey="time" tick={axis} axisLine={false} tickLine={false} minTickGap={32} />
              <YAxis yAxisId="hit" domain={[0, 1]} tickFormatter={(value) => `${Math.round(value * 100)}%`} tick={axis} axisLine={false} tickLine={false} label={{ value: "hit rate", angle: -90, position: "insideLeft", fill: "#8a939e", fontSize: 10 }} />
              <YAxis yAxisId="churn" orientation="right" tick={axis} axisLine={false} tickLine={false} label={{ value: "rows / window", angle: 90, position: "insideRight", fill: "#8a939e", fontSize: 10 }} />
              <Tooltip {...tooltip} />
              {markers.map((marker) => <ReferenceLine key={`${marker.time}-${marker.label}`} yAxisId="hit" x={marker.time} stroke="#f1c37d" strokeDasharray="4 4" label={{ value: marker.label, fill: "#f1c37d", fontSize: 10, position: "insideTopRight" }} />)}
              <Bar yAxisId="churn" dataKey="churn" name="Admissions + evictions (rows)" fill="#bd9dff55" />
              <Line yAxisId="hit" type="monotone" dataKey="hit" name="Occurrence hit rate" stroke="#b5ec66" dot={false} connectNulls isAnimationActive={false} />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
        <div className="chart-wrap short" aria-label="Batch latency percentiles over time">
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={points} syncId="live" margin={{ top: 8, right: 8, left: -10, bottom: 0 }}>
              <CartesianGrid stroke="#252b33" vertical={false} />
              <XAxis dataKey="time" tick={axis} axisLine={false} tickLine={false} minTickGap={32} />
              <YAxis tick={axis} axisLine={false} tickLine={false} label={{ value: "ms", angle: -90, position: "insideLeft", fill: "#8a939e", fontSize: 10 }} />
              <Tooltip {...tooltip} />
              {markers.map((marker) => <ReferenceLine key={`${marker.time}-${marker.label}`} x={marker.time} stroke="#f1c37d" strokeDasharray="4 4" />)}
              <Line type="monotone" dataKey="p95" name="Batch p95 (ms)" stroke="#7aa7ff" dot={false} connectNulls isAnimationActive={false} />
              <Line type="monotone" dataKey="p99" name="Batch p99 (ms)" stroke="#ffb56b" dot={false} connectNulls isAnimationActive={false} />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      </>}
      <div className="chart-foot"><span>Bounded to 600 points · dashed lines mark scenario phase changes</span><span>Latency is per batch (engine scope), not per request round trip</span></div>
    </section>
  );
}
