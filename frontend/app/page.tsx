"use client";

import { useMemo, useRef, useState } from "react";

type ForecastPoint = {
  timestamp: string;
  cgm: number;
  status: "hypo" | "euglycemia" | "hyper";
  insulin_effect: number;
};

type ForecastEvent = {
  timestamp: string;
  event_type: "hypo_risk" | "hyper_risk" | "rapid_drop" | "rapid_rise";
  severity: "low" | "medium" | "high";
  message: string;
};

type ForecastPayload = {
  request_id: string;
  input: {
    history_points: { timestamp: string; glucose: number }[];
    horizon_hours: number;
    step_minutes: number;
    insulin_units: number | null;
    include_without_insulin: boolean;
  };
  with_insulin: ForecastPoint[] | null;
  without_insulin: ForecastPoint[] | null;
  with_insulin_events: ForecastEvent[];
  without_insulin_events: ForecastEvent[];
  recommended_insulin: {
    units: number;
    expected_time_in_range_pct: number;
    expected_min_cgm: number;
    expected_max_cgm: number;
    objective_score: number;
    note: string;
  };
  model: {
    model_active: boolean;
    model_name: string;
    notes: string;
  };
};

type WsTick = {
  event: "tick";
  index: number;
  with_insulin: ForecastPoint | null;
  without_insulin: ForecastPoint | null;
};

type HistoryRow = {
  id: string;
  timestampLocal: string;
  glucose: number;
};

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";
const WS_BASE =
  process.env.NEXT_PUBLIC_WS_BASE_URL ??
  API_BASE.replace("https://", "wss://").replace("http://", "ws://");

function formatLocalInput(date: Date): string {
  const copy = new Date(date);
  copy.setMinutes(copy.getMinutes() - copy.getTimezoneOffset());
  return copy.toISOString().slice(0, 16);
}

function buildDefaultSeries(): HistoryRow[] {
  const now = new Date();
  const values = [158, 153, 149, 145, 142, 140];
  const rows: HistoryRow[] = [];
  for (let i = 0; i < values.length; i += 1) {
    const d = new Date(now.getTime() - (values.length - 1 - i) * 5 * 60 * 1000);
    rows.push({
      id: `h-${i}`,
      timestampLocal: formatLocalInput(d),
      glucose: values[i]
    });
  }
  return rows;
}

function yFor(value: number, min: number, max: number, height: number): number {
  const range = Math.max(1, max - min);
  return height - ((value - min) / range) * height;
}

function buildPath(
  values: number[],
  startX: number,
  span: number,
  height: number,
  min: number,
  max: number
): string {
  if (values.length < 2) return "";
  return values
    .map((v, idx) => {
      const x = startX + (idx / Math.max(1, values.length - 1)) * span;
      const y = yFor(v, min, max, height);
      return `${idx === 0 ? "M" : "L"} ${x.toFixed(2)} ${y.toFixed(2)}`;
    })
    .join(" ");
}

export default function Page() {
  const [historyRows, setHistoryRows] = useState<HistoryRow[]>(buildDefaultSeries());
  const [horizonHours, setHorizonHours] = useState(6);
  const [stepMinutes, setStepMinutes] = useState(5);
  const [useInsulin, setUseInsulin] = useState(true);
  const [insulinUnits, setInsulinUnits] = useState(2);
  const [loading, setLoading] = useState(false);
  const [socketState, setSocketState] = useState("idle");
  const [result, setResult] = useState<ForecastPayload | null>(null);
  const [ticks, setTicks] = useState<WsTick[]>([]);
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null);

  const wsRef = useRef<WebSocket | null>(null);

  const sortedHistory = useMemo(
    () =>
      [...historyRows].sort(
        (a, b) => new Date(a.timestampLocal).getTime() - new Date(b.timestampLocal).getTime()
      ),
    [historyRows]
  );

  const without = result?.without_insulin ?? [];
  const withInsulin = result?.with_insulin ?? [];
  const historyValues = sortedHistory.map((p) => p.glucose);
  const withoutValues = without.map((p) => p.cgm);
  const withValues = withInsulin.map((p) => p.cgm);
  const allValues = [...historyValues, ...withoutValues, ...withValues];
  const minY = allValues.length ? Math.min(...allValues) - 10 : 40;
  const maxY = allValues.length ? Math.max(...allValues) + 10 : 220;

  const chartWidth = 860;
  const chartHeight = 330;
  const historyWidth = 300;
  const futureWidth = chartWidth - historyWidth;
  const futureCount = Math.max(withoutValues.length, withValues.length);
  const activeIndex = hoveredIndex === null ? null : Math.min(futureCount - 1, hoveredIndex);
  const activeNo = activeIndex === null ? null : (without[activeIndex] ?? null);
  const activeWith = activeIndex === null ? null : (withInsulin[activeIndex] ?? null);

  const connectSocketIfNeeded = () => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) return wsRef.current;
    const ws = new WebSocket(`${WS_BASE}/ws/forecast`);
    ws.onopen = () => setSocketState("connected");
    ws.onclose = () => setSocketState("closed");
    ws.onerror = () => setSocketState("error");
    ws.onmessage = (event) => {
      const msg = JSON.parse(event.data);
      if (msg.event === "tick") setTicks((prev) => [...prev, msg as WsTick]);
    };
    wsRef.current = ws;
    setSocketState("connecting");
    return ws;
  };

  const addRow = () => {
    const last = sortedHistory[sortedHistory.length - 1];
    const lastDate = new Date(last.timestampLocal);
    const nextDate = new Date(lastDate.getTime() + stepMinutes * 60 * 1000);
    setHistoryRows((prev) => [
      ...prev,
      {
        id: `h-${Date.now()}`,
        timestampLocal: formatLocalInput(nextDate),
        glucose: last.glucose
      }
    ]);
  };

  const removeRow = (id: string) => {
    if (historyRows.length <= 3) return;
    setHistoryRows((prev) => prev.filter((r) => r.id !== id));
  };

  const patchRow = (id: string, patch: Partial<HistoryRow>) => {
    setHistoryRows((prev) => prev.map((r) => (r.id === id ? { ...r, ...patch } : r)));
  };

  const submit = async () => {
    if (sortedHistory.length < 3) {
      alert("Add at least 3 glucose points.");
      return;
    }
    setLoading(true);
    setTicks([]);

    const payload = {
      history_points: sortedHistory.map((p) => ({
        timestamp: new Date(p.timestampLocal).toISOString(),
        glucose: Number(p.glucose)
      })),
      horizon_hours: horizonHours,
      step_minutes: stepMinutes,
      insulin_units: useInsulin ? insulinUnits : null,
      include_without_insulin: true
    };

    try {
      const ws = connectSocketIfNeeded();
      ws.onopen = () => {
        setSocketState("connected");
        ws.send(JSON.stringify(payload));
      };
      if (ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(payload));

      const response = await fetch(`${API_BASE}/api/v1/forecast`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      if (!response.ok) throw new Error(`Forecast failed (${response.status})`);
      setResult((await response.json()) as ForecastPayload);
    } catch (error) {
      console.error(error);
      alert("Forecast failed. Check API health and request values.");
    } finally {
      setLoading(false);
    }
  };

  const onChartMouseMove = (event: React.MouseEvent<SVGSVGElement>) => {
    if (futureCount < 2) return;
    const rect = event.currentTarget.getBoundingClientRect();
    const x = ((event.clientX - rect.left) / rect.width) * chartWidth;
    if (x <= historyWidth) {
      setHoveredIndex(null);
      return;
    }
    const ratio = (x - historyWidth) / futureWidth;
    const idx = Math.round(Math.max(0, Math.min(1, ratio)) * (futureCount - 1));
    setHoveredIndex(idx);
  };

  return (
    <main className="page">
      <section className="hero">
        <div>
          <h1>Series-Based Glucose Forecast</h1>
          <p>
            Enter a timestamped glucose series, then predict next events and optimal insulin dosage.
            Hover the curve to inspect minute intervals.
          </p>
        </div>
        <div className="pill">WebSocket: {socketState}</div>
      </section>

      <section className="grid">
        <article className="card">
          <h2>Input Series</h2>
          <div className="series-head">
            <button className="mini-btn" onClick={addRow}>
              + Add Point
            </button>
            <span className="hint">Minimum 3 points required</span>
          </div>
          <div className="series-table-wrap">
            <table className="series-table">
              <thead>
                <tr>
                  <th>Time</th>
                  <th>Glucose</th>
                  <th />
                </tr>
              </thead>
              <tbody>
                {historyRows.map((row) => (
                  <tr key={row.id}>
                    <td>
                      <input
                        type="datetime-local"
                        value={row.timestampLocal}
                        onChange={(e) => patchRow(row.id, { timestampLocal: e.target.value })}
                      />
                    </td>
                    <td>
                      <input
                        type="number"
                        min={40}
                        max={500}
                        value={row.glucose}
                        onChange={(e) => patchRow(row.id, { glucose: Number(e.target.value) })}
                      />
                    </td>
                    <td>
                      <button className="danger-btn" onClick={() => removeRow(row.id)}>
                        x
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="row">
            <div className="field">
              <label>Horizon (hours)</label>
              <input
                type="number"
                min={1}
                max={24}
                value={horizonHours}
                onChange={(e) => setHorizonHours(Number(e.target.value))}
              />
            </div>
            <div className="field">
              <label>Step (minutes)</label>
              <input
                type="number"
                min={5}
                max={60}
                step={5}
                value={stepMinutes}
                onChange={(e) => setStepMinutes(Number(e.target.value))}
              />
            </div>
          </div>

          <div className="switch">
            <input
              id="insulin"
              type="checkbox"
              checked={useInsulin}
              onChange={(e) => setUseInsulin(e.target.checked)}
            />
            <label htmlFor="insulin">Compare with insulin scenario</label>
          </div>
          {useInsulin && (
            <div className="field">
              <label>Insulin Units</label>
              <input
                type="number"
                min={0}
                max={50}
                step={0.1}
                value={insulinUnits}
                onChange={(e) => setInsulinUnits(Number(e.target.value))}
              />
            </div>
          )}

          <button className="btn" onClick={submit} disabled={loading}>
            {loading ? "Running Forecast..." : "Predict Next Events + Dosage"}
          </button>
          {result && (
            <div className="reco">
              <p>
                Recommended insulin: <strong>{result.recommended_insulin.units} U</strong>
              </p>
              <p>
                Expected TIR: <strong>{result.recommended_insulin.expected_time_in_range_pct}%</strong>
              </p>
              <p className="hint">{result.recommended_insulin.note}</p>
              <button
                className="btn btn-secondary"
                onClick={() => {
                  setUseInsulin(true);
                  setInsulinUnits(result.recommended_insulin.units);
                }}
              >
                Apply Recommended Dose
              </button>
            </div>
          )}
        </article>

        <article className="card chart-shell">
          <h2>History + Forecast Timeline</h2>
          <div className="legend">
            <span>
              <span className="dot history" /> Input history
            </span>
            <span>
              <span className="dot without" /> Future (no insulin)
            </span>
            <span>
              <span className="dot with" /> Future (with insulin)
            </span>
            <span className="hint">Realtime ticks: {ticks.length}</span>
          </div>

          <svg
            viewBox={`0 0 ${chartWidth} ${chartHeight}`}
            width="100%"
            height="360"
            onMouseMove={onChartMouseMove}
            onMouseLeave={() => setHoveredIndex(null)}
          >
            <rect x={0} y={0} width={chartWidth} height={chartHeight} fill="#f8fcfa" stroke="#d2e4de" />
            <line x1={historyWidth} y1={0} x2={historyWidth} y2={chartHeight} stroke="#cad8d4" strokeDasharray="5 6" />
            <line x1={0} y1={chartHeight * 0.74} x2={chartWidth} y2={chartHeight * 0.74} stroke="#ead4c4" />
            <line x1={0} y1={chartHeight * 0.42} x2={chartWidth} y2={chartHeight * 0.42} stroke="#d8e5df" />

            {historyValues.length > 1 && (
              <path
                d={buildPath(historyValues, 0, historyWidth, chartHeight, minY, maxY)}
                fill="none"
                stroke="#6d7f88"
                strokeWidth={2.5}
              />
            )}
            {withoutValues.length > 1 && (
              <path
                d={buildPath(withoutValues, historyWidth, futureWidth, chartHeight, minY, maxY)}
                fill="none"
                stroke="#1a9b8b"
                strokeWidth={3}
              />
            )}
            {withValues.length > 1 && (
              <path
                d={buildPath(withValues, historyWidth, futureWidth, chartHeight, minY, maxY)}
                fill="none"
                stroke="#ef8d3c"
                strokeWidth={3}
              />
            )}
            {activeIndex !== null && (
              <line
                x1={historyWidth + (activeIndex / Math.max(1, futureCount - 1)) * futureWidth}
                y1={0}
                x2={historyWidth + (activeIndex / Math.max(1, futureCount - 1)) * futureWidth}
                y2={chartHeight}
                stroke="#5d747e"
                strokeDasharray="4 4"
              />
            )}
          </svg>

          {activeIndex !== null && (
            <div className="hovercard">
              <strong>Interval: T+{(activeIndex + 1) * stepMinutes} min</strong>
              <span>
                Time: {new Date((activeNo ?? activeWith)?.timestamp ?? new Date().toISOString()).toLocaleString()}
              </span>
              <span>
                No insulin: {activeNo ? `${activeNo.cgm} mg/dL` : "-"} | With insulin:{" "}
                {activeWith ? `${activeWith.cgm} mg/dL` : "-"}
              </span>
            </div>
          )}

          <div className="event-grid">
            <div className="event-card">
              <h3>Events Without Insulin</h3>
              <ul className="feed">
                {result?.without_insulin_events.length ? (
                  result.without_insulin_events.slice(0, 8).map((evt, i) => (
                    <li key={`n-${i}`}>
                      [{evt.severity}] {new Date(evt.timestamp).toLocaleTimeString()} - {evt.message}
                    </li>
                  ))
                ) : (
                  <li>No events detected.</li>
                )}
              </ul>
            </div>
            <div className="event-card">
              <h3>Events With Insulin</h3>
              <ul className="feed">
                {result?.with_insulin_events.length ? (
                  result.with_insulin_events.slice(0, 8).map((evt, i) => (
                    <li key={`w-${i}`}>
                      [{evt.severity}] {new Date(evt.timestamp).toLocaleTimeString()} - {evt.message}
                    </li>
                  ))
                ) : (
                  <li>No events detected.</li>
                )}
              </ul>
            </div>
          </div>
        </article>
      </section>
    </main>
  );
}
