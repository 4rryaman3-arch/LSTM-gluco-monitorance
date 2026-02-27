"use client";

import { useMemo, useRef, useState } from "react";

type ForecastPoint = {
  timestamp: string;
  cgm: number;
  status: "hypo" | "euglycemia" | "hyper";
  insulin_effect: number;
};

type ForecastPayload = {
  request_id: string;
  input: {
    timestamp: string;
    current_glucose: number;
    horizon_hours: number;
    step_minutes: number;
    insulin_units: number | null;
    include_without_insulin: boolean;
  };
  with_insulin: ForecastPoint[] | null;
  without_insulin: ForecastPoint[] | null;
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

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";
const WS_BASE =
  process.env.NEXT_PUBLIC_WS_BASE_URL ??
  API_BASE.replace("https://", "wss://").replace("http://", "ws://");

function toSvgPath(values: number[], width: number, height: number): string {
  if (values.length === 0) return "";
  const min = Math.min(...values) - 10;
  const max = Math.max(...values) + 10;
  const range = Math.max(1, max - min);
  return values
    .map((value, idx) => {
      const x = (idx / Math.max(1, values.length - 1)) * width;
      const y = height - ((value - min) / range) * height;
      return `${idx === 0 ? "M" : "L"} ${x.toFixed(2)} ${y.toFixed(2)}`;
    })
    .join(" ");
}

function toSvgPathScaled(
  values: number[],
  width: number,
  height: number,
  min: number,
  max: number
): string {
  if (values.length === 0) return "";
  const range = Math.max(1, max - min);
  return values
    .map((value, idx) => {
      const x = (idx / Math.max(1, values.length - 1)) * width;
      const y = height - ((value - min) / range) * height;
      return `${idx === 0 ? "M" : "L"} ${x.toFixed(2)} ${y.toFixed(2)}`;
    })
    .join(" ");
}

export default function Page() {
  const now = new Date();
  now.setMinutes(now.getMinutes() - now.getTimezoneOffset());
  const defaultTs = now.toISOString().slice(0, 16);

  const [timestamp, setTimestamp] = useState(defaultTs);
  const [currentGlucose, setCurrentGlucose] = useState(140);
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

  const chartData = useMemo(() => {
    return {
      without: result?.without_insulin ?? [],
      withInsulin: result?.with_insulin ?? []
    };
  }, [result]);

  const connectSocketIfNeeded = () => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) return wsRef.current;
    const ws = new WebSocket(`${WS_BASE}/ws/forecast`);
    ws.onopen = () => setSocketState("connected");
    ws.onclose = () => setSocketState("closed");
    ws.onerror = () => setSocketState("error");
    ws.onmessage = (event) => {
      const msg = JSON.parse(event.data);
      if (msg.event === "tick") {
        setTicks((prev) => [...prev, msg as WsTick]);
      }
    };
    wsRef.current = ws;
    setSocketState("connecting");
    return ws;
  };

  const submit = async () => {
    setLoading(true);
    setTicks([]);
    const payload = {
      timestamp: new Date(timestamp).toISOString(),
      current_glucose: currentGlucose,
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
      if (ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify(payload));
      }

      const response = await fetch(`${API_BASE}/api/v1/forecast`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      if (!response.ok) throw new Error(`Forecast failed (${response.status})`);
      const data = (await response.json()) as ForecastPayload;
      setResult(data);
    } catch (error) {
      console.error(error);
      alert("Forecast failed. Check backend URL or payload.");
    } finally {
      setLoading(false);
    }
  };

  const withoutValues = chartData.without.map((p) => p.cgm);
  const withValues = chartData.withInsulin.map((p) => p.cgm);
  const allValues = [...withoutValues, ...withValues];
  const chartMin = allValues.length ? Math.min(...allValues) - 10 : 40;
  const chartMax = allValues.length ? Math.max(...allValues) + 10 : 220;
  const maxPoints = Math.max(withoutValues.length, withValues.length);
  const activeIndex = hoveredIndex === null ? null : Math.min(maxPoints - 1, hoveredIndex);
  const activeWithout = activeIndex === null ? null : (chartData.without[activeIndex] ?? null);
  const activeWith = activeIndex === null ? null : (chartData.withInsulin[activeIndex] ?? null);
  const width = 820;
  const height = 320;

  const yFromValue = (value: number) => {
    const range = Math.max(1, chartMax - chartMin);
    return height - ((value - chartMin) / range) * height;
  };

  const xFromIndex = (index: number, count: number) =>
    (index / Math.max(1, count - 1)) * width;

  const onChartMouseMove = (event: React.MouseEvent<SVGSVGElement>) => {
    if (maxPoints < 2) return;
    const rect = event.currentTarget.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const ratio = Math.max(0, Math.min(1, x / rect.width));
    const idx = Math.round(ratio * (maxPoints - 1));
    setHoveredIndex(idx);
  };

  return (
    <main className="page">
      <section className="hero">
        <div>
          <h1>Realtime Gluco Projection</h1>
          <p>
            Choose time and current sugar level, then compare next-hour CGM trajectory with
            and without insulin dosage from the backend forecast engine.
          </p>
        </div>
        <div className="pill">WebSocket: {socketState}</div>
      </section>

      <section className="grid">
        <article className="card">
          <h2>Forecast Inputs</h2>
          <div className="field">
            <label>Timestamp</label>
            <input
              type="datetime-local"
              value={timestamp}
              onChange={(e) => setTimestamp(e.target.value)}
            />
          </div>

          <div className="field">
            <label>Current Sugar (mg/dL)</label>
            <input
              type="number"
              min={40}
              max={500}
              value={currentGlucose}
              onChange={(e) => setCurrentGlucose(Number(e.target.value))}
            />
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
            <label htmlFor="insulin">Include insulin dosage scenario</label>
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
            {loading ? "Forecasting..." : "Map Next Successive Hours"}
          </button>
          <p className="status">
            {result
              ? `Model: ${result.model.model_name} | Active: ${String(result.model.model_active)}`
              : "No forecast yet"}
          </p>
          {result && (
            <div className="reco">
              <p>
                Optimal insulin (model sweep): <strong>{result.recommended_insulin.units} U</strong>
              </p>
              <p>
                Expected TIR: <strong>{result.recommended_insulin.expected_time_in_range_pct}%</strong> | Range:{" "}
                <strong>
                  {result.recommended_insulin.expected_min_cgm}-{result.recommended_insulin.expected_max_cgm} mg/dL
                </strong>
              </p>
              <button
                className="btn btn-secondary"
                onClick={() => {
                  setUseInsulin(true);
                  setInsulinUnits(result.recommended_insulin.units);
                }}
              >
                Use Recommended Dose
              </button>
            </div>
          )}
        </article>

        <article className="card chart-shell">
          <h2>Projected CGM Curve</h2>
          <div className="legend">
            <span>
              <span className="dot without" /> Without insulin
            </span>
            <span>
              <span className="dot with" /> With insulin
            </span>
            <span className="hint">Realtime ticks: {ticks.length}</span>
          </div>

          <svg
            viewBox={`0 0 ${width} ${height}`}
            width="100%"
            height="360"
            role="img"
            onMouseMove={onChartMouseMove}
            onMouseLeave={() => setHoveredIndex(null)}
          >
            <rect x={0} y={0} width={width} height={height} fill="#f8fcfa" stroke="#d2e4de" />
            <line x1={0} y1={height * 0.75} x2={width} y2={height * 0.75} stroke="#e9d1bf" />
            <line x1={0} y1={height * 0.42} x2={width} y2={height * 0.42} stroke="#d2e4de" />
            {withoutValues.length > 1 && (
              <path
                d={toSvgPathScaled(withoutValues, width, height, chartMin, chartMax)}
                fill="none"
                stroke="#1a9b8b"
                strokeWidth={3}
              />
            )}
            {withValues.length > 1 && (
              <path
                d={toSvgPathScaled(withValues, width, height, chartMin, chartMax)}
                fill="none"
                stroke="#ef8d3c"
                strokeWidth={3}
              />
            )}
            {activeIndex !== null && (
              <line
                x1={xFromIndex(activeIndex, maxPoints)}
                y1={0}
                x2={xFromIndex(activeIndex, maxPoints)}
                y2={height}
                stroke="#6a7c83"
                strokeDasharray="4 4"
              />
            )}
            {activeWithout && (
              <circle
                cx={xFromIndex(activeIndex ?? 0, maxPoints)}
                cy={yFromValue(activeWithout.cgm)}
                r={4}
                fill="#1a9b8b"
              />
            )}
            {activeWith && (
              <circle
                cx={xFromIndex(activeIndex ?? 0, maxPoints)}
                cy={yFromValue(activeWith.cgm)}
                r={4}
                fill="#ef8d3c"
              />
            )}
          </svg>
          {activeIndex !== null && (
            <div className="hovercard">
              <strong>
                Interval: T+{(activeIndex + 1) * stepMinutes} min
              </strong>
              <span>
                Time: {new Date((activeWithout ?? activeWith)?.timestamp ?? timestamp).toLocaleString()}
              </span>
              <span>
                No insulin: {activeWithout ? `${activeWithout.cgm} mg/dL` : "-"} | With insulin:{" "}
                {activeWith ? `${activeWith.cgm} mg/dL` : "-"}
              </span>
            </div>
          )}

          <ul className="feed">
            {ticks.slice(-8).map((tick) => (
              <li key={`${tick.index}-${tick.without_insulin?.timestamp ?? "none"}`}>
                Step {tick.index + 1}: no-insulin {tick.without_insulin?.cgm ?? "-"} | insulin{" "}
                {tick.with_insulin?.cgm ?? "-"}
              </li>
            ))}
          </ul>
        </article>
      </section>
    </main>
  );
}
