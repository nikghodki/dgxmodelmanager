import { finiteSeries } from "../logic";
import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

export interface LineDef { key: string; label: string; color: string }
const tick = (ts: number) => new Date(ts * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });

export function MetricChart({ data, lines, height = 180, fmt = (v: number) => v.toFixed(1) }: {
  data: any[]; lines: LineDef[]; height?: number; fmt?: (v: number) => string;
}) {
  if (!data.length) return <div className="flex h-24 items-center justify-center text-xs text-slate-400">No data yet</div>;
  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={finiteSeries(data, lines.map((l) => l.key))} margin={{ top: 4, right: 8, bottom: 0, left: 0 }}>
        <CartesianGrid stroke="#e2e8f0" strokeDasharray="3 3" />
        <XAxis dataKey="ts" tickFormatter={tick} tick={{ fontSize: 10 }} minTickGap={40} />
        <YAxis tick={{ fontSize: 10 }} tickFormatter={fmt} width={48} />
        <Tooltip labelFormatter={(v) => new Date((v as number) * 1000).toLocaleString()} formatter={(v: number) => (Number.isFinite(v) ? fmt(v) : "—")} />
        <Legend wrapperStyle={{ fontSize: 11 }} />
        {lines.map((l) => <Line key={l.key} type="monotone" dataKey={l.key} name={l.label} stroke={l.color} dot={false} isAnimationActive={false} connectNulls />)}
      </LineChart>
    </ResponsiveContainer>
  );
}
