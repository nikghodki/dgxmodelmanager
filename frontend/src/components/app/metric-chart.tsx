import { useState, type ReactNode } from "react";
import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { BarChart3, Table2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { useChartColors } from "@/lib/chart-colors";
import { finiteSeries } from "@/logic";

export interface SeriesDef { key: string; label: string; slot: number }

const timeTick = (ts: number) => new Date(ts * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
const timeFull = (ts: number) => new Date(ts * 1000).toLocaleString();
const defaultFmt = (v: number) => v.toFixed(1);

export function MetricChart({ data, series, height = 200, fmt = defaultFmt, xKey = "ts", xFmt = timeTick }: {
  data: Record<string, any>[]; series: SeriesDef[]; height?: number; fmt?: (v: number) => string; xKey?: string; xFmt?: (v: any) => string;
}) {
  const c = useChartColors();
  if (!data.length) {
    return <div className="flex items-center justify-center rounded-md border border-dashed text-xs text-muted-foreground" style={{ height }}>No data yet</div>;
  }
  const clean = finiteSeries(data, series.map((s) => s.key));
  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={clean} margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
        <CartesianGrid stroke={c.grid} vertical={false} />
        <XAxis dataKey={xKey} tickFormatter={xFmt} tick={{ fontSize: 11, fill: c.axis }} axisLine={false} tickLine={false} minTickGap={48} />
        <YAxis tickFormatter={fmt} tick={{ fontSize: 11, fill: c.axis }} axisLine={false} tickLine={false} width={52} />
        <Tooltip
          cursor={{ stroke: c.axis, strokeDasharray: "3 3" }}
          contentStyle={{ background: c.tooltipBg, border: `1px solid ${c.tooltipBorder}`, borderRadius: 8, fontSize: 12, color: c.text }}
          labelFormatter={(v) => (xKey === "ts" ? timeFull(v as number) : String(v))}
          formatter={(v: number, name) => [Number.isFinite(v) ? fmt(v) : "—", name]}
        />
        {series.length >= 2 && <Legend iconType="plainline" wrapperStyle={{ fontSize: 12, color: c.axis }} />}
        {series.map((s) => (
          <Line key={s.key} type="monotone" dataKey={s.key} name={s.label} stroke={c.series[(s.slot - 1) % 8]} strokeWidth={2}
            dot={false} activeDot={{ r: 4, strokeWidth: 2, stroke: c.tooltipBg }} isAnimationActive={false} connectNulls />
        ))}
      </LineChart>
    </ResponsiveContainer>
  );
}

export function ChartCard({ title, description, actions, data, series, fmt = defaultFmt, height, xKey = "ts", xFmt }: {
  title: string; description?: string; actions?: ReactNode; data: Record<string, any>[]; series: SeriesDef[];
  fmt?: (v: number) => string; height?: number; xKey?: string; xFmt?: (v: any) => string;
}) {
  const [asTable, setAsTable] = useState(false);
  const clean = finiteSeries(data, series.map((s) => s.key));
  const xLabel = xKey === "ts" ? timeFull : (v: any) => String(v);
  return (
    <Card className="shadow-card">
      <CardHeader className="flex flex-row items-start justify-between gap-2 space-y-0 pb-2">
        <div>
          <CardTitle className="text-sm font-semibold">{title}</CardTitle>
          {description && <CardDescription className="text-xs">{description}</CardDescription>}
        </div>
        <div className="flex items-center gap-1">
          {actions}
          <Button variant="ghost" size="sm" className="h-7 text-xs" onClick={() => setAsTable(!asTable)} aria-pressed={asTable}>
            {asTable ? <BarChart3 className="mr-1 h-3.5 w-3.5" /> : <Table2 className="mr-1 h-3.5 w-3.5" />}{asTable ? "Chart" : "Table"}
          </Button>
        </div>
      </CardHeader>
      <CardContent>
        {asTable ? (
          <div className="max-h-72 overflow-auto">
            <table className="w-full text-xs">
              <thead className="sticky top-0 bg-card">
                <tr className="border-b text-left text-muted-foreground">
                  <th className="py-1.5 pr-3 font-medium">{xKey === "ts" ? "Time" : xKey}</th>
                  {series.map((s) => <th key={s.key} className="py-1.5 pr-3 text-right font-medium">{s.label}</th>)}
                </tr>
              </thead>
              <tbody>
                {clean.slice().reverse().map((row, i) => (
                  <tr key={i} className="border-b last:border-0">
                    <td className="py-1 pr-3 tabular-nums text-muted-foreground">{xLabel(row[xKey])}</td>
                    {series.map((s) => <td key={s.key} className="py-1 pr-3 text-right tabular-nums">{row[s.key] == null ? "—" : fmt(row[s.key])}</td>)}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <MetricChart data={data} series={series} fmt={fmt} height={height} xKey={xKey} xFmt={xFmt} />
        )}
      </CardContent>
    </Card>
  );
}
