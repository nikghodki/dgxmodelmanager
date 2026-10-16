export interface FleetKpis {
  genTps: number | null; genSeries: (number | null)[];
  ttftP50: number | null; ttftSeries: (number | null)[];
  kv: number | null; waiting: number | null; running: number | null;
}

const num = (v: unknown): number | null => (typeof v === "number" && Number.isFinite(v) ? v : null);
const agg = (vals: (number | null)[], how: "sum" | "max") => {
  const xs = vals.filter((v): v is number => v != null);
  if (!xs.length) return null;
  return how === "sum" ? xs.reduce((a, b) => a + b, 0) : Math.max(...xs);
};

/** Aggregate live samples (one array per ready deployment, oldest→newest) into dashboard KPIs. */
export function fleetKpis(samples: any[][], points = 30): FleetKpis {
  const lasts = samples.map((s) => s[s.length - 1]).filter(Boolean);
  const pick = (k: string) => lasts.map((x) => num(x[k]));
  const series = (k: string, how: "sum" | "max") => Array.from({ length: points }, (_, i) =>
    agg(samples.map((s) => num(s[s.length - points + i]?.[k])), how));
  return {
    genTps: agg(pick("gen_tps"), "sum"), genSeries: series("gen_tps", "sum"),
    ttftP50: agg(pick("ttft_p50"), "max"), ttftSeries: series("ttft_p50", "max"),
    kv: agg(pick("kv_usage"), "max"), waiting: agg(pick("waiting"), "sum"), running: agg(pick("running"), "sum"),
  };
}
