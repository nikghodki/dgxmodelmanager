import type { ReactNode } from "react";
import { Card } from "@/components/ui/card";

export function StatTile({ label, value, sub, children }: { label: string; value: ReactNode; sub?: ReactNode; children?: ReactNode }) {
  return (
    <Card className="shadow-card p-4">
      <div className="text-[11px] font-medium uppercase tracking-wider text-muted-foreground">{label}</div>
      <div className="mt-1 text-2xl font-semibold tabular-nums tracking-tight">{value}</div>
      {sub && <div className="mt-0.5 text-xs text-muted-foreground">{sub}</div>}
      {children && <div className="mt-3">{children}</div>}
    </Card>
  );
}

/** Minimal 2px line, no axes. `values` oldest→newest; nulls are gaps. */
export function Sparkline({ values, label, color = "var(--chart-1)" }: { values: (number | null)[]; label: string; color?: string }) {
  const nums = values.filter((v): v is number => v != null && Number.isFinite(v));
  if (nums.length < 2) return <div className="h-6" aria-hidden />;
  const max = Math.max(...nums), min = Math.min(...nums), span = max - min || 1;
  const step = 100 / (values.length - 1);
  const parts: string[] = [];
  values.forEach((v, i) => {
    if (v == null || !Number.isFinite(v)) return;
    parts.push(`${(i * step).toFixed(2)},${(22 - ((v - min) / span) * 20).toFixed(2)}`);
  });
  return (
    <svg viewBox="0 0 100 24" preserveAspectRatio="none" className="h-6 w-full" role="img" aria-label={label}>
      <title>{label}</title>
      <polyline fill="none" stroke={color} strokeWidth={2} vectorEffect="non-scaling-stroke" strokeLinejoin="round" points={parts.join(" ")} />
    </svg>
  );
}
