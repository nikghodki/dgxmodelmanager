import type { Budget } from "../types";
import { fmtBytes } from "../format";

const COLORS = ["bg-sky-500", "bg-indigo-500", "bg-teal-500", "bg-fuchsia-500", "bg-cyan-600", "bg-blue-700"];

export function MemoryBar({ budget }: { budget: Budget }) {
  const t = budget.total || 1;
  const managed = budget.deployments.filter((d) => ["starting", "loading", "ready", "sleeping", "unknown", "stopping"].includes(d.state));
  const segs = [
    ...managed.map((d, i) => ({
      label: `${d.name} (${d.state})`,
      bytes: d.state === "sleeping" ? d.actual : Math.max(d.reserved, d.actual),
      cls: d.state === "sleeping" ? "bg-violet-400" : COLORS[i % COLORS.length],
    })),
    { label: "External processes", bytes: budget.external, cls: "bg-amber-400" },
    { label: "Safety margin + OS reserve", bytes: budget.margin + budget.os_reserve, cls: "bg-slate-400" },
    { label: "Free for new deployments", bytes: Math.max(budget.free_for_new, 0), cls: "bg-emerald-200" },
  ];
  return (
    <div>
      <div className="flex h-6 w-full overflow-hidden rounded-md border border-slate-200" role="img" aria-label="Memory usage">
        {segs.map((s) => (
          <div key={s.label} className={s.cls} style={{ width: `${(s.bytes / t) * 100}%` }} title={`${s.label}: ${fmtBytes(s.bytes)}`} />
        ))}
      </div>
      <ul className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-xs text-slate-600">
        {segs.map((s) => (
          <li key={s.label} className="flex items-center gap-1.5">
            <span className={`inline-block h-2.5 w-2.5 rounded-sm ${s.cls}`} />
            {s.label}: <span className="tabular-nums">{fmtBytes(s.bytes)}</span>
          </li>
        ))}
      </ul>
      {budget.free_for_new < 0 && (
        <p className="mt-2 text-xs text-rose-700">Over-committed by {fmtBytes(-budget.free_for_new)}. New deployments are blocked.</p>
      )}
    </div>
  );
}
