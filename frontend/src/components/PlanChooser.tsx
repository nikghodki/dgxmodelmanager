import type { PlanAction, PlanOption } from "../types";
import { fmtBytes } from "../format";
import { Badge } from "./ui";

export function PlanChooser({ plans, selected, onSelect }: { plans: PlanOption[]; selected: PlanAction[]; onSelect: (p: PlanOption) => void }) {
  if (!plans.length) return <p className="mt-3 text-sm text-slate-500">No automatic plan frees enough memory. Stop something manually or pick a smaller configuration.</p>;
  const key = (a: PlanAction[]) => JSON.stringify(a);
  return (
    <div className="mt-4 space-y-2">
      <h3 className="text-sm font-semibold">Ways to make it fit</h3>
      {plans.map((p) => (
        <button key={p.id} onClick={() => onSelect(p)}
          className={`block w-full rounded-md border px-3 py-2 text-left text-sm hover:bg-slate-50 ${key(p.actions.filter((a) => a.type !== "patch")) === key(selected) && selected.length ? "border-slate-900" : "border-slate-200"}`}>
          <div className="flex items-center justify-between">
            <span className="font-medium">{p.title}</span>
            {p.fits ? <Badge tone="green">fits</Badge> : <Badge tone="amber">still short</Badge>}
          </div>
          <div className="text-xs text-slate-600">{p.detail}</div>
          <div className="text-xs text-slate-500">needs {fmtBytes(p.new_required)}{p.frees ? ` · frees ${fmtBytes(p.frees)}` : ""}</div>
        </button>
      ))}
    </div>
  );
}
