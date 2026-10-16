import { CheckCircle2, CircleDashed } from "lucide-react";
import { fmtBytes } from "@/format";
import { cn } from "@/lib/utils";
import type { PlanOption } from "@/types";

export function PlanChooser({ plans, selectedId, onSelect }: { plans: PlanOption[]; selectedId: string | null; onSelect: (p: PlanOption) => void }) {
  if (!plans.length) return <p className="text-sm text-muted-foreground">No automatic plan frees enough memory. Stop something manually or choose a smaller configuration.</p>;
  return (
    <div className="space-y-2" role="radiogroup" aria-label="Ways to make it fit">
      <h3 className="text-sm font-semibold">Ways to make it fit</h3>
      {plans.map((p) => {
        const sel = selectedId === p.id;
        return (
          <button key={p.id} type="button" role="radio" aria-checked={sel} onClick={() => onSelect(p)}
            className={cn("block w-full rounded-lg border p-3 text-left transition-colors hover:bg-accent",
              sel ? "border-primary ring-1 ring-primary" : "border-border")}>
            <div className="flex items-center justify-between gap-2">
              <span className="text-sm font-medium">{p.title}</span>
              {p.fits
                ? <span className="inline-flex items-center gap-1 text-xs text-status-good"><CheckCircle2 className="h-3.5 w-3.5" />fits</span>
                : <span className="inline-flex items-center gap-1 text-xs text-status-warning"><CircleDashed className="h-3.5 w-3.5" />still short</span>}
            </div>
            <p className="mt-1 text-xs text-muted-foreground">{p.detail}</p>
            <p className="mt-1 text-xs tabular-nums text-muted-foreground">needs {fmtBytes(p.new_required)}{p.frees ? ` · frees ${fmtBytes(p.frees)}` : ""}</p>
          </button>
        );
      })}
    </div>
  );
}
