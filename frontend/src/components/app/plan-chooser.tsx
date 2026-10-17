import { AlertTriangle, CheckCircle2, CircleDashed, Info, Star } from "lucide-react";
import { fmtBytes } from "@/format";
import { sortPlans } from "@/lib/decider";
import { cn } from "@/lib/utils";
import type { PlanOption } from "@/types";
import { Link } from "react-router-dom";
import { DecisionChip } from "./decision-chip";

/** Hints that are not plan actions, e.g. how to free the decision model's memory (plans never touch it). */
function PlanNotes({ notes }: { notes?: string[] }) {
  if (!notes?.length) return null;
  return <>{notes.map((n) => (
    <p key={n} className="flex items-start gap-1.5 text-xs text-muted-foreground"><Info aria-hidden className="mt-0.5 h-3.5 w-3.5 shrink-0" />
      <span className="min-w-0 break-words">{n}{n.includes("Decision model page") && <> · <Link className="text-primary hover:underline" to="/decision-model">Open it</Link></>}</span></p>
  ))}</>;
}

export function PlanChooser({ plans, selectedId, onSelect, notes }: { plans: PlanOption[]; selectedId: string | null; onSelect: (p: PlanOption) => void; notes?: string[] }) {
  if (!plans.length) return <div className="space-y-2"><p className="text-sm text-muted-foreground">No automatic plan frees enough memory. Stop something manually or choose a smaller configuration.</p><PlanNotes notes={notes} /></div>;
  return (
    <div className="space-y-2" role="radiogroup" aria-label="Ways to make it fit">
      <h3 className="text-sm font-semibold">Ways to make it fit</h3>
      <PlanNotes notes={notes} />
      {sortPlans(plans).map((p) => {
        const sel = selectedId === p.id;
        return (
          <button key={p.id} type="button" role="radio" aria-checked={sel} onClick={() => onSelect(p)}
            className={cn("block w-full rounded-lg border p-3 text-left transition-colors hover:bg-accent",
              sel ? "border-primary ring-1 ring-primary" : "border-border")}>
            <div className="flex items-center justify-between gap-2">
              <span className="flex min-w-0 flex-wrap items-center gap-1.5">
                <span className="text-sm font-medium">{p.title}</span>
                {p.recommended && <span className="inline-flex items-center gap-1 rounded-full bg-primary px-2 py-0.5 text-[11px] font-semibold text-primary-foreground"><Star aria-hidden className="h-3 w-3" />Recommended</span>}
              </span>
              {p.fits
                ? <span className="inline-flex items-center gap-1 text-xs text-status-good"><CheckCircle2 className="h-3.5 w-3.5" />fits</span>
                : <span className="inline-flex items-center gap-1 text-xs text-status-warning"><CircleDashed className="h-3.5 w-3.5" />still short</span>}
            </div>
            <p className="mt-1 text-xs text-muted-foreground">{p.detail}</p>
            <p className="mt-1 text-xs tabular-nums text-muted-foreground">needs {fmtBytes(p.new_required)}{p.frees ? ` · frees ${fmtBytes(p.frees)}` : ""}</p>
            {p.recommended && p.recommendation && (
              <p className="mt-1.5 flex flex-wrap items-center gap-1.5 text-xs">
                <span className="min-w-0 break-words">{p.recommendation.reason}</span>
                <DecisionChip source={p.recommendation.source} confidence={p.recommendation.confidence} />
              </p>
            )}
            {p.unsafe_reason && (
              <p className="mt-1.5 flex items-start gap-1 text-xs text-status-warning">
                <AlertTriangle aria-hidden className="mt-0.5 h-3.5 w-3.5 shrink-0" /><span className="min-w-0 break-words">Unsafe: {p.unsafe_reason}</span>
              </p>
            )}
          </button>
        );
      })}
    </div>
  );
}
