import { useQuery } from "@tanstack/react-query";
import { Sparkles } from "lucide-react";
import { useState } from "react";
import { api } from "@/api";
import { Button } from "@/components/ui/button";
import { fillUntouched, goalLabel, shownOptions } from "@/lib/advice";
import type { ConfigAdvice, ConfigOption, DeployGoal } from "@/types";
import { DecisionChip } from "./decision-chip";

/**
 * Deploy-time config advice (POST /decider/config-advice): sizings from the memory calculator that fit the free memory,
 * ranked for the goal. With a recommendation (model ≥ threshold) a "Suggested config" card; otherwise the top three
 * with their trade-offs. "Fill" only sets fields the user has not touched — typed values are never overwritten.
 */
export function ConfigAdviceCard({ modelRef, engine, goal, config, touched, onFill, fits }: {
  modelRef: string; engine: string; goal: DeployGoal; config: Record<string, any>; touched: Iterable<string>;
  onFill: (next: Record<string, any>, changed: string[]) => void;
  /** The memory check of the current config (the wizard's preview): a fill that kept typed values made a combination
   * the advice never sized, so its fit is reported from this re-check. */
  fits?: boolean | null;
}) {
  const q = useQuery({
    queryKey: ["config-advice", modelRef, engine, goal],
    queryFn: () => api<ConfigAdvice>("/decider/config-advice", { method: "POST", body: { model_ref: modelRef, engine, goal } }),
    enabled: !!modelRef && !!engine, retry: false, staleTime: 60_000, refetchOnWindowFocus: false,
  });
  const [filled, setFilled] = useState<{ key: string; n: number; kept: string[] } | null>(null);
  const a = q.data && !q.error && Array.isArray(q.data.options) ? q.data : null;
  if (!a || a.disabled) return null;
  if (a.reason === "nothing_fits") return <p className="text-xs text-muted-foreground" data-config-advice="none">Config advice: no sizing of this model fits the memory free right now.</p>;
  const { recommended, top } = shownOptions(a);
  if (!recommended && !top.length) return null;
  const fill = (o: ConfigOption) => {
    const r = fillUntouched(config, o.config, touched);
    onFill(r.config, r.changed);
    const t = new Set(touched);
    const kept = Object.keys(o.config).filter((k) => t.has(k) && config[k] !== o.config[k]);
    setFilled({ key: o.key, n: r.changed.length, kept });
  };
  const status = (o: ConfigOption) => {
    if (filled?.key !== o.key) return null;
    const head = filled.n ? `Filled ${filled.n} field${filled.n === 1 ? "" : "s"}` : "Nothing to fill: those fields are yours";
    if (!filled.kept.length) return <span role="status" className="text-muted-foreground">{head}</span>;
    // a mix of typed values and the option: only the memory check below has sized it
    const check = fits == null ? "the memory check is re-checking it" : fits ? "re-checked: it fits" : "re-checked: it does not fit the free memory";
    return (
      <span role="status" className={fits === false ? "text-status-critical" : "text-muted-foreground"}>
        {head}; kept your {filled.kept.join(", ")} — {check}
      </span>
    );
  };
  return (
    <div className="space-y-2 rounded-md border border-primary/20 bg-primary/5 p-3 text-xs" data-config-advice={recommended ? "recommended" : "top"}>
      <div className="flex flex-wrap items-center gap-2">
        <Sparkles aria-hidden className="h-3.5 w-3.5 text-primary" />
        <span className="font-medium">{recommended ? "Suggested config" : "Configs that fit"} for {goalLabel(goal).toLowerCase()}</span>
        {recommended && <DecisionChip source={a.source} confidence={a.confidence} title="Ranked by the decision model" />}
      </div>
      {recommended ? (
        <div className="flex flex-wrap items-center gap-2">
          <span className="tabular-nums">{recommended.summary}</span>
          <Button type="button" size="sm" variant="outline" className="h-6 px-2 text-xs" aria-label={`Fill untouched fields with the suggested config: ${recommended.summary}`} onClick={() => fill(recommended)}>Fill</Button>
          {status(recommended)}
        </div>
      ) : (
        <ul className="space-y-1">
          {top.map((o) => (
            <li key={o.key} className="flex flex-wrap items-center gap-2">
              <span className="tabular-nums">{o.summary}</span>
              <Button type="button" size="sm" variant="outline" className="h-6 px-2 text-xs" aria-label={`Use ${o.summary}`} onClick={() => fill(o)}>Use</Button>
              {status(o)}
            </li>
          ))}
        </ul>
      )}
      <p className="text-muted-foreground">Fills only fields you have not changed. Each option, as offered, fits the memory free now.</p>
    </div>
  );
}
