import { useQuery } from "@tanstack/react-query";
import { api } from "@/api";
import { Button } from "@/components/ui/button";
import { engineLabel } from "@/format";
import { engineReason } from "@/lib/advice";
import type { DeployGoal, EngineSuggestion } from "@/types";
import { DecisionChip } from "./decision-chip";

/** Engine choice (POST /decider/engine-suggest) next to the engine selector; "Use" switches engine (and, for the other
 * format, the model ref) — nothing changes without the click. Silent when no engine is suggested. */
export function EngineChip({ modelRef, goal, engine, onUse }: {
  modelRef: string; goal: DeployGoal; engine: string; onUse: (engine: string, modelRef: string) => void;
}) {
  const q = useQuery({
    queryKey: ["engine-suggest", modelRef, goal],
    queryFn: () => api<EngineSuggestion>("/decider/engine-suggest", { method: "POST", body: { model_ref: modelRef, goal } }),
    enabled: !!modelRef, retry: false, staleTime: 60_000, refetchOnWindowFocus: false,
  });
  const s = q.data && !q.error && typeof q.data === "object" && "engine" in q.data ? q.data : null;
  if (!s || s.disabled || !s.engine || s.source === "none") return null;
  const ref = s.model_ref ?? modelRef;
  const differs = s.engine !== engine || ref !== modelRef;
  return (
    <div className="flex flex-wrap items-center gap-2 text-xs" data-engine-suggestion={s.engine}>
      <DecisionChip source={s.source} confidence={s.confidence} prefix={`Suggested: ${engineLabel(s.engine)}`} title={engineReason(s.reason) || undefined} />
      {differs && <Button type="button" variant="outline" size="sm" className="h-6 px-2 text-xs" aria-label={`Use ${engineLabel(s.engine)}${ref !== modelRef ? ` with ${ref}` : ""}`} onClick={() => onUse(s.engine!, ref)}>Use</Button>}
      {differs && ref !== modelRef && <span className="break-all text-muted-foreground">uses {ref.split("/").pop()}</span>}
    </div>
  );
}
