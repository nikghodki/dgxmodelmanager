import { useQuery } from "@tanstack/react-query";
import { HelpCircle, Stethoscope, Wrench } from "lucide-react";
import { api } from "@/api";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { causeLabel } from "@/lib/decider-labels";
import type { Diagnosis, DiagnosisResponse } from "@/types";
import { DecisionChip } from "./decision-chip";
import { FeedbackButtons } from "./feedback-buttons";

/** Latest failure diagnosis of a deployment; refreshed on the SSE `diagnosis` topic (see stream.ts). */
export function useDiagnosis(id: number) {
  return useQuery({ queryKey: ["diagnosis", id], queryFn: () => api<DiagnosisResponse>(`/deployments/${id}/diagnosis`), retry: false });
}

/** A diagnosis is usable when someone (rules or the model, above threshold) named a cause. */
export const confidentDiagnosis = (d: Diagnosis) => d.source !== "none" && !!d.cause;

function LogExcerpt({ text }: { text: string }) {
  return <pre aria-label="Log excerpt" className="max-h-48 overflow-auto whitespace-pre-wrap break-all rounded-md bg-zinc-950 p-2 font-mono text-xs text-zinc-100">{text}</pre>;
}

export function DiagnosisCard({ diagnosis: d, admin, onApplyFix, className }: {
  diagnosis: Diagnosis; admin: boolean; onApplyFix?: (patch: Record<string, unknown>) => void; className?: string;
}) {
  const confident = confidentDiagnosis(d);
  return (
    <Card data-card className={`min-w-0 border-status-critical/30 shadow-card ${className ?? ""}`}>
      <CardHeader className="flex-row flex-wrap items-start justify-between gap-2 space-y-0 pb-2">
        <div className="min-w-0 space-y-1.5">
          <CardTitle className="flex items-center gap-2 text-sm"><Stethoscope aria-hidden className="h-4 w-4" /><h2>Diagnosis</h2></CardTitle>
          <CardDescription className="text-xs">Why the last start failed{d.created_at ? ` · ${new Date(d.created_at).toLocaleString()}` : ""}</CardDescription>
        </div>
        {admin && d.decision_id != null && <FeedbackButtons decisionId={d.decision_id} feedback={d.feedback} invalidate={[["diagnosis"]]} />}
      </CardHeader>
      <CardContent className="space-y-3 text-sm">
        {confident ? (
          <>
            <div className="flex flex-wrap items-center gap-2">
              <span className="font-semibold">{causeLabel(d.cause, d.label)}</span>
              <DecisionChip source={d.source} confidence={d.confidence}
                title={d.source === "model" ? "Diagnosed by the decision model" : "Matched a known error pattern"} />
            </div>
            {d.summary && <p className="break-words text-muted-foreground">{d.summary}</p>}
            {d.fix_hint && (
              <p className="flex items-start gap-1.5 break-words"><Wrench aria-hidden className="mt-0.5 h-3.5 w-3.5 shrink-0 text-muted-foreground" />
                <span><span className="sr-only">Suggested fix: </span>{d.fix_hint}</span></p>
            )}
            {admin && d.patch && onApplyFix && (
              <Button size="sm" variant="outline" onClick={() => onApplyFix(d.patch!)}>Apply fix…</Button>
            )}
            {d.log_excerpt && <details><summary className="cursor-pointer text-xs text-primary">Log excerpt</summary><div className="mt-2"><LogExcerpt text={d.log_excerpt} /></div></details>}
          </>
        ) : (
          <>
            <p className="flex items-start gap-1.5"><HelpCircle aria-hidden className="mt-0.5 h-4 w-4 shrink-0 text-muted-foreground" />
              Couldn't diagnose confidently — see the log.</p>
            {d.log_excerpt && <LogExcerpt text={d.log_excerpt} />}
          </>
        )}
      </CardContent>
    </Card>
  );
}
