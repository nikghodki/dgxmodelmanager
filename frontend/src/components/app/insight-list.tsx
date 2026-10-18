import { BellOff, Clock, X } from "lucide-react";
import { Link } from "react-router-dom";
import { useIsAdmin } from "@/auth";
import { Button } from "@/components/ui/button";
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuLabel, DropdownMenuTrigger } from "@/components/ui/dropdown-menu";
import { fmtRelative } from "@/format";
import { useInsightActions } from "@/hooks/use-insights";
import { featureLabel } from "@/lib/decider-labels";
import { SNOOZE_HOURS, severityLabel, severityTone, statusLabel } from "@/lib/insights";
import { cn } from "@/lib/utils";
import type { Insight } from "@/types";
import { DecisionChip } from "./decision-chip";
import { InsightBody } from "./insight-body";
import { InsightActionButton } from "./insight-action";

export function SeverityBadge({ severity }: { severity: string }) {
  return <span data-severity={severity} className={cn("inline-flex items-center rounded-full px-2 py-0.5 text-[11px] font-medium", severityTone(severity))}>{severityLabel(severity)}</span>;
}

function subjectLink(subject: string) {
  const m = /^deployment:(\d+)$/.exec(subject);
  return m ? <Link to={`/deployments/${m[1]}`} className="font-mono text-xs hover:underline">{subject}</Link> : null;
}

/** One insight: severity, title, templated body, source / confidence chip, and — for admins on live insights — the
 *  action button, snooze and dismiss. */
export function InsightItem({ insight: i, showSubject = true }: { insight: Insight; showSubject?: boolean }) {
  const admin = useIsAdmin();
  const { dismiss, snooze } = useInsightActions();
  const live = i.status === "open" || i.status === "snoozed";
  const busy = dismiss.isPending || snooze.isPending;
  return (
    <li data-insight={i.id} className="flex flex-col gap-2 py-3 sm:flex-row sm:items-start sm:justify-between">
      <div className="min-w-0 space-y-1">
        <div className="flex flex-wrap items-center gap-2">
          <SeverityBadge severity={i.severity} />
          <span className="text-sm font-medium">{i.title}</span>
          <DecisionChip source={i.source} confidence={i.confidence} title={i.source === "model" ? "Found by the decision model" : "Found by DGXMM's rules"} />
          {i.status !== "open" && <span className="text-[11px] text-muted-foreground">{statusLabel(i.status)}{i.status === "snoozed" && i.snooze_until ? ` until ${new Date(i.snooze_until).toLocaleString()}` : ""}</span>}
        </div>
        {i.body && <InsightBody body={i.body} />}
        <p className="flex flex-wrap items-center gap-x-2 text-xs text-muted-foreground">
          <span>{featureLabel(i.feature)}</span>
          {showSubject && subjectLink(i.subject)}
          <span title={new Date(i.updated_at).toLocaleString()}>{fmtRelative(i.updated_at)}</span>
        </p>
      </div>
      {admin && live && (
        <div className="flex shrink-0 flex-wrap items-center gap-1.5">
          <InsightActionButton insight={i} />
          <DropdownMenu>
            <DropdownMenuTrigger asChild><Button variant="ghost" size="sm" disabled={busy} aria-label={`Snooze ${i.title}`}><Clock className="mr-1 h-3.5 w-3.5" />Snooze</Button></DropdownMenuTrigger>
            <DropdownMenuContent align="end">
              <DropdownMenuLabel className="text-xs">Snooze for</DropdownMenuLabel>
              {SNOOZE_HOURS.map((s) => <DropdownMenuItem key={s.hours} onSelect={() => snooze.mutate({ id: i.id, hours: s.hours })}>{s.label}</DropdownMenuItem>)}
            </DropdownMenuContent>
          </DropdownMenu>
          <Button variant="ghost" size="sm" disabled={busy} aria-label={`Dismiss ${i.title}`} onClick={() => dismiss.mutate(i.id)}><X className="mr-1 h-3.5 w-3.5" />Dismiss</Button>
        </div>
      )}
    </li>
  );
}

export function InsightList({ insights, showSubject = true, empty }: { insights: Insight[]; showSubject?: boolean; empty?: React.ReactNode }) {
  if (!insights.length) return <>{empty ?? <p className="flex items-center gap-2 py-3 text-sm text-muted-foreground"><BellOff aria-hidden className="h-4 w-4" />No insights</p>}</>;
  return <ul className="divide-y" aria-label="Insights">{insights.map((i) => <InsightItem key={i.id} insight={i} showSubject={showSubject} />)}</ul>;
}
