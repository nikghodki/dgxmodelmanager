import { useMemo, useState } from "react";
import { Lightbulb } from "lucide-react";
import { PageHeader } from "@/components/app/page-header";
import { InsightList } from "@/components/app/insight-list";
import { EmptyState, ErrorState } from "@/components/app/states";
import { Card, CardContent } from "@/components/ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Skeleton } from "@/components/ui/skeleton";
import { useInsights } from "@/hooks/use-insights";
import { featureLabel } from "@/lib/decider-labels";
import { INSIGHT_STATUSES, LIVE_STATUSES, severityLabel, statusLabel } from "@/lib/insights";

const ALL = "__all";
const SEVERITIES = ["critical", "warning", "info"];

export function Insights() {
  const [status, setStatus] = useState(LIVE_STATUSES);
  const [severity, setSeverity] = useState(ALL);
  const [feature, setFeature] = useState(ALL);
  const q = useInsights({ status: status === ALL ? undefined : status });
  const features = useMemo(() => [...new Set((q.data ?? []).map((i) => i.feature))].sort(), [q.data]);
  const rows = (q.data ?? []).filter((i) => (severity === ALL || i.severity === severity) && (feature === ALL || i.feature === feature));
  const filtered = status !== LIVE_STATUSES || severity !== ALL || feature !== ALL;
  return (
    <div className="space-y-4">
      <PageHeader title="Insights" description="Findings from DGXMM's rules and the decision model: deployment health, proposals and advice. Advisory only — nothing changes until you confirm it." />
      <div className="flex flex-wrap gap-2" role="group" aria-label="Filters">
        <Select value={status} onValueChange={setStatus}>
          <SelectTrigger className="h-9 w-44" aria-label="Filter by status"><SelectValue /></SelectTrigger>
          <SelectContent>
            <SelectItem value={LIVE_STATUSES}>Open and snoozed</SelectItem>
            {INSIGHT_STATUSES.map((s) => <SelectItem key={s} value={s}>{statusLabel(s)}</SelectItem>)}
            <SelectItem value={ALL}>All</SelectItem>
          </SelectContent>
        </Select>
        <Select value={severity} onValueChange={setSeverity}>
          <SelectTrigger className="h-9 w-40" aria-label="Filter by severity"><SelectValue /></SelectTrigger>
          <SelectContent><SelectItem value={ALL}>All severities</SelectItem>{SEVERITIES.map((s) => <SelectItem key={s} value={s}>{severityLabel(s)}</SelectItem>)}</SelectContent>
        </Select>
        <Select value={feature} onValueChange={setFeature}>
          <SelectTrigger className="h-9 w-52" aria-label="Filter by feature"><SelectValue /></SelectTrigger>
          <SelectContent><SelectItem value={ALL}>All features</SelectItem>{features.map((f) => <SelectItem key={f} value={f}>{featureLabel(f)}</SelectItem>)}</SelectContent>
        </Select>
      </div>
      <ErrorState error={q.error} onRetry={() => q.refetch()} />
      {q.isLoading ? <div className="space-y-2" aria-busy="true" aria-label="Loading insights">{[0, 1, 2].map((i) => <Skeleton key={i} className="h-16" />)}</div>
        : !q.error && (rows.length === 0
          ? <EmptyState icon={Lightbulb} title={filtered ? "No insights match these filters" : "No open insights"}
              description={filtered ? "Try another status, severity or feature." : "DGXMM raises insights here when its background checks find something worth a look."} />
          : <Card className="shadow-card"><CardContent className="px-4 py-1"><InsightList insights={rows} /></CardContent></Card>)}
    </div>
  );
}
