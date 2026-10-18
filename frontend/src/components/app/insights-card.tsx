import { Link } from "react-router-dom";
import { Lightbulb } from "lucide-react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { useInsights } from "@/hooks/use-insights";
import { LIVE_STATUSES, deploymentSubject } from "@/lib/insights";
import { InsightList } from "./insight-list";
import { ErrorState } from "./states";

/** Deployment detail (Overview): this deployment's open and snoozed insights. */
export function DeploymentInsightsCard({ deploymentId, className }: { deploymentId: number; className?: string }) {
  const q = useInsights({ status: LIVE_STATUSES, subject: deploymentSubject(deploymentId) });
  return (
    <Card data-card className={`min-w-0 shadow-card ${className ?? ""}`}>
      <CardHeader className="flex-row items-start justify-between space-y-0 pb-2">
        <div className="space-y-1.5">
          <CardTitle className="flex items-center gap-2 text-sm"><Lightbulb aria-hidden className="h-4 w-4" /><h2>Insights</h2></CardTitle>
          <CardDescription className="text-xs">What DGXMM noticed about this deployment. Advisory only.</CardDescription>
        </div>
        <Link to="/insights" className="text-xs text-primary hover:underline">All insights</Link>
      </CardHeader>
      <CardContent>
        {q.isLoading ? <Skeleton className="h-12" aria-label="Loading insights" />
          : q.error ? <ErrorState error={q.error} onRetry={() => q.refetch()} />
          : <InsightList insights={q.data ?? []} showSubject={false} empty={<p className="text-sm text-muted-foreground">Nothing to report.</p>} />}
      </CardContent>
    </Card>
  );
}
