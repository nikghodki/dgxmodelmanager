import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import { Plus, Rocket } from "lucide-react";
import { api } from "@/api";
import { useIsAdmin } from "@/auth";
import { DeploymentsTable } from "@/components/app/deployments-table";
import { PageHeader } from "@/components/app/page-header";
import { EmptyState, ErrorState } from "@/components/app/states";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import type { Deployment } from "@/types";
import { SwitchDialog } from "@/components/app/switch-dialog";

const FACETS = [
  { key: "all", label: "All", match: () => true },
  { key: "ready", label: "Ready", match: (s: string) => s === "ready" },
  { key: "sleeping", label: "Sleeping", match: (s: string) => s === "sleeping" },
  { key: "starting", label: "Starting", match: (s: string) => ["planned", "starting", "loading"].includes(s) },
  { key: "failed", label: "Failed", match: (s: string) => s === "failed" },
  { key: "stopped", label: "Stopped", match: (s: string) => ["stopped", "unknown", "stopping"].includes(s) },
];

export function Deployments() {
  const admin = useIsAdmin();
  const [facet, setFacet] = useState("all");
  const [dialog, setDialog] = useState<{ mode: "switch" | "reconfigure"; d: Deployment } | null>(null);
  const deps = useQuery({ queryKey: ["deployments"], queryFn: () => api<Deployment[]>("/deployments") });
  const all = deps.data ?? [];
  const counts = useMemo(() => Object.fromEntries(FACETS.map((f) => [f.key, all.filter((d) => f.match(d.state)).length])), [all]);
  const shown = all.filter((d) => FACETS.find((f) => f.key === facet)!.match(d.state));
  return (
    <div className="space-y-4">
      <PageHeader title="Deployments" description="Every model container DGXMM manages."
        actions={admin && <Button asChild size="sm"><Link to="/deploy"><Plus className="mr-1 h-4 w-4" />Deploy model</Link></Button>} />
      <ErrorState error={deps.error} onRetry={() => deps.refetch()} />
      <div className="flex flex-wrap gap-1.5" role="tablist" aria-label="Filter by state">
        {FACETS.map((f) => (
          <button key={f.key} role="tab" aria-selected={facet === f.key} onClick={() => setFacet(f.key)}
            className={cn("inline-flex h-8 items-center gap-1.5 rounded-full border px-3 text-xs font-medium transition-colors",
              facet === f.key ? "border-primary bg-primary text-primary-foreground" : "bg-card text-muted-foreground hover:text-foreground")}>
            {f.label}<span className="tabular-nums opacity-70">{counts[f.key] ?? 0}</span>
          </button>
        ))}
      </div>
      {deps.data && all.length === 0 ? (
        <EmptyState icon={Rocket} title="No deployments yet" action={admin && <Button asChild size="sm"><Link to="/deploy">Deploy a model</Link></Button>} />
      ) : (
        <DeploymentsTable deployments={shown} loading={deps.isLoading} isAdmin={admin} searchable
          onSwitch={(d) => setDialog({ mode: "switch", d })} onReconfigure={(d) => setDialog({ mode: "reconfigure", d })} />
      )}
      {dialog && <SwitchDialog mode={dialog.mode} deployment={dialog.d} onClose={() => setDialog(null)} />}
    </div>
  );
}
