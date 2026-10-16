import { useState } from "react";
import { useQueries, useQuery } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import { ChevronDown, ChevronRight, Plus, Puzzle, Rocket } from "lucide-react";
import { api } from "@/api";
import { useIsAdmin } from "@/auth";
import { DeploymentsTable } from "@/components/app/deployments-table";
import { ChartCard } from "@/components/app/metric-chart";
import { MemoryBar } from "@/components/app/memory-bar";
import { PageHeader } from "@/components/app/page-header";
import { Sparkline, StatTile } from "@/components/app/stat-tile";
import { EmptyState, ErrorState, PageSkeleton } from "@/components/app/states";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { fmtBytes, fmtMs, fmtNum, fmtPct, GIB } from "@/format";
import { useActionMutation } from "@/hooks/use-action-mutation";
import { fleetKpis } from "@/lib/fleet";
import { cn } from "@/lib/utils";
import type { Deployment, HostInfo } from "@/types";
import { SwitchDialog } from "../components/SwitchDialog";

export function Dashboard() {
  const admin = useIsAdmin();
  const [dialog, setDialog] = useState<{ mode: "switch" | "reconfigure"; d: Deployment } | null>(null);
  const host = useQuery({ queryKey: ["host"], queryFn: () => api<HostInfo>("/host") });
  const deps = useQuery({ queryKey: ["deployments"], queryFn: () => api<Deployment[]>("/deployments") });
  const hostLive = useQuery({ queryKey: ["host-metrics"], queryFn: () => api<any[]>("/host/metrics?window=live"), refetchInterval: 10000 });
  const ready = (deps.data ?? []).filter((d) => d.state === "ready");
  const live = useQueries({ queries: ready.map((d) => ({ queryKey: ["live", d.id], queryFn: () => api<any[]>(`/deployments/${d.id}/metrics?window=live`), staleTime: Infinity })) });
  const k = fleetKpis(live.map((q) => q.data ?? []));
  const b = host.data?.budget;

  if (host.isLoading) return <PageSkeleton />;
  const freeTone = !b ? "" : b.free_for_new < 0 ? "text-status-critical" : b.free_for_new < 8 * GIB ? "text-status-warning" : "text-status-good";

  return (
    <div className="space-y-6">
      <PageHeader title="Dashboard" description="Live memory, throughput and deployments on this DGX Spark." />
      <ErrorState error={host.error || deps.error} onRetry={() => { host.refetch(); deps.refetch(); }} />

      {b && (
        <Card className="shadow-card">
          <CardContent className="grid gap-6 p-6 md:grid-cols-[minmax(0,1fr)_minmax(0,2fr)] md:items-center">
            <div>
              <div className="text-sm text-muted-foreground">Free for new deployments</div>
              <div className={cn("mt-1 text-4xl font-semibold tabular-nums tracking-tight", freeTone)}>{fmtBytes(b.free_for_new)}</div>
              <div className="mt-1 text-xs text-muted-foreground">of {fmtBytes(b.total)} · {fmtBytes(b.margin)} safety margin kept</div>
              <div className="mt-4 flex flex-wrap gap-2">
                {admin && <Button asChild size="sm"><Link to="/deploy"><Plus className="mr-1 h-4 w-4" />Deploy model</Link></Button>}
                <Button asChild size="sm" variant="outline"><Link to="/planner"><Puzzle className="mr-1 h-4 w-4" />Plan combinations</Link></Button>
              </div>
            </div>
            <div>
              <div className="mb-3 flex items-baseline justify-between">
                <span className="text-sm font-semibold">Unified memory</span>
                <span className="text-xs text-muted-foreground tabular-nums">
                  {host.data?.gpu?.util != null ? `GPU ${host.data.gpu.util}% · ${host.data.gpu.temp_c ?? "?"}°C · ` : ""}disk {fmtBytes(host.data?.disk?.free)} free
                </span>
              </div>
              <MemoryBar budget={b} />
            </div>
          </CardContent>
        </Card>
      )}

      <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
        <StatTile label="Generation" value={k.genTps == null ? "—" : `${fmtNum(k.genTps, 0)} tok/s`} sub={`${ready.length} ready`}>
          <Sparkline values={k.genSeries} label="Generation tokens per second, last minutes" />
        </StatTile>
        <StatTile label="TTFT p50" value={fmtMs(k.ttftP50)} sub="worst ready deployment">
          <Sparkline values={k.ttftSeries} label="Time to first token, last minutes" color="var(--chart-2)" />
        </StatTile>
        <StatTile label="KV cache" value={fmtPct(k.kv)} sub="highest usage">
          <Progress value={k.kv == null ? 0 : k.kv * 100} aria-label="KV cache usage" className="h-1.5" />
        </StatTile>
        <StatTile label="Queue" value={k.waiting == null ? "—" : `${k.waiting} waiting`} sub={k.running == null ? "no live data" : `${k.running} running`} />
      </div>

      <section className="space-y-3">
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-semibold">Deployments</h2>
          <Button asChild variant="ghost" size="sm"><Link to="/deployments">View all</Link></Button>
        </div>
        {deps.data && deps.data.length === 0 ? (
          <EmptyState icon={Rocket} title="Nothing deployed yet" description="Deploy a downloaded model to start serving it through LiteLLM."
            action={admin && <Button asChild size="sm"><Link to="/deploy">Deploy a model</Link></Button>} />
        ) : (
          <DeploymentsTable deployments={deps.data ?? []} loading={deps.isLoading} isAdmin={admin}
            onSwitch={(d) => setDialog({ mode: "switch", d })} onReconfigure={(d) => setDialog({ mode: "reconfigure", d })} />
        )}
      </section>

      <ChartCard title="Memory over the last hour" description="Used, external and free-for-new memory" data={hostLive.data ?? []}
        fmt={(v) => fmtBytes(v, 0)} series={[{ key: "used", label: "Used", slot: 1 }, { key: "external", label: "External", slot: 2 }, { key: "free_for_new", label: "Free for new", slot: 3 }]} />

      <div className="grid gap-4 lg:grid-cols-2">
        <AdoptCard />
        {host.data && host.data.external_top.length > 0 && (
          <Card className="shadow-card">
            <CardHeader className="pb-2"><CardTitle className="text-sm">Largest external memory users</CardTitle>
              <CardDescription className="text-xs">Processes DGXMM does not manage. They are never touched.</CardDescription></CardHeader>
            <CardContent>
              <ul className="divide-y text-sm">
                {host.data.external_top.map((p) => (
                  <li key={p.pid} className="flex items-center justify-between gap-3 py-2">
                    <span className="min-w-0 truncate"><span className="font-medium">{p.name}</span> <span className="text-xs text-muted-foreground">pid {p.pid}</span></span>
                    <span className="shrink-0 tabular-nums">{fmtBytes(p.rss + p.gpu)}</span>
                  </li>
                ))}
              </ul>
            </CardContent>
          </Card>
        )}
      </div>

      {dialog && <SwitchDialog mode={dialog.mode} deployment={dialog.d} onClose={() => setDialog(null)} />}
    </div>
  );
}

function AdoptCard() {
  const admin = useIsAdmin();
  const [open, setOpen] = useState(false);
  const rows = useQuery({ queryKey: ["adoptable"], queryFn: () => api<any[]>("/containers/adoptable"), enabled: admin });
  const adopt = useActionMutation(
    (c: { name: string }) => api(`/containers/${encodeURIComponent(c.name)}/adopt`, { method: "POST", body: { name: c.name.replace(/[^a-z0-9-]/g, "-").replace(/^dmm-/, "").slice(0, 40) } }),
    { success: (_d, c) => `Adopted settings from ${c.name}`, invalidate: [["adoptable"], ["deployments"]] },
  );
  if (!admin || !rows.data?.length) return null;
  return (
    <Card className="shadow-card">
      <CardHeader className="pb-2">
        <button type="button" className="flex w-full items-center justify-between text-left" onClick={() => setOpen(!open)} aria-expanded={open}>
          <span><CardTitle className="text-sm">Containers not managed by DGXMM</CardTitle>
            <CardDescription className="text-xs">{rows.data.length} vLLM container(s) found</CardDescription></span>
          {open ? <ChevronDown className="h-4 w-4" /> : <ChevronRight className="h-4 w-4" />}
        </button>
      </CardHeader>
      {open && (
        <CardContent className="space-y-3">
          <ul className="divide-y text-sm">
            {rows.data.map((c) => (
              <li key={c.id} className="flex flex-wrap items-center justify-between gap-2 py-2">
                <div className="min-w-0"><div className="truncate font-mono text-xs">{c.name}</div><div className="text-xs text-muted-foreground">{c.status} · {c.image}</div></div>
                {c.adopted_as ? <Badge variant="secondary">adopted as {c.adopted_as}</Badge>
                  : c.adoptable === false ? <span className="text-xs text-muted-foreground">{c.reason}</span>
                  : <Button size="sm" variant="outline" disabled={adopt.isPending} onClick={() => adopt.mutate(c)}>Adopt settings</Button>}
              </li>
            ))}
          </ul>
          <p className="text-xs text-muted-foreground">Adopting copies the container's settings into a stopped DGXMM deployment. The original container is left as it is; stop it before starting the adopted one.</p>
        </CardContent>
      )}
    </Card>
  );
}
