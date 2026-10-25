import { useEffect, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Link, useNavigate, useParams, useSearchParams } from "react-router-dom";
import { Activity, ArrowLeftRight, BrainCircuit, Clock, Cpu, FileText, Network, Gauge, Lightbulb, Loader2, MoreHorizontal, Moon, Play, Power, RefreshCw, Settings2, Sun, Trash2 } from "lucide-react";
import { api } from "@/api";
import { useIsAdmin } from "@/auth";
import { ConfirmDialog } from "@/components/app/confirm-dialog";
import { ApiAccessCard, KV, LiteLLMTab, useDeploymentLiteLLM } from "@/components/app/deployment-litellm";
import { ModelNameTaken } from "@/components/app/model-name-taken";
import { nameFixHref } from "@/lib/litellm";
import { ChartCard } from "@/components/app/metric-chart";
import { PageHeader } from "@/components/app/page-header";
import { StatusBadge } from "@/components/app/status-badge";
import { EmptyState, ErrorState, PageSkeleton } from "@/components/app/states";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import {
  DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuSeparator, DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Sheet, SheetContent, SheetDescription, SheetHeader, SheetTitle } from "@/components/ui/sheet";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { engineLabel, fmtBytes, fmtMs, fmtNum } from "@/format";
import { useActionMutation } from "@/hooks/use-action-mutation";
import { benchmarkSeries, plausibleRows, readingBadge } from "@/lib/benchmarks";
import { estimateParts } from "@/lib/estimate";
import { SYSTEM_REMOVE_TEXT, adviceApplicable, deploymentActions } from "@/logic";
import { useLive } from "@/stream";
import type { Deployment, Workload } from "@/types";
import { SwitchDialog } from "@/components/app/switch-dialog";
import { DecisionChip, SystemBadge } from "@/components/app/decision-chip";
import { DiagnosisCard, useDiagnosis } from "@/components/app/diagnosis-card";
import { DeploymentInsightsCard } from "@/components/app/insights-card";
import { DeploymentSchedulesCard } from "@/components/app/schedules-card";
import { DeploymentModeCard } from "@/components/app/mode-card";
import { DeploymentStateBadge } from "@/components/app/deployments-table";

/** How long after a failure the page says "Diagnosing…" while no (fresh) diagnosis has arrived. */
const DIAGNOSIS_WAIT_MS = 5 * 60_000;
import { causeLabel, splitAdvice, workloadLabel, workloadRefetchMs } from "@/lib/decider-labels";

const WINDOWS = ["live", "24h", "7d"] as const;
const TABS = ["overview", "litellm", "performance", "advice", "benchmarks", "events"];

function WindowPicker({ value, onChange }: { value: string; onChange: (w: string) => void }) {
  return (
    <div className="flex rounded-md border p-0.5" role="radiogroup" aria-label="Time window">
      {WINDOWS.map((w) => (
        <button key={w} role="radio" aria-checked={value === w} onClick={() => onChange(w)}
          className={`h-6 rounded px-2 text-xs ${value === w ? "bg-primary text-primary-foreground" : "text-muted-foreground hover:text-foreground"}`}>{w}</button>
      ))}
    </div>
  );
}

function Performance({ id }: { id: number }) {
  const [win, setWin] = useState("live");
  const live = useLive(id);
  const hist = useQuery({ queryKey: ["hist", id, win], queryFn: () => api<any[]>(`/deployments/${id}/metrics?window=${win}`), enabled: win !== "live" });
  const data = (win === "live" ? live.data : hist.data) ?? [];
  const picker = <WindowPicker value={win} onChange={setWin} />;
  return (
    <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
      <ChartCard title="Throughput" description="tokens / second" actions={picker} data={data}
        series={[{ key: "gen_tps", label: "Generation", slot: 1 }, { key: "prompt_tps", label: "Prompt", slot: 3 }]} />
      <ChartCard title="Time to first token" data={data} fmt={(v) => fmtMs(v)}
        series={[{ key: "ttft_p50", label: "p50", slot: 1 }, { key: "ttft_p95", label: "p95", slot: 2 }]} />
      <ChartCard title="KV cache usage" description="fraction of the KV pool in use" data={data} fmt={(v) => `${Math.round(v * 100)}%`}
        series={[{ key: "kv_usage", label: "KV used", slot: 1 }]} />
      <ChartCard title="Queue" description="requests waiting / running" data={data} fmt={(v) => fmtNum(v, 0)}
        series={[{ key: "waiting", label: "Waiting", slot: 2 }, { key: "running", label: "Running", slot: 1 }]} />
      <ChartCard title="Inter-token latency" data={data} fmt={(v) => fmtMs(v)} series={[{ key: "itl_p50", label: "ITL p50", slot: 7 }]} />
      <ChartCard title="Speculative decoding acceptance" data={data} fmt={(v) => `${Math.round(v * 100)}%`}
        series={[{ key: "spec_accept_rate", label: "Accepted", slot: 3 }]} />
    </div>
  );
}

function AdviceCard({ a, d, onApply }: { a: any; d: Deployment; onApply: (patch: Record<string, any>) => void }) {
  const admin = useIsAdmin();
  const tone = (s: string) => (s === "critical" ? "bg-status-critical-bg text-status-critical" : s === "warning" ? "bg-status-warning-bg text-status-warning" : s === "info" ? "bg-status-neutral-bg text-status-neutral" : "bg-primary/10 text-primary");
  return (
    <Card className="min-w-0 shadow-card">
      <CardHeader className="space-y-1.5 pb-2">
        <div className="flex flex-wrap items-center gap-2">
          <span className={`rounded-full px-2 py-0.5 text-[11px] font-medium capitalize ${tone(a.severity)}`}>{a.severity}</span>
          <Badge variant="outline" className="text-[11px]">{a.confidence}</Badge>
        </div>
        <CardTitle className="text-sm">{a.title}</CardTitle>
      </CardHeader>
      <CardContent className="space-y-2 text-sm">
        <p className="text-muted-foreground">{a.rationale}</p>
        <p className="text-xs"><span className="text-muted-foreground">Expected:</span> {a.expected_effect}</p>
        {admin && adviceApplicable(a, d) && (
          <Button size="sm" variant="outline" onClick={() => onApply({ ...(a.patch ?? {}), ...(a.model_ref ? { model_ref: a.model_ref } : {}) })}>Apply…</Button>
        )}
      </CardContent>
    </Card>
  );
}

/** Advice arrives ordered by `relevance` for the deployment's workload (server-side); with a known workload the
 *  relevant items get their own heading. */
function Advice({ d, onApply }: { d: Deployment; onApply: (patch: Record<string, any>) => void }) {
  const wl = useQuery({ queryKey: ["workload", d.id], queryFn: () => api<Workload | null>(`/deployments/${d.id}/workload`), retry: false,
    refetchInterval: (query) => workloadRefetchMs(query.state.data) });
  const workload = wl.data && wl.data.source !== "none" && wl.data.label ? wl.data : null;
  // Keyed by the workload label: once the background computation lands, the advice is re-fetched in its order.
  const q = useQuery({ queryKey: ["advice", d.id, workload?.label ?? null], queryFn: () => api<any[]>(`/deployments/${d.id}/advice`),
    refetchInterval: 60000, placeholderData: (prev) => prev });
  if (q.error) return <ErrorState error={q.error} onRetry={() => q.refetch()} />;
  const badge = workload && (
    <DecisionChip source={workload.source} confidence={workload.confidence} prefix={`Workload: ${workloadLabel(workload.label)}`}
      className="text-xs" title={workload.source === "model" ? "Workload classified by the decision model from this deployment's metrics" : "Workload from this deployment's metrics"} />
  );
  if (!q.data?.length) return (
    <div className="space-y-3">{badge && <div>{badge}</div>}
      <EmptyState icon={Lightbulb} title="No suggestions right now" description="Advice appears as DGXMM collects metrics and benchmarks for this deployment." /></div>
  );
  const { top, rest } = splitAdvice(q.data, !!workload);
  const grid = (items: any[]) => <div className="grid grid-cols-1 gap-3 md:grid-cols-2">{items.map((a) => <AdviceCard key={a.id} a={a} d={d} onApply={onApply} />)}</div>;
  const heading = (t: string) => <h3 className="text-xs font-medium uppercase tracking-wider text-muted-foreground">{t}</h3>;
  return (
    <div className="space-y-3">
      {badge && <div>{badge}</div>}
      {top.length ? <>
        {heading("Most relevant for this workload")}{grid(top)}
        <div className="border-t pt-3">{heading("Other suggestions")}</div>{grid(rest)}
      </> : grid(rest)}
    </div>
  );
}

function ReadingBadge({ label, source }: { label?: string | null; source?: string | null }) {
  const r = readingBadge(label, source);
  if (!r) return <span className="text-xs text-muted-foreground">—</span>;
  return <span title={r.hint} aria-label={`Reading: ${r.text}. ${r.hint}`} className={`inline-flex items-center rounded-full px-2 py-0.5 text-[11px] font-medium ${r.tone}`}>{r.text}</span>;
}

function Benchmarks({ d }: { d: Deployment }) {
  const admin = useIsAdmin();
  const q = useQuery({ queryKey: ["benchmarks", d.id], queryFn: () => api<any[]>(`/deployments/${d.id}/benchmarks`), refetchInterval: 10000 });
  const run = useActionMutation(() => api(`/deployments/${d.id}/benchmark`, { method: "POST", body: {} }), {
    success: "Benchmark started — results appear as each level finishes (up to 4 minutes)", invalidate: [["benchmarks", d.id]],
  });
  const rows = q.data ?? [];
  const { data, keys } = benchmarkSeries(plausibleRows(rows));  // readings labelled other than plausible are not charted
  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <p className="text-sm text-muted-foreground">Short load test at concurrency 1, 4, 8 and 16 against this deployment.</p>
        {admin && <Button size="sm" disabled={d.state !== "ready" || run.isPending} onClick={() => run.mutate(undefined)}><Gauge className="mr-1 h-4 w-4" />Run benchmark</Button>}
      </div>
      {rows.length === 0 ? <EmptyState icon={Gauge} title="No benchmarks yet" /> : (
        <>
          <ChartCard title="Total throughput by concurrency" description="tok/s, one line per configuration" data={data} xKey="concurrency"
            xFmt={(v) => `c=${v}`} series={keys.map((k, i) => ({ key: k.hash, label: k.label, slot: i + 1 }))} />
          <Card className="shadow-card overflow-x-auto">
            <table className="w-full text-sm">
              <thead><tr className="border-b text-left text-xs uppercase tracking-wider text-muted-foreground">
                {["When", "Config", "Concurrency", "Total tok/s", "Per-request tok/s", "TTFT p50", "ITL p50", "Reading"].map((h) => <th key={h} className="px-4 py-2 font-medium">{h}</th>)}
              </tr></thead>
              <tbody>
                {rows.map((b) => (
                  <tr key={b.id} className="border-b last:border-0">
                    <td className="whitespace-nowrap px-4 py-2 text-xs text-muted-foreground">{new Date(b.ts).toLocaleString()}</td>
                    <td className="px-4 py-2 font-mono text-xs">{b.config_hash.slice(0, 8)} {b.current && <Badge variant="secondary" className="ml-1">current</Badge>}</td>
                    <td className="px-4 py-2 tabular-nums">{b.concurrency}</td>
                    <td className="px-4 py-2 tabular-nums">{fmtNum(b.tps_total, 1)}</td>
                    <td className="px-4 py-2 tabular-nums">{fmtNum(b.tps_per_req, 1)}</td>
                    <td className="px-4 py-2 tabular-nums">{fmtMs(b.ttft_p50)}</td>
                    <td className="px-4 py-2 tabular-nums">{fmtMs(b.itl_p50)}</td>
                    <td className="px-4 py-2"><ReadingBadge label={b.label} source={b.label_source} /></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>
        </>
      )}
    </div>
  );
}

function LogsSheet({ id, open, onOpenChange }: { id: number; open: boolean; onOpenChange: (o: boolean) => void }) {
  const q = useQuery({ queryKey: ["logs", id], queryFn: () => api<{ logs: string }>(`/deployments/${id}/logs?tail=500`), enabled: open });
  const pre = useRef<HTMLPreElement>(null);
  useEffect(() => { if (pre.current) pre.current.scrollTop = pre.current.scrollHeight; }, [q.data]);
  return (
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetContent side="right" className="flex w-full flex-col gap-3 sm:max-w-2xl">
        <SheetHeader className="flex-row items-center justify-between space-y-0">
          <div><SheetTitle>Container logs</SheetTitle><SheetDescription>Last 500 lines</SheetDescription></div>
          <Button variant="outline" size="sm" onClick={() => q.refetch()} disabled={q.isFetching}><RefreshCw className={`mr-1 h-3.5 w-3.5 ${q.isFetching ? "animate-spin" : ""}`} />Refresh</Button>
        </SheetHeader>
        <ErrorState error={q.error} />
        <pre ref={pre} className="flex-1 overflow-auto whitespace-pre-wrap break-all rounded-md bg-zinc-950 p-3 font-mono text-xs text-zinc-100">{q.data?.logs || (q.isLoading ? "Loading…" : "(empty)")}</pre>
      </SheetContent>
    </Sheet>
  );
}

export function DeploymentDetail() {
  const id = Number(useParams().id);
  const [params, setParams] = useSearchParams();
  const tab = TABS.includes(params.get("tab") ?? "") ? params.get("tab")! : "overview";
  const admin = useIsAdmin();
  const nav = useNavigate();
  const dep = useQuery({ queryKey: ["deployment", id], queryFn: () => api<Deployment>(`/deployments/${id}`) });
  const litellm = useDeploymentLiteLLM(id);
  const diagnosis = useDiagnosis(id).data?.diagnosis ?? null;
  const [logs, setLogs] = useState(false);
  const [dialog, setDialog] = useState<null | { mode: "switch" | "reconfigure"; patch?: Record<string, any>; source?: string }>(null);
  const [confirm, setConfirm] = useState<null | "stop" | "remove">(null);
  const act = useActionMutation((action: string) => api(`/deployments/${id}/${action}`, { method: "POST" }), {
    success: (_r, a) => `${a[0].toUpperCase()}${a.slice(1)} requested`, invalidate: [["deployment", id], ["deployments"]],
  });
  const remove = useActionMutation(() => api(`/deployments/${id}`, { method: "DELETE" }), {
    success: "Deployment removed", invalidate: [["deployments"]], onSuccess: () => nav("/deployments"),
  });

  // A start/wake model-name conflict notice is stale once the user moves on (e.g. follows its fix link).
  const resetAct = act.reset;
  useEffect(() => { resetAct(); }, [tab]); // eslint-disable-line react-hooks/exhaustive-deps

  if (dep.isLoading) return <PageSkeleton />;
  const d = dep.data;
  if (!d) return <ErrorState error={dep.error ?? new Error("Deployment not found")} onRetry={() => dep.refetch()} />;
  const acts = deploymentActions(d, admin);
  const busy = act.isPending || remove.isPending;
  // The in-flight action shows a spinner in place of its icon; every header action is disabled meanwhile.
  const icon = (a: string, Icon: typeof Moon) => (act.isPending && act.variables === a
    ? <Loader2 className="mr-1 h-4 w-4 animate-spin" aria-hidden /> : <Icon className="mr-1 h-4 w-4" />);
  const parts = estimateParts(d.estimate ?? {});
  const applyFix = acts.includes("reconfigure") ? (patch: Record<string, unknown>) => setDialog({ mode: "reconfigure", patch, source: "the diagnosis" }) : undefined;
  // A diagnosis older than the latest failure belongs to an earlier failure: the current one is still being diagnosed.
  const failedAt = Math.max(0, ...(d.events ?? []).filter((e) => e.to === "failed").map((e) => Date.parse(e.ts) || 0));
  const staleDiagnosis = !!diagnosis && failedAt > 0 && (Date.parse(diagnosis.created_at) || 0) < failedAt;
  const currentDiagnosis = d.state === "failed" && diagnosis && !staleDiagnosis ? diagnosis : null;
  const diagnosing = d.state === "failed" && !currentDiagnosis && failedAt > 0 && Date.now() - failedAt < DIAGNOSIS_WAIT_MS;

  return (
    <div className="space-y-6">
      <PageHeader
        title={<span className="flex items-center gap-3">{d.name}<DeploymentStateBadge d={d} />{d.system && <SystemBadge />}</span>}
        description={<span className="break-all">{d.model_ref} · {engineLabel(d.engine)}</span>}
        actions={<>
          {acts.includes("sleep") && <Button variant="outline" size="sm" disabled={busy} onClick={() => act.mutate("sleep")}>{icon("sleep", Moon)}Sleep</Button>}
          {acts.includes("wake") && <Button variant="outline" size="sm" disabled={busy} onClick={() => act.mutate("wake")}>{icon("wake", Sun)}Wake</Button>}
          {acts.includes("start") && <Button size="sm" disabled={busy} onClick={() => act.mutate("start")}>{icon("start", Play)}Start</Button>}
          {acts.includes("manage") && <Button asChild variant="outline" size="sm"><Link to="/decision-model"><BrainCircuit className="mr-1 h-4 w-4" />Decision model page</Link></Button>}
          {acts.includes("reconfigure") && <Button variant="outline" size="sm" disabled={busy} onClick={() => setDialog({ mode: "reconfigure" })}><Settings2 className="mr-1 h-4 w-4" />Reconfigure</Button>}
          {admin && <Button variant="outline" size="sm" onClick={() => setLogs(true)}><FileText className="mr-1 h-4 w-4" />Logs</Button>}
          {(acts.includes("stop") || acts.includes("remove") || acts.includes("switch")) && (
            <DropdownMenu>
              <DropdownMenuTrigger asChild><Button variant="outline" size="icon" className="h-8 w-8" disabled={busy} aria-label="More actions"><MoreHorizontal className="h-4 w-4" /></Button></DropdownMenuTrigger>
              <DropdownMenuContent align="end">
                {acts.includes("switch") && <DropdownMenuItem onSelect={() => setDialog({ mode: "switch" })}><ArrowLeftRight className="mr-2 h-4 w-4" />Switch model…</DropdownMenuItem>}
                {acts.includes("switch") && <DropdownMenuSeparator />}
                {acts.includes("stop") && <DropdownMenuItem className="text-status-critical focus:text-status-critical" onSelect={() => setConfirm("stop")}><Power className="mr-2 h-4 w-4" />Stop</DropdownMenuItem>}
                {acts.includes("remove") && <DropdownMenuItem className="text-status-critical focus:text-status-critical" onSelect={() => setConfirm("remove")}><Trash2 className="mr-2 h-4 w-4" />Remove</DropdownMenuItem>}
              </DropdownMenuContent>
            </DropdownMenu>
          )}
        </>}
      />
      {d.startup_queue && (
        <p role="status" className="flex items-start gap-2 rounded-lg border bg-muted/50 px-4 py-3 text-sm text-muted-foreground">
          <Clock aria-hidden className="mt-0.5 h-4 w-4 shrink-0" />
          <span><span className="font-medium text-foreground">{d.startup_queue.kind === "wake" ? "Waiting to wake up" : "Queued"}: </span>
            {d.startup_queue.reason}. Its memory stays reserved{d.startup_queue.kind === "wake" ? "" : " and the fit is checked again when its turn comes"}.</span>
        </p>
      )}
      {d.last_error && <ErrorState error={new Error(d.last_error)} />}
      <p aria-live="polite" data-diagnosis-live className="sr-only">{currentDiagnosis ? `Diagnosis: ${currentDiagnosis.source === "none" || !currentDiagnosis.cause ? "not confident" : causeLabel(currentDiagnosis.cause, currentDiagnosis.label)}` : ""}</p>
      {d.system && (
        <p className="flex items-start gap-2 rounded-lg border bg-muted/50 px-4 py-3 text-sm text-muted-foreground">
          <Cpu aria-hidden className="mt-0.5 h-4 w-4 shrink-0" />
          <span>Used by the decision model and managed on the Decision model page. Plans never stop or sleep it to make room, and it is not registered in LiteLLM. Under memory pressure the watchdog stops it before any of your deployments (and always when it uses far more memory than it reserved, even with auto-protect off); the AI suggestions fall back to rules until it runs again.</span>
        </p>
      )}
      <ModelNameTaken error={act.error} fixHref={admin ? nameFixHref(d) : undefined} fixLabel={`Set a model name for ${d.name}`} />

      <Tabs value={tab} onValueChange={(t) => setParams(t === "overview" ? {} : { tab: t }, { replace: true })}>
        <TabsList className="w-full justify-start overflow-x-auto sm:w-auto">
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="litellm"><Network className="mr-1 h-3.5 w-3.5" />LiteLLM</TabsTrigger>
          <TabsTrigger value="performance"><Activity className="mr-1 h-3.5 w-3.5" />Performance</TabsTrigger>
          <TabsTrigger value="advice"><Lightbulb className="mr-1 h-3.5 w-3.5" />Advice</TabsTrigger>
          <TabsTrigger value="benchmarks"><Gauge className="mr-1 h-3.5 w-3.5" />Benchmarks</TabsTrigger>
          <TabsTrigger value="events">Events</TabsTrigger>
        </TabsList>

        <TabsContent value="overview" className="mt-4 grid grid-cols-1 gap-4 lg:grid-cols-3">
          {currentDiagnosis && <DiagnosisCard diagnosis={currentDiagnosis} admin={admin} className="lg:col-span-3" onApplyFix={applyFix} />}
          {diagnosing && <p className="flex items-center gap-2 text-sm text-muted-foreground lg:col-span-3"><Loader2 aria-hidden className="h-4 w-4 animate-spin" />Diagnosing…</p>}
          <DeploymentInsightsCard deploymentId={d.id} className="lg:col-span-3" />
          <DeploymentSchedulesCard deployment={d} className="lg:col-span-3" />
          <ApiAccessCard d={d} info={litellm.data} loading={litellm.isLoading} error={litellm.error} />
          <DeploymentModeCard d={d} admin={admin} />
          <Card className="min-w-0 shadow-card">
            <CardHeader className="pb-2"><CardTitle className="text-sm">Memory</CardTitle>
              <CardDescription className="text-xs">confidence: {(d.estimate as any)?.confidence ?? "—"}</CardDescription></CardHeader>
            <CardContent><dl>
              <KV label="Reserved" value={<span className="tabular-nums">{fmtBytes(d.reserved_bytes)}</span>} />
              <KV label="Measured" value={<span className="tabular-nums">{fmtBytes(d.actual_bytes)}</span>} />
              {parts.map((p) => <KV key={p.key} label={p.label} value={<span className="tabular-nums">{fmtBytes(p.bytes)}</span>} />)}
            </dl></CardContent>
          </Card>
          <Card className="min-w-0 shadow-card lg:col-span-2">
            <CardHeader className="pb-2"><CardTitle className="text-sm">Configuration</CardTitle></CardHeader>
            <CardContent><dl>
              {Object.entries(d.config ?? {}).filter(([, v]) => v !== null && !(Array.isArray(v) && !v.length)).map(([k, v]) => (
                <KV key={k} label={k} mono value={typeof v === "object" ? JSON.stringify(v) : String(v)} />
              ))}
            </dl></CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="litellm" className="mt-4"><LiteLLMTab d={d} info={litellm.data} loading={litellm.isLoading} error={litellm.error} admin={admin} suggested={params.get("alias")} onRenamed={() => act.reset()} /></TabsContent>
        <TabsContent value="performance" className="mt-4"><Performance id={id} /></TabsContent>
        <TabsContent value="advice" className="mt-4"><Advice d={d} onApply={(patch) => setDialog({ mode: "reconfigure", patch, source: "the advice" })} /></TabsContent>
        <TabsContent value="benchmarks" className="mt-4"><Benchmarks d={d} /></TabsContent>
        <TabsContent value="events" className="mt-4 space-y-4">
          {diagnosis && <DiagnosisCard diagnosis={diagnosis} admin={admin} onApplyFix={d.state === "failed" ? applyFix : undefined} />}
          <Card className="shadow-card"><CardContent className="p-0">
            <ol className="divide-y">
              {(d.events ?? []).map((e, i) => (
                <li key={i} className="flex flex-col gap-1 px-4 py-3 text-sm sm:flex-row sm:items-start sm:gap-4">
                  <span className="shrink-0 font-mono text-xs text-muted-foreground sm:w-44">{new Date(e.ts).toLocaleString()}</span>
                  <StatusBadge state={e.to} className="w-fit" />
                  <div className="min-w-0 flex-1">
                    <p className="break-words">{e.message}</p>
                    {e.log_tail && <details className="mt-1"><summary className="cursor-pointer text-xs text-primary">Log tail</summary>
                      <pre className="mt-2 max-h-64 overflow-auto whitespace-pre-wrap break-all rounded-md bg-zinc-950 p-2 font-mono text-xs text-zinc-100">{e.log_tail}</pre></details>}
                  </div>
                </li>
              ))}
            </ol>
          </CardContent></Card>
        </TabsContent>
      </Tabs>

      {admin && <LogsSheet id={id} open={logs} onOpenChange={setLogs} />}
      <ConfirmDialog open={!!confirm} onOpenChange={(o) => !o && setConfirm(null)}
        title={confirm === "stop" ? `Stop ${d.name}?` : `Remove ${d.name}?`}
        description={confirm === "stop" ? "The container is stopped and its memory released." : `Its settings, events and history are deleted. This cannot be undone.${d.system ? ` ${SYSTEM_REMOVE_TEXT}` : ""}`}
        confirmLabel={confirm === "stop" ? "Stop" : d.system ? "Remove and turn off" : "Remove"} destructive pending={act.isPending || remove.isPending}
        requireText={confirm === "stop" && d.state === "ready" ? d.name : undefined}
        onConfirm={() => { if (confirm === "stop") act.mutate("stop", { onSettled: () => setConfirm(null) }); else remove.mutate(undefined); }} />
      {dialog && <SwitchDialog mode={dialog.mode} deployment={d} initialPatch={dialog.patch} patchSource={dialog.source} onClose={() => setDialog(null)} />}
    </div>
  );
}
