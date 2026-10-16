import { useEffect, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { useNavigate, useParams } from "react-router-dom";
import { Activity, ArrowLeftRight, FileText, Gauge, Lightbulb, MoreHorizontal, Moon, Play, Power, RefreshCw, Settings2, Sun, Trash2 } from "lucide-react";
import { api } from "@/api";
import { useIsAdmin } from "@/auth";
import { ConfirmDialog } from "@/components/app/confirm-dialog";
import { CopyButton } from "@/components/app/copy-button";
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
import { fmtBytes, fmtMs, fmtNum } from "@/format";
import { useActionMutation } from "@/hooks/use-action-mutation";
import { benchmarkSeries } from "@/lib/benchmarks";
import { estimateParts } from "@/lib/estimate";
import { adviceApplicable, deploymentActions } from "@/logic";
import { useLive } from "@/stream";
import type { Deployment } from "@/types";
import { SwitchDialog } from "@/components/app/switch-dialog";

const WINDOWS = ["live", "24h", "7d"] as const;

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
    <div className="grid gap-4 lg:grid-cols-2">
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

function Advice({ d, onApply }: { d: Deployment; onApply: (patch: Record<string, any>) => void }) {
  const admin = useIsAdmin();
  const q = useQuery({ queryKey: ["advice", d.id], queryFn: () => api<any[]>(`/deployments/${d.id}/advice`), refetchInterval: 60000 });
  if (q.error) return <ErrorState error={q.error} onRetry={() => q.refetch()} />;
  if (!q.data?.length) return <EmptyState icon={Lightbulb} title="No suggestions right now" description="Advice appears as DGXMM collects metrics and benchmarks for this deployment." />;
  const tone = (s: string) => (s === "warning" ? "bg-status-critical-bg text-status-critical" : s === "info" ? "bg-status-neutral-bg text-status-neutral" : "bg-primary/10 text-primary");
  return (
    <div className="grid gap-3 md:grid-cols-2">
      {q.data.map((a) => (
        <Card key={a.id} className="shadow-card">
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
      ))}
    </div>
  );
}

function Benchmarks({ d }: { d: Deployment }) {
  const admin = useIsAdmin();
  const q = useQuery({ queryKey: ["benchmarks", d.id], queryFn: () => api<any[]>(`/deployments/${d.id}/benchmarks`), refetchInterval: 10000 });
  const run = useActionMutation(() => api(`/deployments/${d.id}/benchmark`, { method: "POST", body: {} }), {
    success: "Benchmark started — results appear as each level finishes (up to 4 minutes)", invalidate: [["benchmarks", d.id]],
  });
  const rows = q.data ?? [];
  const { data, keys } = benchmarkSeries(rows);
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
                {["When", "Config", "Concurrency", "Total tok/s", "Per-request tok/s", "TTFT p50", "ITL p50"].map((h) => <th key={h} className="px-4 py-2 font-medium">{h}</th>)}
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

function KV({ label, value, mono, copy }: { label: string; value: React.ReactNode; mono?: boolean; copy?: string | null }) {
  return (
    <div className="flex items-center justify-between gap-3 border-b py-2 last:border-0">
      <dt className="shrink-0 text-sm text-muted-foreground">{label}</dt>
      <dd className={`flex min-w-0 items-center gap-1 text-right text-sm ${mono ? "font-mono text-xs" : ""}`}>
        <span className="min-w-0 break-all">{value ?? "—"}</span>{copy && <CopyButton value={copy} label={`Copy ${label}`} />}
      </dd>
    </div>
  );
}

export function DeploymentDetail() {
  const id = Number(useParams().id);
  const admin = useIsAdmin();
  const nav = useNavigate();
  const dep = useQuery({ queryKey: ["deployment", id], queryFn: () => api<Deployment>(`/deployments/${id}`) });
  const [logs, setLogs] = useState(false);
  const [dialog, setDialog] = useState<null | { mode: "switch" | "reconfigure"; patch?: Record<string, any> }>(null);
  const [confirm, setConfirm] = useState<null | "stop" | "remove">(null);
  const act = useActionMutation((action: string) => api(`/deployments/${id}/${action}`, { method: "POST" }), {
    success: (_r, a) => `${a[0].toUpperCase()}${a.slice(1)} requested`, invalidate: [["deployment", id], ["deployments"]],
  });
  const remove = useActionMutation(() => api(`/deployments/${id}`, { method: "DELETE" }), {
    success: "Deployment removed", invalidate: [["deployments"]], onSuccess: () => nav("/deployments"),
  });

  if (dep.isLoading) return <PageSkeleton />;
  const d = dep.data;
  if (!d) return <ErrorState error={dep.error ?? new Error("Deployment not found")} onRetry={() => dep.refetch()} />;
  const acts = deploymentActions(d, admin);
  const parts = estimateParts(d.estimate ?? {});

  return (
    <div className="space-y-6">
      <PageHeader
        title={<span className="flex items-center gap-3">{d.name}<StatusBadge state={d.state} /></span>}
        description={<span className="break-all">{d.model_ref} · {d.engine === "llamacpp" ? "llama.cpp" : "vLLM"}</span>}
        actions={<>
          {acts.includes("sleep") && <Button variant="outline" size="sm" onClick={() => act.mutate("sleep")}><Moon className="mr-1 h-4 w-4" />Sleep</Button>}
          {acts.includes("wake") && <Button variant="outline" size="sm" onClick={() => act.mutate("wake")}><Sun className="mr-1 h-4 w-4" />Wake</Button>}
          {acts.includes("start") && <Button size="sm" onClick={() => act.mutate("start")}><Play className="mr-1 h-4 w-4" />Start</Button>}
          {acts.includes("reconfigure") && <Button variant="outline" size="sm" onClick={() => setDialog({ mode: "reconfigure" })}><Settings2 className="mr-1 h-4 w-4" />Reconfigure</Button>}
          {admin && <Button variant="outline" size="sm" onClick={() => setLogs(true)}><FileText className="mr-1 h-4 w-4" />Logs</Button>}
          {(acts.includes("stop") || acts.includes("remove") || acts.includes("switch")) && (
            <DropdownMenu>
              <DropdownMenuTrigger asChild><Button variant="outline" size="icon" className="h-8 w-8" aria-label="More actions"><MoreHorizontal className="h-4 w-4" /></Button></DropdownMenuTrigger>
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
      {d.last_error && <ErrorState error={new Error(d.last_error)} />}

      <Tabs defaultValue="overview">
        <TabsList className="w-full justify-start overflow-x-auto sm:w-auto">
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="performance"><Activity className="mr-1 h-3.5 w-3.5" />Performance</TabsTrigger>
          <TabsTrigger value="advice"><Lightbulb className="mr-1 h-3.5 w-3.5" />Advice</TabsTrigger>
          <TabsTrigger value="benchmarks"><Gauge className="mr-1 h-3.5 w-3.5" />Benchmarks</TabsTrigger>
          <TabsTrigger value="events">Events</TabsTrigger>
        </TabsList>

        <TabsContent value="overview" className="mt-4 grid gap-4 lg:grid-cols-3">
          <Card className="shadow-card">
            <CardHeader className="pb-2"><CardTitle className="text-sm">Endpoints</CardTitle><CardDescription className="text-xs">OpenAI-compatible</CardDescription></CardHeader>
            <CardContent><dl>
              <KV label="Served name" value={d.served_name} mono copy={d.served_name} />
              <KV label="Gateway network" value={d.endpoint} mono copy={d.endpoint} />
              <KV label="On the DGX" value={d.host_endpoint} mono copy={d.host_endpoint} />
              <KV label="LiteLLM" value={<span className="inline-flex items-center gap-2"><StatusBadge state={d.litellm_status} />{d.alias && <span className="font-mono text-xs">{d.alias}</span>}</span>} />
            </dl></CardContent>
          </Card>
          <Card className="shadow-card">
            <CardHeader className="pb-2"><CardTitle className="text-sm">Memory</CardTitle>
              <CardDescription className="text-xs">confidence: {(d.estimate as any)?.confidence ?? "—"}</CardDescription></CardHeader>
            <CardContent><dl>
              <KV label="Reserved" value={<span className="tabular-nums">{fmtBytes(d.reserved_bytes)}</span>} />
              <KV label="Measured" value={<span className="tabular-nums">{fmtBytes(d.actual_bytes)}</span>} />
              {parts.map((p) => <KV key={p.key} label={p.label} value={<span className="tabular-nums">{fmtBytes(p.bytes)}</span>} />)}
            </dl></CardContent>
          </Card>
          <Card className="shadow-card">
            <CardHeader className="pb-2"><CardTitle className="text-sm">Configuration</CardTitle></CardHeader>
            <CardContent><dl>
              {Object.entries(d.config ?? {}).filter(([, v]) => v !== null && !(Array.isArray(v) && !v.length)).map(([k, v]) => (
                <KV key={k} label={k} mono value={typeof v === "object" ? JSON.stringify(v) : String(v)} />
              ))}
            </dl></CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="performance" className="mt-4"><Performance id={id} /></TabsContent>
        <TabsContent value="advice" className="mt-4"><Advice d={d} onApply={(patch) => setDialog({ mode: "reconfigure", patch })} /></TabsContent>
        <TabsContent value="benchmarks" className="mt-4"><Benchmarks d={d} /></TabsContent>
        <TabsContent value="events" className="mt-4">
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
        description={confirm === "stop" ? "The container is stopped and its memory released." : "Its settings, events and history are deleted. This cannot be undone."}
        confirmLabel={confirm === "stop" ? "Stop" : "Remove"} destructive pending={act.isPending || remove.isPending}
        requireText={confirm === "stop" && d.state === "ready" ? d.name : undefined}
        onConfirm={() => { if (confirm === "stop") act.mutate("stop", { onSettled: () => setConfirm(null) }); else remove.mutate(undefined); }} />
      {dialog && <SwitchDialog mode={dialog.mode} deployment={d} initialPatch={dialog.patch} onClose={() => setDialog(null)} />}
    </div>
  );
}
