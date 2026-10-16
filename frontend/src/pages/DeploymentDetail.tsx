import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { useParams } from "react-router-dom";
import { api } from "../api";
import { useIsAdmin } from "../auth";
import { SwitchDialog } from "../components/SwitchDialog";
import { MetricChart } from "../components/Charts";
import { Badge, Button, Card, ErrorBox, Table } from "../components/ui";
import { fmtBytes, fmtMs, fmtNum, stateTone } from "../format";
import type { Deployment } from "../types";
import { useLive } from "../stream";

function Performance({ id }: { id: number }) {
  const [win, setWin] = useState("live");
  const live = useLive(id);
  const hist = useQuery({ queryKey: ["hist", id, win], queryFn: () => api<any[]>(`/deployments/${id}/metrics?window=${win}`), enabled: win !== "live" });
  const data = (win === "live" ? live.data : hist.data) ?? [];
  return (
    <Card title="Performance" actions={["live", "24h", "7d"].map((w) => <Button key={w} variant={w === win ? "primary" : "ghost"} onClick={() => setWin(w)}>{w}</Button>)}>
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <div><h4 className="text-xs font-semibold text-slate-500">Throughput (tok/s)</h4>
          <MetricChart data={data} lines={[{ key: "gen_tps", label: "generation", color: "#2563eb" }, { key: "prompt_tps", label: "prompt", color: "#0d9488" }]} /></div>
        <div><h4 className="text-xs font-semibold text-slate-500">Time to first token</h4>
          <MetricChart data={data} fmt={(v) => fmtMs(v)} lines={[{ key: "ttft_p50", label: "p50", color: "#7c3aed" }, { key: "ttft_p95", label: "p95", color: "#db2777" }]} /></div>
        <div><h4 className="text-xs font-semibold text-slate-500">KV cache and queue</h4>
          <MetricChart data={data} fmt={(v) => fmtNum(v, 2)} lines={[{ key: "kv_usage", label: "KV used (0–1)", color: "#ea580c" }, { key: "waiting", label: "waiting requests", color: "#475569" }]} /></div>
        <div><h4 className="text-xs font-semibold text-slate-500">Inter-token latency / spec-decode acceptance</h4>
          <MetricChart data={data} fmt={(v) => fmtNum(v, 3)} lines={[{ key: "itl_p50", label: "ITL p50 (s)", color: "#7c3aed" }, { key: "spec_accept_rate", label: "accept rate", color: "#0d9488" }]} /></div>
      </div>
    </Card>
  );
}

function AdviceCard({ id, onApply }: { id: number; onApply: (patch: Record<string, any>) => void }) {
  const q = useQuery({ queryKey: ["advice", id], queryFn: () => api<any[]>(`/deployments/${id}/advice`), refetchInterval: 60000 });
  const admin = useIsAdmin();
  return (
    <Card title="Tuning advice">
      <ErrorBox error={q.error} />
      {(q.data ?? []).length === 0 && <p className="text-sm text-slate-500">No suggestions right now.</p>}
      <ul className="space-y-3">
        {(q.data ?? []).map((a) => (
          <li key={a.id} className="rounded-md border border-slate-200 p-3 text-sm">
            <div className="flex items-center gap-2">
              <Badge tone={a.severity === "warning" ? "red" : a.severity === "info" ? "slate" : "blue"}>{a.severity}</Badge>
              <span className="font-medium">{a.title}</span>
              <Badge tone={a.confidence === "measured" ? "green" : "slate"}>{a.confidence}</Badge>
              {admin && (a.patch || a.model_ref) && <Button className="ml-auto" onClick={() => onApply({ ...(a.patch ?? {}), ...(a.model_ref ? { model_ref: a.model_ref } : {}) })}>Apply…</Button>}
            </div>
            <p className="mt-1 text-slate-600">{a.rationale}</p>
            <p className="text-xs text-slate-500">Expected: {a.expected_effect}</p>
          </li>
        ))}
      </ul>
    </Card>
  );
}

function Benchmarks({ id, ready }: { id: number; ready: boolean }) {
  const admin = useIsAdmin();
  const qc = useQueryClient();
  const q = useQuery({ queryKey: ["benchmarks", id], queryFn: () => api<any[]>(`/deployments/${id}/benchmarks`), refetchInterval: 10000 });
  const run = useMutation({ mutationFn: () => api(`/deployments/${id}/benchmark`, { method: "POST", body: {} }), onSettled: () => qc.invalidateQueries({ queryKey: ["benchmarks", id] }) });
  const rows = q.data ?? [];
  return (
    <Card title="Benchmarks" actions={admin && <Button disabled={!ready || run.isPending} onClick={() => run.mutate()}>Run benchmark (1, 4, 8, 16)</Button>}>
      <ErrorBox error={run.error} />
      {run.isSuccess && <p className="mb-2 text-xs text-slate-500">Running… takes up to 4 minutes; results appear below as each level finishes.</p>}
      <Table head={["When", "Config", "Concurrency", "Total tok/s", "Per-request tok/s", "TTFT p50", "ITL p50"]}>
        {rows.map((b) => (
          <tr key={b.id}>
            <td className="td whitespace-nowrap text-xs">{new Date(b.ts).toLocaleString()}</td>
            <td className="td font-mono text-xs">{b.config_hash.slice(0, 8)}{b.current && <Badge tone="green">current</Badge>}</td>
            <td className="td tabular-nums">{b.concurrency}</td>
            <td className="td tabular-nums">{fmtNum(b.tps_total, 1)}</td>
            <td className="td tabular-nums">{fmtNum(b.tps_per_req, 1)}</td>
            <td className="td tabular-nums">{fmtMs(b.ttft_p50)}</td>
            <td className="td tabular-nums">{fmtMs(b.itl_p50)}</td>
          </tr>
        ))}
      </Table>
    </Card>
  );
}

export function DeploymentDetail() {
  const id = Number(useParams().id);
  const admin = useIsAdmin();
  const qc = useQueryClient();
  const dep = useQuery({ queryKey: ["deployment", id], queryFn: () => api<Deployment>(`/deployments/${id}`) });
  const [logs, setLogs] = useState<string | null>(null);
  const [dialog, setDialog] = useState<null | { patch?: Record<string, any> }>(null);
  const act = useMutation({
    mutationFn: (action: string) => api(`/deployments/${id}/${action}`, { method: "POST" }),
    onSettled: () => qc.invalidateQueries({ queryKey: ["deployment", id] }),
  });
  const d = dep.data;
  if (!d) return <ErrorBox error={dep.error} />;
  return (
    <div className="space-y-4">
      <div className="flex items-center gap-3">
        <h1 className="text-xl font-semibold">{d.name}</h1>
        <Badge tone={stateTone(d.state)}>{d.state}</Badge>
        <span className="text-sm text-slate-500">{d.model_ref} · {d.engine}</span>
        <div className="ml-auto space-x-2">
          {admin && d.state === "ready" && d.sleep_capable && <Button onClick={() => act.mutate("sleep")}>Sleep</Button>}
          {admin && d.state === "sleeping" && <Button onClick={() => act.mutate("wake")}>Wake</Button>}
          {admin && <Button onClick={() => setDialog({})}>Reconfigure…</Button>}
          {admin && ["ready", "sleeping", "starting", "loading", "unknown"].includes(d.state) && <Button variant="danger" onClick={() => act.mutate("stop")}>Stop</Button>}
          {admin && ["stopped", "failed"].includes(d.state) && <Button onClick={() => act.mutate("start")}>Start</Button>}
          {admin && <Button onClick={async () => setLogs((await api<{ logs: string }>(`/deployments/${id}/logs?tail=300`)).logs)}>Logs</Button>}
        </div>
      </div>
      <ErrorBox error={act.error} />
      {d.last_error && <div className="rounded-md bg-rose-50 px-3 py-2 text-sm text-rose-800">{d.last_error}</div>}
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
        <Card title="Endpoint">
          <dl className="space-y-1 text-sm">
            <div><dt className="text-slate-500">Served name</dt><dd className="font-mono">{d.served_name}</dd></div>
            <div><dt className="text-slate-500">On gateway network</dt><dd className="font-mono">{d.endpoint}</dd></div>
            <div><dt className="text-slate-500">On the DGX</dt><dd className="font-mono">{d.host_endpoint}</dd></div>
            <div><dt className="text-slate-500">LiteLLM</dt><dd>{d.litellm_status}{d.alias && <> · alias <span className="font-mono">{d.alias}</span></>}</dd></div>
          </dl>
        </Card>
        <Card title="Memory">
          <p className="text-sm">Reserved {fmtBytes(d.reserved_bytes)} · measured {fmtBytes(d.actual_bytes)}</p>
          {"total" in d.estimate && <p className="mt-1 text-xs text-slate-500">estimate confidence: {(d.estimate as any).confidence}</p>}
        </Card>
        <Card title="Configuration">
          <pre className="max-h-64 overflow-auto text-xs">{JSON.stringify(d.config, null, 2)}</pre>
        </Card>
      </div>
      <Performance id={id} />
      <AdviceCard id={id} onApply={(patch) => setDialog({ patch })} />
      <Benchmarks id={id} ready={d.state === "ready"} />
      <Card title="Events">
        <ol className="space-y-2 text-sm">
          {(d.events ?? []).map((e, i) => (
            <li key={i}>
              <span className="font-mono text-xs text-slate-500">{new Date(e.ts).toLocaleString()}</span>{" "}
              <Badge tone={stateTone(e.to)}>{e.to}</Badge> {e.message}
              {e.log_tail && <details className="mt-1"><summary className="cursor-pointer text-xs underline">log tail</summary><pre className="max-h-64 overflow-auto bg-slate-900 p-2 text-xs text-slate-100">{e.log_tail}</pre></details>}
            </li>
          ))}
        </ol>
      </Card>
      {logs !== null && (
        <Card title="Container logs (last 300 lines)" actions={<Button variant="ghost" onClick={() => setLogs(null)}>Close</Button>}>
          <pre className="max-h-[32rem] overflow-auto bg-slate-900 p-3 text-xs text-slate-100">{logs || "(empty)"}</pre>
        </Card>
      )}
      {dialog && <SwitchDialog mode="reconfigure" deployment={d} initialPatch={dialog.patch} onClose={() => setDialog(null)} />}
    </div>
  );
}
