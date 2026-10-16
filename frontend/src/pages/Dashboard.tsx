import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import { api } from "../api";
import { useIsAdmin } from "../auth";
import { MemoryBar } from "../components/MemoryBar";
import { SwitchDialog } from "../components/SwitchDialog";
import { MetricChart } from "../components/Charts";
import { Badge, Button, Card, ErrorBox, Stat, Table } from "../components/ui";
import { fmtBytes, fmtNum, fmtPct, stateTone } from "../format";
import { type Deployment, type HostInfo } from "../types";
import { useLive } from "../stream";
import { canRemove } from "../logic";

function LiveCell({ id, field, fmt }: { id: number; field: string; fmt: (v: any) => string }) {
  const q = useLive(id);
  const last = q.data?.[q.data.length - 1];
  return <span className="tabular-nums">{last ? fmt(last[field]) : "—"}</span>;
}

export function Dashboard() {
  const admin = useIsAdmin();
  const qc = useQueryClient();
  const [dialog, setDialog] = useState<Deployment | null>(null);
  const host = useQuery({ queryKey: ["host"], queryFn: () => api<HostInfo>("/host") });
  const deps = useQuery({ queryKey: ["deployments"], queryFn: () => api<Deployment[]>("/deployments") });
  const act = useMutation({
    mutationFn: ({ id, action }: { id: number; action: string }) =>
      action === "remove" ? api(`/deployments/${id}`, { method: "DELETE" }) : api(`/deployments/${id}/${action}`, { method: "POST" }),
    onSettled: () => qc.invalidateQueries({ queryKey: ["deployments"] }),
  });
  const hostLive = useQuery({ queryKey: ["host-metrics"], queryFn: () => api<any[]>("/host/metrics?window=live"), refetchInterval: 10000 });
  const b = host.data?.budget;
  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h1 className="text-xl font-semibold">Dashboard</h1>
        {admin && <Link to="/deploy"><Button variant="primary">Deploy a model</Button></Link>}
      </div>
      {b && (
        <>
          <div className="grid grid-cols-2 gap-3 md:grid-cols-6">
            <Stat label="Total memory" value={fmtBytes(b.total)} />
            <Stat label="Available (kernel)" value={fmtBytes(b.available)} />
            <Stat label="Free for new" value={fmtBytes(b.free_for_new)} sub={`after ${fmtBytes(b.margin)} safety margin`} />
            <Stat label="Managed models" value={fmtBytes(b.committed)} sub={`${fmtBytes(b.managed_actual)} measured`} />
            <Stat label="External" value={fmtBytes(b.external)} sub="processes DGXMM does not manage" />
            <Stat label="Model disk free" value={fmtBytes(host.data?.disk?.free)} sub={host.data?.gpu?.util != null ? `GPU ${host.data.gpu.util}% · ${host.data.gpu.temp_c ?? "?"}°C` : undefined} />
          </div>
          <Card title="Unified memory"><MemoryBar budget={b} /></Card>
          <Card title="Memory over the last hour">
            <MetricChart data={hostLive.data ?? []} fmt={(v) => fmtBytes(v, 0)} lines={[{ key: "used", label: "used", color: "#2563eb" }, { key: "external", label: "external", color: "#ea580c" }, { key: "free_for_new", label: "free for new", color: "#0d9488" }]} />
          </Card>
        </>
      )}
      <ErrorBox error={host.error || deps.error || act.error} />
      <Card title="Deployments">
        <Table head={["Name", "Model", "Engine", "State", "Reserved / actual", "Gen tok/s", "KV", "Queue", "LiteLLM", ""]}>
          {(deps.data ?? []).map((d) => (
            <tr key={d.id}>
              <td className="td font-medium"><Link className="underline" to={`/deployments/${d.id}`}>{d.name}</Link></td>
              <td className="td">{d.model_ref}</td>
              <td className="td">{d.engine}</td>
              <td className="td"><Badge tone={stateTone(d.state)}>{d.state}</Badge>{d.last_error && <div className="mt-1 max-w-xs text-xs text-rose-700">{d.last_error}</div>}</td>
              <td className="td tabular-nums">{fmtBytes(d.reserved_bytes)} / {fmtBytes(d.actual_bytes)}</td>
              <td className="td"><LiveCell id={d.id} field="gen_tps" fmt={(v) => fmtNum(v, 1)} /></td>
              <td className="td"><LiveCell id={d.id} field="kv_usage" fmt={fmtPct} /></td>
              <td className="td"><LiveCell id={d.id} field="waiting" fmt={(v) => (v == null ? "—" : String(v))} /></td>
              <td className="td"><Badge tone={d.litellm_status === "synced" ? "green" : d.litellm_status === "error" ? "red" : "slate"}>{d.litellm_status}</Badge></td>
              <td className="td whitespace-nowrap space-x-1">
                {admin && d.state === "ready" && d.sleep_capable && <Button onClick={() => act.mutate({ id: d.id, action: "sleep" })}>Sleep</Button>}
                {admin && d.state === "sleeping" && <Button onClick={() => act.mutate({ id: d.id, action: "wake" })}>Wake</Button>}
                {admin && ["ready", "sleeping"].includes(d.state) && <Button onClick={() => setDialog(d)}>Switch…</Button>}
                {admin && ["ready", "sleeping", "starting", "loading", "unknown"].includes(d.state) && (
                  <Button variant="danger" onClick={() => confirm(`Stop ${d.name}?`) && act.mutate({ id: d.id, action: "stop" })}>Stop</Button>
                )}
                {admin && ["stopped", "failed"].includes(d.state) && (
                  <Button onClick={() => act.mutate({ id: d.id, action: "start" })}>Start</Button>
                )}
                {admin && canRemove(d.state) && (
                  <Button variant="danger" onClick={() => confirm(`Remove ${d.name}? Its settings and history are deleted.`) && act.mutate({ id: d.id, action: "remove" })}>Remove</Button>
                )}
              </td>
            </tr>
          ))}
        </Table>
        {deps.data?.length === 0 && <p className="text-sm text-slate-500">Nothing deployed yet.</p>}
      </Card>
      {dialog && <SwitchDialog mode="switch" deployment={dialog} onClose={() => setDialog(null)} />}
      <AdoptCard />
      {host.data && host.data.external_top.length > 0 && (
        <Card title="Largest external memory users (not managed; never touched)">
          <Table head={["PID", "Process", "RSS", "GPU"]}>
            {host.data.external_top.map((p) => (
              <tr key={p.pid}>
                <td className="td tabular-nums">{p.pid}</td><td className="td">{p.name}</td>
                <td className="td tabular-nums">{fmtBytes(p.rss)}</td><td className="td tabular-nums">{fmtBytes(p.gpu)}</td>
              </tr>
            ))}
          </Table>
        </Card>
      )}
    </div>
  );
}

function AdoptCard() {
  const admin = useIsAdmin();
  const qc = useQueryClient();
  const rows = useQuery({ queryKey: ["adoptable"], queryFn: () => api<any[]>("/containers/adoptable") });
  const adopt = useMutation({
    mutationFn: (c: { name: string }) => api(`/containers/${encodeURIComponent(c.name)}/adopt`, { method: "POST", body: { name: c.name.replace(/[^a-z0-9-]/g, "-").replace(/^dmm-/, "").slice(0, 40) } }),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ["adoptable"] }); qc.invalidateQueries({ queryKey: ["deployments"] }); },
  });
  if (!rows.data?.length) return null;
  return (
    <Card title="vLLM containers not managed by DGXMM">
      <ErrorBox error={adopt.error} />
      <Table head={["Container", "Status", "Image", ""]}>
        {rows.data.map((c) => (
          <tr key={c.id}>
            <td className="td font-mono">{c.name}</td><td className="td">{c.status}</td><td className="td text-xs">{c.image}</td>
            <td className="td">{c.adopted_as ? <Badge tone="green">adopted as {c.adopted_as}</Badge>
              : c.adoptable === false ? <span className="text-xs text-slate-500">{c.reason}</span>
              : admin && <Button onClick={() => adopt.mutate(c)}>Adopt settings</Button>}</td>
          </tr>
        ))}
      </Table>
      <p className="mt-2 text-xs text-slate-500">Adopting copies the container's settings into a stopped DGXMM deployment. The original container is left as it is; stop it before starting the adopted one.</p>
    </Card>
  );
}
