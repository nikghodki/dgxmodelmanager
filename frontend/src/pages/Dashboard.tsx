import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import { api } from "../api";
import { useIsAdmin } from "../auth";
import { MemoryBar } from "../components/MemoryBar";
import { Badge, Button, Card, ErrorBox, Stat, Table } from "../components/ui";
import { fmtBytes, fmtNum, fmtPct, stateTone } from "../format";
import type { Deployment, HostInfo } from "../types";

function LiveCell({ id, field, fmt }: { id: number; field: string; fmt: (v: any) => string }) {
  const q = useQuery<any[]>({ queryKey: ["live", id], queryFn: () => Promise.resolve([]), staleTime: Infinity });
  const last = q.data?.[q.data.length - 1];
  return <span className="tabular-nums">{last ? fmt(last[field]) : "—"}</span>;
}

export function Dashboard() {
  const admin = useIsAdmin();
  const qc = useQueryClient();
  const host = useQuery({ queryKey: ["host"], queryFn: () => api<HostInfo>("/host") });
  const deps = useQuery({ queryKey: ["deployments"], queryFn: () => api<Deployment[]>("/deployments") });
  const act = useMutation({
    mutationFn: ({ id, action }: { id: number; action: string }) => api(`/deployments/${id}/${action}`, { method: "POST" }),
    onSettled: () => qc.invalidateQueries({ queryKey: ["deployments"] }),
  });
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
                {admin && ["ready", "sleeping", "starting", "loading", "unknown"].includes(d.state) && (
                  <Button variant="danger" onClick={() => confirm(`Stop ${d.name}?`) && act.mutate({ id: d.id, action: "stop" })}>Stop</Button>
                )}
                {admin && ["stopped", "failed"].includes(d.state) && (
                  <Button onClick={() => act.mutate({ id: d.id, action: "start" })}>Start</Button>
                )}
              </td>
            </tr>
          ))}
        </Table>
        {deps.data?.length === 0 && <p className="text-sm text-slate-500">Nothing deployed yet.</p>}
      </Card>
      {host.data && host.data.external_top.length > 0 && (
        <Card title="Largest external memory users (not managed; never touched)">
          <Table head={["PID", "Process", "RSS", "GPU", "Command"]}>
            {host.data.external_top.map((p) => (
              <tr key={p.pid}>
                <td className="td tabular-nums">{p.pid}</td><td className="td">{p.name}</td>
                <td className="td tabular-nums">{fmtBytes(p.rss)}</td><td className="td tabular-nums">{fmtBytes(p.gpu)}</td>
                <td className="td max-w-md truncate font-mono text-xs" title={p.cmdline}>{p.cmdline}</td>
              </tr>
            ))}
          </Table>
        </Card>
      )}
    </div>
  );
}
