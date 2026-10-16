import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { useParams } from "react-router-dom";
import { api } from "../api";
import { useIsAdmin } from "../auth";
import { Badge, Button, Card, ErrorBox } from "../components/ui";
import { fmtBytes, stateTone } from "../format";
import type { Deployment } from "../types";

export function DeploymentDetail() {
  const id = Number(useParams().id);
  const admin = useIsAdmin();
  const qc = useQueryClient();
  const dep = useQuery({ queryKey: ["deployment", id], queryFn: () => api<Deployment>(`/deployments/${id}`) });
  const [logs, setLogs] = useState<string | null>(null);
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
    </div>
  );
}
