import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import { api } from "../api";
import { useIsAdmin } from "../auth";
import { Badge, Button, Card, ErrorBox, Table } from "../components/ui";
import { fmtBytes } from "../format";
import type { Artifact } from "../types";

export function Models() {
  const admin = useIsAdmin();
  const qc = useQueryClient();
  const inv = useQuery({ queryKey: ["inventory"], queryFn: () => api<Artifact[]>("/inventory") });
  const refresh = useMutation({
    mutationFn: () => api<Artifact[]>("/inventory/refresh", { method: "POST" }),
    onSuccess: (d) => qc.setQueryData(["inventory"], d),
  });
  const rows = inv.data ?? [];
  const total = rows.reduce((s, a) => s + a.size_bytes, 0);
  return (
    <Card title={`Models on disk (${rows.length}, ${fmtBytes(total)})`}
      actions={admin && <Button onClick={() => refresh.mutate()} disabled={refresh.isPending}>Rescan</Button>}>
      <ErrorBox error={inv.error || refresh.error} />
      <Table head={["Model", "Kind", "Format / quant", "Weights", "Context", "Architecture", "Status", ""]}>
        {rows.map((a) => (
          <tr key={a.ref}>
            <td className="td font-medium break-all">{a.ref}{a.revision && <div className="font-mono text-xs text-slate-400">{a.revision.slice(0, 8)}</div>}</td>
            <td className="td"><Badge tone={a.kind === "llm" ? "blue" : a.kind === "draft" ? "violet" : "slate"}>{a.kind}</Badge></td>
            <td className="td">{a.format} {a.quant && <Badge>{a.quant}</Badge>} {a.has_mtp && <Badge tone="violet">MTP</Badge>}</td>
            <td className="td tabular-nums">{fmtBytes(a.weights_bytes)}</td>
            <td className="td tabular-nums">{a.max_context?.toLocaleString() ?? "—"}</td>
            <td className="td text-xs">{a.architecture}{a.is_moe && <Badge tone="amber">MoE</Badge>}</td>
            <td className="td">
              {a.complete ? <Badge tone="green">complete</Badge> : <Badge tone="red">incomplete</Badge>}
              {a.problems.map((p) => <div key={p} className="mt-1 text-xs text-amber-700">{p}</div>)}
            </td>
            <td className="td">
              {admin && a.complete && a.kind === "llm" && a.engines.length > 0 && (
                <Link to={`/deploy?model=${encodeURIComponent(a.ref)}`}><Button>Deploy</Button></Link>
              )}
            </td>
          </tr>
        ))}
      </Table>
    </Card>
  );
}
