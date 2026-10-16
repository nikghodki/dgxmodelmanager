import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "../api";
import { useIsAdmin } from "../auth";
import { Badge, Button, Card, ErrorBox, Table } from "../components/ui";
import { fmtBytes } from "../format";

export function Downloads() {
  const admin = useIsAdmin();
  const qc = useQueryClient();
  const q = useQuery({ queryKey: ["downloads"], queryFn: () => api<any[]>("/downloads") });
  const cancel = useMutation({ mutationFn: (id: number) => api(`/downloads/${id}/cancel`, { method: "POST" }), onSettled: () => qc.invalidateQueries({ queryKey: ["downloads"] }) });
  return (
    <Card title="Downloads">
      <ErrorBox error={q.error || cancel.error} />
      <Table head={["Repo", "Files", "Progress", "State", ""]}>
        {(q.data ?? []).map((d) => {
          const pct = d.bytes_total ? Math.min(100, (d.bytes_done / d.bytes_total) * 100) : 0;
          return (
            <tr key={d.id}>
              <td className="td font-medium">{d.repo_id}</td>
              <td className="td text-xs">{d.files.length} files</td>
              <td className="td w-64">
                <div className="h-2 w-full rounded bg-slate-100"><div className="h-2 rounded bg-sky-500" style={{ width: `${pct}%` }} /></div>
                <div className="text-xs tabular-nums text-slate-500">{fmtBytes(d.bytes_done)} / {fmtBytes(d.bytes_total)}</div>
              </td>
              <td className="td"><Badge tone={d.state === "done" ? "green" : d.state === "failed" ? "red" : d.state === "running" ? "blue" : "slate"}>{d.state}</Badge>{d.error && <div className="mt-1 max-w-md text-xs text-rose-700">{d.error}</div>}</td>
              <td className="td">{admin && ["queued", "running"].includes(d.state) && <Button onClick={() => cancel.mutate(d.id)}>Cancel</Button>}</td>
            </tr>
          );
        })}
      </Table>
      <p className="mt-2 text-xs text-slate-500">Cancelling takes effect after the file currently downloading finishes. Interrupted downloads resume when restarted.</p>
    </Card>
  );
}
