import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "../api";
import { useIsAdmin } from "../auth";
import { Badge, Button, Card, ErrorBox, Input, Table } from "../components/ui";
import { fmtBytes } from "../format";
import { favouritePath } from "../logic";

export function Planner() {
  const admin = useIsAdmin();
  const qc = useQueryClient();
  const [ctx, setCtx] = useState(32768);
  const [must, setMust] = useState<string[]>([]);
  const q = useQuery({ queryKey: ["planner", ctx, must], queryFn: () => api<any>(`/planner?ctx=${ctx}${must.map((m) => `&must=${encodeURIComponent(m)}`).join("")}`) });
  const fav = useMutation({
    mutationFn: ({ ref, on }: { ref: string; on: boolean }) => api(favouritePath(ref), { method: on ? "PUT" : "DELETE" }),
    onSettled: () => qc.invalidateQueries({ queryKey: ["planner"] }),
  });
  const v = q.data;
  return (
    <div className="space-y-4">
      <Card title="What can run together?">
        <div className="flex items-end gap-3 text-sm">
          <label>Context per model <Input className="!w-28" type="number" value={ctx} onChange={(e) => setCtx(parseInt(e.target.value || "32768", 10))} /></label>
          {v && <span className="text-slate-600">Capacity {fmtBytes(v.capacity)} (total minus margin, OS reserve and {fmtBytes(v.external)} external use; ignores what is deployed and ready now)</span>}
        </div>
        <ErrorBox error={q.error} />
      </Card>
      <Card title="Best combinations">
        {(v?.combos ?? []).map((c: any, i: number) => (
          <div key={i} className="flex items-center justify-between border-b py-2 text-sm last:border-0">
            <div className="flex flex-wrap gap-2">{c.models.map((m: any) => <Badge key={m.ref} tone={m.favourite ? "violet" : "blue"}>{m.ref} · {fmtBytes(m.required)}</Badge>)}</div>
            <div className="tabular-nums text-slate-600">{fmtBytes(c.total)} · {fmtBytes(c.headroom)} spare</div>
          </div>
        ))}
        {v && !v.combos.length && <p className="text-sm text-slate-500">No downloaded model fits in the current capacity.</p>}
      </Card>
      <Card title="Candidates (default configuration)">
        <Table head={["Model", "Engine", "Needs", "Favourite", "Must include"]}>
          {(v?.candidates ?? []).map((c: any) => (
            <tr key={c.ref}>
              <td className="td">{c.ref}</td><td className="td">{c.engine}</td><td className="td tabular-nums">{fmtBytes(c.required)}</td>
              <td className="td">{admin ? <input type="checkbox" checked={c.favourite} onChange={(e) => fav.mutate({ ref: c.ref, on: e.target.checked })} /> : c.favourite ? "★" : ""}</td>
              <td className="td"><input type="checkbox" checked={must.includes(c.ref)} onChange={(e) => setMust(e.target.checked ? [...must, c.ref] : must.filter((m) => m !== c.ref))} /></td>
            </tr>
          ))}
        </Table>
        <Button className="mt-2" variant="ghost" onClick={() => setMust([])}>Clear "must include"</Button>
      </Card>
    </div>
  );
}
