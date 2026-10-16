import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import type { ColumnDef } from "@tanstack/react-table";
import { Puzzle, Star } from "lucide-react";
import { api } from "@/api";
import { useIsAdmin } from "@/auth";
import { DataTable } from "@/components/app/data-table";
import { PageHeader } from "@/components/app/page-header";
import { EmptyState, ErrorState } from "@/components/app/states";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { fmtBytes } from "@/format";
import { useActionMutation } from "@/hooks/use-action-mutation";
import { favouritePath } from "@/logic";
import { cn } from "@/lib/utils";

const short = (ref: string) => (ref.startsWith("/") ? ref.split("/").pop()! : ref);

export function Planner() {
  const admin = useIsAdmin();
  const [ctx, setCtx] = useState(32768);
  const [must, setMust] = useState<string[]>([]);
  const q = useQuery({ queryKey: ["planner", ctx, must], queryFn: () => api<any>(`/planner?ctx=${ctx}${must.map((m) => `&must=${encodeURIComponent(m)}`).join("")}`) });
  const fav = useActionMutation(({ ref, on }: { ref: string; on: boolean }) => api(favouritePath(ref), { method: on ? "PUT" : "DELETE" }), { invalidate: [["planner"]] });
  const v = q.data;
  const slotOf = useMemo(() => new Map<string, number>((v?.candidates ?? []).map((c: any, i: number) => [c.ref, (i % 8) + 1])), [v?.candidates]);
  const columns = useMemo<ColumnDef<any, any>[]>(() => [
    { accessorKey: "ref", header: "Model", cell: ({ row }) => <span className="block max-w-[40ch] truncate font-medium" title={row.original.ref}>{short(row.original.ref)}</span> },
    { accessorKey: "engine", header: "Engine", cell: ({ getValue }) => (getValue() === "llamacpp" ? "llama.cpp" : "vLLM") },
    { accessorKey: "required", header: "Needs", cell: ({ getValue }) => <span className="tabular-nums">{fmtBytes(getValue())}</span> },
    { id: "favourite", header: "Favourite", enableSorting: false, cell: ({ row }) => (
      <Button variant="ghost" size="icon" className="h-8 w-8" disabled={!admin || fav.isPending} aria-pressed={row.original.favourite}
        aria-label={row.original.favourite ? `Unfavourite ${row.original.ref}` : `Favourite ${row.original.ref}`}
        onClick={() => fav.mutate({ ref: row.original.ref, on: !row.original.favourite })}>
        <Star className={cn("h-4 w-4", row.original.favourite && "fill-current text-status-warning")} />
      </Button>) },
    { id: "must", header: "Must include", enableSorting: false, cell: ({ row }) => (
      <Checkbox aria-label={`Must include ${row.original.ref}`} checked={must.includes(row.original.ref)}
        onCheckedChange={(c) => setMust(c ? [...must, row.original.ref] : must.filter((m) => m !== row.original.ref))} />) },
  ], [admin, fav, must]);
  return (
    <div className="space-y-6">
      <PageHeader title="Planner" description="Which downloaded models can run together, ranked by count, favourites and headroom." />
      <ErrorState error={q.error} onRetry={() => q.refetch()} />
      <Card className="shadow-card"><CardContent className="flex flex-wrap items-end gap-6 p-6">
        <div><div className="text-sm text-muted-foreground">Capacity</div><div className="text-3xl font-semibold tabular-nums">{fmtBytes(v?.capacity)}</div>
          <div className="text-xs text-muted-foreground">usable memory minus {fmtBytes(v?.external)} external use; ignores current deployments</div></div>
        <div className="w-40 space-y-1.5"><Label htmlFor="ctx">Context per model</Label><Input id="ctx" type="number" value={ctx} onChange={(e) => setCtx(parseInt(e.target.value || "32768", 10))} /></div>
      </CardContent></Card>
      <section className="space-y-3">
        <h2 className="text-sm font-semibold">Best combinations</h2>
        {v && !v.combos.length ? <EmptyState icon={Puzzle} title="No combination fits" description="Free memory or lower the context per model." /> : (
          <div className="grid gap-3 md:grid-cols-2">
            {(v?.combos ?? []).map((c: any, i: number) => (
              <Card key={i} className="shadow-card">
                <CardHeader className="pb-2"><CardTitle className="flex items-baseline justify-between text-sm"><span>#{i + 1} · {c.models.length} model{c.models.length > 1 ? "s" : ""}</span>
                  <span className="text-xs font-normal tabular-nums text-muted-foreground">{fmtBytes(c.total)} · {fmtBytes(c.headroom)} spare</span></CardTitle></CardHeader>
                <CardContent className="space-y-3">
                  <div className="flex h-2.5 gap-[2px]" role="img" aria-label="Capacity used by this combination">
                    {c.models.map((m: any) => <div key={m.ref} className="h-full rounded-[3px]" style={{ width: `${(m.required / (v.capacity || 1)) * 100}%`, background: `var(--chart-${slotOf.get(m.ref) ?? 1})` }} />)}
                    <div className="h-full flex-1 rounded-[3px] bg-muted" />
                  </div>
                  <ul className="space-y-1 text-xs">{c.models.map((m: any) => (
                    <li key={m.ref} className="flex items-center justify-between gap-2"><span className="flex min-w-0 items-center gap-1.5"><span aria-hidden className="h-2 w-2 shrink-0 rounded-[2px]" style={{ background: `var(--chart-${slotOf.get(m.ref) ?? 1})` }} />
                      <span className="truncate" title={m.ref}>{short(m.ref)}</span>{m.favourite && <Star className="h-3 w-3 shrink-0 fill-current text-status-warning" aria-label="favourite" />}</span>
                      <span className="shrink-0 tabular-nums text-muted-foreground">{fmtBytes(m.required)}</span></li>))}</ul>
                </CardContent>
              </Card>
            ))}
          </div>
        )}
      </section>
      <section className="space-y-3">
        <div className="flex items-center justify-between"><h2 className="text-sm font-semibold">Candidates (default configuration)</h2>
          {must.length > 0 && <Button variant="ghost" size="sm" onClick={() => setMust([])}>Clear "must include"</Button>}</div>
        <DataTable columns={columns} data={v?.candidates ?? []} loading={q.isLoading} getRowId={(c) => c.ref} />
      </section>
    </div>
  );
}
