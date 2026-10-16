import { useMemo, useRef } from "react";
import { useQuery } from "@tanstack/react-query";
import type { ColumnDef } from "@tanstack/react-table";
import { Download, X } from "lucide-react";
import { Link } from "react-router-dom";
import { api } from "@/api";
import { useIsAdmin } from "@/auth";
import { DataTable } from "@/components/app/data-table";
import { PageHeader } from "@/components/app/page-header";
import { StatusBadge } from "@/components/app/status-badge";
import { EmptyState, ErrorState } from "@/components/app/states";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { fmtBytes } from "@/format";
import { useActionMutation } from "@/hooks/use-action-mutation";
import { etaSeconds, fmtDuration, nextRate, type Rate } from "@/lib/progress";

export function Downloads() {
  const admin = useIsAdmin();
  const q = useQuery({ queryKey: ["downloads"], queryFn: () => api<any[]>("/downloads") });
  const cancel = useActionMutation((id: number) => api(`/downloads/${id}/cancel`, { method: "POST" }), { success: "Cancel requested", invalidate: [["downloads"]] });
  const rates = useRef(new Map<number, Rate>());
  const now = Date.now() / 1000;
  for (const d of q.data ?? []) if (d.state === "running") rates.current.set(d.id, nextRate(rates.current.get(d.id), d.bytes_done, now));
  const columns = useMemo<ColumnDef<any, any>[]>(() => [
    { accessorKey: "repo_id", header: "Repository", cell: ({ row }) => <div className="min-w-0"><div className="max-w-[40ch] truncate font-medium" title={row.original.repo_id}>{row.original.repo_id}</div><div className="text-xs text-muted-foreground">{row.original.files.length} files</div></div> },
    { id: "progress", header: "Progress", enableSorting: false, cell: ({ row }) => {
      const d = row.original;
      const pct = d.bytes_total ? Math.min(100, (d.bytes_done / d.bytes_total) * 100) : 0;
      const r = rates.current.get(d.id);
      return <div className="w-56 space-y-1"><Progress value={pct} aria-label={`${pct.toFixed(0)}%`} className="h-1.5" />
        <div className="flex justify-between text-xs tabular-nums text-muted-foreground"><span>{fmtBytes(d.bytes_done)} / {fmtBytes(d.bytes_total)}</span>
          {d.state === "running" && <span>{r?.bps ? `${fmtBytes(r.bps)}/s · ${fmtDuration(etaSeconds(r, d.bytes_total - d.bytes_done))}` : "…"}</span>}</div></div>;
    } },
    { accessorKey: "state", header: "State", cell: ({ row }) => <div className="space-y-1"><StatusBadge state={row.original.state} />{row.original.error && <p className="max-w-sm break-words text-xs text-status-critical">{row.original.error}</p>}</div> },
  ], []); // eslint-disable-line react-hooks/exhaustive-deps
  return (
    <div className="space-y-4">
      <PageHeader title="Downloads" description="Cancelling takes effect after the current file. Interrupted downloads are marked failed after a restart." />
      <ErrorState error={q.error} onRetry={() => q.refetch()} />
      <DataTable columns={columns} data={q.data ?? []} loading={q.isLoading} getRowId={(d) => String(d.id)}
        rowActions={(d) => admin && ["queued", "running"].includes(d.state)
          ? <Button size="sm" variant="ghost" disabled={cancel.isPending} onClick={() => cancel.mutate(d.id)}><X className="mr-1 h-3.5 w-3.5" />Cancel</Button> : null}
        empty={<EmptyState icon={Download} title="No downloads yet" description="Start one from the Catalog." action={<Button asChild size="sm" variant="outline"><Link to="/catalog">Open catalog</Link></Button>} />} />
    </div>
  );
}
