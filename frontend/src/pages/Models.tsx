import { useMemo, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import type { ColumnDef } from "@tanstack/react-table";
import { AlertTriangle, CheckCircle2, HardDrive, RefreshCw, Rocket } from "lucide-react";
import { api } from "@/api";
import { useIsAdmin } from "@/auth";
import { DataTable } from "@/components/app/data-table";
import { PageHeader } from "@/components/app/page-header";
import { EmptyState, ErrorState } from "@/components/app/states";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { fmtBytes, shortRef } from "@/format";
import { useActionMutation } from "@/hooks/use-action-mutation";
import { cn } from "@/lib/utils";
import type { Artifact } from "@/types";

const KINDS = [{ key: "all", label: "All" }, { key: "llm", label: "LLM" }, { key: "draft", label: "Draft" }, { key: "other", label: "Other" }];

export function Models() {
  const admin = useIsAdmin();
  const qc = useQueryClient();
  const [kind, setKind] = useState("all");
  const inv = useQuery({ queryKey: ["inventory"], queryFn: () => api<Artifact[]>("/inventory") });
  const refresh = useActionMutation(() => api<Artifact[]>("/inventory/refresh", { method: "POST" }), {
    success: (d) => `Rescanned: ${d.length} models`, onSuccess: (d) => qc.setQueryData(["inventory"], d),
  });
  const all = inv.data ?? [];
  const rows = kind === "all" ? all : all.filter((a) => a.kind === kind);
  const total = all.reduce((s, a) => s + a.size_bytes, 0);
  const columns = useMemo<ColumnDef<Artifact, any>[]>(() => [
    { accessorKey: "ref", header: "Model", cell: ({ row }) => (
      <div className="min-w-0"><div className="max-w-[40ch] truncate font-medium" title={row.original.ref}>{shortRef(row.original.ref)}</div>
        {row.original.revision && <div className="font-mono text-[11px] text-muted-foreground">{row.original.revision.slice(0, 8)}</div>}</div>) },
    { accessorKey: "kind", header: "Kind", cell: ({ getValue }) => <Badge variant={getValue() === "llm" ? "default" : "secondary"} className="capitalize">{getValue()}</Badge> },
    { id: "format", header: "Format", accessorFn: (a) => `${a.format} ${a.quant ?? ""}`, cell: ({ row }) => (
      <div className="flex flex-wrap gap-1"><Badge variant="outline">{row.original.format}</Badge>{row.original.quant && <Badge variant="outline">{row.original.quant}</Badge>}
        {row.original.has_mtp && <Badge variant="outline">MTP</Badge>}{row.original.is_moe && <Badge variant="outline">MoE</Badge>}</div>) },
    { accessorKey: "weights_bytes", header: "Weights", cell: ({ getValue }) => <span className="tabular-nums">{fmtBytes(getValue())}</span> },
    { accessorKey: "max_context", header: "Context", meta: { hideOnMobile: true }, cell: ({ getValue }) => <span className="tabular-nums">{getValue()?.toLocaleString() ?? "—"}</span> },
    { accessorKey: "architecture", header: "Architecture", meta: { hideOnMobile: true, className: "text-xs text-muted-foreground" } },
    { id: "status", header: "Status", accessorFn: (a) => (a.complete ? 1 : 0), cell: ({ row }) => (
      <div className="flex items-center gap-1.5">
        {row.original.complete
          ? <span className="inline-flex items-center gap-1 text-xs text-status-good"><CheckCircle2 className="h-3.5 w-3.5" aria-hidden />Complete</span>
          : <span className="inline-flex items-center gap-1 text-xs text-status-warning"><AlertTriangle className="h-3.5 w-3.5" aria-hidden />Incomplete</span>}
        {row.original.problems.length > 0 && <AlertTriangle className="h-3.5 w-3.5 text-status-warning" aria-label={`${row.original.problems.length} problem(s)`} />}</div>) },
  ], []);
  return (
    <div className="space-y-4">
      <PageHeader title="Models on disk" description={`${all.length} models · ${fmtBytes(total)} in the Hugging Face cache and GGUF folders`}
        actions={admin && <Button variant="outline" size="sm" disabled={refresh.isPending} onClick={() => refresh.mutate(undefined)}><RefreshCw className={cn("mr-1 h-4 w-4", refresh.isPending && "animate-spin")} />Rescan</Button>} />
      <ErrorState error={inv.error} onRetry={() => inv.refetch()} />
      <DataTable columns={columns} data={rows} loading={inv.isLoading} getRowId={(a) => a.ref}
        searchPlaceholder="Search models" searchText={(a) => `${a.ref} ${a.architecture ?? ""} ${a.quant ?? ""}`}
        toolbar={<div className="flex gap-1" role="tablist" aria-label="Filter by kind">{KINDS.map((k) => (
          <button key={k.key} role="tab" aria-selected={kind === k.key} onClick={() => setKind(k.key)}
            className={cn("h-8 rounded-full border px-3 text-xs font-medium", kind === k.key ? "border-primary bg-primary text-primary-foreground" : "bg-card text-muted-foreground hover:text-foreground")}>{k.label}</button>))}</div>}
        renderDetail={(a) => a.problems.length ? <ul className="space-y-1 text-xs text-status-warning">{a.problems.map((p) => <li key={p}>• {p}</li>)}</ul> : null}
        rowActions={(a) => admin && a.complete && a.kind === "llm" && a.engines.length > 0
          ? <Button asChild size="sm" variant="outline"><Link to={`/deploy?model=${encodeURIComponent(a.ref)}`}><Rocket className="mr-1 h-3.5 w-3.5" />Deploy</Link></Button> : null}
        empty={<EmptyState icon={HardDrive} title="No models found" description="Download one from the Catalog." action={<Button asChild size="sm" variant="outline"><Link to="/catalog">Open catalog</Link></Button>} />} />
    </div>
  );
}
