import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Navigate } from "react-router-dom";
import type { ColumnDef } from "@tanstack/react-table";
import { api } from "@/api";
import { useIsAdmin } from "@/auth";
import { DataTable } from "@/components/app/data-table";
import { PageHeader } from "@/components/app/page-header";
import { ErrorState } from "@/components/app/states";
import { Badge } from "@/components/ui/badge";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { fmtRelative } from "@/format";

const ALL = "__all__";

export function Audit() {
  const admin = useIsAdmin();
  if (!admin) return <Navigate to="/" replace />;
  const q = useQuery({ queryKey: ["audit"], queryFn: () => api<any[]>("/audit?limit=500"), refetchInterval: 15000 });
  const [action, setAction] = useState(ALL);
  const rows = q.data ?? [];
  const actions = useMemo(() => [...new Set(rows.map((r) => r.action))].sort(), [rows]);
  const shown = action === ALL ? rows : rows.filter((r) => r.action === action);
  const columns = useMemo<ColumnDef<any, any>[]>(() => [
    { accessorKey: "ts", header: "When", cell: ({ getValue }) => (
      <Tooltip><TooltipTrigger asChild><span className="whitespace-nowrap text-muted-foreground">{fmtRelative(getValue())}</span></TooltipTrigger>
        <TooltipContent>{new Date(getValue()).toLocaleString()}</TooltipContent></Tooltip>) },
    { accessorKey: "user", header: "User" },
    { accessorKey: "action", header: "Action", cell: ({ getValue }) => <Badge variant="outline" className="font-mono text-[11px]">{getValue()}</Badge> },
    { accessorKey: "target", header: "Target", cell: ({ getValue }) => <span className="block max-w-[32ch] truncate" title={getValue()}>{getValue()}</span> },
  ], []);
  return (
    <div className="space-y-4">
      <PageHeader title="Audit log" description="Every sign-in and change, newest first." />
      <ErrorState error={q.error} onRetry={() => q.refetch()} />
      <DataTable columns={columns} data={shown} loading={q.isLoading} getRowId={(r) => `${r.ts}-${r.action}-${r.target}`}
        searchPlaceholder="Search user, action, target" searchText={(r) => `${r.user} ${r.action} ${r.target}`}
        toolbar={<Select value={action} onValueChange={setAction}><SelectTrigger className="h-9 w-48" aria-label="Filter by action"><SelectValue /></SelectTrigger>
          <SelectContent><SelectItem value={ALL}>All actions</SelectItem>{actions.map((a) => <SelectItem key={a} value={a}>{a}</SelectItem>)}</SelectContent></Select>}
        renderDetail={(r) => Object.keys(r.detail ?? {}).length ? <pre className="overflow-x-auto whitespace-pre-wrap break-all font-mono text-xs">{JSON.stringify(r.detail, null, 2)}</pre> : null} />
    </div>
  );
}
