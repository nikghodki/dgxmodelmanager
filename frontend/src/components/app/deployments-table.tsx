import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import type { ColumnDef } from "@tanstack/react-table";
import { ArrowLeftRight, ExternalLink, Moon, MoreHorizontal, Play, Power, Settings2, Sun, Trash2 } from "lucide-react";
import { api } from "@/api";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuSeparator, DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { engineLabel, fmtBytes, fmtNum, fmtPct, shortRef } from "@/format";
import { useActionMutation } from "@/hooks/use-action-mutation";
import { deploymentActions } from "@/logic";
import { useLive } from "@/stream";
import type { Deployment } from "@/types";
import { ConfirmDialog } from "./confirm-dialog";
import { DataTable } from "./data-table";
import { StatusBadge } from "./status-badge";

function Live({ id, field, fmt }: { id: number; field: string; fmt: (v: any) => string }) {
  const q = useLive(id);
  const last = q.data?.[q.data.length - 1];
  const v = last?.[field];
  return <span className="tabular-nums">{typeof v === "number" && Number.isFinite(v) ? fmt(v) : "—"}</span>;
}

type Pending = { kind: "stop" | "remove"; d: Deployment } | null;

export function DeploymentsTable({ deployments, isAdmin, loading, searchable, onSwitch, onReconfigure }: {
  deployments: Deployment[]; isAdmin: boolean; loading?: boolean; searchable?: boolean;
  onSwitch: (d: Deployment) => void; onReconfigure: (d: Deployment) => void;
}) {
  const nav = useNavigate();
  const [pending, setPending] = useState<Pending>(null);
  const act = useActionMutation(
    ({ d, action }: { d: Deployment; action: string }) =>
      action === "remove" ? api(`/deployments/${d.id}`, { method: "DELETE" }) : api(`/deployments/${d.id}/${action}`, { method: "POST" }),
    { success: (_r, v) => `${v.d.name}: ${v.action} requested`, invalidate: [["deployments"], ["host"]] },
  );

  const columns: ColumnDef<Deployment, any>[] = [
    { accessorKey: "name", header: "Name", cell: ({ row }) => (
      <Link to={`/deployments/${row.original.id}`} className="font-medium hover:underline">{row.original.name}</Link>) },
    { accessorKey: "model_ref", header: "Model", cell: ({ row }) => (
      <span title={row.original.model_ref} className="block max-w-[26ch] truncate text-muted-foreground">{shortRef(row.original.model_ref)}</span>) },
    { accessorKey: "engine", header: "Engine", meta: { hideOnMobile: true }, cell: ({ getValue }) => engineLabel(getValue()) },
    { accessorKey: "state", header: "State", cell: ({ row }) => (
      row.original.last_error ? (
        <Tooltip><TooltipTrigger asChild><span><StatusBadge state={row.original.state} /></span></TooltipTrigger>
          <TooltipContent className="max-w-xs break-words">{row.original.last_error}</TooltipContent></Tooltip>
      ) : <StatusBadge state={row.original.state} />) },
    { id: "memory", header: "Memory", accessorFn: (d) => d.reserved_bytes, cell: ({ row }) => (
      <span className="tabular-nums">{fmtBytes(row.original.actual_bytes)} <span className="text-muted-foreground">/ {fmtBytes(row.original.reserved_bytes)}</span></span>) },
    { id: "tps", header: "Tok/s", enableSorting: false, cell: ({ row }) => <Live id={row.original.id} field="gen_tps" fmt={(v) => fmtNum(v, 1)} /> },
    { id: "kv", header: "KV", enableSorting: false, meta: { hideOnMobile: true }, cell: ({ row }) => <Live id={row.original.id} field="kv_usage" fmt={fmtPct} /> },
    { id: "litellm", header: "LiteLLM", meta: { hideOnMobile: true }, accessorFn: (d) => d.model_name ?? d.alias ?? d.served_name ?? "", cell: ({ row }) => {
      const name = row.original.model_name ?? row.original.alias ?? row.original.served_name;
      return (
        <span className="flex flex-col items-start gap-1">
          {name && <span title={name} className="block max-w-[22ch] truncate font-mono text-xs">{name}</span>}
          <StatusBadge state={row.original.litellm_status} />
        </span>
      );
    } },
  ];

  const menu = (d: Deployment) => {
    const acts = deploymentActions(d, isAdmin);
    return (
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <Button variant="ghost" size="icon" className="h-8 w-8" aria-label={`Actions for ${d.name}`}><MoreHorizontal className="h-4 w-4" /></Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="end" className="w-48">
          <DropdownMenuItem onSelect={() => nav(`/deployments/${d.id}`)}><ExternalLink className="mr-2 h-4 w-4" />Open</DropdownMenuItem>
          {acts.includes("sleep") && <DropdownMenuItem onSelect={() => act.mutate({ d, action: "sleep" })}><Moon className="mr-2 h-4 w-4" />Sleep</DropdownMenuItem>}
          {acts.includes("wake") && <DropdownMenuItem onSelect={() => act.mutate({ d, action: "wake" })}><Sun className="mr-2 h-4 w-4" />Wake</DropdownMenuItem>}
          {acts.includes("start") && <DropdownMenuItem onSelect={() => act.mutate({ d, action: "start" })}><Play className="mr-2 h-4 w-4" />Start</DropdownMenuItem>}
          {acts.includes("switch") && <DropdownMenuItem onSelect={() => onSwitch(d)}><ArrowLeftRight className="mr-2 h-4 w-4" />Switch model…</DropdownMenuItem>}
          {acts.includes("reconfigure") && <DropdownMenuItem onSelect={() => onReconfigure(d)}><Settings2 className="mr-2 h-4 w-4" />Reconfigure…</DropdownMenuItem>}
          {(acts.includes("stop") || acts.includes("remove")) && <DropdownMenuSeparator />}
          {acts.includes("stop") && <DropdownMenuItem className="text-status-critical focus:text-status-critical" onSelect={() => setPending({ kind: "stop", d })}><Power className="mr-2 h-4 w-4" />Stop</DropdownMenuItem>}
          {acts.includes("remove") && <DropdownMenuItem className="text-status-critical focus:text-status-critical" onSelect={() => setPending({ kind: "remove", d })}><Trash2 className="mr-2 h-4 w-4" />Remove</DropdownMenuItem>}
        </DropdownMenuContent>
      </DropdownMenu>
    );
  };

  return (
    <>
      <DataTable columns={columns} data={deployments} getRowId={(d) => String(d.id)} loading={loading} rowActions={menu}
        searchPlaceholder={searchable ? "Search deployments" : undefined}
        searchText={(d) => `${d.name} ${d.model_ref} ${d.engine} ${d.state} ${d.model_name ?? ""}`}
        mobileTitle={(d) => <div className="flex items-center gap-2"><Link to={`/deployments/${d.id}`} className="hover:underline">{d.name}</Link></div>} />
      <ConfirmDialog open={!!pending} onOpenChange={(o) => !o && setPending(null)}
        title={pending?.kind === "stop" ? `Stop ${pending?.d.name}?` : `Remove ${pending?.d.name}?`}
        description={pending?.kind === "stop"
          ? "The container is stopped and its memory is released. LiteLLM routes to it are removed. You can start it again later."
          : "Its settings, events and history are deleted. This cannot be undone."}
        confirmLabel={pending?.kind === "stop" ? "Stop" : "Remove"} destructive pending={act.isPending}
        requireText={pending?.kind === "stop" && pending.d.state === "ready" ? pending.d.name : undefined}
        onConfirm={() => pending && act.mutate({ d: pending.d, action: pending.kind }, { onSettled: () => setPending(null) })} />
    </>
  );
}
