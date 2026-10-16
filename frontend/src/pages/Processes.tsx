import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import type { ColumnDef } from "@tanstack/react-table";
import { Box, Monitor, Pause, Play } from "lucide-react";
import { api } from "@/api";
import { CopyButton } from "@/components/app/copy-button";
import { DataTable } from "@/components/app/data-table";
import { PageHeader } from "@/components/app/page-header";
import { ErrorState } from "@/components/app/states";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { fmtBytes } from "@/format";
import { containerLabel, filterScope, groupByContainer, processSearchText, type ProcessGroup, type ProcessScope } from "@/lib/processes";
import { cn } from "@/lib/utils";
import type { ContainerRef, ManagedRef, ProcessesResponse, ProcessInfo } from "@/types";

const LIMIT = 1000; // fetch everything (the server caps at 1000); scope/search are applied client-side
const REFRESH_MS = 5000;
const SCOPES: { key: ProcessScope; label: string }[] = [
  { key: "all", label: "All" },
  { key: "external", label: "External" },
  { key: "managed", label: "DGXMM-managed" },
];

function ManagedBadge({ managed }: { managed: ManagedRef | null }) {
  if (!managed) return null;
  return <Badge variant="secondary" className="text-[11px]" title={managed.name ? `Deployment ${managed.name}` : undefined}>DGXMM</Badge>;
}

function ContainerCell({ container, managed }: { container: ContainerRef | null; managed: ManagedRef | null }) {
  const Icon = container ? Box : Monitor;
  return (
    <div className="flex min-w-0 flex-wrap items-center gap-1.5">
      <Icon className="h-3.5 w-3.5 shrink-0 text-muted-foreground" aria-hidden />
      <span className="font-medium">{containerLabel(container)}</span>
      {container?.image && <Badge variant="outline" className="max-w-[28ch] truncate font-mono text-[11px]" title={container.image}>{container.image}</Badge>}
      <ManagedBadge managed={managed} />
    </div>
  );
}

function CommandText({ command, clamp = true }: { command: string; clamp?: boolean }) {
  if (!command) return <span className="text-muted-foreground">(no command line)</span>;
  return <span className={cn("block whitespace-pre-wrap break-all font-mono text-xs", clamp && "line-clamp-2")} title={command}>{command}</span>;
}

function ProcessDetail({ p }: { p: ProcessInfo }) {
  return (
    <div className="space-y-2">
      <div className="flex items-start gap-2">
        <pre className="min-w-0 flex-1 overflow-x-auto whitespace-pre-wrap break-all font-mono text-xs">{p.command || "(no command line)"}</pre>
        {p.command && <CopyButton value={p.command} label={`Copy command of pid ${p.pid}`} />}
      </div>
      <div className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-muted-foreground">
        <span>pid {p.pid}</span><span>parent {p.ppid}</span><span>exe {p.exe}</span><span>uid {p.uid ?? "?"}</span>
        {p.container && <span>container {p.container.id}</span>}
        {p.managed && <span>deployment {p.managed.name ?? p.managed.deployment_id}</span>}
        {p.command_truncated && <span>command truncated at 4 KiB</span>}
      </div>
      <p className="text-xs text-muted-foreground">Secrets (keys, tokens, passwords, credentials in URLs) are replaced with *** on the server.</p>
    </div>
  );
}

const memCols = <T extends { rss: number; gpu: number }>(): ColumnDef<T, any>[] => [
  { id: "rss", accessorFn: (r) => r.rss, header: "RAM", cell: ({ getValue }) => <span className="whitespace-nowrap">{fmtBytes(getValue())}</span>, meta: { className: "text-right" } },
  { id: "gpu", accessorFn: (r) => r.gpu, header: "GPU", cell: ({ getValue }) => <span className="whitespace-nowrap">{getValue() ? fmtBytes(getValue()) : "—"}</span>, meta: { className: "text-right" } },
];

export function Processes() {
  const [view, setView] = useState<"container" | "process">("container");
  const [scope, setScope] = useState<ProcessScope>("all");
  const [paused, setPaused] = useState(false);
  const q = useQuery({
    queryKey: ["processes"],
    queryFn: () => api<ProcessesResponse>(`/host/processes?scope=all&limit=${LIMIT}`),
    refetchInterval: paused ? false : REFRESH_MS,
  });
  const all = q.data?.processes ?? [];
  const counts = useMemo(() => Object.fromEntries(SCOPES.map((s) => [s.key, filterScope(all, s.key).length])), [all]);
  const shown = useMemo(() => filterScope(all, scope), [all, scope]);
  const groups = useMemo(() => groupByContainer(shown), [shown]);

  const processCols = useMemo<ColumnDef<ProcessInfo, any>[]>(() => [
    { id: "command", accessorFn: (r) => r.command, header: "Command", enableSorting: false, cell: ({ row }) => <CommandText command={row.original.command} />, meta: { className: "min-w-[24ch] max-w-[70ch]" } },
    { id: "container", accessorFn: (r) => containerLabel(r.container), header: "Container", cell: ({ row }) => <ContainerCell container={row.original.container} managed={row.original.managed} /> },
    { accessorKey: "user", header: "User", cell: ({ getValue }) => <span className="whitespace-nowrap">{getValue()}</span> },
    ...memCols<ProcessInfo>(),
    { accessorKey: "pid", header: "PID", meta: { className: "text-right" } },
  ], []);

  const groupCols = useMemo<ColumnDef<ProcessGroup, any>[]>(() => [
    { id: "name", accessorFn: (g) => g.name, header: "Container", cell: ({ row }) => (
      <ContainerCell container={row.original.key === "host" ? null : { id: row.original.key, name: row.original.name, image: row.original.image }} managed={row.original.managed} />) },
    { id: "count", accessorFn: (g) => g.processes.length, header: "Processes", meta: { className: "text-right" } },
    ...memCols<ProcessGroup>(),
    { id: "top", accessorFn: (g) => g.processes[0]?.command ?? "", header: "Largest process", enableSorting: false,
      cell: ({ getValue }) => <CommandText command={getValue()} />, meta: { className: "min-w-[24ch] max-w-[60ch]", hideOnMobile: true } },
  ], []);

  const scopeBar = (
    <div className="flex flex-wrap gap-1.5" role="radiogroup" aria-label="Filter by scope">
      {SCOPES.map((s) => (
        <button key={s.key} type="button" role="radio" aria-checked={scope === s.key} onClick={() => setScope(s.key)}
          className={cn("inline-flex h-8 items-center gap-1.5 rounded-full border px-3 text-xs font-medium transition-colors",
            scope === s.key ? "border-primary bg-primary text-primary-foreground" : "bg-card text-muted-foreground hover:text-foreground")}>
          {s.label}<span className="tabular-nums opacity-70">{counts[s.key] ?? 0}</span>
        </button>
      ))}
    </div>
  );

  const truncatedNote = q.data && q.data.total > all.length
    ? <p className="text-xs text-muted-foreground">Showing the {all.length} largest of {q.data.total} processes.</p> : null;

  return (
    <div className="space-y-4">
      <PageHeader title="Processes" description="Exactly what is running on this DGX: every process using memory, its container and command line (secrets redacted)."
        actions={
          <>
            {q.data && <span className="text-xs text-muted-foreground tabular-nums">{q.data.total} processes · scanned in {Math.round(q.data.duration_ms)} ms</span>}
            <Button size="sm" variant="outline" aria-pressed={paused} aria-label={paused ? "Resume auto-refresh" : "Pause auto-refresh"} onClick={() => setPaused(!paused)}>
              {paused ? <Play className="mr-1 h-4 w-4" /> : <Pause className="mr-1 h-4 w-4" />}{paused ? "Resume" : "Pause"}
            </Button>
          </>
        } />
      <ErrorState error={q.error} onRetry={() => q.refetch()} />
      <div className="flex flex-wrap items-center justify-between gap-2">
        <Tabs value={view} onValueChange={(v) => setView(v as "container" | "process")}>
          <TabsList aria-label="Group processes">
            <TabsTrigger value="container">By container</TabsTrigger>
            <TabsTrigger value="process">By process</TabsTrigger>
          </TabsList>
        </Tabs>
        {scopeBar}
      </div>
      {truncatedNote}
      {view === "process" ? (
        <DataTable key="process" columns={processCols} data={shown} loading={q.isLoading} getRowId={(r) => String(r.pid)}
          searchPlaceholder="Search command, container, user, pid" searchText={processSearchText}
          initialSort={{ id: "rss", desc: true }} renderDetail={(p) => <ProcessDetail p={p} />}
          mobileTitle={(p) => <CommandText command={p.command} />} />
      ) : (
        <DataTable key="container" columns={groupCols} data={groups} loading={q.isLoading} getRowId={(g) => g.key}
          searchPlaceholder="Search container, image, command" searchText={(g) => `${g.name} ${g.image ?? ""} ${g.processes.map(processSearchText).join(" ")}`}
          renderDetail={(g) => (
            <ul className="divide-y">
              {g.processes.map((p) => (
                <li key={p.pid} className="grid grid-cols-1 gap-1 py-2 sm:grid-cols-[minmax(0,1fr)_auto] sm:gap-4">
                  <CommandText command={p.command} clamp={false} />
                  <span className="whitespace-nowrap text-xs text-muted-foreground tabular-nums">
                    pid {p.pid} · {p.user} · RAM {fmtBytes(p.rss)}{p.gpu ? ` · GPU ${fmtBytes(p.gpu)}` : ""}
                  </span>
                </li>
              ))}
            </ul>
          )} />
      )}
    </div>
  );
}
