import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import type { ColumnDef } from "@tanstack/react-table";
import { AlertTriangle, Download, Search } from "lucide-react";
import { api } from "@/api";
import { useIsAdmin } from "@/auth";
import { DataTable } from "@/components/app/data-table";
import { PageHeader } from "@/components/app/page-header";
import { ErrorState } from "@/components/app/states";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Sheet, SheetContent, SheetDescription, SheetHeader, SheetTitle } from "@/components/ui/sheet";
import { Switch } from "@/components/ui/switch";
import { fmtBytes } from "@/format";
import { useActionMutation } from "@/hooks/use-action-mutation";
import { cn } from "@/lib/utils";

const FIT: Record<string, { label: string; cls: string }> = {
  fits_now: { label: "Fits now", cls: "bg-status-good-bg text-status-good" },
  fits_after_evict: { label: "Fits if you evict", cls: "bg-status-warning-bg text-status-warning" },
  too_big: { label: "Too big", cls: "bg-status-critical-bg text-status-critical" },
  unknown: { label: "Size unknown", cls: "bg-status-neutral-bg text-status-neutral" },
};
const FitBadge = ({ badge }: { badge: string }) => {
  const f = FIT[badge] ?? FIT.unknown;
  return <span className={cn("inline-flex whitespace-nowrap rounded-full px-2 py-0.5 text-xs font-medium", f.cls)}>{f.label}</span>;
};
const SUPPORT = { yes: ["✓", "supported"], no: ["✗", "not supported"], unknown: ["?", "unknown"] } as const;
const Support = ({ v }: { v: boolean | null }) => {
  const [mark, text] = SUPPORT[v === null ? "unknown" : v ? "yes" : "no"];
  return <><span aria-hidden>{mark}</span><span className="sr-only">{text}</span></>;
};

export function Catalog() {
  const admin = useIsAdmin();
  const [q, setQ] = useState("qwen");
  const [term, setTerm] = useState("qwen");
  const [gguf, setGguf] = useState(false);
  const [ctx, setCtx] = useState(32768);
  const [open, setOpen] = useState<string | null>(null);
  const res = useQuery({ queryKey: ["catalog", term, gguf, ctx], queryFn: () => api<any>(`/catalog/search?q=${encodeURIComponent(term)}&gguf=${gguf}&ctx=${ctx}`) });
  const columns = useMemo<ColumnDef<any, any>[]>(() => [
    { accessorKey: "repo_id", header: "Model", cell: ({ row }) => {
      const m = row.original;
      // A real button so details are reachable from the keyboard; the row click stays as a pointer shortcut.
      return <div className="min-w-0"><button type="button" className="block max-w-full truncate rounded-sm md:max-w-[44ch] text-left font-medium hover:underline focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
        title={m.repo_id} aria-haspopup="dialog" onClick={(e) => { e.stopPropagation(); setOpen(m.repo_id); }}>{m.repo_id}</button>
        <div className="mt-1 flex flex-wrap gap-1">{m.quant && <Badge variant="outline">{m.quant}</Badge>}{m.gated && <Badge variant="outline" className="text-status-warning">gated</Badge>}{m.downloaded && <Badge variant="secondary">on disk</Badge>}</div></div>;
    } },
    { accessorKey: "downloads", header: "Downloads", cell: ({ getValue }) => <span className="tabular-nums">{Number(getValue()).toLocaleString()}</span> },
    { accessorKey: "est_bytes", header: "Approx size", cell: ({ getValue }) => <span className="tabular-nums">{fmtBytes(getValue())}</span> },
    { id: "fit", header: "Fit", accessorFn: (m) => m.fit.badge, cell: ({ row }) => <FitBadge badge={row.original.fit.badge} /> },
    { id: "engines", header: "Engines", enableSorting: false, meta: { hideOnMobile: true }, cell: ({ row }) => (
      <span className="text-xs text-muted-foreground">vLLM <Support v={row.original.support.vllm} /> · llama.cpp <Support v={row.original.support.llamacpp} /></span>) },
    { id: "llmfit", header: "llmfit", enableSorting: false, meta: { hideOnMobile: true }, cell: ({ row }) => {
      const l = row.original.llmfit;
      return l ? <span className="text-xs">{l.fit_level ?? ""} {l.estimated_tps ? `· ~${l.estimated_tps} tok/s` : ""}{l.mismatch && <AlertTriangle className="ml-1 inline h-3 w-3 text-status-warning" aria-label="differs from DGXMM by over 20%" />}</span> : <span className="text-muted-foreground">—</span>;
    } },
  ], []);
  return (
    <div className="space-y-4">
      <PageHeader title="Catalog" description="Browse Hugging Face. Fit badges use the memory free right now." />
      <Card className="shadow-card"><CardContent className="p-4">
        <form className="flex flex-wrap items-end gap-3" onSubmit={(e) => { e.preventDefault(); setTerm(q.trim()); }}>
          <div className="relative min-w-[16rem] flex-1">
            <Search className="pointer-events-none absolute left-2.5 top-2.5 h-4 w-4 text-muted-foreground" />
            <Input className="pl-8" aria-label="Search Hugging Face" placeholder="e.g. qwen3 nvfp4" value={q} onChange={(e) => setQ(e.target.value)} />
          </div>
          <div className="flex items-center gap-2"><Switch id="gguf" checked={gguf} onCheckedChange={setGguf} /><Label htmlFor="gguf">GGUF (llama.cpp)</Label></div>
          <div className="w-32 space-y-1.5"><Label htmlFor="ctx">Context</Label><Input id="ctx" type="number" value={ctx} onChange={(e) => setCtx(parseInt(e.target.value || "32768", 10))} /></div>
          <Button type="submit">Search</Button>
        </form>
        {res.data?.banner && <p className="mt-3 flex items-center gap-1.5 text-xs text-status-warning"><AlertTriangle className="h-3.5 w-3.5" />{res.data.banner}</p>}
      </CardContent></Card>
      <ErrorState error={res.error} onRetry={() => res.refetch()} />
      {res.data && <p className="text-xs text-muted-foreground">{res.data.results.length} results · {fmtBytes(res.data.free_for_new)} free now</p>}
      <DataTable columns={columns} data={res.data?.results ?? []} loading={res.isLoading} getRowId={(m) => m.repo_id} onRowClick={(m) => setOpen(m.repo_id)} />
      <Sheet open={!!open} onOpenChange={(o) => !o && setOpen(null)}>
        <SheetContent side="right" className="w-full overflow-y-auto sm:max-w-xl">{open && <Details repo={open} ctx={ctx} admin={admin} />}</SheetContent>
      </Sheet>
    </div>
  );
}

function Details({ repo, ctx, admin }: { repo: string; ctx: number; admin: boolean }) {
  const d = useQuery({ queryKey: ["catalog-details", repo, ctx], queryFn: () => api<any>(`/catalog/details?repo=${encodeURIComponent(repo)}&ctx=${ctx}`) });
  const dl = useActionMutation((include?: string[]) => api("/downloads", { method: "POST", body: { repo_id: repo, include } }), {
    success: "Download queued — follow it on the Downloads page", invalidate: [["downloads"]],
  });
  const x = d.data;
  return (
    <div className="space-y-5">
      <SheetHeader><SheetTitle className="break-all">{repo}</SheetTitle>
        <SheetDescription>{x ? `${x.architecture ?? "unknown architecture"} · ${x.quant ?? "?"} · max context ${x.max_context?.toLocaleString() ?? "?"}` : "Loading…"}</SheetDescription></SheetHeader>
      <ErrorState error={d.error} onRetry={() => d.refetch()} />
      {x && (
        <>
          {x.gated && <p className="rounded-md bg-status-warning-bg px-3 py-2 text-xs text-status-warning">Gated repository: an HF token with access is required.</p>}
          {x.estimate && (
            <div className="space-y-2 rounded-lg border p-4">
              <div className="flex items-center justify-between"><span className="text-sm font-medium">vLLM estimate</span><FitBadge badge={x.fit.badge} /></div>
              <div className="text-2xl font-semibold tabular-nums">{fmtBytes(x.estimate.total)}</div>
              <p className="text-xs text-muted-foreground">at {ctx.toLocaleString()} context, default sequences</p>
              {x.llmfit && <p className="text-xs text-muted-foreground">llmfit: {x.llmfit.memory_required_gb ?? "?"} GB · {x.llmfit.estimated_tps ?? "?"} tok/s{x.llmfit.mismatch && <span className="text-status-warning"> · differs by over 20%</span>}</p>}
            </div>
          )}
          <p className="text-sm">vLLM: <b>{x.support.vllm === null ? "unknown (registry loading)" : x.support.vllm ? "supported" : "not supported"}</b> · llama.cpp: <b>{x.support.llamacpp ? "GGUF available" : "no GGUF"}</b></p>
          {admin && x.estimate && !x.downloaded && (
            <Button disabled={dl.isPending} onClick={() => dl.mutate(undefined)}><Download className="mr-1 h-4 w-4" />Download safetensors</Button>
          )}
          {x.gguf_files.length > 0 && (
            <div className="space-y-2">
              <h4 className="text-sm font-semibold">GGUF files</h4>
              <ul className="divide-y rounded-lg border">
                {x.gguf_files.map((f: any) => (
                  <li key={f.name} className="flex items-center justify-between gap-3 px-3 py-2">
                    <div className="min-w-0"><div className="truncate font-mono text-xs" title={f.name}>{f.name}</div>
                      <div className="mt-1 flex items-center gap-2 text-xs text-muted-foreground"><span className="tabular-nums">{fmtBytes(f.size)}</span>{f.quant && <Badge variant="outline" className="text-[10px]">{f.quant}</Badge>}<FitBadge badge={f.fit.badge} /></div></div>
                    {admin && <Button size="sm" variant="outline" disabled={dl.isPending} onClick={() => dl.mutate([f.name])} aria-label={`Download ${f.name}`}><Download className="h-3.5 w-3.5" /></Button>}
                  </li>
                ))}
              </ul>
            </div>
          )}
          {x.variants.length > 0 && (
            <div className="space-y-2"><h4 className="text-sm font-semibold">Other variants</h4>
              <ul className="space-y-1 text-sm">{x.variants.map((v: any) => <li key={v.repo_id} className="flex items-center gap-2"><span className="truncate">{v.repo_id}</span>{v.quant && <Badge variant="outline" className="text-[10px]">{v.quant}</Badge>}{v.est_bytes && <span className="text-xs tabular-nums text-muted-foreground">{fmtBytes(v.est_bytes)}</span>}</li>)}</ul></div>
          )}
        </>
      )}
    </div>
  );
}
