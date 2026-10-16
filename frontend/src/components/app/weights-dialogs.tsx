import { useEffect, useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { toast } from "sonner";
import { AlertTriangle, Ban, Loader2 } from "lucide-react";
import { api } from "@/api";
import { ConfirmDialog } from "@/components/app/confirm-dialog";
import { ResponsiveDialog } from "@/components/app/responsive-dialog";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { fmtBytes, shortRef } from "@/format";
import { useActionMutation } from "@/hooks/use-action-mutation";
import type { Artifact, CleanupItem, CleanupPreview, CleanupResult, DeletePreview, DeleteResult, DeploymentRef } from "@/types";

/** Short model name the user types to confirm a deletion (last path segment of the ref). */
export const modelShortName = (ref: string) => ref.split("/").filter(Boolean).pop() ?? ref;

const depList = (ds: DeploymentRef[]) => ds.map((d) => `${d.name} (${d.state})`).join(", ");

export function cleanupToast(d: CleanupResult) {
  const parts = [`Cleaned up ${d.deleted.length} model${d.deleted.length === 1 ? "" : "s"} · freed ${fmtBytes(d.freed_bytes)}`];
  if (d.skipped.length) parts.push(`${d.skipped.length} skipped`);
  if (d.deleted.length) toast.success(parts.join(" · "));
  else if (d.skipped.length) toast.warning(`Nothing removed: ${d.skipped.map((s) => s.message).join("; ")}`);
  for (const f of d.failed) toast.error(f.message);
}

function Notice({ tone, children }: { tone: "warning" | "critical"; children: React.ReactNode }) {
  const Icon = tone === "critical" ? Ban : AlertTriangle;
  return (
    <div className={tone === "critical" ? "flex gap-2 rounded-md bg-status-critical-bg p-2 text-status-critical" : "flex gap-2 rounded-md bg-status-warning-bg p-2 text-status-warning"}>
      <Icon className="mt-0.5 h-4 w-4 shrink-0" aria-hidden /><div>{children}</div>
    </div>
  );
}

function PreviewBody({ p, mode }: { p: DeletePreview; mode: "delete" | "cleanup" }) {
  return (
    <div className="space-y-3">
      <p>{mode === "delete" ? "Frees" : "Removes the partial download and frees"} <strong className="text-foreground">{fmtBytes(p.bytes)}</strong> on the DGX. This cannot be undone.</p>
      <div>
        <div className="text-xs font-medium text-foreground">Deleted from disk</div>
        <ul className="mt-1 space-y-0.5">{p.paths.map((x) => <li key={x} className="break-all font-mono text-xs">{x}</li>)}</ul>
      </div>
      {p.deployments_blocking.length > 0 && (
        <Notice tone="critical">In use by {depList(p.deployments_blocking)}. {p.deployments_blocking.length === 1 ? "Stop it first." : "Stop them first."}</Notice>
      )}
      {p.download_active && <Notice tone="critical">A download for this model is still running. Cancel it on the Downloads page first.</Notice>}
      {p.deployments_affected.length > 0 && (
        <Notice tone="warning">Used by stopped deployment{p.deployments_affected.length === 1 ? "" : "s"} {depList(p.deployments_affected)}. They will fail to start until the model is downloaded again.</Notice>
      )}
    </div>
  );
}

/** Confirm deleting one model's weights (mode "delete") or cleaning up one incomplete download (mode "cleanup"). */
export function DeleteWeightsDialog({ artifact, mode, onClose }: { artifact: Artifact | null; mode: "delete" | "cleanup"; onClose: () => void }) {
  const ref = artifact?.ref ?? "";
  const preview = useQuery({
    queryKey: ["inventory", "delete-preview", ref],
    queryFn: () => api<DeletePreview>(`/inventory/delete-preview?ref=${encodeURIComponent(ref)}`),
    enabled: !!artifact, staleTime: 0, gcTime: 0,
  });
  const invalidate = [["inventory"]];
  const del = useActionMutation(() => api<DeleteResult>(`/inventory?ref=${encodeURIComponent(ref)}`, { method: "DELETE" }), {
    success: (d) => `Deleted ${shortRef(ref)} · freed ${fmtBytes(d.freed_bytes)}`, invalidate, onSuccess: onClose,
  });
  const clean = useActionMutation(() => api<CleanupResult>("/inventory/cleanup-incomplete", { method: "POST", body: { refs: [ref] } }), {
    invalidate, onSuccess: (d) => { cleanupToast(d); onClose(); },
  });
  const p = preview.data;
  const description = preview.isLoading
    ? <span className="inline-flex items-center gap-2"><Loader2 className="h-4 w-4 animate-spin" aria-hidden />Checking what will be deleted…</span>
    : preview.error ? <Notice tone="critical">{preview.error.message}</Notice>
    : p ? <PreviewBody p={p} mode={mode} /> : null;
  return (
    <ConfirmDialog open={!!artifact} onOpenChange={(o) => !o && onClose()}
      title={mode === "delete" ? `Delete ${shortRef(ref)}?` : `Clean up ${shortRef(ref)}?`}
      description={description} destructive
      confirmLabel={mode === "delete" ? "Delete weights" : "Clean up"}
      requireText={mode === "delete" && p ? p.name : undefined}
      disabled={!p || p.blocked} pending={del.isPending || clean.isPending}
      onConfirm={() => (mode === "delete" ? del.mutate(undefined) : clean.mutate(undefined))} />
  );
}

function skipText(i: CleanupItem): string {
  if (i.skip_reason === "download_active") return "Skipped: a download is still running.";
  if (i.skip_reason === "model_in_use") return `Skipped: used by ${depList(i.deployments_blocking)}. Stop it first.`;
  return `Skipped: ${i.message ?? i.skip_reason}`;
}

/** Bulk clean-up of incomplete downloads, all eligible ones pre-selected. */
export function CleanupIncompleteDialog({ open, onOpenChange, preview }: { open: boolean; onOpenChange: (o: boolean) => void; preview: CleanupPreview | undefined }) {
  const items = useMemo(() => preview?.items ?? [], [preview]);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  useEffect(() => { if (open) setSelected(new Set(items.filter((i) => i.eligible).map((i) => i.ref))); }, [open, items]);
  const chosen = items.filter((i) => i.eligible && selected.has(i.ref));
  const bytes = chosen.reduce((s, i) => s + i.bytes, 0);
  const run = useActionMutation(() => api<CleanupResult>("/inventory/cleanup-incomplete", { method: "POST", body: { refs: chosen.map((i) => i.ref) } }), {
    invalidate: [["inventory"]], onSuccess: (d) => { cleanupToast(d); onOpenChange(false); },
  });
  const toggle = (ref: string, on: boolean) => setSelected((s) => { const n = new Set(s); if (on) n.add(ref); else n.delete(ref); return n; });
  return (
    <ResponsiveDialog open={open} onOpenChange={onOpenChange} title="Clean up incomplete downloads" className="max-w-2xl"
      description="Interrupted downloads and models with missing files. The selected ones are deleted from disk.">
      <ul className="divide-y rounded-md border">
        {items.map((i) => {
          const id = `cleanup-${i.ref.replace(/[^A-Za-z0-9_-]/g, "_")}`;
          return (
            <li key={i.ref} className="flex items-start gap-3 p-3">
              <Checkbox id={id} className="mt-0.5" checked={i.eligible && selected.has(i.ref)} disabled={!i.eligible}
                onCheckedChange={(v) => toggle(i.ref, v === true)} />
              <div className="min-w-0 flex-1 text-sm">
                <label htmlFor={id} className="flex flex-wrap items-baseline justify-between gap-x-3">
                  <span className="break-all font-medium">{i.ref}</span><span className="tabular-nums text-muted-foreground">{fmtBytes(i.bytes)}</span>
                </label>
                {i.problems.length > 0 && <div className="text-xs text-muted-foreground">{i.problems.join(" · ")}</div>}
                {!i.eligible && <div className="mt-1 inline-flex items-center gap-1 text-xs text-status-warning"><AlertTriangle className="h-3.5 w-3.5" aria-hidden />{skipText(i)}</div>}
              </div>
            </li>
          );
        })}
      </ul>
      <div className="mt-4 flex flex-col-reverse gap-2 sm:flex-row sm:justify-end">
        <Button variant="outline" onClick={() => onOpenChange(false)}>Cancel</Button>
        <Button variant="destructive" disabled={chosen.length === 0 || run.isPending} onClick={() => run.mutate(undefined)}>
          {run.isPending && <Loader2 className="mr-2 h-4 w-4 animate-spin" aria-hidden />}Clean up {chosen.length} · {fmtBytes(bytes)}
        </Button>
      </div>
    </ResponsiveDialog>
  );
}
