import { useEffect, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Loader2 } from "lucide-react";
import { ApiError, api } from "@/api";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { fmtBytes } from "@/format";
import { useActionMutation } from "@/hooks/use-action-mutation";
import type { CommandInterpretation, CommandResult, Deployment, FreePlans, PlanOption } from "@/types";
import { ConfirmDialog } from "./confirm-dialog";
import { PlanChooser } from "./plan-chooser";
import { ResponsiveDialog } from "./responsive-dialog";
import { ErrorState } from "./states";
import { SwitchDialog } from "./switch-dialog";

const GIB = 1024 ** 3;

/** Ask the server to interpret a typed command (POST /decider/command). Interpretation only: nothing runs. */
export const interpretCommand = (text: string) => api<CommandResult>("/decider/command", { method: "POST", body: { text } });

/** Where a navigate / deploy interpretation leads (the palette navigates; the wizard is the confirmation). */
export function interpretationHref(i: CommandInterpretation): string | null {
  if (i.action === "navigate") return i.target?.kind === "page" ? i.target.path ?? null : i.target?.id != null ? `/deployments/${i.target.id}` : null;
  if (i.action === "deploy" && i.params.model_ref) {
    const q = new URLSearchParams({ model: i.params.model_ref });
    if (i.params.context) q.set("ctx", String(i.params.context));
    return `/deploy?${q}`;
  }
  return null;
}

const VERB: Record<string, string> = { start: "Start", sleep: "Put to sleep", wake: "Wake", stop: "Stop", restart: "Restart" };
const WHAT: Record<string, string> = {
  start: "It is started with its current configuration (fit check and start-up queue as usual).",
  sleep: "Its KV cache is freed and the weights offloaded; waking it up is fast.",
  wake: "It is woken up and serves requests again.",
  stop: "The container is stopped and its memory released. You can start it again later.",
  restart: "Two steps: it is stopped, then started again with its current configuration. If the start fails (e.g. it no longer fits), it stays stopped.",
};

/** Lifecycle confirmation for start / sleep / wake / stop / restart; the request is sent only on confirm. */
function LifecycleConfirm({ interp, onClose }: { interp: CommandInterpretation; onClose: () => void }) {
  const t = interp.target!;
  const a = interp.action;
  const run = useActionMutation(async () => {
    if (a === "restart") {
      if (t.state !== "stopped" && t.state !== "failed") await api(`/deployments/${t.id}/stop`, { method: "POST" });
      return api(`/deployments/${t.id}/start`, { method: "POST" });
    }
    return api(`/deployments/${t.id}/${a}`, { method: "POST" });
  }, { success: `${VERB[a]} ${t.name}: requested`, invalidate: [["deployments"], ["deployment", t.id]], onSuccess: onClose });
  const destructive = a === "stop" || a === "restart";
  return (
    <ConfirmDialog open onOpenChange={(o) => !o && onClose()} title={`${VERB[a]} ${t.name}?`} description={WHAT[a]}
      confirmLabel={VERB[a]} destructive={destructive} pending={run.isPending}
      requireText={destructive && t.state === "ready" ? t.name : undefined} onConfirm={() => run.mutate(undefined)} />
  );
}

function ReconfigureFromCommand({ interp, onClose }: { interp: CommandInterpretation; onClose: () => void }) {
  const id = interp.target!.id!;
  const dep = useQuery({ queryKey: ["deployment", id], queryFn: () => api<Deployment>(`/deployments/${id}`), retry: false });
  if (dep.data) return <SwitchDialog mode="reconfigure" deployment={dep.data} initialPatch={interp.params.patch} patchSource="your command" onClose={onClose} />;
  return (
    <ResponsiveDialog open onOpenChange={(o) => !o && onClose()} title="Reconfigure" className="max-w-md">
      {dep.error ? <ErrorState error={dep.error} /> : <p className="flex items-center gap-2 text-sm text-muted-foreground"><Loader2 aria-hidden className="h-4 w-4 animate-spin" />Loading the deployment…</p>}
    </ResponsiveDialog>
  );
}

/** "Free N GiB": the plan chooser of a deploy that does not fit (same safety annotations). The chosen plan is applied
 *  by the server through the deploy plan path (undone if an action fails); an unsafe plan needs an explicit
 *  acknowledgement. Nothing is applied before "Apply plan". */
export function FreeMemoryDialog({ bytes, onClose }: { bytes?: number; onClose: () => void }) {
  const [gib, setGib] = useState(bytes ? String(Math.round((bytes / GIB) * 10) / 10) : "");
  const n = Number(gib);
  const valid = Number.isFinite(n) && n > 0 && n <= 4096;
  const q = useQuery({ queryKey: ["free-plans", n], queryFn: () => api<FreePlans>(`/decider/free-plans?gib=${n}`), enabled: valid, retry: false });
  const [plan, setPlan] = useState<PlanOption | null>(null);
  const [ack, setAck] = useState(false);
  const apply = useActionMutation((p: PlanOption) => api("/decider/free-memory", { method: "POST", body: { gib: n, plan_id: p.id, acknowledge_unsafe: ack ? p.unsafe_fingerprint ?? null : null } }), {
    success: (_d, p) => `${p.title}: done`, invalidate: [["deployments"], ["host"], ["free-plans"]], onSuccess: onClose,
  });
  // The plan changed since it was listed: unsafe now (acknowledge it here, inline) or gone (choose again).
  useEffect(() => {
    const e = apply.error;
    if (!(e instanceof ApiError)) return;
    if (e.code === "plan_unsafe" && e.detail?.plan) { setPlan(e.detail.plan as PlanOption); setAck(false); }
    if (e.code === "plan_gone") { setPlan(null); setAck(false); q.refetch(); }
  }, [apply.error]); // eslint-disable-line react-hooks/exhaustive-deps
  const v = q.data;
  const unsafe = plan?.unsafe_reason;
  return (
    <ResponsiveDialog open onOpenChange={(o) => !o && onClose()} title="Free memory" className="max-w-lg">
      <div className="space-y-4">
        <div className="flex items-end gap-2">
          <div className="space-y-1.5"><Label htmlFor="free-gib">GiB to free</Label>
            <Input id="free-gib" type="number" min={0.1} step={0.5} value={gib} onChange={(e) => { setGib(e.target.value); setPlan(null); setAck(false); }} className="w-32" /></div>
          {v && <p className="pb-2 text-xs text-muted-foreground">{fmtBytes(v.free_for_new)} free now</p>}
        </div>
        <ErrorState error={q.error ?? apply.error} />
        {q.isFetching && <p className="flex items-center gap-2 text-sm text-muted-foreground"><Loader2 aria-hidden className="h-4 w-4 animate-spin" />Finding plans…</p>}
        {v && v.shortfall === 0 && <p role="status" className="text-sm">{fmtBytes(v.bytes)} is already free: nothing needs to change.</p>}
        {v && v.shortfall > 0 && <PlanChooser plans={v.plans} selectedId={plan?.id ?? null} onSelect={(p) => { setPlan(p); setAck(false); }} />}
        {unsafe && (
          <label className="flex items-start gap-2 rounded-md border border-status-warning/50 p-2 text-sm">
            <input type="checkbox" checked={ack} onChange={(e) => setAck(e.target.checked)} className="mt-1" />
            <span>Apply it anyway, although it is unsafe: {unsafe}</span>
          </label>
        )}
        <div className="flex justify-end gap-2">
          <Button variant="outline" size="sm" onClick={onClose}>Cancel</Button>
          <Button size="sm" disabled={!plan || apply.isPending || (!!unsafe && !ack)} onClick={() => plan && apply.mutate(plan)}>Apply plan</Button>
        </div>
      </div>
    </ResponsiveDialog>
  );
}

/** The dialog an interpretation opens (navigate / deploy are links and open none). */
export function InterpretationDialog({ interp, onClose }: { interp: CommandInterpretation; onClose: () => void }) {
  if (interp.action === "free_memory") return <FreeMemoryDialog bytes={interp.params.bytes} onClose={onClose} />;
  if (interp.action === "reconfigure" && interp.target?.id != null) return <ReconfigureFromCommand interp={interp} onClose={onClose} />;
  if (interp.target?.kind === "deployment" && interp.target.id != null && VERB[interp.action]) return <LifecycleConfirm interp={interp} onClose={onClose} />;
  return null;
}
