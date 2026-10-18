import { useState, type ComponentType } from "react";
import { useQuery } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import { ArrowRight, Loader2 } from "lucide-react";
import { api } from "@/api";
import { useIsAdmin } from "@/auth";
import { Button } from "@/components/ui/button";
import { useActionMutation } from "@/hooks/use-action-mutation";
import { useInsightActions } from "@/hooks/use-insights";
import { fmtConfidence } from "@/lib/decider";
import { actionLabel, parseAction } from "@/lib/insights";
import type { DeciderStatus, Deployment, Insight, InsightAction } from "@/types";
import { ConfirmDialog } from "./confirm-dialog";
import { CopyButton } from "./copy-button";
import { ResponsiveDialog } from "./responsive-dialog";
import { ErrorState } from "./states";
import { SwitchDialog } from "./switch-dialog";

/** Props of the dialog an insight's action opens. Call `onApplied` once the change went through its normal endpoint
 *  (the insight is then marked applied); `onClose` closes the dialog either way. */
export interface InsightActionDialogProps<A extends InsightAction = InsightAction> {
  insight: Insight; action: A; onClose: () => void; onApplied: () => void;
}
type Dialog = ComponentType<InsightActionDialogProps<any>>;

function ReconfigureDialog({ action, onClose, onApplied }: InsightActionDialogProps<Extract<InsightAction, { type: "reconfigure" }>>) {
  const dep = useQuery({ queryKey: ["deployment", action.deployment_id], queryFn: () => api<Deployment>(`/deployments/${action.deployment_id}`), retry: false });
  if (dep.data) return <SwitchDialog mode="reconfigure" deployment={dep.data} initialPatch={action.patch} patchSource="the insight" onClose={onClose} onDone={onApplied} />;
  return (
    <ResponsiveDialog open onOpenChange={(o) => !o && onClose()} title="Reconfigure" className="max-w-md">
      {dep.error ? <ErrorState error={dep.error} /> : <p className="flex items-center gap-2 text-sm text-muted-foreground"><Loader2 aria-hidden className="h-4 w-4 animate-spin" />Loading the deployment…</p>}
    </ResponsiveDialog>
  );
}

function ThresholdDialog({ action, onClose, onApplied }: InsightActionDialogProps<Extract<InsightAction, { type: "threshold" }>>) {
  const st = useQuery({ queryKey: ["decider-status"], queryFn: () => api<DeciderStatus>("/decider/status") });
  const label = st.data?.feature_specs?.find((f) => f.name === action.feature)?.label ?? action.feature;
  const current = st.data?.features?.[action.feature]?.threshold;
  const save = useActionMutation(() => api("/decider/settings", { method: "PUT", body: { features: { [action.feature]: { threshold: action.value } } } }), {
    success: `${label}: threshold set to ${fmtConfidence(action.value)}`, invalidate: [["decider-status"]], onSuccess: () => { onApplied(); onClose(); },
  });
  return (
    <ConfirmDialog open onOpenChange={(o) => !o && onClose()} title={`Set the ${label} threshold to ${fmtConfidence(action.value)}?`}
      description={<>Model answers for this feature are used only at or above this confidence{current != null ? <> (now {fmtConfidence(current)})</> : null}. You can change it back on the Decision model page.</>}
      confirmLabel="Set threshold" pending={save.isPending} onConfirm={() => save.mutate(undefined)} />
  );
}

function CopyDialog({ insight, action, onClose, onApplied }: InsightActionDialogProps<Extract<InsightAction, { type: "copy" }>>) {
  return (
    <ResponsiveDialog open onOpenChange={(o) => !o && onClose()} title={insight.title} className="max-w-2xl">
      <div className="space-y-3">
        <div className="flex items-start gap-2">
          <pre aria-label="Snippet" className="max-h-80 min-w-0 flex-1 overflow-auto whitespace-pre-wrap break-all rounded-md bg-muted p-3 font-mono text-xs">{action.text}</pre>
          <CopyButton value={action.text} label="Copy snippet" />
        </div>
        <p className="text-xs text-muted-foreground">DGXMM changes nothing by itself: paste this where it belongs, then mark the insight as applied.</p>
        <div className="flex justify-end gap-2">
          <Button variant="outline" size="sm" onClick={onClose}>Close</Button>
          <Button size="sm" onClick={() => { onApplied(); onClose(); }}>Mark as applied</Button>
        </div>
      </div>
    </ResponsiveDialog>
  );
}

/** Placeholder until schedules exist: shows the proposal, accepts nothing. */
function ScheduleDialog({ insight, action, onClose }: InsightActionDialogProps<Extract<InsightAction, { type: "schedule" }>>) {
  return (
    <ResponsiveDialog open onOpenChange={(o) => !o && onClose()} title={insight.title} className="max-w-md">
      <div className="space-y-3 text-sm">
        <dl className="space-y-1 rounded-md border bg-muted/50 p-3 font-mono text-xs">
          {Object.entries(action.proposal).map(([k, v]) => <div key={k} className="flex gap-2"><dt className="text-muted-foreground">{k}</dt><dd className="break-all">{typeof v === "object" ? JSON.stringify(v) : String(v)}</dd></div>)}
        </dl>
        <p className="text-muted-foreground">Schedules cannot be created from an insight in this version yet.</p>
        <div className="flex justify-end"><Button variant="outline" size="sm" onClick={onClose}>Close</Button></div>
      </div>
    </ResponsiveDialog>
  );
}

const DIALOGS: Partial<Record<InsightAction["type"], Dialog>> = {
  reconfigure: ReconfigureDialog, threshold: ThresholdDialog, copy: CopyDialog, schedule: ScheduleDialog,
};

/** Replace the dialog of an action type (feature lanes complete e.g. "schedule"). `open` actions are plain links. */
export function registerInsightActionDialog<T extends InsightAction["type"]>(type: T, dialog: ComponentType<InsightActionDialogProps<Extract<InsightAction, { type: T }>>>) {
  DIALOGS[type] = dialog as Dialog;
}

/**
 * The button of an insight's action, for admins only (viewers never get actions; the server strips them too). It opens
 * the existing dialog for the action type; nothing is applied without that dialog's confirmation. Renders nothing for a
 * missing, unknown or malformed action, or a closed insight.
 */
export function InsightActionButton({ insight, size = "sm" }: { insight: Insight; size?: "sm" | "default" }) {
  const admin = useIsAdmin();
  const { applied } = useInsightActions();
  const [open, setOpen] = useState(false);
  const action = parseAction(insight.action);
  if (!admin || !action || !["open", "snoozed"].includes(insight.status)) return null;
  if (action.type === "open") {
    return <Button asChild variant="outline" size={size}><Link to={action.path}>{actionLabel(action)}<ArrowRight aria-hidden className="ml-1 h-3.5 w-3.5" /></Link></Button>;
  }
  const D = DIALOGS[action.type];
  if (!D) return null;
  return (
    <>
      <Button variant="outline" size={size} onClick={() => setOpen(true)}>{actionLabel(action)}</Button>
      {open && <D insight={insight} action={action} onClose={() => setOpen(false)} onApplied={() => applied.mutate(insight.id)} />}
    </>
  );
}
