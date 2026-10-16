import { useEffect, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Loader2 } from "lucide-react";
import { ApiError, api } from "@/api";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { useActionMutation } from "@/hooks/use-action-mutation";
import { submitState } from "@/logic";
import type { Artifact, Deployment, PlanAction, Preview } from "@/types";
import { ConfigForm } from "./config-form";
import { EstimatePanel } from "./estimate-panel";
import { ModelSelect } from "./model-select";
import { PlanChooser } from "./plan-chooser";
import { ResponsiveDialog } from "./responsive-dialog";
import { ErrorState } from "./states";

export function SwitchDialog({ mode, deployment, initialPatch, onClose }: {
  mode: "switch" | "reconfigure"; deployment: Deployment; initialPatch?: Record<string, any>; onClose: () => void;
}) {
  const inv = useQuery({ queryKey: ["inventory"], queryFn: () => api<Artifact[]>("/inventory") });
  const [ref, setRef] = useState<string>(initialPatch?.model_ref ?? deployment.model_ref);
  const [config, setConfig] = useState<Record<string, any>>(() => {
    const { model_ref: _ignored, ...patch } = initialPatch ?? {};
    return { ...deployment.config, ...patch };
  });
  const [name, setName] = useState(`${deployment.name}-2`.slice(0, 40));
  const [preview, setPreview] = useState<Preview | null>(null);
  const [err, setErr] = useState<unknown>(null);
  const [plan, setPlan] = useState<{ id: string; actions: PlanAction[] } | null>(null);
  const [advanced, setAdvanced] = useState(false);
  const choices = (inv.data ?? []).filter((a) => a.complete && a.kind === "llm" && a.engines.includes(deployment.engine));

  useEffect(() => {
    let live = true; // a slower, older preview must not overwrite the current one
    const t = setTimeout(async () => {
      try {
        const p = await api<Preview>("/deployments/preview", { method: "POST", body: { engine: deployment.engine, model_ref: ref, config, replacing: deployment.id } });
        if (live) { setPreview(p); setErr(null); }
      } catch (e) { if (live) { setErr(e); setPreview(null); } }
    }, 350);
    return () => { live = false; clearTimeout(t); };
  }, [ref, config, deployment.engine, deployment.id]);

  const actions = plan?.actions ?? [];
  const go = useActionMutation(() => (mode === "switch"
    ? api(`/deployments/${deployment.id}/switch`, { method: "POST", body: { request: { name, engine: deployment.engine, model_ref: ref, config, priority: deployment.priority }, actions } })
    : api(`/deployments/${deployment.id}/reconfigure`, { method: "POST", body: { config, model_ref: ref === deployment.model_ref ? null : ref, actions } })), {
    success: mode === "switch" ? "Switched — the new deployment is ready" : "Reconfigured and restarted",
    invalidate: [["deployments"], ["deployment", deployment.id]],
    onSuccess: onClose,
  });

  // A 409 carries a fresh fit + plans: show them and make the user choose again (depends only on the error).
  useEffect(() => {
    const e = go.error;
    if (!(e instanceof ApiError) || e.code !== "insufficient_memory") return;
    setPlan(null);
    const { fit, plans } = e.detail ?? {};
    if (fit && plans) setPreview((p) => p && { ...p, fit, plans });
  }, [go.error]);

  const submit = submitState(preview, actions, go.isPending, mode);
  const memoryChanged = go.error instanceof ApiError && go.error.code === "insufficient_memory";

  return (
    <ResponsiveDialog open onOpenChange={(o) => !o && !go.isPending && onClose()}
      title={mode === "switch" ? `Switch ${deployment.name} to another model` : `Reconfigure ${deployment.name}`}
      description={mode === "switch" ? `${deployment.name} is put to sleep (or stopped) and its LiteLLM alias moves to the new deployment.` : "The deployment restarts with the new settings."}>
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-[minmax(0,1fr)_340px]">
        <div className="space-y-4">
          <div className="space-y-1.5"><Label id="switch-model-label">Model</Label><ModelSelect id="switch-model" labelledBy="switch-model-label" artifacts={choices} value={ref} onChange={(r) => { setRef(r); setPlan(null); }} /></div>
          {mode === "switch" && <div className="space-y-1.5"><Label htmlFor="new-name">New deployment name</Label><Input id="new-name" value={name} onChange={(e) => setName(e.target.value)} /></div>}
          <ConfigForm engine={deployment.engine} value={config} onChange={(c) => { setConfig(c); setPlan(null); }} showAdvanced={advanced} />
          <Button variant="link" className="h-auto p-0 text-xs" onClick={() => setAdvanced(!advanced)}>{advanced ? "Hide" : "Show"} advanced settings</Button>
          <p className="text-xs text-muted-foreground">If the new deployment fails to become ready, DGXMM rolls back automatically.</p>
        </div>
        <div className="space-y-4 lg:sticky lg:top-0 lg:self-start">
          <ErrorState error={err} />
          {preview ? <EstimatePanel preview={preview} /> : !err && <p className="text-sm text-muted-foreground">Calculating…</p>}
          {preview && !preview.fit.fits && (
            <PlanChooser plans={preview.plans ?? []} selectedId={plan?.id ?? null}
              onSelect={(o) => { setPlan({ id: o.id, actions: o.actions.filter((a) => a.type !== "patch") }); if (o.patch) setConfig({ ...config, ...o.patch }); if (o.model_ref) setRef(o.model_ref); }} />
          )}
          {memoryChanged && <p className="text-xs text-status-critical">Memory changed since the preview — choose a plan again.</p>}
          <Button className="w-full" disabled={submit.disabled} onClick={() => go.mutate(undefined)}>
            {go.isPending && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}{submit.label}
          </Button>
        </div>
      </div>
    </ResponsiveDialog>
  );
}
