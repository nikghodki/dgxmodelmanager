import { useEffect, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "../api";
import type { Artifact, Deployment, PlanAction, Preview } from "../types";
import { ConfigForm } from "./ConfigForm";
import { EstimatePanel } from "./EstimatePanel";
import { PlanChooser } from "./PlanChooser";
import { Button, ErrorBox, Field, Input, Modal, Select } from "./ui";

export function SwitchDialog({ mode, deployment, initialPatch, onClose }: {
  mode: "switch" | "reconfigure"; deployment: Deployment; initialPatch?: Record<string, any>; onClose: () => void;
}) {
  const qc = useQueryClient();
  const inv = useQuery({ queryKey: ["inventory"], queryFn: () => api<Artifact[]>("/inventory") });
  const [ref, setRef] = useState(initialPatch?.model_ref ?? deployment.model_ref);
  const [config, setConfig] = useState<Record<string, any>>(() => {
    const { model_ref: _ignored, ...patch } = initialPatch ?? {};
    return { ...deployment.config, ...patch };
  });
  const [name, setName] = useState(`${deployment.name}-2`.slice(0, 40));
  const [preview, setPreview] = useState<Preview | null>(null);
  const [err, setErr] = useState<unknown>(null);
  const [actions, setActions] = useState<PlanAction[]>([]);
  const [advanced, setAdvanced] = useState(false);
  const choices = (inv.data ?? []).filter((a) => a.complete && a.kind === "llm" && a.engines.includes(deployment.engine));

  useEffect(() => {
    const t = setTimeout(async () => {
      try {
        setPreview(await api<Preview>("/deployments/preview", { method: "POST", body: { engine: deployment.engine, model_ref: ref, config, replacing: deployment.id } }));
        setErr(null);
      } catch (e) {
        setErr(e);
        setPreview(null);
      }
    }, 350);
    return () => clearTimeout(t);
  }, [ref, config, deployment.engine, deployment.id]);

  const go = useMutation({
    mutationFn: () => mode === "switch"
      ? api(`/deployments/${deployment.id}/switch`, { method: "POST", body: { request: { name, engine: deployment.engine, model_ref: ref, config, priority: deployment.priority }, actions } })
      : api(`/deployments/${deployment.id}/reconfigure`, { method: "POST", body: { config, model_ref: ref === deployment.model_ref ? null : ref, actions } }),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ["deployments"] }); onClose(); },
  });

  return (
    <Modal open wide title={mode === "switch" ? `Switch ${deployment.name} to another model` : `Reconfigure ${deployment.name}`} onClose={onClose}>
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
        <div className="space-y-3 lg:col-span-2">
          <Field label="Model">
            <Select value={ref} onChange={(e) => setRef(e.target.value)}>
              {choices.map((a) => <option key={a.ref} value={a.ref}>{a.ref} ({a.quant})</option>)}
            </Select>
          </Field>
          {mode === "switch" && <Field label="New deployment name" help={`${deployment.name} is put to sleep (or stopped). Its LiteLLM alias moves to the new deployment.`}><Input value={name} onChange={(e) => setName(e.target.value)} /></Field>}
          <ConfigForm engine={deployment.engine} value={config} onChange={setConfig} showAdvanced={advanced} />
          <button className="text-xs underline" onClick={() => setAdvanced(!advanced)}>{advanced ? "Hide" : "Show"} advanced</button>
          <p className="text-xs text-slate-500">If the new deployment fails to become ready, DGXMM rolls back automatically.</p>
          <ErrorBox error={go.error} />
          <Button variant="primary" disabled={!preview || go.isPending || (!preview.fit.fits && !actions.length)} onClick={() => go.mutate()}>
            {go.isPending ? "Working… (this can take minutes)" : mode === "switch" ? "Switch" : "Apply and restart"}
          </Button>
        </div>
        <div>
          <ErrorBox error={err} />
          {preview && <EstimatePanel preview={preview} />}
          {preview && !preview.fit.fits && (
            <PlanChooser plans={preview.plans ?? []} selected={actions}
              onSelect={(o) => { setActions(o.actions.filter((a) => a.type !== "patch")); if (o.patch) setConfig({ ...config, ...o.patch }); if (o.model_ref) setRef(o.model_ref); }} />
          )}
        </div>
      </div>
    </Modal>
  );
}
