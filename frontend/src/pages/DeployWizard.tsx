import { useEffect, useMemo, useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { useNavigate, useSearchParams } from "react-router-dom";
import { ApiError, api } from "../api";
import { ConfigForm } from "../components/ConfigForm";
import { EstimatePanel } from "../components/EstimatePanel";
import { PlanChooser } from "../components/PlanChooser";
import { Button, Card, ErrorBox, Field, Input, Select } from "../components/ui";
import type { Artifact, EngineInfo, PlanAction, Preview } from "../types";

const slug = (ref: string) => ref.split("/").pop()!.toLowerCase().replace(/\.gguf$/, "").replace(/[^a-z0-9-]+/g, "-").replace(/^-+|-+$/g, "").slice(0, 40) || "model";

export function DeployWizard() {
  const [params] = useSearchParams();
  const nav = useNavigate();
  const inv = useQuery({ queryKey: ["inventory"], queryFn: () => api<Artifact[]>("/inventory") });
  const engines = useQuery({ queryKey: ["engines"], queryFn: () => api<EngineInfo[]>("/engines") });
  const deployable = (inv.data ?? []).filter((a) => a.complete && a.kind === "llm" && a.engines.length > 0);
  const [ref, setRef] = useState(params.get("model") ?? "");
  const artifact = deployable.find((a) => a.ref === ref);
  const [engine, setEngine] = useState("");
  const [name, setName] = useState("");
  const [priority, setPriority] = useState(0);
  const [alias, setAlias] = useState("");
  const [config, setConfig] = useState<Record<string, any>>({});
  const [advanced, setAdvanced] = useState(false);
  const [preview, setPreview] = useState<Preview | null>(null);
  const [previewErr, setPreviewErr] = useState<unknown>(null);
  const [actions, setActions] = useState<PlanAction[]>([]);

  useEffect(() => {
    if (!artifact) return;
    const eng = artifact.engines[0];
    setEngine(eng);
    setName(slug(artifact.ref));
    const defaults = { ...(engines.data?.find((e) => e.name === eng)?.defaults ?? {}) };
    const ctxKey = eng === "vllm" ? "max_model_len" : "ctx_size";
    if (artifact.max_context && defaults[ctxKey] > artifact.max_context) defaults[ctxKey] = artifact.max_context;
    for (const k of Object.keys(defaults)) if (defaults[k] === null || (Array.isArray(defaults[k]) && !defaults[k].length)) delete defaults[k];
    setConfig(defaults);
    setActions([]);
  }, [artifact?.ref, engines.data]); // eslint-disable-line react-hooks/exhaustive-deps

  const body = useMemo(() => ({ engine, model_ref: ref, config }), [engine, ref, config]);
  useEffect(() => {
    if (!engine || !ref) return;
    const t = setTimeout(async () => {
      try {
        setPreview(await api<Preview>("/deployments/preview", { method: "POST", body }));
        setPreviewErr(null);
      } catch (e) {
        setPreviewErr(e);
        setPreview(null);
      }
    }, 350);
    return () => clearTimeout(t);
  }, [body, engine, ref]);

  const deploy = useMutation({
    mutationFn: () => api<{ id: number }>("/deployments", {
      method: "POST",
      body: { name, engine, model_ref: ref, config, priority, alias: alias || null, actions },
    }),
    onSuccess: (d) => nav(`/deployments/${d.id}`),
  });

  return (
    <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
      <Card title="Deploy a model" className="lg:col-span-2">
        <div className="space-y-4">
          <Field label="Model">
            <Select value={ref} onChange={(e) => setRef(e.target.value)}>
              <option value="">Choose a downloaded model…</option>
              {deployable.map((a) => <option key={a.ref} value={a.ref}>{a.ref} ({a.quant})</option>)}
            </Select>
          </Field>
          {artifact && (
            <>
              <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
                <Field label="Engine">
                  <Select value={engine} onChange={(e) => { setEngine(e.target.value); setConfig(engines.data?.find((x) => x.name === e.target.value)?.defaults ?? {}); }}>
                    {artifact.engines.map((e) => <option key={e}>{e}</option>)}
                  </Select>
                </Field>
                <Field label="Deployment name"><Input value={name} onChange={(e) => setName(e.target.value)} /></Field>
                <Field label="Priority" help="Lower is evicted first."><Input type="number" value={priority} onChange={(e) => setPriority(parseInt(e.target.value || "0", 10))} /></Field>
                <Field label="LiteLLM alias" help="Optional stable name."><Input value={alias} onChange={(e) => setAlias(e.target.value)} /></Field>
              </div>
              <ConfigForm engine={engine} value={config} onChange={setConfig} showAdvanced={advanced} />
              <button className="text-xs underline" onClick={() => setAdvanced(!advanced)}>{advanced ? "Hide" : "Show"} advanced settings</button>
            </>
          )}
          <ErrorBox error={deploy.error} />
          <Button variant="primary" disabled={!preview || deploy.isPending || (!preview.fit.fits && actions.length === 0)} onClick={() => deploy.mutate()}>
            {actions.length ? "Apply plan and deploy" : "Deploy"}
          </Button>
          {deploy.error instanceof ApiError && deploy.error.code === "insufficient_memory" && (
            <p className="text-xs text-rose-700">Memory changed since the preview. Choose a plan again.</p>
          )}
        </div>
      </Card>
      <Card title="Memory check">
        <ErrorBox error={previewErr} />
        {preview ? <EstimatePanel preview={preview} /> : <p className="text-sm text-slate-500">Choose a model to see the estimate.</p>}
        {preview && !preview.fit.fits && (
          <PlanChooser plans={preview.plans ?? []} selected={actions}
            onSelect={(opt) => { setActions(opt.actions.filter((a) => a.type !== "patch")); if (opt.patch) setConfig({ ...config, ...opt.patch }); }} />
        )}
      </Card>
    </div>
  );
}
