import { useEffect, useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Navigate, useNavigate, useSearchParams } from "react-router-dom";
import { Loader2 } from "lucide-react";
import { ApiError, api } from "@/api";
import { useIsAdmin } from "@/auth";
import { ConfigForm } from "@/components/app/config-form";
import { EstimatePanel } from "@/components/app/estimate-panel";
import { ModelSelect } from "@/components/app/model-select";
import { PageHeader } from "@/components/app/page-header";
import { PlanChooser } from "@/components/app/plan-chooser";
import { ErrorState } from "@/components/app/states";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { fmtBytes } from "@/format";
import { useActionMutation } from "@/hooks/use-action-mutation";
import { submitState } from "@/logic";
import type { Artifact, EngineInfo, PlanAction, Preview } from "@/types";

const slug = (ref: string) => ref.split("/").pop()!.toLowerCase().replace(/\.gguf$/, "").replace(/[^a-z0-9-]+/g, "-").replace(/^-+|-+$/g, "").slice(0, 40) || "model";
const engineLabel = (e: string) => (e === "llamacpp" ? "llama.cpp" : "vLLM");

function Section({ n, title, description, children }: { n: number; title: string; description?: string; children: React.ReactNode }) {
  return (
    <Card className="shadow-card">
      <CardHeader className="pb-3">
        <CardTitle className="flex items-center gap-2 text-sm"><span className="flex h-5 w-5 items-center justify-center rounded-full bg-primary/10 text-[11px] font-semibold text-primary">{n}</span>{title}</CardTitle>
        {description && <CardDescription className="text-xs">{description}</CardDescription>}
      </CardHeader>
      <CardContent>{children}</CardContent>
    </Card>
  );
}

function DeployWizardPage() {
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
  const [plan, setPlan] = useState<{ id: string; actions: PlanAction[] } | null>(null);

  const defaultsFor = (eng: string, a?: Artifact) => {
    const d = { ...(engines.data?.find((e) => e.name === eng)?.defaults ?? {}) };
    const ctxKey = eng === "vllm" ? "max_model_len" : "ctx_size";
    if (a?.max_context && d[ctxKey] > a.max_context) d[ctxKey] = a.max_context;
    for (const k of Object.keys(d)) if (d[k] === null || (Array.isArray(d[k]) && !d[k].length)) delete d[k];
    return d;
  };

  useEffect(() => {
    if (!artifact) return;
    const eng = artifact.engines[0];
    setEngine(eng); setName(slug(artifact.ref)); setConfig(defaultsFor(eng, artifact)); setPlan(null);
  }, [artifact?.ref, engines.data]); // eslint-disable-line react-hooks/exhaustive-deps

  const body = useMemo(() => ({ engine, model_ref: ref, config }), [engine, ref, config]);
  useEffect(() => {
    if (!engine || !ref) return;
    const t = setTimeout(async () => {
      try { setPreview(await api<Preview>("/deployments/preview", { method: "POST", body })); setPreviewErr(null); }
      catch (e) { setPreviewErr(e); setPreview(null); }
    }, 350);
    return () => clearTimeout(t);
  }, [body, engine, ref]);

  const actions = plan?.actions ?? [];
  const deploy = useActionMutation(() => api<{ id: number }>("/deployments", {
    method: "POST", body: { name, engine, model_ref: ref, config, priority, alias: alias || null, actions },
  }), {
    success: (d: any) => `Deploying ${name} (id ${d.id})`,
    invalidate: [["deployments"], ["host"]],
    onSuccess: (d: any) => nav(`/deployments/${d.id}`),
  });

  useEffect(() => {
    if (deploy.error instanceof ApiError && deploy.error.code === "insufficient_memory") {
      setPlan(null);
      if (deploy.error.detail?.fit && deploy.error.detail?.plans) {
        setPreview({ ...preview!, fit: deploy.error.detail.fit, plans: deploy.error.detail.plans });
      }
    }
  }, [deploy.error, preview]);

  const submit = submitState(preview, actions, deploy.isPending, "deploy");
  const memoryChanged = deploy.error instanceof ApiError && deploy.error.code === "insufficient_memory";

  const check = (
    <div className="space-y-4">
      <ErrorState error={previewErr} />
      {preview ? <EstimatePanel preview={preview} /> : <p className="text-sm text-muted-foreground">Choose a model to see the memory check.</p>}
      {preview && !preview.fit.fits && (
        <PlanChooser plans={preview.plans ?? []} selectedId={plan?.id ?? null}
          onSelect={(o) => {
            setPlan({ id: o.id, actions: o.actions.filter((a) => a.type !== "patch") });
            if (o.patch) setConfig({ ...config, ...o.patch });
            if (o.model_ref) setRef(o.model_ref); // triggers artifact effect → resets config/plan, but user sees model change
          }} />
      )}
      {memoryChanged && <p className="text-xs text-status-critical">Memory changed since the preview — choose a plan again.</p>}
    </div>
  );

  return (
    <div className="pb-[calc(44px+env(safe-area-inset-bottom)+1rem)] md:pb-0">
      <PageHeader title="Deploy a model" description="DGXMM checks memory before anything starts. Nothing is evicted without your confirmation." />
      <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_380px]">
        <div className="space-y-4">
          <Section n={1} title="Model" description="Only complete, downloaded models are listed.">
            <ModelSelect artifacts={deployable} value={ref} onChange={setRef} />
            {artifact && <p className="mt-2 text-xs text-muted-foreground tabular-nums">{artifact.architecture} · {artifact.quant} · {fmtBytes(artifact.weights_bytes)} weights · max context {artifact.max_context?.toLocaleString() ?? "?"}</p>}
          </Section>
          {artifact && (
            <>
              <Section n={2} title="Engine and name">
                <div className="grid gap-4 sm:grid-cols-2">
                  <div className="space-y-1.5"><Label>Engine</Label>
                    <Select value={engine} onValueChange={(e) => { setEngine(e); setConfig(defaultsFor(e, artifact)); setPlan(null); }}>
                      <SelectTrigger aria-label="Engine"><SelectValue /></SelectTrigger>
                      <SelectContent>{artifact.engines.map((e) => <SelectItem key={e} value={e}>{engineLabel(e)}</SelectItem>)}</SelectContent>
                    </Select></div>
                  <div className="space-y-1.5"><Label htmlFor="dep-name">Deployment name</Label><Input id="dep-name" value={name} onChange={(e) => setName(e.target.value)} /></div>
                  <div className="space-y-1.5"><Label htmlFor="dep-prio">Priority</Label><Input id="dep-prio" type="number" value={priority} onChange={(e) => setPriority(parseInt(e.target.value || "0", 10))} />
                    <p className="text-xs text-muted-foreground">Lower priority is evicted first.</p></div>
                  <div className="space-y-1.5"><Label htmlFor="dep-alias">LiteLLM alias</Label><Input id="dep-alias" placeholder="optional, e.g. default-chat" value={alias} onChange={(e) => setAlias(e.target.value)} /></div>
                </div>
              </Section>
              <Section n={3} title="Sizing and features">
                <ConfigForm engine={engine} value={config} onChange={(c) => { setConfig(c); setPlan(null); }} showAdvanced={advanced} />
                <Button variant="link" className="mt-3 h-auto p-0 text-xs" onClick={() => setAdvanced(!advanced)}>{advanced ? "Hide" : "Show"} advanced settings</Button>
              </Section>
            </>
          )}
        </div>
        <div className="hidden lg:block">
          <Card className="shadow-card lg:sticky lg:top-20">
            <CardHeader className="pb-3"><CardTitle className="text-sm">Memory check</CardTitle></CardHeader>
            <CardContent className="space-y-4">
              {check}
              <Button className="w-full" disabled={submit.disabled} onClick={() => deploy.mutate(undefined)}>
                {deploy.isPending && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}{submit.label}
              </Button>
            </CardContent>
          </Card>
        </div>
        <div className="lg:hidden">{artifact && <Card className="shadow-card"><CardHeader className="pb-3"><CardTitle className="text-sm">Memory check</CardTitle></CardHeader><CardContent className="space-y-4">{check}<Button className="hidden w-full md:flex lg:hidden" disabled={submit.disabled} onClick={() => deploy.mutate(undefined)}>{deploy.isPending && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}{submit.label}</Button></CardContent></Card>}</div>
      </div>
      {artifact && (
        <div className="fixed inset-x-0 bottom-[calc(44px+env(safe-area-inset-bottom))] z-30 border-t bg-background/95 px-4 py-3 backdrop-blur md:hidden">
          <div className="flex items-center gap-3">
            <div className="min-w-0 flex-1 text-xs">
              {preview ? <span className={preview.fit.fits ? "text-status-good" : "text-status-critical"}>{preview.fit.fits ? "Fits" : "Does not fit"} · needs {fmtBytes(preview.estimate.total)}</span> : <span className="text-muted-foreground">Checking memory…</span>}
            </div>
            <Button size="sm" disabled={submit.disabled} onClick={() => deploy.mutate(undefined)}>{deploy.isPending && <Loader2 className="mr-1 h-4 w-4 animate-spin" />}{submit.label}</Button>
          </div>
        </div>
      )}
    </div>
  );
}

export function DeployWizard() {
  return useIsAdmin() ? <DeployWizardPage /> : <Navigate to="/" replace />;
}
