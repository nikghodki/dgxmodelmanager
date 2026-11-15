import { useEffect, useMemo, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Navigate, useNavigate, useSearchParams } from "react-router-dom";
import { Loader2 } from "lucide-react";
import { ApiError, api } from "@/api";
import { useIsAdmin } from "@/auth";
import { ConfigForm } from "@/components/app/config-form";
import { EstimatePanel } from "@/components/app/estimate-panel";
import { ModelSelect } from "@/components/app/model-select";
import { ModeFields, ONDEMAND_NEEDS_LITELLM, idleInvalid, useIdleDefaultMinutes, useOnDemandReady } from "@/components/app/mode-fields";
import { PageHeader } from "@/components/app/page-header";
import { PlanChooser } from "@/components/app/plan-chooser";
import { useParserSuggestions } from "@/components/app/parser-suggestions";
import { CompatLine } from "@/components/app/compat-line";
import { ConfigAdviceCard } from "@/components/app/config-advice";
import { EngineChip } from "@/components/app/engine-chip";
import { RemoteCodeFindings, RemoteCodeWarning } from "@/components/app/remote-code-panel";
import { SpeculativeField } from "@/components/app/speculative-field";
import { ErrorState } from "@/components/app/states";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { engineLabel, fmtBytes } from "@/format";
import { useActionMutation } from "@/hooks/use-action-mutation";
import { ModelNameTaken } from "@/components/app/model-name-taken";
import { CopyWeightsPrompt } from "@/components/app/copy-weights";
import { PlacementChooser } from "@/components/app/placement-chooser";
import { useNodes } from "@/hooks/use-nodes";
import { isMultiNode, placementKey } from "@/lib/placement";
import { aliasProblem, isModelNameTaken, suggestAlias } from "@/lib/litellm";
import { changedKeys, defaultGoal, GOALS, REPO_ID } from "@/lib/advice";
import { DEFAULT_IDLE_MINUTES, fmtMinutes, minutesToSeconds, parseIdleMinutes } from "@/lib/ondemand";
import { submitState } from "@/logic";
import { ENGINE_FIELDS } from "@/engineFields";
import type { Artifact, DeployGoal, DeploymentMode, EngineInfo, LiteLLMSettings, PlacementOptionInfo, PlanAction, Preview } from "@/types";

const slug = (ref: string) => ref.split("/").pop()!.toLowerCase().replace(/\.gguf$/, "").replace(/[^a-z0-9-]+/g, "-").replace(/^-+|-+$/g, "").slice(0, 40) || "model";

function Section({ n, title, titleId, description, children }: { n: number; title: string; titleId?: string; description?: string; children: React.ReactNode }) {
  return (
    <Card className="shadow-card">
      <CardHeader className="pb-3">
        <CardTitle id={titleId} className="flex items-center gap-2 text-sm"><span aria-hidden className="flex h-5 w-5 items-center justify-center rounded-full bg-primary/10 text-[11px] font-semibold text-primary">{n}</span>{title}</CardTitle>
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
  const litellm = useQuery({ queryKey: ["litellm"], queryFn: () => api<LiteLLMSettings>("/integrations/litellm?probe=false"), staleTime: 60000 });
  const deployable = (inv.data ?? []).filter((a) => a.complete && a.kind === "llm" && a.engines.length > 0);
  const [ref, setRef] = useState(params.get("model") ?? "");
  const artifact = deployable.find((a) => a.ref === ref);
  const [engine, setEngine] = useState("");
  const [name, setName] = useState("");
  const [priority, setPriority] = useState(0);
  const [alias, setAlias] = useState("");
  // On demand by default only when LiteLLM can start it (integration on, last sync good); always on otherwise
  const ondemandReady = useOnDemandReady();
  const [modeChosen, setMode] = useState<DeploymentMode | null>(null);
  const mode: DeploymentMode = modeChosen ?? (ondemandReady ? "on_demand" : "always_on");
  // Pre-filled with the global default; sent as null (= follow the default) unless the user changes it.
  const idleDefault = fmtMinutes(useIdleDefaultMinutes() ?? DEFAULT_IDLE_MINUTES);
  const [idleTyped, setIdle] = useState<string | null>(null);
  const idle = idleTyped ?? idleDefault;
  const aliasInput = useRef<HTMLInputElement>(null);
  const [config, setConfig] = useState<Record<string, any>>({});
  const [advanced, setAdvanced] = useState(false);
  const [preview, setPreview] = useState<Preview | null>(null);
  const [previewErr, setPreviewErr] = useState<unknown>(null);
  const [plan, setPlan] = useState<{ id: string; actions: PlanAction[] } | null>(null);
  // Deploy-time advice: the goal (default from the tool parser) and the config fields the user changed (advice
  // fills only the others; reset with the model / engine).
  const [goalPick, setGoalPick] = useState<DeployGoal | null>(null);
  const goal = goalPick ?? defaultGoal(config);
  const [touched, setTouched] = useState<string[]>([]);
  const userEdit = (c: Record<string, any>) => {
    const keys = changedKeys(config, c);
    if (keys.length) setTouched((t) => [...new Set([...t, ...keys])]);
    parsers.noteChange(config, c); setConfig(c); setPlan(null);
  };
  // Typed values that stay valid for an engine (its own fields), carried over an engine / model switch from the chip.
  const carryFor = (eng: string): Record<string, any> => {
    const keys = new Set((ENGINE_FIELDS[eng] ?? []).map((f) => f.key).concat(eng === "vllm" ? ["speculative"] : []));
    return Object.fromEntries(touched.filter((k) => keys.has(k) && k in config).map((k) => [k, config[k]]));
  };
  const carry = useRef<Record<string, any> | null>(null);
  const pickEngine = (e: string, keep: Record<string, any> = {}) => {
    setEngine(e); setConfig({ ...defaultsFor(e, artifact), ...keep }); setTouched(Object.keys(keep)); setPlan(null);
  };

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
    const keep = carry.current ?? {}; // typed values carried from the engine chip's "Use"
    carry.current = null;
    const cfg = defaultsFor(eng, artifact);
    // ?ctx=N (a ⌘K command "deploy X with 32k context"): the context length, capped at the model's own
    const ctx = Number(params.get("ctx"));
    const ctxKey = eng === "vllm" ? "max_model_len" : "ctx_size";
    const typed = Object.keys(keep);
    if (Number.isInteger(ctx) && ctx >= 512 && !(ctxKey in keep)) {
      cfg[ctxKey] = artifact.max_context ? Math.min(ctx, artifact.max_context) : ctx;
      typed.push(ctxKey); // asked for by the user: config advice never overwrites it
    }
    setEngine(eng); setName(slug(artifact.ref)); setConfig({ ...cfg, ...keep });
    setTouched(typed); setPlan(null);
  }, [artifact?.ref, engines.data]); // eslint-disable-line react-hooks/exhaustive-deps

  // Decision-model parser suggestions (declared after the defaults effect so a reset lands before an auto-fill).
  const parsers = useParserSuggestions({ modelRef: artifact ? ref : "", engine, config, setConfig, onUserEdit: () => setPlan(null) });

  const body = useMemo(() => ({ engine, model_ref: ref, config, priority }), [engine, ref, config, priority]);

  // Where to run (only with more than one node): the placement options for the current request, the recommended
  // one pre-selected unless the user picked another that is still offered.
  const nodes = useNodes();
  const multiNode = isMultiNode(nodes.data);
  const [options, setOptions] = useState<PlacementOptionInfo[] | null>(null);
  const [optionsErr, setOptionsErr] = useState<unknown>(null);
  const [placementPick, setPlacementPick] = useState<string | null>(null);
  const [optionsTry, setOptionsTry] = useState(0); // bumped by the error's Retry
  useEffect(() => {
    if (!multiNode || !engine || !ref) { setOptions(null); return; }
    let live = true;
    const t = setTimeout(async () => {
      try {
        // the name is ignored by placement-options; a valid one from the model keeps typing in the name field free
        const o = await api<PlacementOptionInfo[]>("/deployments/placement-options", { method: "POST", body: { ...body, name: slug(ref) } });
        if (live) { setOptions(o); setOptionsErr(null); }
      } catch (e) { if (live) { setOptionsErr(e); setOptions(null); } }
    }, 350);
    return () => { live = false; clearTimeout(t); };
  }, [body, engine, ref, multiNode, optionsTry]);
  const chosen = options?.find((o) => placementKey(o.placement) === placementPick) ?? options?.find((o) => o.recommended) ?? null;
  const placement = multiNode && chosen ? chosen.placement : null;
  const placementBlocked = multiNode && (!chosen || chosen.reason === "node_stale");

  // the first preview waits for the options (it follows the pre-selected placement); later ones run on each change
  // of the request or the placement, never again just because the options were re-read
  const optionsReady = !multiNode || options !== null || optionsErr !== null;
  const previewBody = useMemo(() => (placement ? { ...body, placement } : body), [body, placement?.kind, placement?.nodes.join(",")]); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => {
    if (!engine || !ref) return;
    if (!optionsReady) return;
    let live = true; // a slower, older preview must not overwrite the current one
    const t = setTimeout(async () => {
      try {
        const p = await api<Preview>("/deployments/preview", { method: "POST", body: previewBody });
        if (live) { setPreview(p); setPreviewErr(null); }
      } catch (e) { if (live) { setPreviewErr(e); setPreview(null); } }
    }, 350);
    return () => { live = false; clearTimeout(t); };
  }, [previewBody, engine, ref, optionsReady]);

  const actions = plan?.actions ?? [];
  const deploy = useActionMutation(() => api<{ id: number }>("/deployments", {
    method: "POST", body: {
      name, engine, model_ref: ref, config, priority, alias: alias.trim() || null, actions, mode,
      idle_timeout_s: mode === "on_demand" && idle !== idleDefault ? minutesToSeconds(parseIdleMinutes(idle)!) : null,
      ...(placement ? { placement } : {}),
    },
  }), {
    success: (d: any) => `Deploying ${name} (id ${d.id})`,
    invalidate: [["deployments"], ["host"]],
    onSuccess: (d: any) => nav(`/deployments/${d.id}`),
  });

  // A 409 carries a fresh fit + plans: show them and make the user choose again. Depends only on the error,
  // so choosing a plan afterwards sticks (and the functional update cannot re-trigger this effect).
  useEffect(() => {
    const e = deploy.error;
    if (!(e instanceof ApiError) || e.code !== "insufficient_memory") return;
    setPlan(null);
    const { fit, plans, notes } = e.detail ?? {};
    if (fit && plans) setPreview((p) => p && { ...p, fit, plans, notes: notes ?? p.notes });
  }, [deploy.error]);

  const aliasErr = aliasProblem(alias);
  const baseSubmit = submitState(preview, actions, deploy.isPending, "deploy");
  // a placement that does not fit needs a plan, like a preview that does not fit (its preview says so too, unless
  // memory changed since the options were read)
  // Whether the chosen placement fits is the preview's (it follows the placement and is re-read on every change;
  // the options' `fits` may be older). Blocked here only: nodes not known yet, no placement chosen, or a stale node.
  const submit = deploy.isPending ? baseSubmit
    : nodes.isPending ? { disabled: true, label: "Checking nodes…" }
    : placementBlocked ? { disabled: true, label: "Choose where to run" }
    : { ...baseSubmit, disabled: baseSubmit.disabled || !!aliasErr || idleInvalid(mode, idle) };
  const memoryChanged = deploy.error instanceof ApiError && deploy.error.code === "insufficient_memory";

  const check = (
    <div className="space-y-4">
      {nodes.isError && (
        <div role="alert" className="space-y-1 rounded-lg border border-status-warning/40 bg-status-warning-bg/40 p-3 text-xs">
          <p>Could not read the configured nodes, so this deploys on this DGX only and the Where to run step is hidden.</p>
          <Button size="sm" variant="outline" className="h-7 text-xs" onClick={() => nodes.refetch()}>Retry</Button>
        </div>
      )}
      <ErrorState error={previewErr} />
      {preview ? <EstimatePanel preview={preview} /> : <p className="text-sm text-muted-foreground">Choose a model to see the memory check.</p>}
      {preview && !preview.fit.fits && (
        <PlanChooser plans={preview.plans ?? []} notes={preview.notes} selectedId={plan?.id ?? null}
          onSelect={(o) => {
            setPlan({ id: o.id, actions: o.actions.filter((a) => a.type !== "patch") });
            if (o.patch) setConfig({ ...config, ...o.patch });
            if (o.model_ref) setRef(o.model_ref); // triggers artifact effect → resets config/plan, but user sees model change
          }} />
      )}
      {memoryChanged && <p className="text-xs text-status-critical">Memory changed since the preview — choose a plan again.</p>}
      <CopyWeightsPrompt error={deploy.error} nodes={nodes.data} />
    </div>
  );

  return (
    <div className="pb-[calc(44px+env(safe-area-inset-bottom)+1rem)] md:pb-0">
      <PageHeader title="Deploy a model" description="DGXMM checks memory before anything starts. Nothing is evicted without your confirmation." />
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-[minmax(0,1fr)_380px]">
        <div className="space-y-4">
          <Section n={1} title="Model" titleId="dep-model-label" description="Only complete, downloaded models are listed.">
            <ModelSelect id="dep-model" labelledBy="dep-model-label" artifacts={deployable} value={ref} onChange={setRef} />
            {artifact && <p className="mt-2 text-xs text-muted-foreground tabular-nums">{artifact.architecture} · {artifact.quant} · {fmtBytes(artifact.weights_bytes)} weights · max context {artifact.max_context?.toLocaleString() ?? "?"}</p>}
            {!artifact && inv.isSuccess && REPO_ID.test(ref) && !(inv.data ?? []).some((a) => a.ref === ref) && (
              <div className="mt-3 space-y-2" data-not-downloaded>
                <p className="text-xs text-muted-foreground">{ref} is not downloaded yet. Download it from the Catalog first; this is how it would run here:</p>
                <CompatLine repo={ref} />
              </div>
            )}
          </Section>
          {artifact && (
            <>
              <Section n={2} title="Engine and name">
                <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
                  <div className="space-y-1.5"><Label id="dep-engine-label">Engine</Label>
                    <Select value={engine} onValueChange={(e) => pickEngine(e)}>
                      <SelectTrigger id="dep-engine" aria-labelledby="dep-engine-label dep-engine-value"><SelectValue id="dep-engine-value" /></SelectTrigger>
                      <SelectContent>{artifact.engines.map((e) => <SelectItem key={e} value={e}>{engineLabel(e)}</SelectItem>)}</SelectContent>
                    </Select>
                    <EngineChip modelRef={ref} goal={goal} engine={engine}
                      onUse={(e, r) => {
                        if (r !== ref) {
                          const next = deployable.find((a) => a.ref === r);
                          carry.current = carryFor(next?.engines[0] ?? e); setRef(r);
                        } else if (artifact.engines.includes(e)) pickEngine(e, carryFor(e));
                      }} /></div>
                  <div className="space-y-1.5"><Label htmlFor="dep-name">Deployment name</Label><Input id="dep-name" value={name} onChange={(e) => setName(e.target.value)} /></div>
                  <div className="space-y-1.5"><Label htmlFor="dep-prio">Priority</Label><Input id="dep-prio" type="number" value={priority} onChange={(e) => { const n = parseInt(e.target.value, 10); setPriority(Number.isFinite(n) ? n : 0); }} />
                    <p className="text-xs text-muted-foreground">Lower priority is evicted first.</p></div>
                  <div className="space-y-1.5"><Label htmlFor="dep-alias">Model name (LiteLLM)</Label>
                    <Input ref={aliasInput} id="dep-alias" placeholder={preview?.served_name ?? "optional, e.g. default-chat"} value={alias} aria-invalid={!!aliasErr}
                      aria-describedby="dep-alias-help" onChange={(e) => setAlias(e.target.value)} />
                    {isModelNameTaken(deploy.error) && <ModelNameTaken error={deploy.error} suggestion={suggestAlias(preview?.served_name, name)}
                      onUse={(a) => { setAlias(a); deploy.reset(); aliasInput.current?.focus(); }} />}
                    {aliasErr ? <p id="dep-alias-help" className="text-xs text-status-critical">{aliasErr}</p> : (
                      <p id="dep-alias-help" className="break-words text-xs text-muted-foreground">
                        Clients call <code className="font-mono">{litellm.data?.endpoint ?? "the LiteLLM endpoint"}</code> with model: <code className="font-mono">{alias.trim() || preview?.served_name || "<served name>"}</code>. Defaults to the served name.
                      </p>)}
                  </div>
                </div>
                <div className="mt-4 border-t pt-4"><ModeFields id="dep-mode" mode={mode} onMode={setMode} idle={idle} onIdle={setIdle} hint={ondemandReady === false ? ONDEMAND_NEEDS_LITELLM : null} /></div>
              </Section>
              <Section n={3} title="Sizing and features">
                <div className="mb-4 space-y-3">
                  <div className="max-w-xs space-y-1.5"><Label id="dep-goal-label">Goal</Label>
                    <Select value={goal} onValueChange={(g) => setGoalPick(g as DeployGoal)}>
                      <SelectTrigger id="dep-goal" aria-labelledby="dep-goal-label dep-goal-value"><SelectValue id="dep-goal-value" /></SelectTrigger>
                      <SelectContent>{GOALS.map((g) => <SelectItem key={g.value} value={g.value}>{g.label}</SelectItem>)}</SelectContent>
                    </Select>
                    <p className="text-xs text-muted-foreground">{GOALS.find((g) => g.value === goal)?.help}</p></div>
                  <ConfigAdviceCard modelRef={ref} engine={engine} goal={goal} config={config} touched={touched}
                    fits={preview ? preview.fit.fits : null} onFill={(c) => { setConfig(c); setPlan(null); }} />
                </div>
                <ConfigForm engine={engine} value={config} onChange={userEdit} showAdvanced={advanced} hints={parsers.hints} />
                {engine === "vllm" && (
                  <div className="mt-4"><SpeculativeField modelRef={ref} value={config.speculative} drafts={inv.data ?? []}
                    onChange={(v) => { const { speculative: _old, ...rest } = config; userEdit(v ? { ...rest, speculative: v } : rest); }} /></div>
                )}
                {config.trust_remote_code === true && (
                  <div className="mt-4 space-y-2"><RemoteCodeWarning /><RemoteCodeFindings modelRef={ref} enabled={!!artifact.has_remote_code} /></div>
                )}
                <Button variant="link" className="mt-3 h-auto p-0 text-xs" onClick={() => setAdvanced(!advanced)}>{advanced ? "Hide" : "Show"} advanced settings</Button>
              </Section>
              {multiNode && (
                <Section n={4} title="Where to run" description="One DGX, or the DGXs together as a cluster for a model too big for one. The memory check follows your choice.">
                  <ErrorState error={optionsErr} onRetry={() => setOptionsTry((n) => n + 1)} />
                  {options ? (
                    <PlacementChooser options={options} nodes={nodes.data} selected={chosen ? placementKey(chosen.placement) : null}
                      onSelect={(o) => { setPlacementPick(placementKey(o.placement)); setPlan(null); }} />
                  ) : !optionsErr && <p className="text-sm text-muted-foreground">Checking where it fits…</p>}
                </Section>
              )}
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
