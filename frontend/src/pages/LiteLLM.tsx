import { useMemo, useState, type ReactNode } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { Link, Navigate } from "react-router-dom";
import type { ColumnDef } from "@tanstack/react-table";
import { AlertTriangle, Boxes, KeyRound, Loader2, Network, PlugZap, RefreshCw, RotateCw } from "lucide-react";
import { api } from "@/api";
import { useIsAdmin } from "@/auth";
import { CopyButton } from "@/components/app/copy-button";
import { DataTable } from "@/components/app/data-table";
import { KeyManager } from "@/components/app/litellm-keys";
import { PageHeader } from "@/components/app/page-header";
import { PasswordInput } from "@/components/app/password-input";
import { EmptyState, ErrorState, PageSkeleton } from "@/components/app/states";
import { StatusBadge } from "@/components/app/status-badge";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { useActionMutation } from "@/hooks/use-action-mutation";
import { connectionPatch, endpointPreview, litellmProblem, modelNameOptions, syncMessage, urlProblem, type KeyMode } from "@/lib/litellm";
import type { Deployment, LiteLLMModel, LiteLLMKey, LiteLLMSettings, LiteLLMSyncResult, LiteLLMTestResult } from "@/types";

const SOURCE: Record<LiteLLMSettings["master_key_source"], string> = { settings: "Set (from settings)", secret_file: "Set (from secret file)", none: "Not set" };

function Notice({ tone, title, children }: { tone: "warning" | "critical"; title: string; children: ReactNode }) {
  const cls = tone === "critical" ? "border-status-critical/30 bg-status-critical-bg text-status-critical" : "border-status-warning/30 bg-status-warning-bg text-status-warning";
  return (
    <div role="alert" className={`flex items-start gap-3 rounded-lg border px-4 py-3 text-sm ${cls}`}>
      <AlertTriangle aria-hidden className="mt-0.5 h-4 w-4 shrink-0" />
      <div className="min-w-0 break-words"><p className="font-medium">{title}</p><div className="mt-0.5 opacity-90">{children}</div></div>
    </div>
  );
}

function Row({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="flex items-center justify-between gap-3 border-b py-1.5 last:border-0">
      <dt className="text-muted-foreground">{label}</dt><dd className="min-w-0 break-all text-right">{children}</dd>
    </div>
  );
}
const yesNo = (v: boolean | null | undefined, unknown = "Unknown") => (v === true ? "Yes" : v === false ? "No" : unknown);

function TestResult({ r }: { r: LiteLLMTestResult }) {
  const good = r.reachable && r.auth_ok;
  return (
    <div role="status" aria-label="Connection test result" className="rounded-md border p-3 text-sm">
      <p className="mb-2 flex items-center gap-2 font-medium"><StatusBadge state={good ? "ready" : "error"} />{good ? "Connected" : "Connection problem"}</p>
      <dl>
        <Row label="Reachable">{yesNo(r.reachable)}</Row>
        <Row label="Master key accepted">{yesNo(r.auth_ok, "Not checked")}</Row>
        <Row label="Version">{r.version ?? "—"}</Row>
        <Row label="Database">{r.db ?? "Unknown"}</Row>
        <Row label="Models">{r.models ?? "—"}</Row>
      </dl>
      {r.error && <p className="mt-2 break-words text-xs text-status-critical">{r.error}</p>}
    </div>
  );
}

function ConnectionCard({ s }: { s: LiteLLMSettings }) {
  const qc = useQueryClient();
  const [url, setUrl] = useState(s.url);
  const [pub, setPub] = useState(s.public_url);
  const [enabled, setEnabled] = useState(s.enabled);
  const [keyMode, setKeyMode] = useState<KeyMode>("keep");
  const [masterKey, setMasterKey] = useState("");
  const [test, setTest] = useState<LiteLLMTestResult | null>(null);
  const [sync, setSync] = useState<LiteLLMSyncResult | null>(null);
  const { body, keyDetaches } = connectionPatch(s, { enabled, url, public_url: pub, keyMode, masterKey });
  const dirty = Object.keys(body).length > 0;
  // Keys are bound to the URL they were entered for: prompt for one when the URL changes or the server says so.
  const hint = keyDetaches
    ? `The current master key is only used with ${s.url}. Enter the master key for the new URL, otherwise DGXMM has no key for it.`
    : s.master_key_hint;
  const prompt = !!hint && keyMode !== "clear";
  const urlErr = urlProblem(url);
  const pubErr = urlProblem(pub);
  const preview = endpointPreview(url, pub);
  const keyInput = keyMode === "replace" || prompt;
  const reset = (next: LiteLLMSettings) => {
    setUrl(next.url); setPub(next.public_url); setEnabled(next.enabled); setKeyMode("keep"); setMasterKey("");
  };
  const save = useActionMutation(() => api<LiteLLMSettings>("/integrations/litellm", { method: "PUT", body }), {
    success: "LiteLLM settings saved", invalidate: [["litellm-models"], ["litellm-keys"], ["deployments"]], gcTime: 0,
    onSuccess: (next) => { qc.setQueryData(["litellm"], next); reset(next); setTest(null); },
  });
  const runTest = useActionMutation(() => api<LiteLLMTestResult>("/integrations/litellm/test", { method: "POST" }), { onSuccess: setTest });
  const runSync = useActionMutation(() => api<LiteLLMSyncResult>("/integrations/litellm/sync", { method: "POST" }), {
    success: "Sync finished", invalidate: [["litellm-models"], ["deployments"], ["deployment-litellm"]], onSuccess: setSync,
  });
  const saveErr = save.error as (Error & { code?: string; detail?: any }) | null;
  const fieldErr = (f: string) => (saveErr?.detail?.field === f ? saveErr.message : undefined);

  return (
    <Card className="min-w-0 shadow-card">
      <CardHeader className="pb-3">
        <CardTitle className="flex items-center gap-2 text-sm"><PlugZap aria-hidden className="h-4 w-4" />Connection</CardTitle>
        <CardDescription className="text-xs">How DGXMM reaches LiteLLM, and the endpoint clients are told to use.</CardDescription>
      </CardHeader>
      <CardContent>
        <form className="space-y-4" noValidate onSubmit={(e) => { e.preventDefault(); if (dirty && !urlErr && !pubErr) save.mutate(undefined); }}>
          <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
            <div className="space-y-1.5">
              <Label htmlFor="ll-url">Internal URL</Label>
              <Input id="ll-url" autoComplete="off" value={url} onChange={(e) => setUrl(e.target.value)} aria-invalid={!!urlErr} placeholder="http://litellm-proxy:4000" />
              <p className={`text-xs ${urlErr || fieldErr("url") ? "text-status-critical" : "text-muted-foreground"}`}>
                {urlErr ?? fieldErr("url") ?? "Where DGXMM reaches LiteLLM, e.g. http://litellm-proxy:4000. Empty: the config default."}</p>
            </div>
            <div className="space-y-1.5">
              <Label htmlFor="ll-public">Public endpoint URL</Label>
              <Input id="ll-public" autoComplete="off" value={pub} onChange={(e) => setPub(e.target.value)} aria-invalid={!!pubErr} placeholder="https://llm.example.com" />
              <p className={`text-xs ${pubErr || fieldErr("public_url") ? "text-status-critical" : "text-muted-foreground"}`}>
                {pubErr ?? fieldErr("public_url") ?? (pub.trim() ? "What clients on other machines use." : "Not set: clients are shown the internal URL, which other machines may not reach.")}</p>
            </div>
          </div>
          <div className="flex min-w-0 items-center gap-2 rounded-md bg-muted/50 px-3 py-2 text-sm">
            <span className="shrink-0 text-muted-foreground">Clients call</span>
            <code data-testid="endpoint-preview" className="min-w-0 flex-1 break-all font-mono text-xs">{preview || "—"}</code>
            {preview && <CopyButton value={preview} label="Copy endpoint" />}
          </div>

          <div className="space-y-2 rounded-md border p-3">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <div>
                <p className="text-sm font-medium">Master key</p>
                <p className="flex items-center gap-2 text-xs text-muted-foreground">
                  {s.master_key_set ? <KeyRound aria-hidden className="h-3.5 w-3.5 text-status-good" /> : <AlertTriangle aria-hidden className="h-3.5 w-3.5 text-status-warning" />}
                  {keyMode === "clear" ? "Will be cleared on save" : SOURCE[s.master_key_source]}
                </p>
              </div>
              <div className="flex gap-2">
                {keyMode === "keep" && !prompt && <Button type="button" variant="outline" size="sm" onClick={() => setKeyMode("replace")}>{s.master_key_set ? "Replace" : "Set key"}</Button>}
                {keyMode === "keep" && s.master_key_source === "settings" && <Button type="button" variant="outline" size="sm" onClick={() => setKeyMode("clear")}>Clear</Button>}
                {keyMode !== "keep" && !prompt && <Button type="button" variant="ghost" size="sm" onClick={() => { setKeyMode("keep"); setMasterKey(""); }}>Cancel</Button>}
              </div>
            </div>
            {hint && (
              <p data-testid="master-key-hint" role="note" className="flex items-start gap-2 rounded-md border border-status-warning/30 bg-status-warning-bg px-3 py-2 text-sm text-status-warning">
                <AlertTriangle aria-hidden className="mt-0.5 h-4 w-4 shrink-0" /><span className="min-w-0 break-words">{hint}</span>
              </p>
            )}
            {keyMode === "clear" && <p className="text-xs text-muted-foreground">The key stored in DGXMM is removed; DGXMM then uses the secret file key if there is one.</p>}
            {keyInput && (
              <div className="space-y-1.5">
                <Label htmlFor="ll-key">New master key</Label>
                <PasswordInput id="ll-key" autoComplete="new-password" spellCheck={false} value={masterKey}
                  onChange={(e) => { setMasterKey(e.target.value); setKeyMode("replace"); }} aria-describedby="ll-key-help" />
                <p id="ll-key-help" className="text-xs text-muted-foreground">Write-only: DGXMM stores it for this URL and never shows it again.</p>
              </div>
            )}
            {fieldErr("master_key") && <p className="text-xs text-status-critical">{fieldErr("master_key")}</p>}
          </div>

          <div className="flex items-center justify-between gap-3 rounded-md border p-3">
            <div><Label htmlFor="ll-enabled">Integration enabled</Label>
              <p className="text-xs text-muted-foreground">When on, ready deployments are registered in LiteLLM automatically.</p></div>
            <Switch id="ll-enabled" checked={enabled} onCheckedChange={setEnabled} />
          </div>

          {saveErr && !saveErr.detail?.field && <ErrorState error={saveErr} />}
          <div className="flex flex-wrap items-center gap-2">
            <Button type="submit" size="sm" disabled={!dirty || !!urlErr || !!pubErr || save.isPending}>
              {save.isPending && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}Save</Button>
            {dirty && <Button type="button" variant="ghost" size="sm" onClick={() => reset(s)}>Discard changes</Button>}
            <span className="flex-1" />
            <Button type="button" variant="outline" size="sm" disabled={runTest.isPending} onClick={() => runTest.mutate(undefined)}>
              {runTest.isPending ? <Loader2 className="mr-1 h-4 w-4 animate-spin" /> : <PlugZap className="mr-1 h-4 w-4" />}Test connection</Button>
            <Button type="button" variant="outline" size="sm" disabled={runSync.isPending} onClick={() => runSync.mutate(undefined)}>
              {runSync.isPending ? <Loader2 className="mr-1 h-4 w-4 animate-spin" /> : <RefreshCw className="mr-1 h-4 w-4" />}Sync now</Button>
          </div>
          {dirty && <p className="text-xs text-muted-foreground">Test connection and Sync now use the saved settings. Save first to try new values.</p>}
          {test && <TestResult r={test} />}
          {sync && (
            <p role="status" className="text-sm text-muted-foreground">
              {syncMessage(sync)}
              {sync.error && <span className="block text-status-critical">{sync.error}</span>}
            </p>
          )}
        </form>
      </CardContent>
    </Card>
  );
}

/** Why models / keys cannot be listed right now, or null if they can. */
function blocker(s: LiteLLMSettings, needsDb: boolean): { title: string; description: string } | null {
  if (!s.master_key_set) return litellmProblem({ code: "litellm_not_configured", message: s.master_key_hint ?? "" });
  if (!s.enabled) return litellmProblem({ code: "litellm_disabled", message: "" });
  if (!s.status.reachable) return litellmProblem({ code: "litellm_unreachable", message: s.status.error ?? "LiteLLM did not answer." });
  if (needsDb && s.status.db && s.status.db !== "connected") return litellmProblem({ code: "litellm_no_db", message: "" });
  return null;
}

function Problem({ icon, p, onRetry }: { icon: typeof Boxes; p: { title: string; description: string }; onRetry?: () => void }) {
  return <EmptyState icon={icon} title={p.title} description={p.description}
    action={onRetry && <Button variant="outline" size="sm" onClick={onRetry}><RotateCw className="mr-1 h-3.5 w-3.5" />Retry</Button>} />;
}

function ModelsCard({ s }: { s: LiteLLMSettings }) {
  const block = blocker(s, false);
  const q = useQuery({ queryKey: ["litellm-models"], queryFn: () => api<LiteLLMModel[]>("/integrations/litellm/models"), enabled: !block });
  const columns = useMemo<ColumnDef<LiteLLMModel, any>[]>(() => [
    { accessorKey: "model_name", header: "Model name", meta: { mono: true } },
    { id: "managed", header: "Managed by", accessorFn: (m) => m.deployment?.name ?? (m.managed_by_dgxmm ? "~orphan" : "~external"), cell: ({ row }) => {
      const m = row.original;
      if (m.deployment) return <span className="flex flex-wrap items-center gap-2"><Link to={`/deployments/${m.deployment.id}`} className="font-medium hover:underline">{m.deployment.name}</Link><StatusBadge state={m.deployment.state} /></span>;
      if (m.managed_by_dgxmm) return <Badge variant="outline" className="text-status-warning">DGXMM (deployment removed)</Badge>;
      return <Badge variant="secondary">external</Badge>;
    } },
    { accessorKey: "model", header: "Backend model", meta: { mono: true, hideOnMobile: true } },
    { accessorKey: "api_base", header: "API base", meta: { mono: true, hideOnMobile: true }, cell: ({ getValue }) => getValue() ?? "—" },
  ], []);
  return (
    <Card className="min-w-0 shadow-card">
      <CardHeader className="pb-3">
        <CardTitle className="flex items-center gap-2 text-sm"><Boxes aria-hidden className="h-4 w-4" />Models</CardTitle>
        <CardDescription className="text-xs">Every model LiteLLM serves. DGXMM registers each ready deployment once, under its model name.</CardDescription>
      </CardHeader>
      <CardContent>
        {block ? <Problem icon={Boxes} p={block} />
          : q.error ? <Problem icon={Boxes} p={litellmProblem(q.error)} onRetry={() => q.refetch()} />
          : <DataTable columns={columns} data={q.data ?? []} loading={q.isLoading} getRowId={(m) => m.id}
              searchPlaceholder={(q.data?.length ?? 0) > 8 ? "Search models" : undefined}
              empty={<EmptyState icon={Boxes} title="No models in LiteLLM" description="Ready deployments appear here after the next sync." />} />}
      </CardContent>
    </Card>
  );
}

function KeysCard({ s }: { s: LiteLLMSettings }) {
  const block = blocker(s, true);
  const q = useQuery({ queryKey: ["litellm-keys"], queryFn: () => api<LiteLLMKey[]>("/integrations/litellm/keys"), enabled: !block });
  const deps = useQuery({ queryKey: ["deployments"], queryFn: () => api<Deployment[]>("/deployments"), enabled: !block });
  const options = modelNameOptions(deps.data ?? []);
  return (
    <Card className="min-w-0 shadow-card">
      <CardHeader className="pb-3">
        <CardTitle className="flex items-center gap-2 text-sm"><KeyRound aria-hidden className="h-4 w-4" />Virtual keys</CardTitle>
        <CardDescription className="text-xs">Give apps and people their own key, limited to some models, a budget or a rate.</CardDescription>
      </CardHeader>
      <CardContent>
        {/* KeyManager stays mounted through errors: it owns the one-time created-key dialog. */}
        <KeyManager keys={q.data ?? []} loading={q.isLoading} modelOptions={options} endpoint={s.endpoint}
          problem={block ? <Problem icon={KeyRound} p={block} />
            : q.error ? <Problem icon={KeyRound} p={litellmProblem(q.error)} onRetry={() => q.refetch()} /> : undefined} />
      </CardContent>
    </Card>
  );
}

function LiteLLMPage() {
  const q = useQuery({ queryKey: ["litellm"], queryFn: () => api<LiteLLMSettings>("/integrations/litellm") });
  if (q.isLoading) return <PageSkeleton />;
  const s = q.data;
  const header = <PageHeader title="LiteLLM" description="One OpenAI-compatible endpoint for every deployment: clients pick a deployment with the model field." />;
  if (!s) return <div className="space-y-4">{header}<ErrorState error={q.error ?? new Error("Could not load LiteLLM settings")} onRetry={() => q.refetch()} /></div>;
  const saved = `${s.url}|${s.public_url}|${s.enabled}|${s.master_key_source}`;
  return (
    <div className="space-y-4">
      {header}
      {!s.status.reachable && <Notice tone="critical" title="Cannot reach LiteLLM">{s.status.error ?? `No answer from ${s.url}.`} Check the internal URL and that the proxy is running.</Notice>}
      {!s.master_key_set && <Notice tone="warning" title="No master key set">{s.master_key_hint ?? "DGXMM needs the LiteLLM master key to register deployments and manage keys. Set it below."}</Notice>}
      {s.master_key_set && !s.enabled && <Notice tone="warning" title="Integration disabled">Deployments are not registered in LiteLLM while the integration is off.</Notice>}
      {s.status.reachable && (
        <p className="flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
          <Network aria-hidden className="h-3.5 w-3.5" />LiteLLM {s.status.version ?? "(version unknown)"} · database {s.status.db ?? "unknown"}
        </p>
      )}
      <ConnectionCard key={saved} s={s} />
      <ModelsCard s={s} />
      <KeysCard s={s} />
    </div>
  );
}

export function LiteLLM() {
  return useIsAdmin() ? <LiteLLMPage /> : <Navigate to="/" replace />;
}
