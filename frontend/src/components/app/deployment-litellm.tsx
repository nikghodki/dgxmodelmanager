import { useEffect, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import { AlertTriangle, KeyRound, Loader2, RefreshCw } from "lucide-react";
import { api } from "@/api";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { useActionMutation } from "@/hooks/use-action-mutation";
import { aliasProblem, curlSnippet, litellmProblem, modelNameOptions, pythonSnippet } from "@/lib/litellm";
import type { Deployment, DeploymentLiteLLM } from "@/types";
import { CopyButton } from "./copy-button";
import { CodeBlock, KeyManager } from "./litellm-keys";
import { EmptyState, ErrorState } from "./states";
import { StatusBadge } from "./status-badge";

export function KV({ label, value, mono, copy }: { label: string; value: React.ReactNode; mono?: boolean; copy?: string | null }) {
  return (
    <div className="flex items-center justify-between gap-3 border-b py-2 last:border-0">
      <dt className="shrink-0 text-sm text-muted-foreground">{label}</dt>
      <dd className={`flex min-w-0 items-center gap-1 text-right text-sm ${mono ? "font-mono text-xs" : ""}`}>
        <span className="min-w-0 break-all">{value ?? "—"}</span>{copy && <CopyButton value={copy} label={`Copy ${label}`} />}
      </dd>
    </div>
  );
}

export const modelNameOf = (d: Deployment, info?: DeploymentLiteLLM) => info?.model_name ?? d.model_name ?? d.alias ?? d.served_name ?? "";

export function useDeploymentLiteLLM(id: number) {
  return useQuery({ queryKey: ["deployment-litellm", id], queryFn: () => api<DeploymentLiteLLM>(`/deployments/${id}/litellm`) });
}

function Warn({ children }: { children: React.ReactNode }) {
  return <p className="flex items-start gap-2 text-xs text-status-warning"><AlertTriangle aria-hidden className="mt-0.5 h-3.5 w-3.5 shrink-0" /><span>{children}</span></p>;
}

/** Overview card: the one way to call this deployment (LiteLLM endpoint + model), with the raw engine URLs tucked away. */
export function ApiAccessCard({ d, info, loading, error }: { d: Deployment; info?: DeploymentLiteLLM; loading: boolean; error: unknown }) {
  const model = modelNameOf(d, info);
  const endpoint = info?.endpoint ?? "";
  return (
    <Card data-card className="min-w-0 shadow-card lg:col-span-3">
      <CardHeader className="flex-row flex-wrap items-start justify-between gap-2 space-y-0 pb-2">
        <div className="space-y-1.5">
          <CardTitle className="text-sm"><h2>API access</h2></CardTitle>
          <CardDescription className="text-xs">OpenAI-compatible, through LiteLLM. Every deployment shares this endpoint; the model field picks this one.</CardDescription>
        </div>
        <StatusBadge state={info?.status ?? d.litellm_status} />
      </CardHeader>
      <CardContent className="space-y-4">
        <dl>
          <KV label="Endpoint" mono value={endpoint || (loading ? "Loading…" : "—")} copy={endpoint || null} />
          <KV label="Model" mono value={model} copy={model || null} />
        </dl>
        {!!error && <ErrorState error={error} />}
        {info && !info.enabled && <Warn>The LiteLLM integration is disabled, so this model is not served through the endpoint.</Warn>}
        {info?.enabled && info.registered === false && <Warn>This model is not registered in LiteLLM yet; requests will fail until the next sync.</Warn>}
        {info?.error && info.registered === null && <Warn>Could not check LiteLLM: {litellmProblem(info.error).description}</Warn>}
        {endpoint && model && (
          <div className="grid grid-cols-1 gap-3 xl:grid-cols-2">
            <div className="min-w-0 space-y-1.5"><p className="text-xs font-medium text-muted-foreground">curl</p><CodeBlock code={curlSnippet(endpoint, model)} label="curl example" /></div>
            <div className="min-w-0 space-y-1.5"><p className="text-xs font-medium text-muted-foreground">Python (OpenAI SDK)</p><CodeBlock code={pythonSnippet(endpoint, model)} label="Python example" /></div>
          </div>
        )}
        <details className="group rounded-md border px-3 py-2">
          <summary className="cursor-pointer text-sm font-medium">Direct engine endpoints (advanced)</summary>
          <p className="mt-1 text-xs text-muted-foreground">These bypass LiteLLM: no virtual keys, budgets or rate limits apply.</p>
          <dl className="mt-1">
            <KV label="Served name" value={d.served_name} mono copy={d.served_name} />
            <KV label="Gateway network" value={d.endpoint} mono copy={d.endpoint} />
            <KV label="On the DGX" value={d.host_endpoint} mono copy={d.host_endpoint} />
          </dl>
        </details>
      </CardContent>
    </Card>
  );
}

function ModelNameForm({ d }: { d: Deployment }) {
  const [value, setValue] = useState(d.alias ?? "");
  useEffect(() => setValue(d.alias ?? ""), [d.alias]);
  const save = useActionMutation((alias: string) => api<Deployment>(`/deployments/${d.id}`, { method: "PATCH", body: { alias } }), {
    success: (_r, a) => (a ? `Model name set to ${a}` : "Model name reset to the served name"),
    invalidate: [["deployment", d.id], ["deployments"], ["deployment-litellm", d.id], ["litellm-models"]],
  });
  const problem = aliasProblem(value);
  const changed = value.trim() !== (d.alias ?? "");
  return (
    <form className="space-y-1.5" noValidate onSubmit={(e) => { e.preventDefault(); if (!problem && changed) save.mutate(value.trim()); }}>
      <Label htmlFor="dep-model-name">Model name</Label>
      <div className="flex flex-col gap-2 sm:flex-row">
        <Input id="dep-model-name" autoComplete="off" spellCheck={false} className="font-mono" value={value} placeholder={d.served_name ?? ""}
          aria-invalid={!!problem || !!save.error} aria-describedby="dep-model-name-help"
          onChange={(e) => { setValue(e.target.value); if (save.error) save.reset(); }} />
        <Button type="submit" size="sm" className="h-9 shrink-0" aria-label="Save model name" disabled={!!problem || !changed || save.isPending}>
          {save.isPending && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}Save
        </Button>
      </div>
      <div id="dep-model-name-help">
        {problem ? <p className="text-xs text-status-critical">{problem}</p>
          : save.error ? <p role="alert" className="text-xs text-status-critical">{save.error.message}</p>
          : <p className="text-xs text-muted-foreground">Clients send this as model. Leave empty to use the served name ({d.served_name ?? "—"}). Renaming re-registers it in LiteLLM; clients using the old name stop working.</p>}
      </div>
    </form>
  );
}

/** The deployment's LiteLLM tab. Admins can rename the model, re-sync and manage keys; viewers only read. */
export function LiteLLMTab({ d, info, loading, error, admin }: { d: Deployment; info?: DeploymentLiteLLM; loading: boolean; error: unknown; admin: boolean }) {
  const model = modelNameOf(d, info);
  const deps = useQuery({ queryKey: ["deployments"], queryFn: () => api<Deployment[]>("/deployments"), enabled: admin });
  const sync = useActionMutation(() => api("/integrations/litellm/sync", { method: "POST" }), {
    success: "LiteLLM sync finished", invalidate: [["deployment-litellm", d.id], ["deployment", d.id], ["deployments"]],
  });
  const registered = info?.registered === true ? "Yes" : info?.registered === false ? "No" : loading ? "Checking…" : "Unknown";
  return (
    <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
      <Card className="min-w-0 shadow-card">
        <CardHeader className="pb-2"><CardTitle className="text-sm">Model name</CardTitle>
          <CardDescription className="text-xs">The name this deployment is registered under in LiteLLM.</CardDescription></CardHeader>
        <CardContent>
          {admin ? <ModelNameForm d={d} /> : <dl><KV label="Model name" mono value={model} copy={model || null} /></dl>}
        </CardContent>
      </Card>
      <Card className="min-w-0 shadow-card">
        <CardHeader className="flex-row items-start justify-between gap-2 space-y-0 pb-2">
          <div className="space-y-1.5"><CardTitle className="text-sm">Registration</CardTitle>
            <CardDescription className="text-xs">DGXMM registers ready deployments once, under the model name.</CardDescription></div>
          {admin && <Button variant="outline" size="sm" disabled={sync.isPending} onClick={() => sync.mutate(undefined)}>
            {sync.isPending ? <Loader2 className="mr-1 h-4 w-4 animate-spin" /> : <RefreshCw className="mr-1 h-4 w-4" />}Re-sync</Button>}
        </CardHeader>
        <CardContent className="space-y-2">
          <dl>
            <KV label="Status" value={<StatusBadge state={info?.status ?? d.litellm_status} />} />
            <KV label="Found in LiteLLM" value={registered} />
            <KV label="Endpoint" mono value={info?.endpoint} copy={info?.endpoint} />
          </dl>
          {!!error && <ErrorState error={error} />}
          {info && !info.enabled && <Warn>The integration is disabled.{admin && <> Turn it on in <Link className="underline" to="/litellm">LiteLLM settings</Link>.</>}</Warn>}
          {info?.error && <Warn>{litellmProblem(info.error).title}: {litellmProblem(info.error).description}</Warn>}
        </CardContent>
      </Card>
      {admin && info && (
        <Card className="min-w-0 shadow-card lg:col-span-2">
          <CardHeader className="pb-2"><CardTitle className="text-sm">Keys with access</CardTitle>
            <CardDescription className="text-xs">Virtual keys allowed to call {model}: keys listing it, and keys for all models.</CardDescription></CardHeader>
          <CardContent>
            {info.keys_error ? <EmptyState icon={KeyRound} title={litellmProblem(info.keys_error).title} description={litellmProblem(info.keys_error).description} />
              : <KeyManager keys={info.keys} modelOptions={modelNameOptions(deps.data ?? [], [model])} endpoint={info.endpoint}
                  preselect={model ? [model] : undefined} createLabel="Create key for this deployment"
                  empty={<EmptyState icon={KeyRound} title="No keys can call this model yet" description="Create a key for this deployment, or one for all models." />} />}
          </CardContent>
        </Card>
      )}
    </div>
  );
}
