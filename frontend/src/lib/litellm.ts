import type { ApiProblem, Deployment, DeploymentRef, LiteLLMKey, LiteLLMSyncResult } from "@/types";

/** Same rule as the backend `ALIAS_PATTERN`; the model name clients pass to LiteLLM. */
export const ALIAS_RE = /^[A-Za-z0-9._:/-]{1,64}$/;
const KEY_ALIAS_RE = /^[A-Za-z0-9._:@/ -]{1,128}$/;
const DURATION_RE = /^[1-9][0-9]{0,5}(s|m|h|d|w|mo)$/;
const ALL_PROXY_MODELS = "all-proxy-models";

const trimSlash = (u: string) => u.trim().replace(/\/+$/, "");

/** The single OpenAI-compatible base URL clients use: public URL if set, else the internal one, plus /v1. */
export function endpointPreview(url: string, publicUrl: string): string {
  const base = trimSlash(publicUrl) || trimSlash(url);
  return base ? `${base}/v1` : "";
}

/** Client-side mirror of the backend URL check. Empty is allowed (internal: config default; public: cleared). */
export function urlProblem(raw: string): string | null {
  const v = raw.trim();
  if (!v) return null;
  if (/\s/.test(v)) return "URLs cannot contain spaces";
  let u: URL;
  try { u = new URL(v); } catch { return "Enter a full URL starting with http:// or https://"; }
  if (u.protocol !== "http:" && u.protocol !== "https:") return "Enter a full URL starting with http:// or https://";
  if (!u.hostname) return "The URL needs a host name";
  if (u.username || u.password) return "Do not put credentials in the URL";
  if (u.search || u.hash || v.includes("?") || v.includes("#")) return "Remove the query string or fragment";
  return null;
}

export type KeyMode = "keep" | "replace" | "clear";
export interface ConnectionForm { enabled: boolean; url: string; public_url: string; keyMode: KeyMode; masterKey: string }
export interface ConnectionSaved { enabled: boolean; url: string; public_url: string; master_key_set: boolean; default_url?: string }

/**
 * The PUT body for the connection form (changed fields only). `keyDetaches` is true when the internal URL changes
 * while a master key is in effect and no new key is given: the backend binds each key to the URL it was entered for,
 * so the current key will stop being used and the user should enter the key for the new URL.
 */
export function connectionPatch(saved: ConnectionSaved, form: ConnectionForm): { body: Record<string, unknown>; keyDetaches: boolean } {
  const body: Record<string, unknown> = {};
  if (form.enabled !== saved.enabled) body.enabled = form.enabled;
  const url = trimSlash(form.url);
  // An empty internal URL means "the config default": no change if that is already the effective URL.
  const effective = url || (saved.default_url !== undefined ? trimSlash(saved.default_url) : "");
  const urlChanged = effective !== trimSlash(saved.url);
  if (urlChanged) body.url = url;
  const pub = trimSlash(form.public_url);
  if (pub !== trimSlash(saved.public_url)) body.public_url = pub;
  const newKey = form.keyMode === "replace" ? form.masterKey.trim() : "";
  if (newKey) body.master_key = newKey;
  else if (form.keyMode === "clear" && saved.master_key_set) body.master_key = "";
  return { body, keyDetaches: urlChanged && saved.master_key_set && !newKey };
}

export function aliasProblem(alias: string): string | null {
  const v = alias.trim();
  if (!v) return null;
  if (v.length > 64) return "Use at most 64 characters";
  if (!ALIAS_RE.test(v)) return "Use letters, digits and . _ : / - only (no spaces)";
  return null;
}

export function curlSnippet(endpoint: string, model: string, key?: string): string {
  return [
    `curl ${endpoint}/chat/completions \\`,
    `  -H "Authorization: Bearer ${key ?? "$LITELLM_API_KEY"}" \\`,
    `  -H "Content-Type: application/json" \\`,
    `  -d '{"model": "${model}", "messages": [{"role": "user", "content": "Hello"}]}'`,
  ].join("\n");
}

export function pythonSnippet(endpoint: string, model: string): string {
  return [
    "import os",
    "from openai import OpenAI",
    "",
    `client = OpenAI(base_url="${endpoint}", api_key=os.environ["LITELLM_API_KEY"])`,
    "resp = client.chat.completions.create(",
    `    model="${model}",`,
    `    messages=[{"role": "user", "content": "Hello"}],`,
    ")",
    "print(resp.choices[0].message.content)",
  ].join("\n");
}

export const isAllModels = (models: string[] | null | undefined) => !models?.length || models.includes(ALL_PROXY_MODELS);

export interface KeyForm {
  alias: string; allModels: boolean; models: string[]; maxBudget: string; budgetDuration: string;
  rpm: string; tpm: string; duration: string; blocked: boolean;
}

export const emptyKeyForm = (models: string[] = []): KeyForm => ({
  alias: "", allModels: models.length === 0, models, maxBudget: "", budgetDuration: "", rpm: "", tpm: "", duration: "", blocked: false,
});

const str = (v: number | string | null | undefined) => (v === null || v === undefined ? "" : String(v));

export function keyFormFromKey(k: LiteLLMKey): KeyForm {
  return {
    alias: k.key_alias ?? "", allModels: isAllModels(k.models), models: (k.models ?? []).filter((m) => m !== ALL_PROXY_MODELS),
    maxBudget: str(k.max_budget), budgetDuration: k.budget_duration ?? "", rpm: str(k.rpm_limit), tpm: str(k.tpm_limit),
    duration: "", blocked: !!k.blocked,
  };
}

export type KeyFormProblems = Partial<Record<keyof KeyForm, string>>;

export function keyFormProblems(f: KeyForm, mode: "create" | "edit", original?: LiteLLMKey | null): KeyFormProblems {
  const p: KeyFormProblems = {};
  const keepsNoAlias = mode === "edit" && !f.alias.trim() && !original?.key_alias;
  if (!keepsNoAlias && !KEY_ALIAS_RE.test(f.alias.trim())) p.alias = "Required: up to 128 letters, digits, spaces and . _ : @ / -";
  if (!f.allModels && f.models.length === 0) p.models = "Choose at least one model, or allow all models";
  const budget = f.maxBudget.trim();
  if (budget && !(Number.isFinite(Number(budget)) && Number(budget) >= 0)) p.maxBudget = "Enter an amount of 0 or more";
  for (const k of ["rpm", "tpm"] as const) {
    const v = f[k].trim();
    if (v && !/^\d+$/.test(v)) p[k] = "Enter a whole number of 0 or more";
  }
  const durHelp = "Use a number and unit, e.g. 30d, 12h, 1mo";
  if (f.budgetDuration.trim() && !DURATION_RE.test(f.budgetDuration.trim())) p.budgetDuration = durHelp;
  if (mode === "create" && f.duration.trim() && !DURATION_RE.test(f.duration.trim())) p.duration = durHelp;
  return p;
}

export function buildKeyCreateBody(f: KeyForm): Record<string, unknown> {
  const body: Record<string, unknown> = { key_alias: f.alias.trim(), models: f.allModels ? [] : f.models };
  if (f.maxBudget.trim()) body.max_budget = Number(f.maxBudget);
  if (f.budgetDuration.trim()) body.budget_duration = f.budgetDuration.trim();
  if (f.rpm.trim()) body.rpm_limit = Number(f.rpm);
  if (f.tpm.trim()) body.tpm_limit = Number(f.tpm);
  if (f.duration.trim()) body.duration = f.duration.trim();
  return body;
}

const LIMITS = [
  ["maxBudget", "max_budget", "max budget"], ["budgetDuration", "budget_duration", "budget period"],
  ["rpm", "rpm_limit", "RPM limit"], ["tpm", "tpm_limit", "TPM limit"],
] as const;

/** Limits the user emptied in the edit form. LiteLLM ignores cleared values, so they are kept, not removed. */
export function clearedLimits(k: LiteLLMKey, f: KeyForm): string[] {
  return LIMITS.filter(([field, apiField]) => !f[field].trim() && k[apiField] !== null && k[apiField] !== undefined).map(([, , label]) => label);
}

/** PATCH body with only the changed fields. Emptied limits are not sent (see clearedLimits). */
export function buildKeyPatchBody(k: LiteLLMKey, f: KeyForm): Record<string, unknown> {
  const body: Record<string, unknown> = {};
  const alias = f.alias.trim();
  if (alias && alias !== (k.key_alias ?? "")) body.key_alias = alias;
  const before = isAllModels(k.models) ? [] : [...k.models].sort();
  const after = f.allModels ? [] : [...f.models].sort();
  if (JSON.stringify(before) !== JSON.stringify(after)) body.models = f.allModels ? [] : f.models;
  for (const [field, apiField] of LIMITS) {
    const v = f[field].trim();
    if (!v) continue;
    const next = field === "budgetDuration" ? v : Number(v);
    if (next !== k[apiField]) body[apiField] = next;
  }
  if (f.blocked !== !!k.blocked) body.blocked = f.blocked;
  return body;
}

/** Model names offered in the key dialog: every DGXMM deployment's LiteLLM model name, plus any extra names. */
export function modelNameOptions(deps: Pick<Deployment, "model_name" | "alias" | "served_name">[], extras: string[] = []): string[] {
  const names = deps.map((d) => d.model_name ?? d.alias ?? d.served_name).concat(extras);
  return [...new Set(names.filter((n): n is string => !!n && n !== ALL_PROXY_MODELS))].sort();
}

/** Friendly title + explanation for LiteLLM integration errors (ApiError or {code, message}). */
export function litellmProblem(err: unknown): { title: string; description: string } {
  const e = (err ?? {}) as Partial<ApiProblem>;
  const message = e.message ?? String(err);
  switch (e.code) {
    case "litellm_no_db":
      return {
        title: "LiteLLM has no database",
        description: "Virtual keys need LiteLLM to run with a Postgres database: set DATABASE_URL (and STORE_MODEL_IN_DB=True so models added by DGXMM persist) on the LiteLLM proxy, then restart it.",
      };
    case "litellm_not_configured":
      return { title: "No master key set", description: e.message || "Add the LiteLLM master key in Connection so DGXMM can manage models and keys." };
    case "model_name_conflict":
      return { title: "Model name conflict", description: message };
    case "litellm_name_conflict":
      return { title: "Name also used outside DGXMM", description: e.host ? `${message} The other entry's API base host is ${e.host}.` : message };
    case "litellm_disabled":
      return { title: "Integration disabled", description: "Turn the integration on in Connection to manage LiteLLM from DGXMM." };
    case "litellm_unreachable":
      return { title: "Cannot reach LiteLLM", description: message };
    case "litellm_auth_failed":
      return { title: "LiteLLM rejected the master key", description: "Check the master key in Connection." };
    default:
      return { title: "LiteLLM request failed", description: message };
  }
}

export const fmtUsd = (v: number | null | undefined) => (v === null || v === undefined ? "—" : `$${v.toFixed(2)}`);

export function fmtSpend(k: Pick<LiteLLMKey, "spend" | "max_budget" | "budget_duration">): string {
  const spend = fmtUsd(k.spend ?? 0);
  if (k.max_budget === null || k.max_budget === undefined) return `${spend} (no budget)`;
  return `${spend} / ${fmtUsd(k.max_budget)}${k.budget_duration ? ` per ${k.budget_duration}` : ""}`;
}

export function fmtLimits(k: Pick<LiteLLMKey, "rpm_limit" | "tpm_limit">): string {
  const parts = [k.rpm_limit != null && `${k.rpm_limit.toLocaleString()} rpm`, k.tpm_limit != null && `${k.tpm_limit.toLocaleString()} tpm`].filter(Boolean);
  return parts.length ? parts.join(" · ") : "No limits";
}

export const fmtDate = (iso: string | null | undefined) => (iso ? new Date(iso).toLocaleDateString() : "—");

/** The deployment that already owns a model name, from a 409 `model_name_taken` (deploy / alias / start / wake). */
export function takenHolder(err: unknown): DeploymentRef | null {
  const d = (err as { detail?: any } | null)?.detail;
  return d && typeof d === "object" && d.code === "model_name_taken" && d.holder ? (d.holder as DeploymentRef) : null;
}

/** A 409 `model_name_taken`, held by another deployment (`takenHolder`) or by a LiteLLM entry DGXMM did not create
 * (static config.yaml entry or another tool's; no holder). */
export function isModelNameTaken(err: unknown): boolean {
  const d = (err as { detail?: any } | null)?.detail;
  return !!d && typeof d === "object" && d.code === "model_name_taken";
}

/** Why a model name may not be shared, for the kind of holder in a `model_name_taken` error. */
export function modelNameRule(err: unknown): string {
  return takenHolder(err) ? MODEL_NAME_RULE : EXTERNAL_NAME_RULE;
}

/** Error text for toasts: a model-name conflict always names the deployment holding the name. */
export function apiErrorText(err: unknown): string {
  const msg = (err as Error | null)?.message || "Request failed";
  const h = takenHolder(err);
  return h && !msg.includes(h.name) ? `${msg} (held by deployment ${h.name})` : msg;
}

/** Keys whose model list names `model` explicitly (unrestricted keys are not affected by a rename). */
export const keysScopedTo = (keys: LiteLLMKey[], model: string) => keys.filter((k) => !isAllModels(k.models) && k.models.includes(model));

export const MODEL_NAME_RULE = "Two deployments can't share one LiteLLM model name, otherwise LiteLLM would load-balance requests between them. Give this deployment its own model name.";
export const EXTERNAL_NAME_RULE = "LiteLLM load-balances between entries with the same model name, so requests would also go to that other backend and fail when it is down. DGXMM never changes entries it did not create.";

/** A stable alias that cannot collide: deployment names are unique. Sanitised to the alias pattern. */
export function suggestAlias(servedName: string | null | undefined, deploymentName: string): string {
  const raw = servedName ? `${servedName}-${deploymentName}` : deploymentName;
  return raw.replace(/[^A-Za-z0-9._:/-]+/g, "-").slice(0, 64) || "model";
}

/** The deployment's LiteLLM tab with a suggested model name prefilled. */
export const nameFixHref = (d: { id: number; name: string; served_name?: string | null }) =>
  `/deployments/${d.id}?tab=litellm&alias=${encodeURIComponent(suggestAlias(d.served_name, d.name))}`;

/** `keys_rewritten` from an alias PATCH may be a bool (all affected), a count or the list actually rewritten. */
export function rewrittenCount(v: unknown, affected: number): number {
  if (typeof v === "number") return v;
  if (Array.isArray(v)) return v.length;
  return v === true ? affected : 0;
}

/** One wording for "Sync now" / "Re-sync" results. */
export function syncMessage(r: LiteLLMSyncResult): string {
  if (r.status === "disabled") return "Sync skipped: the integration is disabled or has no master key.";
  return `Sync ${r.ok ? "finished" : "failed"}: ${r.registered} registered, ${r.removed} removed.`;
}
