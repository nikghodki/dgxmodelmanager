import type { CompatResult, CompatVerdict, ConfigAdvice, ConfigOption, DeployGoal, RemoteCodeCheck } from "@/types";

/** Deploy-time advice (decision-model features: config, compat, draft, engine, remote_code). Every text shown to users
 * comes from these templates, never from the model. */
/** A Hugging Face repo id (org/name), as the server's REPO_RE. */
export const REPO_ID = /^[A-Za-z0-9][A-Za-z0-9._-]*\/[A-Za-z0-9][A-Za-z0-9._-]*$/;

export const GOALS: { value: DeployGoal; label: string; help: string }[] = [
  { value: "chat", label: "Chat", help: "A few users at a time, multi-turn conversations" },
  { value: "agent_tool_use", label: "Agents and tool use", help: "Long system prompts with tool schemas, repeated prefixes" },
  { value: "batch_throughput", label: "Batch throughput", help: "As many parallel requests as possible" },
  { value: "long_context", label: "Long context", help: "Long documents in one request" },
];
export const goalLabel = (g: string) => GOALS.find((x) => x.value === g)?.label ?? g;

/** Default goal: agents when the model has a tool-call parser configured, else chat. */
export function defaultGoal(config: Record<string, any>): DeployGoal {
  return config.tool_call_parser ? "agent_tool_use" : "chat";
}

/** Fill the advice's settings into the config, but only fields the user has not touched: typed values are never
 * overwritten. Returns the new config and the keys it changed. */
export function fillUntouched(config: Record<string, any>, patch: Record<string, any>, touched: Iterable<string>): { config: Record<string, any>; changed: string[] } {
  const t = new Set(touched);
  const next = { ...config };
  const changed: string[] = [];
  for (const [k, v] of Object.entries(patch)) {
    if (t.has(k) || next[k] === v) continue;
    next[k] = v;
    changed.push(k);
  }
  return { config: next, changed };
}

/** Keys whose value differs between two configs (user edits, for "touched" tracking). */
export function changedKeys(prev: Record<string, any>, next: Record<string, any>): string[] {
  const keys = new Set([...Object.keys(prev), ...Object.keys(next)]);
  return [...keys].filter((k) => JSON.stringify(prev[k] ?? null) !== JSON.stringify(next[k] ?? null));
}

/** The options to show: the recommended one alone when there is one, else the top three (server order). */
export function shownOptions(a: ConfigAdvice | null | undefined): { recommended: ConfigOption | null; top: ConfigOption[] } {
  if (!a || !Array.isArray(a.options) || !a.options.length) return { recommended: null, top: [] };
  const byKey = new Map(a.options.map((o) => [o.key, o]));
  const rec = a.recommended ? byKey.get(a.recommended) ?? null : null;
  const keys = (Array.isArray(a.top) && a.top.length ? a.top : a.options.map((o) => o.key)).slice(0, 3);
  return { recommended: rec, top: keys.map((k) => byKey.get(k)).filter((o): o is ConfigOption => !!o) };
}

const VERDICT: Record<string, { label: string; tone: "good" | "warning" | "critical" | "neutral" }> = {
  runs: { label: "runs", tone: "good" }, runs_with_flags: { label: "runs with extra settings", tone: "warning" },
  unsupported: { label: "not supported", tone: "critical" }, unknown: { label: "unknown", tone: "neutral" },
};
export const verdictLabel = (v: string) => VERDICT[v]?.label ?? "unknown";
export const verdictTone = (v: string) => VERDICT[v]?.tone ?? "neutral";

export const COMPAT_REASONS: Record<string, string> = {
  supported_architecture: "architecture supported by the pinned vLLM image",
  unsupported_architecture: "architecture not supported",
  unknown_architecture: "architecture not in the pinned vLLM image's list",
  registry_unavailable: "vLLM architecture list still loading",
  trust_remote_code_required: "needs trust_remote_code (custom code)",
  auth_gated: "gated: needs a Hugging Face token with access",
  tokenizer_or_template: "no chat template: chat completions need one",
  no_config: "no config.json",
  no_safetensors: "only GGUF weights (vLLM needs safetensors)",
  vision: "takes images",
  audio: "takes audio",
  gguf_available: "GGUF files available",
  no_gguf: "no GGUF files in this repo",
};
export const compatReason = (r: string) => COMPAT_REASONS[r] ?? r;
export const FLAG_LABELS: Record<string, string> = { trust_remote_code: "trust_remote_code", hf_token: "HF token", chat_template: "chat template" };

/** "vLLM: runs with extra settings (needs trust_remote_code …)" */
export function compatText(engine: "vLLM" | "llama.cpp", v: CompatVerdict | null | undefined): string {
  if (!v) return `${engine}: unknown`;
  const reasons = (v.reasons ?? []).filter((r) => r !== "supported_architecture").map(compatReason);
  return `${engine}: ${verdictLabel(v.verdict)}${reasons.length ? ` (${reasons.join("; ")})` : ""}`;
}

export const isCompat = (x: unknown): x is CompatResult =>
  !!x && typeof x === "object" && "vllm" in (x as any) && "llamacpp" in (x as any) && typeof (x as any).vllm?.verdict === "string";

export const ENGINE_REASONS: Record<string, string> = {
  gguf_only: "only GGUF weights are on disk", safetensors_only: "only safetensors weights are on disk",
  vllm_unsupported: "the pinned vLLM image does not support this architecture; a GGUF is on disk", model: "chosen for the goal",
  only_vllm_fits: "only the vLLM weights fit the free memory", only_llamacpp_fits: "only the GGUF fits the free memory",
  neither_fits: "neither fits the free memory now; the GGUF needs the least freed",
};
export const engineReason = (r: string | null | undefined) => (r ? ENGINE_REASONS[r] ?? "" : "");

export const DRAFT_METHODS = ["draft_model", "eagle", "eagle3", "dflash", "mtp", "ngram"] as const;
export const DRAFT_REASONS: Record<string, string> = {
  same_tokenizer: "identical tokenizer", same_family_vocab: "same family and vocabulary", matching_head: "draft head built for this model",
};

export const FINDING_LABELS: Record<string, string> = {
  network_import: "imports a network module", process_import: "imports a process module", native_import: "loads native code",
  process_call: "runs a process", dynamic_exec: "calls eval / exec / compile", encoded_exec: "executes decoded data",
  file_write_outside: "writes outside the model directory", sensitive_path: "names credentials or shell start-up files",
  credential_env: "reads a credential from the environment", environment_dump: "copies the whole environment",
  long_encoded_literal: "embeds a long encoded string", unparseable: "could not be parsed", oversized_file: "too large to scan",
  unanalysable: "could not be analysed", dynamic_import: "imports a module by a computed name or a file path", builtins_access: "reaches the builtins",
  dynamic_attribute: "looks up a risky function by name", obfuscated_name: "assembles a risky name from pieces",
  unsafe_deserialize: "unpickles / deserialises data", external_code: "loads code from another repository",
  missing_code: "references code that is not in the snapshot", code_object: "builds or patches a code object",
  object_walk: "walks the object graph (__subclasses__ / __globals__)", native_load: "loads native code at run time",
  remote_fetch: "downloads files or code from the Hub at run time",
};
export const findingLabel = (k: string) => FINDING_LABELS[k] ?? k;

/** The remote-code panel's extra warning: rule findings, or the model said unusual. There is no "routine" verdict: a
 * clean scan is "no_findings" and the generic trust_remote_code warning is shown regardless. */
export function remoteCodeWarns(c: RemoteCodeCheck | null | undefined): boolean {
  return !!c && (c.findings?.length > 0 || c.verdict === "unusual");
}
