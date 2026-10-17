/**
 * Decision-model labels used only by lazily loaded pages (deployment detail, Decision model page): kept out of
 * `decider.ts`, which the entry chunk imports.
 */
/** The decision model's section of the usage guide (tests/test_docs_links.py checks that the heading exists). */
/** Measured first start-up (kernel compilation + warm-up) of each decision model, for the "starting" toast. */
const STARTUP: Record<string, string> = { "vllm-sr/d3-nano": "about 90 seconds" };
export const deciderStartupHint = (model: string | null | undefined) => `usually ready in ${STARTUP[model ?? ""] ?? "2–4 minutes"}`;

export const DECIDER_DOCS_URL = "https://github.com/nikghodki/dgxmodelmanager/blob/main/docs/usage.md#decision-model";

const humanize = (key: string) => key.replace(/_/g, " ");

/**
 * Failure-diagnosis taxonomy — keep in sync with the spec ("Feature 1 — Failure diagnosis") and the backend
 * taxonomy. The server also sends a `label`; it wins when present.
 */
export const DIAGNOSIS_CAUSES: Record<string, string> = {
  gpu_oom: "Out of GPU memory",
  kv_cache_too_small: "KV cache too small for the context",
  context_too_long: "Context longer than the model supports",
  unsupported_architecture: "Unsupported model architecture",
  trust_remote_code_required: "Needs trust_remote_code",
  auth_gated: "Gated model: Hugging Face token needed",
  bad_parser_flag: "Invalid tool-call or reasoning parser",
  tokenizer_or_template: "Tokenizer or chat template problem",
  port_in_use: "Port already in use",
  offline_missing_files: "Model files missing (offline)",
  driver_mismatch: "GPU driver / CUDA mismatch",
  disk_full: "Disk full",
  integrity_check_failed: "Decision model files fail the integrity check",
  api_key_missing: "Decision model key not delivered",
  other: "Other",
};
export const causeLabel = (cause: string | null, serverLabel?: string | null) =>
  serverLabel || (cause ? DIAGNOSIS_CAUSES[cause] || humanize(cause) : "Couldn't diagnose confidently");

export const WORKLOAD_LABELS: Record<string, string> = {
  single_user_chat: "Single-user chat",
  agent_tool_use: "Agent / tool use",
  batch_throughput: "Batch throughput",
  long_context_rag: "Long-context RAG",
  idle: "Idle",
  mixed: "Mixed",
};
export const workloadLabel = (w: string | null) => (w ? WORKLOAD_LABELS[w] ?? humanize(w) : "Unknown");

/** Per-feature settings keys (GET /decider/status `features`), in display order. */
export const DECIDER_FEATURES: { key: string; label: string; description: string }[] = [
  { key: "diagnosis", label: "Failure diagnosis", description: "Explain why a deployment failed and suggest a fix." },
  { key: "parser", label: "Parser suggestions", description: "Suggest tool-call and reasoning parsers in the deploy wizard." },
  { key: "plan", label: "Plan recommendation", description: "Mark the best safe plan when a model does not fit." },
  { key: "kind", label: "Model kind", description: "Classify models no rule recognises (embedding, reranker, …)." },
  { key: "workload", label: "Workload-aware advice", description: "Label each deployment's workload and rank its advice." },
];
const FEATURE_LABEL: Record<string, string> = {
  ...Object.fromEntries(DECIDER_FEATURES.map((f) => [f.key, f.label])),
  parser_tool: "Tool-call parser", parser_reasoning: "Reasoning parser",
};
export const featureLabel = (f: string) => FEATURE_LABEL[f] ?? humanize(f);

/** Split server-ordered advice for a divider: items the server marks `workload_fit: "boost"` (favoured for the
 *  deployment's workload) go first. Only when a workload is known and both groups are non-empty. */
export function splitAdvice<T extends { workload_fit?: string | null }>(advice: T[], hasWorkload: boolean): { top: T[]; rest: T[] } {
  if (!hasWorkload) return { top: [], rest: advice };
  const top = advice.filter((a) => a.workload_fit === "boost");
  const rest = advice.filter((a) => a.workload_fit !== "boost");
  return top.length && rest.length ? { top, rest } : { top: [], rest: advice };
}

/** The workload is computed in the background after the first request: poll every 30 s until a label is known,
 *  then every 5 min. */
export function workloadRefetchMs(w: { label?: string | null; source?: string } | null | undefined): number {
  return w && w.label && w.source !== "none" ? 300000 : 30000;
}
