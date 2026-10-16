export interface User { id: number; username: string; role: "admin" | "viewer"; must_change_pw: boolean }
export interface DeploymentUsage { deployment_id: number; name: string; state: string; reserved: number; actual: number }
export interface Budget {
  total: number; available: number; margin: number; os_reserve: number; usable: number; committed: number;
  managed_actual: number; external: number; free_for_new: number; deployments: DeploymentUsage[];
}
export interface ProcMem { pid: number; name: string; rss: number; gpu: number }
export interface Alert { level: "ok" | "warning" | "critical"; message?: string; since?: string; action?: string }
export interface HostInfo { budget: Budget; external_top: ProcMem[]; gpu: Record<string, number | null>; alert: Alert; disk: { total: number; free: number } | null }
export interface Artifact {
  ref: string; source: "hf" | "gguf"; revision: string | null; format: string; weights_bytes: number; size_bytes: number;
  complete: boolean; kind: "llm" | "draft" | "other"; architecture: string | null; quant: string | null; problems: string[];
  max_context: number | null; is_moe: boolean | null; num_layers: number | null; confident: boolean | null;
  has_mtp: boolean; engines: string[];
}
export interface Estimate {
  weights: number; draft_weights: number; kv_cache: number; state: number; activations: number; runtime: number;
  margin: number; confidence: string; notes: string[]; total: number;
}
export interface Fit { fits: boolean; required: number; free_for_new: number; headroom: number; estimate: Estimate }
export interface PlanAction { type: "sleep" | "stop" | "patch"; deployment_id?: number | null; patch?: Record<string, unknown> | null }
export interface PlanOption { id: string; title: string; detail: string; new_required: number; frees: number; fits: boolean; actions: PlanAction[]; patch: Record<string, unknown> | null; model_ref?: string | null }
export interface Preview {
  estimate: Estimate; fit: Fit; budget: Budget; served_name: string; config: Record<string, unknown>;
  gpu_memory_utilization?: number; kv_pool_tokens?: number | null; plans?: PlanOption[]; llmfit?: Record<string, any> | null;
}
export interface DeploymentEvent { ts: string; from: string; to: string; message: string; log_tail: string | null }
export interface Deployment {
  id: number; name: string; engine: string; model_ref: string; state: string; port: number | null; priority: number;
  alias: string | null; reserved_bytes: number; actual_bytes: number | null; estimate: Estimate | Record<string, never>;
  config: Record<string, any>; litellm_status: string; last_error: string | null; exit_code: number | null;
  container_id: string | null; served_name: string | null; endpoint: string | null; host_endpoint: string | null;
  sleep_capable: boolean; adopted_from: string | null; created_at: string; updated_at: string; events?: DeploymentEvent[];
  /** The name clients pass as `model` to LiteLLM: alias if set, else served_name. */
  model_name?: string | null;
}
export interface EngineInfo { name: string; supports_sleep: boolean; defaults: Record<string, any>; image: string | null }

export interface LiteLLMStatus { reachable: boolean; version: string | null; db: string | null; error: string | null; checked_at: string }
export interface LiteLLMSettings {
  enabled: boolean; url: string; public_url: string; endpoint: string;
  master_key_set: boolean; master_key_source: "settings" | "secret_file" | "none"; status: LiteLLMStatus;
  /** Set when a key exists but is not valid for the effective URL (keys are bound to the URL they were entered for). */
  master_key_hint: string | null;
}
export interface LiteLLMTestResult extends LiteLLMStatus { auth_ok: boolean | null; models: number | null }
export interface LiteLLMSyncResult { ok: boolean; status: string; registered: number; removed: number; error: string | null }
export interface LiteLLMModel {
  id: string; model_name: string; model: string; api_base: string | null; managed_by_dgxmm: boolean;
  deployment: { id: number; name: string; state: string } | null; db_model: boolean;
}
/** A LiteLLM virtual key as listed: never the full key. `token` is the 64-hex hash used as the key id. */
export interface LiteLLMKey {
  token: string; key_name: string | null; key_alias: string | null; models: string[]; spend: number | null;
  max_budget: number | null; budget_duration: string | null; rpm_limit: number | null; tpm_limit: number | null;
  expires: string | null; created_at: string | null; blocked: boolean | null; metadata: Record<string, unknown> | null;
  created_by_dgxmm: boolean;
}
/** Only the create response carries the full key. */
export interface LiteLLMCreatedKey extends LiteLLMKey { key: string }
export interface ApiProblem { code: string; message: string }
export interface DeploymentRef { id: number; name: string; state: string }
/** PATCH /deployments/{id} response: the deployment plus what happened to keys scoped to the old model name. */
export interface AliasPatchResult extends Deployment {
  affected_keys: { token: string; key_alias: string | null; models: string[] }[]; keys_rewritten: boolean | number | unknown[]; keys_error: ApiProblem | null;
}
export interface DeploymentLiteLLM {
  deployment_id: number; model_name: string; endpoint: string; registered: boolean | null; status: string; enabled: boolean;
  error: ApiProblem | null; keys: LiteLLMKey[]; keys_visible: boolean; keys_error: ApiProblem | null;
}
