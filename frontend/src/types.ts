export interface User { id: number; username: string; role: "admin" | "viewer"; must_change_pw: boolean }
export interface DeploymentUsage { deployment_id: number; name: string; state: string; reserved: number; actual: number }
export interface Budget {
  total: number; available: number; margin: number; os_reserve: number; usable: number; committed: number;
  managed_actual: number; external: number; free_for_new: number; deployments: DeploymentUsage[];
}
export interface ContainerRef { id: string; name: string | null; image: string | null }
export interface ManagedRef { deployment_id: number; name: string | null }
/** Largest external memory users (GET /host). `command` is redacted server-side; older servers omit the extras. */
export interface ProcMem { pid: number; name: string; rss: number; gpu: number; command?: string; user?: string; user_scope?: UserScope; container?: ContainerRef | null; dgxmm_label?: ManagedRef | null }
/** "container": uid as seen inside the container (root / "uid N"), not a host account. */
export type UserScope = "host" | "container";
export interface ProcessInfo {
  pid: number; ppid: number; uid: number | null; user: string; user_scope: UserScope; name: string; exe: string;
  command: string; command_truncated: boolean; container: ContainerRef | null; managed: ManagedRef | null;
  /** DGXMM-labelled container that is not an active deployment: its memory counts as external. */
  dgxmm_label: ManagedRef | null; rss: number; gpu: number;
}
export interface ProcessGroupSummary extends ContainerRef { rss: number; gpu: number; process_count: number; managed: ManagedRef | null; dgxmm_label: ManagedRef | null }
export interface ProcessesResponse {
  scope: "all" | "external" | "managed"; total: number; processes: ProcessInfo[]; containers: ProcessGroupSummary[];
  host: { rss: number; gpu: number; process_count: number }; duration_ms: number;
}
export interface Alert { level: "ok" | "warning" | "critical"; message?: string; since?: string; action?: string }
export interface HostInfo { budget: Budget; external_top: ProcMem[]; gpu: Record<string, number | null>; alert: Alert; disk: { total: number; free: number } | null }
export interface Artifact {
  ref: string; source: "hf" | "gguf"; revision: string | null; format: string; weights_bytes: number; size_bytes: number;
  complete: boolean; kind: "llm" | "draft" | "embedding" | "reranker" | "vision-llm" | "speech" | "decision" | "other"; architecture: string | null; quant: string | null; problems: string[];
  max_context: number | null; is_moe: boolean | null; num_layers: number | null; confident: boolean | null;
  has_mtp: boolean; engines: string[];
  /** Who classified `kind`: rules, or the decision model when no rule recognised the model. */
  kind_source?: "rules" | "model"; kind_confidence?: number | null;
}
export interface DeploymentRef { id: number; name: string; state: string }
/** GET /inventory/delete-preview — what deleting a model would remove (host paths). */
export interface DeletePreview {
  ref: string; source: "hf" | "gguf"; name: string; paths: string[]; bytes: number;
  deployments_blocking: DeploymentRef[]; deployments_affected: DeploymentRef[]; download_active: boolean; blocked: boolean;
  /** Partial files were written in the last 10 minutes: a clean-up (not a full delete) is refused. */
  recently_active: boolean;
  notes: string[];
}
export interface DeleteResult { ref: string; freed_bytes: number; paths: string[]; method: string; warnings: DeploymentRef[]; notes: string[] }
export interface CleanupItem extends Omit<DeletePreview, "blocked"> {
  problems: string[]; eligible: boolean; skip_reason: string | null; message?: string;
}
export interface CleanupPreview { items: CleanupItem[]; count: number; bytes: number }
export interface CleanupResult {
  deleted: DeleteResult[]; skipped: { ref: string; code: string; message: string }[];
  failed: { ref: string; code: string; message: string }[]; freed_bytes: number;
}
export interface Estimate {
  weights: number; draft_weights: number; kv_cache: number; state: number; activations: number; runtime: number;
  margin: number; confidence: string; notes: string[]; total: number;
}
export interface Fit { fits: boolean; required: number; free_for_new: number; headroom: number; estimate: Estimate }
export interface PlanAction { type: "sleep" | "stop" | "patch"; deployment_id?: number | null; patch?: Record<string, unknown> | null }
export interface PlanOption { id: string; title: string; detail: string; new_required: number; frees: number; fits: boolean; actions: PlanAction[]; patch: Record<string, unknown> | null; model_ref?: string | null;
  /** The decision model's (or the rules') pick among the safe plans. */
  recommended?: boolean; recommendation?: PlanRecommendation | null;
  /** Set when the safety filter excludes this plan from recommendation (still selectable as before). */
  unsafe_reason?: string | null;
}
export interface Preview {
  estimate: Estimate; fit: Fit; budget: Budget; served_name: string; config: Record<string, unknown>;
  gpu_memory_utilization?: number; kv_pool_tokens?: number | null; plans?: PlanOption[]; llmfit?: Record<string, any> | null;
  /** When it does not fit: hints that are not plans (e.g. the decision model's memory, freed only by disabling it). */
  notes?: string[];
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
  /** Why this ready deployment is not (correctly) registered in LiteLLM, as of the last sync. */
  litellm_error?: ApiProblem | null;
  /** A system deployment (the decision model): never a plan candidate, not registered in LiteLLM; the watchdog stops it
   * before any user deployment under memory pressure. */
  system?: boolean;
  /** Set while it waits for a start-up slot (state "queued", or a wake-up waiting): at most N engines start at once. */
  startup_queue?: StartupQueue | null;
}
export interface StartupQueue {
  position: number | null; waiting_for: string[]; reason: string; kind: "launch" | "wake"; since: number; limit: number;
}
export interface EngineInfo { name: string; supports_sleep: boolean; defaults: Record<string, any>; image: string | null }

export interface LiteLLMStatus { reachable: boolean; version: string | null; db: string | null; error: string | null; checked_at: string }
export interface LiteLLMSettings {
  enabled: boolean; url: string; public_url: string; endpoint: string;
  /** null when read with ?probe=false before any probe (the page then reads GET /integrations/litellm/status). */
  master_key_set: boolean; master_key_source: "settings" | "secret_file" | "none"; status: LiteLLMStatus | null;
  /** Set when a key exists but is not valid for the effective URL (keys are bound to the URL they were entered for). */
  master_key_hint: string | null;
  /** The config default internal URL (an empty URL in the form resets to it). */
  default_url?: string;
}
/** PUT /integrations/litellm: the saved settings plus what was removed from the LiteLLM being left (disable/URL change). */
export interface LiteLLMSaveResult extends LiteLLMSettings { deregistered: number; deregister_error: string | null; warning: string | null }
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
  affected_keys: { token: string; key_alias: string | null; models: string[] }[]; keys_rewritten: number | boolean | unknown[]; keys_error: ApiProblem | null;
}
export interface DeploymentLiteLLM {
  deployment_id: number; model_name: string; endpoint: string; registered: boolean | null; status: string; enabled: boolean;
  error: ApiProblem | null; keys: LiteLLMKey[]; keys_visible: boolean; keys_error: ApiProblem | null;
}

// ---- Decision model (decider) ----
/** Who made a decision: deterministic rules, the decision model, or nobody (abstained / unavailable). */
export type DecisionSource = "rules" | "model" | "none";
export interface DeciderFeatureSettings { enabled: boolean; threshold: number }
export interface DeciderImageStatus { present: boolean; building: boolean; error: string | null }
/** GET /decider/status (viewer). */
/** POST /decider/rotate-key (the key itself is never returned). */
export interface DeciderKeyRotation {
  rotated: boolean;
  restarted: boolean;
  deployment_id: number | null;
  state: string | null;
  error: string | null;
}

export interface DeciderStatus {
  enabled: boolean; healthy: boolean; deployment_id: number | null; state: string | null; model: string; revision: string | null;
  p50_ms: number | null; decisions_24h: number; accepted_rate: number | null;
  features: Record<string, DeciderFeatureSettings>; image: DeciderImageStatus;
  /** The running decider container predates the current image / hardening (rebuild + restart). */
  upgrade?: { needed: boolean; reasons: string[] };
  /** The model the next enable runs (`model` is the one answering now). */
  configured_model?: string; restart_required?: boolean; models?: DeciderModelInfo[];
  breaker?: { open: boolean; failures: number; retry_in_s: number } | null;
  /** Client-side back-pressure: requests at the decider, waiting, and answered by rules because it was busy. */
  load?: { max_in_flight: number; in_flight: number; queued: number; shed: number; service_ms: number };
  supervisor?: DeciderSupervisorStatus | null;
}
export interface DeciderModelInfo {
  repo: string; revision: string; served_name: string; params: string; reservation_bytes: number; default: boolean; downloaded: boolean;
}
export interface DeciderSupervisorStatus {
  last_action: string | null; last_reason: string | null; probe: { ok: boolean; latency_ms: number | null; at: number } | null;
  failed_probes: number; restarts_last_hour: number; max_restarts_per_hour: number; backoff_s: number; gave_up: boolean;
  /** Set when the supervisor stopped a hung decider and has not managed to start it again yet. */
  pending_restart?: string | null;
  next_attempt_in_s: number | null;
}
/** PUT /decider/settings body. */
export interface DeciderSettingsPatch { features?: Record<string, Partial<DeciderFeatureSettings>>; model?: string }
/** GET /decider/decisions rows (no raw log text). */
export interface DecisionRow {
  id: number; created_at: string; feature: string; subject: string; source: DecisionSource; choice: string | null;
  confidence: number | null; latency_ms: number | null; feedback: "accepted" | "rejected" | null;
}
/** SSE `decider` topic while the image builds. */
export interface DeciderBuildProgress { phase: string; message: string }
/** GET /deployments/{id}/diagnosis. */
export interface Diagnosis {
  cause: string | null; label: string; source: DecisionSource; confidence: number | null; summary: string; fix_hint: string;
  patch: Record<string, unknown> | null; decision_id: number | null; created_at: string; log_excerpt: string | null;
  /** The admin's 👍/👎 on its decision, so the card shows it after a reload. */
  feedback?: "accepted" | "rejected" | null;
}
export interface DiagnosisResponse { diagnosis: Diagnosis | null }
/** POST /decider/suggest-config: one suggestion per parser field. `value: null` means "no parser". */
export interface Suggestion {
  value: string | null; source: DecisionSource; confidence: number | null; auto_apply: boolean; alternatives: { value: string | null; p: number }[];
  decision_id?: number | null; reason?: string | null;
}
export interface ConfigSuggestions { tool_call_parser: Suggestion | null; reasoning_parser: Suggestion | null }
export interface PlanRecommendation { source: DecisionSource; confidence: number | null; reason: string; decision_id?: number | null }
/** GET /deployments/{id}/workload. */
export interface Workload { label: string | null; source: DecisionSource; confidence: number | null; summary?: Record<string, unknown> | string | null; reason?: string | null; decision_id?: number | null }
