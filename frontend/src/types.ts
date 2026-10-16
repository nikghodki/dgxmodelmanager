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
export interface DeploymentRef { id: number; name: string; state: string }
/** GET /inventory/delete-preview — what deleting a model would remove (host paths). */
export interface DeletePreview {
  ref: string; source: "hf" | "gguf"; name: string; paths: string[]; bytes: number;
  deployments_blocking: DeploymentRef[]; deployments_affected: DeploymentRef[]; download_active: boolean; blocked: boolean;
}
export interface DeleteResult { ref: string; freed_bytes: number; paths: string[]; method: string; warnings: DeploymentRef[] }
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
}
export interface EngineInfo { name: string; supports_sleep: boolean; defaults: Record<string, any>; image: string | null }
