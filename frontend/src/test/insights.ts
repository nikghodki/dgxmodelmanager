import type { Insight } from "@/types";

/** An insight row as GET /insights returns it (test data). */
export const insight = (over: Partial<Insight> = {}): Insight => ({
  id: 1, kind: "health", feature: "health", subject: "deployment:4", severity: "warning", title: "KV cache pressure",
  body: "KV cache usage stayed above 95% with preemptions.", confidence: null, source: "rules", decision_id: null,
  action: { type: "reconfigure", deployment_id: 4, patch: { kv_cache_dtype: "fp8" } }, status: "open", snooze_until: null,
  dedupe_key: "health:kv:4", created_at: "2026-10-17T10:00:00Z", updated_at: "2026-10-17T10:00:00Z", expires_at: null, ...over,
});
