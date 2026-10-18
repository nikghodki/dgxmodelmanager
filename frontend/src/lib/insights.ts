import type { Insight, InsightAction, InsightSeverity, InsightStatus } from "@/types";

/** Insights feed helpers (GET /insights; dgxmm/decider/insights.py). */
export const INSIGHT_STATUSES: InsightStatus[] = ["open", "snoozed", "dismissed", "applied", "expired"];
export const LIVE_STATUSES = "open,snoozed";

const SEVERITY_LABEL: Record<InsightSeverity, string> = { critical: "Critical", warning: "Warning", info: "Info" };
export const severityLabel = (s: string) => SEVERITY_LABEL[s as InsightSeverity] ?? s;
const SEVERITY_TONE: Record<InsightSeverity, string> = {
  critical: "bg-status-critical-bg text-status-critical", warning: "bg-status-warning-bg text-status-warning", info: "bg-status-neutral-bg text-status-neutral",
};
export const severityTone = (s: string) => SEVERITY_TONE[s as InsightSeverity] ?? SEVERITY_TONE.info;

const STATUS_LABEL: Record<InsightStatus, string> = { open: "Open", snoozed: "Snoozed", dismissed: "Dismissed", applied: "Applied", expired: "Expired" };
export const statusLabel = (s: string) => STATUS_LABEL[s as InsightStatus] ?? s;

/** Snooze choices (hours; the server allows 1..168). */
export const SNOOZE_HOURS: { hours: number; label: string }[] = [
  { hours: 1, label: "1 hour" }, { hours: 4, label: "4 hours" }, { hours: 24, label: "1 day" }, { hours: 168, label: "1 week" },
];

/** Patch keys an insight may never carry (server: dgxmm/decider/insights.py FORBIDDEN_PATCH_KEYS): a model switch
 *  and trust_remote_code have their own flows. */
export const FORBIDDEN_PATCH_KEYS = new Set(["model_ref", "trust_remote_code"]);
const isObj = (v: unknown): v is Record<string, unknown> => typeof v === "object" && v !== null && !Array.isArray(v);
const isId = (v: unknown) => typeof v === "number" && Number.isInteger(v) && v >= 1;

/** The action as the UI may use it, or null (missing, unknown type or malformed: no button). Mirrors the server's
 *  validation so a stale or tampered row never opens a dialog with bad input. */
export function parseAction(a: unknown): InsightAction | null {
  if (!isObj(a) || typeof a.type !== "string") return null;
  if (a.label !== undefined && (typeof a.label !== "string" || !a.label)) return null;
  switch (a.type) {
    case "reconfigure": return isId(a.deployment_id) && isObj(a.patch) && Object.keys(a.patch).length > 0
      && !Object.keys(a.patch).some((k) => FORBIDDEN_PATCH_KEYS.has(k)) ? (a as InsightAction) : null;
    case "schedule": return isId(a.deployment_id) && isObj(a.proposal) ? (a as InsightAction) : null;
    case "threshold": return typeof a.feature === "string" && typeof a.value === "number" && a.value >= 0 && a.value <= 1 ? (a as InsightAction) : null;
    case "copy": return typeof a.text === "string" && a.text.length > 0 ? (a as InsightAction) : null;
    case "open": return typeof a.path === "string" && /^\/(?!\/)/.test(a.path) ? (a as InsightAction) : null;
    default: return null;
  }
}

const DEFAULT_LABEL: Record<InsightAction["type"], string> = {
  reconfigure: "Review change…", schedule: "Review schedule…", threshold: "Set threshold…", copy: "Show snippet…", open: "Open",
};
export const actionLabel = (a: InsightAction) => a.label ?? DEFAULT_LABEL[a.type];

/** Query keys: every insights query starts with ["insights"] (the SSE `insights` topic invalidates them all). */
export interface InsightFilters { status?: string; subject?: string; kind?: string }
export const insightsKey = (f: InsightFilters = {}) => ["insights", f.status ?? "", f.subject ?? "", f.kind ?? ""];
export function insightsPath(f: InsightFilters = {}): string {
  const q = new URLSearchParams();
  if (f.status) q.set("status", f.status);
  if (f.subject) q.set("subject", f.subject);
  if (f.kind) q.set("kind", f.kind);
  const s = q.toString();
  return `/insights${s ? `?${s}` : ""}`;
}

export const deploymentSubject = (id: number) => `deployment:${id}`;
/** Open insights count for the top-bar badge. */
export const openCount = (rows: Insight[] | undefined) => (rows ?? []).filter((r) => r.status === "open").length;
