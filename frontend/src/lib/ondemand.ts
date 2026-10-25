import type { Deployment, DeploymentMode } from "@/types";

/** Per-deployment idle timeout bounds in minutes (the API takes 60–86400 seconds). */
export const IDLE_MIN_MINUTES = 1;
export const IDLE_MAX_MINUTES = 1440;
export const DEFAULT_IDLE_MINUTES = 5;
export const IDLE_RANGE_TEXT = `Enter ${IDLE_MIN_MINUTES} to ${IDLE_MAX_MINUTES} minutes.`;

export const MODE_LABELS: Record<DeploymentMode, string> = { on_demand: "On demand", always_on: "Always on" };

export const fmtMinutes = (m: number) => String(Math.round(m * 100) / 100);

/** "" = no value (the default); otherwise the minutes when they are within the bounds, else NaN. */
export function parseIdleMinutes(text: string): number | null {
  if (!text.trim()) return null;
  const m = Number(text);
  return Number.isFinite(m) && m >= IDLE_MIN_MINUTES && m <= IDLE_MAX_MINUTES ? m : NaN;
}

export const minutesToSeconds = (m: number) => Math.round(m * 60);

export const deploymentMode = (d: Pick<Deployment, "mode" | "system">): DeploymentMode =>
  d.system || d.mode !== "on_demand" ? "always_on" : "on_demand";

/** "default (N min)", or "default" while the global default is unknown. */
export const defaultIdleText = (defaultMinutes: number | null | undefined) =>
  defaultMinutes ? `default (${fmtMinutes(defaultMinutes)} min)` : "default";

/** Read-only description of a deployment's mode; `defaultMinutes`: the global idle default. */
export function modeSummary(d: Pick<Deployment, "mode" | "system" | "idle_timeout_s">, defaultMinutes?: number | null): string {
  if (deploymentMode(d) === "always_on") return MODE_LABELS.always_on;
  return `On demand · sleeps after ${d.idle_timeout_s ? `${fmtMinutes(d.idle_timeout_s / 60)} min` : defaultIdleText(defaultMinutes)} idle`;
}

/**
 * The badge an on-demand deployment shows in place of its state: a StatusBadge state (for its colour and icon)
 * and the text. null when the plain state badge applies (always-on, a failed or removed deployment, which LiteLLM no
 * longer lists and requests do not wake, or a phase that says nothing more).
 */
export function ondemandBadge(d: Pick<Deployment, "mode" | "system" | "state" | "ondemand">): { state: string; label: string } | null {
  const o = d.ondemand;
  if (deploymentMode(d) !== "on_demand" || !o || d.state === "failed" || o.phase === "failed") return null;
  switch (o.phase) {
    case "queued": return { state: "queued", label: `${o.queued} queued` };
    case "starting_for_request": return { state: "starting", label: "Starting for a request" };
    case "sleeping": return { state: "sleeping", label: "Sleeping (on demand)" };
    case "serving": return { state: "ready", label: "Ready" };
    default: return d.state === "ready" ? { state: "ready", label: "Ready" } : null;
  }
}
