// Small pure helpers used by pages; unit-tested in logic.test.ts.

import type { PlanAction, Preview } from "./types";

/** I5: only stopped/failed deployments can be removed (DELETE /deployments/{id}). */
export const canRemove = (state: string) => state === "stopped" || state === "failed";

const isGgufRef = (ref: string) => /\.gguf$/i.test(ref) || ref.startsWith("/");

/** M8: advice can be applied as a reconfigure only if it stays on the same engine.
 *  A GGUF model_ref (llama.cpp) on a vLLM deployment is informational. */
export function adviceApplicable(a: { patch?: Record<string, any> | null; model_ref?: string | null }, d: { engine: string }): boolean {
  if (a.model_ref) return d.engine === "llamacpp" ? isGgufRef(a.model_ref) : !isGgufRef(a.model_ref);
  return !!a.patch && Object.keys(a.patch).length > 0;
}

/** M9: refs may be absolute GGUF paths; encode so "/" is not collapsed by the proxy. */
export const favouritePath = (ref: string) => `/planner/favourites/${encodeURIComponent(ref)}`;

/** M12: NaN/Infinity break chart scales; treat them as gaps. */
export function finiteSeries<T extends Record<string, any>>(data: T[], keys: string[]): T[] {
  return data.map((row) => {
    const out: Record<string, any> = { ...row };
    for (const k of keys) {
      const v = out[k];
      if (typeof v === "number" && !Number.isFinite(v)) out[k] = null;
    }
    return out as T;
  });
}

/** One-line summary of why a deployment waits for a start-up slot (the full reason goes in a tooltip). */
export function queueShort(q: { waiting_for: string[]; position: number | null }): string {
  const who = q.waiting_for.length ? `Waiting for ${q.waiting_for.join(", ")}` : "Waiting for a start-up slot";
  return q.position && q.position > 1 ? `${who} (#${q.position} in queue)` : who;
}

export type DeploymentAction = "open" | "manage" | "sleep" | "wake" | "switch" | "reconfigure" | "stop" | "start" | "remove";

/**
 * Which row/header actions a user may take on a deployment in its current state. A system deployment (the decision
 * model) is managed on the Decision model page ("manage"): it is never switched or reconfigured, it starts only by
 * enabling the decision model there, and removing it also turns the decision model off (the backend enforces all three).
 */
export function deploymentActions(d: { state: string; sleep_capable: boolean; system?: boolean }, isAdmin: boolean): DeploymentAction[] {
  const out: DeploymentAction[] = ["open"];
  if (!isAdmin) return out;
  if (d.system) {
    out.push("manage");
    if (["ready", "sleeping", "queued", "starting", "loading", "unknown"].includes(d.state)) out.push("stop");
    if (canRemove(d.state)) out.push("remove");
    return out;
  }
  if (d.state === "ready" && d.sleep_capable) out.push("sleep");
  if (d.state === "sleeping") out.push("wake");
  if (d.state === "ready" || d.state === "sleeping") out.push("switch");
  if (["ready", "sleeping", "stopped", "failed"].includes(d.state)) out.push("reconfigure");
  if (["ready", "sleeping", "queued", "starting", "loading", "unknown"].includes(d.state)) out.push("stop");
  if (d.state === "stopped" || d.state === "failed") out.push("start");
  if (canRemove(d.state)) out.push("remove");
  return out;
}

/** What Remove does to a system deployment, for its confirmation. */
export const SYSTEM_REMOVE_TEXT = "This also turns the decision model off: every feature falls back to rules until you enable it again on the Decision model page.";

export function submitState(preview: Preview | null, actions: PlanAction[], pending: boolean, mode: "deploy" | "switch" | "reconfigure") {
  if (pending) return { disabled: true, label: mode === "deploy" ? "Deploying…" : "Working… this can take minutes" };
  const base = mode === "deploy" ? "Deploy" : mode === "switch" ? "Switch" : "Apply and restart";
  if (!preview) return { disabled: true, label: base };
  if (!preview.fit.fits && actions.length === 0) return { disabled: true, label: "Choose a plan to fit" };
  if (actions.length) return { disabled: false, label: mode === "deploy" ? "Apply plan and deploy" : `Apply plan and ${base.toLowerCase()}` };
  return { disabled: false, label: base };
}
