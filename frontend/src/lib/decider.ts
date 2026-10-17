import type { DeciderStatus, DecisionSource, PlanOption, Suggestion } from "@/types";

/** Confidence (0..1) as a whole percentage. */
export const fmtConfidence = (c: number | null | undefined) => (c == null || Number.isNaN(c) ? "—" : `${Math.round(c * 100)}%`);

const SOURCE_LABEL: Record<DecisionSource, string> = { rules: "Rules", model: "AI", none: "None" };
export const sourceLabel = (s: DecisionSource | string) => SOURCE_LABEL[s as DecisionSource] ?? s;

/** "92% · AI" for model decisions; rules are certain by construction, so just "Rules". */
export function decisionChip(source: DecisionSource | string, confidence: number | null | undefined): string {
  if (source === "model" && confidence != null) return `${fmtConfidence(confidence)} · ${sourceLabel(source)}`;
  return sourceLabel(source);
}

export const KIND_LABELS: Record<string, string> = {
  llm: "LLM", draft: "Draft", embedding: "Embedding", reranker: "Reranker", "vision-llm": "Vision LLM", speech: "Speech",
  decision: "Decision", other: "Other",
};
export const kindLabel = (k: string) => KIND_LABELS[k] ?? k;

/** A parser suggestion's value: null means "no parser". */
export const parserValueLabel = (v: string | null | undefined) => (v == null || v === "" ? "none" : v);

/** The recommended plan first; the rest keep the server's order. */
export function sortPlans(plans: PlanOption[]): PlanOption[] {
  return [...plans.filter((p) => p.recommended), ...plans.filter((p) => !p.recommended)];
}

export type SuggestionState = "autofill" | "filled" | "apply" | "matches";

/**
 * What to do with a parser suggestion given the field's current value.
 * - autofill: empty, untouched field and the server says auto_apply → fill it now.
 * - filled:   we filled it and the value is still ours → show "Filled automatically — undo".
 * - apply:    the suggestion differs from the value → offer an explicit Apply (never overwrite a user-typed value).
 * - matches:  nothing to do.
 */
export function suggestionState(s: Suggestion, current: unknown, f: { touched: boolean; autoFilled: boolean }): SuggestionState {
  const cur = current == null || current === "" ? null : String(current);
  if (cur === s.value) return f.autoFilled && cur !== null ? "filled" : "matches";
  if (cur === null && s.value !== null && s.auto_apply && !f.touched && !f.autoFilled) return "autofill";
  return "apply";
}

export type DeciderHealth = "healthy" | "building" | "starting" | "stopping" | "sleeping" | "stopped" | "disabled" | "error";
const STARTING_STATES = ["planned", "queued", "starting", "loading", "unknown"];
const OWN_WORD: Record<string, { state: DeciderHealth; label: string }> = {
  stopping: { state: "stopping", label: "Stopping" }, sleeping: { state: "sleeping", label: "Sleeping" }, stopped: { state: "stopped", label: "Stopped" },
};

/**
 * One word for the decider's health (status chip). Healthy wins; then an image build in progress (Building); then
 * disabled; then the deployment's own state (Starting / Stopping / Sleeping / Stopped). Error only for a failed
 * deployment, no deployment while enabled, or "ready" but not answering (circuit breaker open).
 */
export function deciderHealth(s: Pick<DeciderStatus, "enabled" | "healthy" | "state" | "image">): { state: DeciderHealth; label: string } {
  if (s.enabled && s.healthy) return { state: "healthy", label: "Healthy" };
  if (s.image?.building) return { state: "building", label: "Building" };
  if (!s.enabled) return { state: "disabled", label: "Disabled" };
  if (s.state && STARTING_STATES.includes(s.state)) return { state: "starting", label: "Starting" };
  if (s.state && OWN_WORD[s.state]) return OWN_WORD[s.state];
  return { state: "error", label: "Error" };
}
