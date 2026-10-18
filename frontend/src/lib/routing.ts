import type { RoutingClassAdvice } from "@/types";

/** LiteLLM routing advice helpers (GET /decider/routing-advice). */
export function sourceText(r: Pick<RoutingClassAdvice, "source" | "confidence">): string {
  if (r.source === "model") return r.confidence != null ? `Decision model · ${Math.round(r.confidence * 100)} %` : "Decision model";
  if (r.source === "rules") return "Rules";
  return "—";
}

/** One class as a copyable LiteLLM router alias (the full snippet comes from the server). */
export function aliasSnippet(r: Pick<RoutingClassAdvice, "alias" | "model_name" | "routable">): string | null {
  if (!r.model_name || !r.routable) return null;  // LiteLLM serves only ready, registered deployments
  return `# merge into your existing router_settings\nrouter_settings:\n  model_group_alias:\n    ${r.alias}: ${JSON.stringify(r.model_name)}`;
}
