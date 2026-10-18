import { useQuery } from "@tanstack/react-query";
import { Loader2 } from "lucide-react";
import { api } from "@/api";
import { compatText, FLAG_LABELS, isCompat, verdictTone } from "@/lib/advice";
import { cn } from "@/lib/utils";
import type { CompatResult } from "@/types";
import { DecisionChip } from "./decision-chip";

const TONE: Record<string, string> = {
  good: "text-status-good", warning: "text-status-warning", critical: "text-status-critical", neutral: "text-muted-foreground",
};

/** GET /catalog/compat: will this Hugging Face repo run here, before it is downloaded (rules, else the decision model). */
export function useCompat(repo: string, enabled = true) {
  return useQuery({
    queryKey: ["compat", repo],
    queryFn: () => api<CompatResult>(`/catalog/compat?repo=${encodeURIComponent(repo)}`),
    enabled: enabled && !!repo, retry: false, staleTime: 10 * 60_000, refetchOnWindowFocus: false,
  });
}

/** One line per engine with the verdict and its templated reasons; silent when the feature is off. */
export function CompatLine({ repo, className, label = "Compatibility", compact = false }: {
  repo: string; className?: string; label?: string;
  /** One line (e.g. next to a download button): "vLLM: runs · llama.cpp: not supported". */
  compact?: boolean;
}) {
  const q = useCompat(repo);
  if (q.isLoading) return <p className={cn("flex items-center gap-1.5 text-xs text-muted-foreground", className)}><Loader2 aria-hidden className="h-3 w-3 animate-spin" />Checking compatibility…</p>;
  const c = isCompat(q.data) ? q.data : null;
  if (q.error || !c) return null;
  if (c.disabled) return null;
  if (c.error === "hub_unavailable") return <p className={cn("text-xs text-muted-foreground", className)}>{label}: unavailable (Hugging Face not reachable)</p>;
  if (compact) {
    return <p className={cn("text-xs text-muted-foreground", className)} data-compat-compact={c.vllm.verdict}>{label}: {compatText("vLLM", { ...c.vllm, reasons: [] })} · {compatText("llama.cpp", { ...c.llamacpp, reasons: [] })}{c.gated ? " · gated" : ""}</p>;
  }
  const flags = c.vllm.flags.filter((f) => f !== "chat_template").map((f) => FLAG_LABELS[f] ?? f);
  return (
    <div className={cn("space-y-1 rounded-md border p-2.5 text-xs", className)} data-compat={c.vllm.verdict}>
      <div className="flex flex-wrap items-center gap-2">
        <span className="font-medium">{label}</span>
        <DecisionChip source={c.source} confidence={c.confidence} title={c.source === "model" ? "vLLM verdict from the decision model" : "From the repo's config and the pinned vLLM image"} />
        {c.gated && <span className="text-status-warning">gated</span>}
      </div>
      <p className={TONE[verdictTone(c.vllm.verdict)]}>{compatText("vLLM", c.vllm)}</p>
      <p className={TONE[verdictTone(c.llamacpp.verdict)]}>{compatText("llama.cpp", c.llamacpp)}</p>
      {flags.length > 0 && <p className="text-muted-foreground">Deploying it needs: {flags.join(", ")}</p>}
    </div>
  );
}
