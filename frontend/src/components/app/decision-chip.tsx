import { CircleSlash, Cpu, ListChecks, Sparkles } from "lucide-react";
import { decisionChip } from "@/lib/decider";
import { cn } from "@/lib/utils";
import type { DecisionSource } from "@/types";

/** "92% · AI" / "Rules" with an icon; the text carries the meaning, colour is decoration only. */
export function DecisionChip({ source, confidence, prefix, className, title }: {
  source: DecisionSource | string; confidence: number | null | undefined; prefix?: string; className?: string; title?: string;
}) {
  const Icon = source === "model" ? Sparkles : source === "none" ? CircleSlash : ListChecks;
  const text = decisionChip(source, confidence);
  return (
    <span data-decision-source={source} title={title}
      className={cn("inline-flex items-center gap-1 whitespace-nowrap rounded-full border px-2 py-0.5 text-[11px] font-medium",
        source === "model" ? "border-primary/30 bg-primary/10 text-primary"
          : source === "none" ? "border-dashed border-muted-foreground/40 bg-transparent text-muted-foreground"
          : "border-border bg-muted text-muted-foreground", className)}>
      <Icon aria-hidden className="h-3 w-3" />
      {prefix ? `${prefix} · ${text}` : text}
    </span>
  );
}

/** A system deployment (the decision model): DGXMM manages it; plans and LiteLLM leave it alone, and the watchdog stops it
 * first under memory pressure (or when its own memory runs away). */
export function SystemBadge({ className }: { className?: string }) {
  return (
    <span title="System deployment used by the decision model"
      className={cn("inline-flex items-center gap-1 whitespace-nowrap rounded-full border border-border bg-muted px-2 py-0.5 text-[11px] font-medium text-muted-foreground", className)}>
      <Cpu aria-hidden className="h-3 w-3" />System
    </span>
  );
}
