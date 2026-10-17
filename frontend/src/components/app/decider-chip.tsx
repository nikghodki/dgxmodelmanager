import { useQuery } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import { BrainCircuit, CheckCircle2, Circle, Hammer, Loader2, Moon, XCircle } from "lucide-react";
import { api } from "@/api";
import { useIsAdmin } from "@/auth";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { deciderHealth, type DeciderHealth } from "@/lib/decider";
import { cn } from "@/lib/utils";
import type { DeciderStatus } from "@/types";

const KIND: Record<DeciderHealth, string> = {
  healthy: "good", building: "warning", starting: "warning", stopping: "warning", sleeping: "sleeping", stopped: "neutral",
  disabled: "neutral", error: "critical",
};
const STYLE: Record<string, string> = {
  good: "bg-status-good-bg text-status-good",
  warning: "bg-status-warning-bg text-status-warning",
  sleeping: "bg-status-sleeping-bg text-status-sleeping",
  neutral: "bg-status-neutral-bg text-status-neutral",
  critical: "bg-status-critical-bg text-status-critical",
};
const ICON: Record<DeciderHealth, typeof Circle> = {
  healthy: CheckCircle2, building: Hammer, starting: Loader2, stopping: Loader2, sleeping: Moon, stopped: Circle, disabled: Circle, error: XCircle,
};
const SPIN: DeciderHealth[] = ["starting", "stopping"];

/**
 * Top-bar status chip for the local decision model (Healthy, Building, Starting, Stopping, Sleeping, Stopped,
 * Disabled, Error). For admins it links to the Decision model page; viewers (who cannot open that page) get a
 * read-only status popover instead. Shares the page's ["decider-status"] query, which the SSE stream invalidates whenever the
 * decider deployment changes; the slow poll only catches a circuit breaker opening or closing.
 */
export function DeciderChip() {
  const admin = useIsAdmin();
  const st = useQuery({ queryKey: ["decider-status"], queryFn: () => api<DeciderStatus>("/decider/status"), refetchInterval: 30000 });
  if (!st.data) return null;
  const h = deciderHealth(st.data);
  const Icon = ICON[h.state];
  const cls = cn("inline-flex h-8 items-center gap-1.5 whitespace-nowrap rounded-full px-3 text-xs font-medium transition-opacity hover:opacity-80 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring", STYLE[KIND[h.state]]);
  const body = <>
    <BrainCircuit aria-hidden className="hidden h-3.5 w-3.5 sm:block" />
    <Icon aria-hidden className={cn("h-3 w-3", SPIN.includes(h.state) && "animate-spin")} />
    <span className="sr-only sm:not-sr-only">{h.label}</span>
  </>;
  if (admin) {
    return <Link to="/decision-model" aria-label={`Decision model: ${h.label}`} data-status={KIND[h.state]} title={`Decision model: ${h.label}`} className={cls}>{body}</Link>;
  }
  return (
    <Popover>
      <PopoverTrigger asChild>
        <button type="button" aria-label={`Decision model: ${h.label}`} data-status={KIND[h.state]} title={`Decision model: ${h.label}`} className={cls}>{body}</button>
      </PopoverTrigger>
      <PopoverContent align="end" className="w-72 space-y-1.5 text-sm">
        <p className="font-medium">Decision model: {h.label}</p>
        <p className="text-xs text-muted-foreground">{st.data.enabled
          ? <>A local model (<span className="font-mono">{st.data.model}</span>) that advises DGXMM's features. Rules come first; it abstains when unsure.</>
          : "Off: every feature uses DGXMM's rules."}</p>
        <p className="text-xs text-muted-foreground">Only admins can change it.</p>
      </PopoverContent>
    </Popover>
  );
}
