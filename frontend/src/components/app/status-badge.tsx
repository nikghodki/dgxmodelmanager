import { AlertTriangle, CheckCircle2, Circle, Loader2, Moon, XCircle } from "lucide-react";
import { cn } from "@/lib/utils";
import { statusInfo, type StatusKind } from "@/lib/status";

const STYLES: Record<StatusKind, string> = {
  good: "bg-status-good-bg text-status-good",
  warning: "bg-status-warning-bg text-status-warning",
  sleeping: "bg-status-sleeping-bg text-status-sleeping",
  critical: "bg-status-critical-bg text-status-critical",
  neutral: "bg-status-neutral-bg text-status-neutral",
};

export function StatusBadge({ state, className }: { state: string; className?: string }) {
  const s = statusInfo(state);
  const Icon = s.kind === "good" ? CheckCircle2 : s.kind === "critical" ? XCircle : s.kind === "sleeping" ? Moon
    : s.kind === "warning" ? (s.spin ? Loader2 : AlertTriangle) : Circle;
  return (
    <span data-status={s.kind}
      className={cn("inline-flex items-center gap-1 whitespace-nowrap rounded-full px-2 py-0.5 text-xs font-medium", STYLES[s.kind], className)}>
      <Icon aria-hidden className={cn("h-3 w-3", s.spin && "animate-spin")} />
      {s.label}
    </span>
  );
}
