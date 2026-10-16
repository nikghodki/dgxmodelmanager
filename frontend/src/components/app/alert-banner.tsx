import { AlertTriangle, OctagonAlert } from "lucide-react";
import { cn } from "@/lib/utils";
import type { Alert } from "@/types";

export function AlertBanner({ alert }: { alert?: Alert }) {
  if (!alert || alert.level === "ok") return null;
  const critical = alert.level === "critical";
  const Icon = critical ? OctagonAlert : AlertTriangle;
  return (
    <div role="alert" className={cn("flex items-start gap-2 border-b px-4 py-2 text-sm md:px-6",
      critical ? "bg-status-critical-bg text-status-critical" : "bg-status-warning-bg text-status-warning")}>
      <Icon className="mt-0.5 h-4 w-4 shrink-0" />
      <p className="min-w-0"><span className="font-semibold">{critical ? "Memory critical" : "Memory warning"}:</span> {alert.message}
        {alert.action && <span className="ml-1 opacity-80">({alert.action})</span>}</p>
    </div>
  );
}
