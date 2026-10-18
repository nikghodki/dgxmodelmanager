import { Link } from "react-router-dom";
import { Bell } from "lucide-react";
import { useInsights } from "@/hooks/use-insights";
import { openCount } from "@/lib/insights";
import { cn } from "@/lib/utils";

/** Top-bar link to the Insights page with the number of open insights (live: the SSE `insights` topic refreshes the
 *  shared ["insights", "open", …] query). */
export function InsightsBadge() {
  const q = useInsights({ status: "open" });
  const n = openCount(q.data);
  const label = n ? `Insights: ${n} open` : "Insights";
  return (
    <Link to="/insights" aria-label={label} title={label}
      className="relative inline-flex h-8 w-8 items-center justify-center rounded-md text-muted-foreground transition-colors hover:bg-accent hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring">
      <Bell aria-hidden className="h-4 w-4" />
      {n > 0 && (
        <span data-count={n} className={cn("absolute -right-0.5 -top-0.5 flex h-4 min-w-4 items-center justify-center rounded-full bg-primary px-1 text-[10px] font-semibold tabular-nums text-primary-foreground")}>
          {n > 99 ? "99+" : n}
        </span>
      )}
    </Link>
  );
}
