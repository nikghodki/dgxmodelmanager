import { useState } from "react";
import { Table2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { fmtBytes } from "@/format";
import { memorySegments } from "@/lib/memory";
import type { Budget } from "@/types";

export function MemoryBar({ budget, compact = false }: { budget: Budget; compact?: boolean }) {
  const { segments, overcommitted } = memorySegments(budget);
  const [asTable, setAsTable] = useState(false);
  return (
    <div className="space-y-3">
      <div className="flex h-4 w-full gap-[2px]" role="img" aria-label="Unified memory usage">
        {segments.filter((s) => s.bytes > 0).map((s) => (
          <Tooltip key={s.key}>
            <TooltipTrigger asChild>
              <div className="h-full min-w-[3px] rounded-[4px] first:rounded-l-[6px] last:rounded-r-[6px]"
                style={{ width: `${s.pct}%`, background: s.color, opacity: s.kind === "sleeping" ? 0.55 : 1 }} />
            </TooltipTrigger>
            <TooltipContent><span className="font-medium">{s.label}</span> · <span className="tabular-nums">{fmtBytes(s.bytes)}</span> ({s.pct.toFixed(0)}%)</TooltipContent>
          </Tooltip>
        ))}
      </div>
      {!compact && (
        <div className="flex flex-wrap items-start justify-between gap-2">
          {asTable ? (
            <table className="w-full max-w-md text-xs">
              <tbody>
                {segments.map((s) => (
                  <tr key={s.key} className="border-b last:border-0">
                    <td className="py-1 pr-3">{s.label}</td>
                    <td className="py-1 pr-3 text-right tabular-nums">{fmtBytes(s.bytes)}</td>
                    <td className="py-1 text-right tabular-nums text-muted-foreground">{s.pct.toFixed(1)}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            <ul className="flex flex-wrap gap-x-4 gap-y-1.5 text-xs text-muted-foreground">
              {segments.map((s) => (
                <li key={s.key} className="flex items-center gap-1.5">
                  <span aria-hidden className="inline-block h-2.5 w-2.5 rounded-[3px]" style={{ background: s.color, opacity: s.kind === "sleeping" ? 0.55 : 1 }} />
                  <span className="text-foreground">{s.label}</span>
                  <span className="tabular-nums">{fmtBytes(s.bytes)}</span>
                </li>
              ))}
            </ul>
          )}
          <Button variant="ghost" size="sm" className="h-7 text-xs" onClick={() => setAsTable(!asTable)}>
            <Table2 className="mr-1 h-3.5 w-3.5" />{asTable ? "Legend" : "Table"}
          </Button>
        </div>
      )}
      {overcommitted > 0 && (
        <p className="text-xs font-medium text-status-critical">Over-committed by {fmtBytes(overcommitted)}. New deployments are blocked.</p>
      )}
    </div>
  );
}
