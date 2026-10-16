import { useQuery } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import { api } from "@/api";
import { Button } from "@/components/ui/button";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { fmtBytes, GIB } from "@/format";
import { cn } from "@/lib/utils";
import type { HostInfo } from "@/types";
import { MemoryBar } from "./memory-bar";

export function MemoryChip() {
  const host = useQuery({ queryKey: ["host"], queryFn: () => api<HostInfo>("/host") });
  const b = host.data?.budget;
  if (!b) return null;
  const tone = b.free_for_new < 0 ? "bg-status-critical" : b.free_for_new < 8 * GIB ? "bg-status-warning" : "bg-status-good";
  return (
    <Popover>
      <PopoverTrigger asChild>
        <Button variant="outline" size="sm" className="h-8 gap-2 rounded-full px-3 text-xs font-medium tabular-nums">
          <span aria-hidden className={cn("h-2 w-2 rounded-full", tone)} />
          <span className="hidden sm:inline">{fmtBytes(b.free_for_new)} free</span>
          <span className="sm:hidden">{fmtBytes(b.free_for_new, 0)}</span>
        </Button>
      </PopoverTrigger>
      <PopoverContent align="end" className="w-[min(92vw,26rem)]">
        <div className="mb-3 flex items-baseline justify-between">
          <span className="text-sm font-semibold">Unified memory</span>
          <span className="text-xs text-muted-foreground tabular-nums">{fmtBytes(b.total)} total</span>
        </div>
        <MemoryBar budget={b} />
        <Button asChild variant="link" className="mt-2 h-auto p-0 text-xs"><Link to="/">Open dashboard</Link></Button>
      </PopoverContent>
    </Popover>
  );
}
