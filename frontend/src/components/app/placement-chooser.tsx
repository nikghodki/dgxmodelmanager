import { useRef } from "react";
import { CheckCircle2, Star } from "lucide-react";
import { fmtBytes } from "@/format";
import { PLACEMENT_BADGES, nodeLabel, optionNodes, placementKey } from "@/lib/placement";
import { cn } from "@/lib/utils";
import type { NodeInfo, PlacementOptionInfo } from "@/types";

const TONES = {
  critical: "bg-status-critical-bg text-status-critical",
  warning: "bg-status-warning-bg text-status-warning",
  neutral: "bg-status-neutral-bg text-status-neutral",
};

const domId = (key: string) => `placement-${key.replace(/[^a-z0-9-]/gi, "-")}`;

/** The deploy wizard's "Where to run": one card per placement option (each node, the nodes as a cluster) with the
 *  memory it needs against what is free on each node. A radio group with a roving tabindex: Tab enters it at the
 *  selected option (the first when none is), the arrow keys move and select. */
export function PlacementChooser({ options, nodes, selected, onSelect }: {
  options: PlacementOptionInfo[]; nodes: NodeInfo[] | undefined; selected: string | null; onSelect: (o: PlacementOptionInfo) => void;
}) {
  const refs = useRef<(HTMLButtonElement | null)[]>([]);
  const none = !options.some((o) => o.recommended);
  const keys = options.map((o) => placementKey(o.placement));
  const tabStop = Math.max(keys.indexOf(selected ?? ""), 0);
  const onKey = (e: React.KeyboardEvent, i: number) => {
    const step = e.key === "ArrowDown" || e.key === "ArrowRight" ? 1 : e.key === "ArrowUp" || e.key === "ArrowLeft" ? -1 : 0;
    const to = e.key === "Home" ? 0 : e.key === "End" ? options.length - 1 : step ? (i + step + options.length) % options.length : -1;
    if (to < 0) return;
    e.preventDefault();
    refs.current[to]?.focus();
    onSelect(options[to]);
  };
  return (
    <div className="space-y-2">
      {none && (
        <p className="text-xs text-status-critical"><span className="font-medium">Nothing fits right now</span>: every placement is short
          of memory or on a node without a recent memory reading. Choose one to see ways to make room.</p>
      )}
      <div className="grid grid-cols-1 gap-2 sm:grid-cols-2" role="radiogroup" aria-label="Where to run">
        {options.map((o, i) => {
          const key = keys[i];
          const id = domId(key);
          const sel = selected === key;
          const badge = o.fits ? null : PLACEMENT_BADGES[o.reason] ?? PLACEMENT_BADGES.insufficient_memory;
          return (
            <button key={key} ref={(el) => { refs.current[i] = el; }} type="button" role="radio" aria-checked={sel}
              tabIndex={i === tabStop ? 0 : -1} aria-labelledby={`${id}-label`} aria-describedby={`${id}-badges ${id}-memory`}
              onClick={() => onSelect(o)} onKeyDown={(e) => onKey(e, i)}
              className={cn("block w-full rounded-lg border p-3 text-left transition-colors hover:bg-accent focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
                sel ? "border-primary ring-1 ring-primary" : "border-border")}>
              <span className="flex flex-wrap items-center justify-between gap-2">
                <span id={`${id}-label`} className="text-sm font-medium">{o.label}</span>
                <span id={`${id}-badges`} className="flex flex-wrap items-center gap-1.5">
                  {o.recommended && <span className="inline-flex items-center gap-1 rounded-full bg-primary px-2 py-0.5 text-[11px] font-semibold text-primary-foreground"><Star aria-hidden className="h-3 w-3" />Recommended</span>}
                  {badge && <span className={cn("rounded-full px-2 py-0.5 text-[11px] font-medium", TONES[badge.tone])}>{badge.text}</span>}
                  {o.fits && !o.recommended && <span className="inline-flex items-center gap-1 text-xs text-status-good"><CheckCircle2 aria-hidden className="h-3.5 w-3.5" />fits</span>}
                </span>
              </span>
              <span id={`${id}-memory`} className="mt-1.5 block space-y-0.5">
                {optionNodes(o).map((n) => (
                  <span key={n.node} className="block text-xs tabular-nums text-muted-foreground">
                    <span className="text-foreground">{nodeLabel(nodes, n.node)}</span>: needs {fmtBytes(n.need)} of {fmtBytes(n.free)} free
                    {o.reason !== "node_stale" && n.short > 0 && <span className="text-status-critical"> · short by {fmtBytes(n.short)}</span>}
                  </span>
                ))}
              </span>
            </button>
          );
        })}
      </div>
    </div>
  );
}
