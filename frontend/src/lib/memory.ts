import type { Budget } from "@/types";

export interface MemorySegment {
  key: string;
  label: string;
  bytes: number;
  pct: number;
  color: string;
  kind: "deployment" | "sleeping" | "external" | "reserve" | "free";
}

const ACTIVE = ["starting", "loading", "ready", "sleeping", "unknown", "stopping"];

/** Segments for the unified-memory bar. Deployment colour = chart slot by deployment id (follows the entity). */
export function memorySegments(b: Budget): { segments: MemorySegment[]; overcommitted: number } {
  const total = b.total || 1;
  const deps = b.deployments.filter((d) => ACTIVE.includes(d.state)).sort((x, y) => x.deployment_id - y.deployment_id);
  const segs: Omit<MemorySegment, "pct">[] = deps.map((d) => {
    const sleeping = d.state === "sleeping";
    return {
      key: `dep-${d.deployment_id}`,
      label: sleeping ? `${d.name} · sleeping` : d.name,
      bytes: sleeping ? d.actual : Math.max(d.reserved, d.actual),
      color: `var(--chart-${((d.deployment_id - 1) % 8 + 8) % 8 + 1})`,
      kind: sleeping ? "sleeping" : "deployment",
    };
  });
  segs.push(
    { key: "external", label: "External processes", bytes: b.external, color: "var(--chart-external)", kind: "external" },
    { key: "reserve", label: "Safety margin + OS reserve", bytes: b.margin + b.os_reserve, color: "var(--chart-reserve)", kind: "reserve" },
    { key: "free", label: "Free for new deployments", bytes: Math.max(b.free_for_new, 0), color: "var(--chart-free)", kind: "free" },
  );
  return {
    segments: segs.map((s) => ({ ...s, pct: (s.bytes / total) * 100 })),
    overcommitted: b.free_for_new < 0 ? -b.free_for_new : 0,
  };
}
