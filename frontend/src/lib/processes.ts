import type { ContainerRef, ManagedRef, ProcessInfo } from "@/types";

export type ProcessScope = "all" | "external" | "managed";

export interface ProcessGroup {
  key: string;            // container id, or "host"
  name: string;           // display label
  image: string | null;
  managed: ManagedRef | null;
  rss: number;
  gpu: number;
  processes: ProcessInfo[];
}

export const containerLabel = (c: ContainerRef | null | undefined) => (c ? c.name ?? c.id : "host");

export function filterScope(ps: ProcessInfo[], scope: ProcessScope): ProcessInfo[] {
  if (scope === "external") return ps.filter((p) => !p.managed);
  if (scope === "managed") return ps.filter((p) => !!p.managed);
  return ps;
}

/** One group per container plus one for host processes, biggest (RAM + GPU) first. */
export function groupByContainer(ps: ProcessInfo[]): ProcessGroup[] {
  const groups = new Map<string, ProcessGroup>();
  for (const p of ps) {
    const key = p.container?.id ?? "host";
    let g = groups.get(key);
    if (!g) {
      g = { key, name: containerLabel(p.container), image: p.container?.image ?? null, managed: p.managed, rss: 0, gpu: 0, processes: [] };
      groups.set(key, g);
    }
    g.rss += p.rss;
    g.gpu += p.gpu;
    g.processes.push(p);
  }
  return [...groups.values()].sort((a, b) => b.rss + b.gpu - (a.rss + a.gpu));
}

export const processSearchText = (p: ProcessInfo) =>
  `${p.command} ${p.name} ${p.exe} ${containerLabel(p.container)} ${p.container?.image ?? ""} ${p.user} ${p.pid} ${p.managed?.name ?? ""}`;
