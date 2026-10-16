export interface Rate { bytes: number; t: number; bps: number | null }

const ALPHA = 0.3;

export function nextRate(prev: Rate | undefined, bytes: number, t: number): Rate {
  if (!prev) return { bytes, t, bps: null };
  const dt = t - prev.t;
  if (dt <= 0) return prev;
  const inst = Math.max(bytes - prev.bytes, 0) / dt;
  return { bytes, t, bps: prev.bps == null ? inst : ALPHA * inst + (1 - ALPHA) * prev.bps };
}

export function etaSeconds(r: Rate | undefined, remaining: number): number | null {
  if (!r || !r.bps || r.bps <= 0) return null;
  return Math.round(remaining / r.bps);
}

export function fmtDuration(s: number | null): string {
  if (s == null || !Number.isFinite(s)) return "—";
  if (s < 60) return `${Math.round(s)}s`;
  if (s < 3600) return `${Math.floor(s / 60)}m ${Math.round(s % 60)}s`;
  return `${Math.floor(s / 3600)}h ${Math.floor((s % 3600) / 60)}m`;
}

export function updateRates(map: Map<number, Rate>, downloads: Array<{ id: number; state: string; bytes_done: number }>, t: number): void {
  const running = new Set<number>();
  for (const d of downloads) {
    if (d.state === "running") {
      running.add(d.id);
      map.set(d.id, nextRate(map.get(d.id), d.bytes_done, t));
    }
  }
  for (const id of map.keys()) {
    if (!running.has(id)) map.delete(id);
  }
}
