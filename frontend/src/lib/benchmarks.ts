interface Row { config_hash: string; concurrency: number; tps_total: number; current?: boolean; ts: string }

export function benchmarkSeries(rows: Row[]) {
  const byHash = new Map<string, { current: boolean; latest: string }>();
  for (const r of rows) {
    const prev = byHash.get(r.config_hash);
    byHash.set(r.config_hash, { current: !!(prev?.current || r.current), latest: prev && prev.latest > r.ts ? prev.latest : r.ts });
  }
  const keys = [...byHash.entries()]
    .sort((a, b) => Number(b[1].current) - Number(a[1].current) || b[1].latest.localeCompare(a[1].latest))
    .map(([hash, v]) => ({ hash, label: `${hash.slice(0, 8)}${v.current ? " (current)" : ""}`, current: v.current }));
  const levels = [...new Set(rows.map((r) => r.concurrency))].sort((a, b) => a - b);
  const data = levels.map((c) => {
    const row: Record<string, number> = { concurrency: c };
    for (const k of keys) {
      const hit = rows.filter((r) => r.config_hash === k.hash && r.concurrency === c).sort((a, b) => b.ts.localeCompare(a.ts))[0];
      if (hit) row[k.hash] = hit.tps_total;
    }
    return row;
  });
  return { data, keys };
}

/** Reading labels of benchmark rows (dgxmm/decider/reading.py): plausible rows are "verified", every other label is
 *  "disputed" and is left out of the chart and of the advice. Unlabelled rows (feature off, or not labelled yet)
 *  show no badge and count as before. */
export type ReadingLabel = "plausible" | "contended" | "cold_cache" | "broken";
const READING: Record<ReadingLabel, { text: string; tone: string; hint: string }> = {
  plausible: { text: "Plausible", tone: "bg-status-good-bg text-status-good", hint: "Consistent with earlier readings" },
  contended: { text: "Contended", tone: "bg-status-warning-bg text-status-warning", hint: "Disturbed by other work on the host; not used for advice" },
  cold_cache: { text: "Cold cache", tone: "bg-status-warning-bg text-status-warning", hint: "First run after the deployment started; not used for advice" },
  broken: { text: "Broken", tone: "bg-status-critical-bg text-status-critical", hint: "Not a real measurement; not used for advice" },
};

export function readingBadge(label: string | null | undefined, source?: string | null) {
  const r = label ? READING[label as ReadingLabel] : undefined;
  if (!r) return null;
  const by = source === "model" ? " (decision model)" : source === "rules" ? " (rules)" : "";
  return { text: r.text, tone: r.tone, hint: r.hint + by };
}

export const plausibleRows = <T extends { quality?: string | null }>(rows: T[]) => rows.filter((r) => r.quality !== "disputed");
