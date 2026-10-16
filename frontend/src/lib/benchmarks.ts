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
