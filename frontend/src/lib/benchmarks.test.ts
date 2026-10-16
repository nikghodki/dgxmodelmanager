import { describe, expect, it } from "vitest";
import { benchmarkSeries } from "@/lib/benchmarks";

describe("benchmarkSeries", () => {
  it("pivots rows into one series per config, current first, sorted by concurrency", () => {
    const r = benchmarkSeries([
      { config_hash: "aaaaaaaa11", concurrency: 4, tps_total: 120, ts: "2026-10-16T01:00:00Z" },
      { config_hash: "bbbbbbbb22", concurrency: 1, tps_total: 50, current: true, ts: "2026-10-16T02:00:00Z" },
      { config_hash: "aaaaaaaa11", concurrency: 1, tps_total: 40, ts: "2026-10-16T01:00:00Z" },
      { config_hash: "bbbbbbbb22", concurrency: 4, tps_total: 180, current: true, ts: "2026-10-16T02:00:00Z" },
    ]);
    expect(r.keys).toEqual([
      { hash: "bbbbbbbb22", label: "bbbbbbbb (current)", current: true },
      { hash: "aaaaaaaa11", label: "aaaaaaaa", current: false },
    ]);
    expect(r.data).toEqual([
      { concurrency: 1, bbbbbbbb22: 50, aaaaaaaa11: 40 },
      { concurrency: 4, bbbbbbbb22: 180, aaaaaaaa11: 120 },
    ]);
  });
});
