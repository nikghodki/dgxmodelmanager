import { describe, expect, it } from "vitest";
import { fleetKpis } from "@/lib/fleet";

describe("fleetKpis", () => {
  it("returns nulls (never NaN) when there are no samples", () => {
    const k = fleetKpis([]);
    expect(k).toMatchObject({ genTps: null, ttftP50: null, kv: null, waiting: null, running: null });
    expect(k.genSeries.every((v) => v === null)).toBe(true);
  });
  it("sums throughput, takes worst TTFT/KV, ignores NaN", () => {
    const a = [{ gen_tps: 10, ttft_p50: 0.2, kv_usage: 0.3, waiting: 1, running: 2 }, { gen_tps: 20, ttft_p50: 0.25, kv_usage: 0.4, waiting: 0, running: 1 }];
    const b = [{ gen_tps: NaN, ttft_p50: 0.5, kv_usage: 0.9, waiting: 2, running: 1 }];
    const k = fleetKpis([a, b], 3);
    expect(k.genTps).toBe(20);
    expect(k.ttftP50).toBe(0.5);
    expect(k.kv).toBe(0.9);
    expect(k.waiting).toBe(2);
    expect(k.running).toBe(2);
    expect(k.genSeries).toEqual([null, 10, 20]);
  });
});
