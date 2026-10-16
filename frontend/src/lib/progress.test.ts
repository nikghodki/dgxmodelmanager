import { describe, expect, it } from "vitest";
import { etaSeconds, fmtDuration, nextRate, updateRates } from "@/lib/progress";

describe("download progress", () => {
  it("needs two samples, then smooths with an EMA", () => {
    const a = nextRate(undefined, 0, 0);
    expect(a.bps).toBeNull();
    const b = nextRate(a, 100, 1);
    expect(b.bps).toBe(100);
    const c = nextRate(b, 400, 2);
    expect(c.bps).toBeCloseTo(0.3 * 300 + 0.7 * 100);
  });
  it("ignores non-advancing time and computes ETA", () => {
    const b = nextRate(nextRate(undefined, 0, 0), 100, 1);
    expect(nextRate(b, 150, 1)).toEqual(b);
    expect(etaSeconds(b, 1000)).toBe(10);
    expect(etaSeconds(undefined, 1000)).toBeNull();
  });
  it("formats durations", () => {
    expect(fmtDuration(null)).toBe("—");
    expect(fmtDuration(42)).toBe("42s");
    expect(fmtDuration(125)).toBe("2m 5s");
    expect(fmtDuration(3725)).toBe("1h 2m");
  });
  it("updateRates: no change on duplicate sample, updates on new sample", () => {
    const map = new Map();
    const dl1 = [{ id: 1, state: "running", bytes_done: 100 }];
    updateRates(map, dl1, 1);
    expect(map.get(1)?.bps).toBeNull();
    const dl2 = [{ id: 1, state: "running", bytes_done: 400 }];
    updateRates(map, dl2, 2);
    expect(map.get(1)?.bps).toBe(300);
    const bpsBefore = map.get(1)?.bps;
    updateRates(map, dl2, 2);
    expect(map.get(1)?.bps).toBe(bpsBefore);
    const dl3 = [{ id: 1, state: "running", bytes_done: 700 }];
    updateRates(map, dl3, 3);
    expect(map.get(1)?.bps).toBeCloseTo(0.3 * 300 + 0.7 * 300);
  });
  it("updateRates: drops non-running downloads", () => {
    const map = new Map();
    updateRates(map, [{ id: 1, state: "running", bytes_done: 100 }], 1);
    updateRates(map, [{ id: 1, state: "running", bytes_done: 400 }], 2);
    expect(map.has(1)).toBe(true);
    updateRates(map, [{ id: 1, state: "done", bytes_done: 400 }], 3);
    expect(map.has(1)).toBe(false);
  });
});
