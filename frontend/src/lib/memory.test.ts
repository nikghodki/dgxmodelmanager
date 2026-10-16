import { describe, expect, it } from "vitest";
import { memorySegments } from "@/lib/memory";
import type { Budget } from "@/types";

const GIB = 1024 ** 3;
const budget = (deployments: Budget["deployments"], free = 30 * GIB): Budget => ({
  total: 120 * GIB, available: 60 * GIB, margin: 8 * GIB, os_reserve: 4 * GIB, usable: 108 * GIB,
  committed: 0, managed_actual: 0, external: 20 * GIB, free_for_new: free, deployments,
});

describe("memorySegments", () => {
  it("orders deployments, external, reserve, free and sizes them against total", () => {
    const { segments } = memorySegments(budget([
      { deployment_id: 2, name: "b", state: "ready", reserved: 30 * GIB, actual: 20 * GIB },
      { deployment_id: 1, name: "a", state: "sleeping", reserved: 40 * GIB, actual: 6 * GIB },
      { deployment_id: 3, name: "c", state: "stopped", reserved: 9 * GIB, actual: 0 },
    ]));
    expect(segments.map((s) => s.key)).toEqual(["dep-1", "dep-2", "external", "reserve", "free"]);
    expect(segments[0]).toMatchObject({ kind: "sleeping", bytes: 6 * GIB, label: "a · sleeping" });
    expect(segments[1]).toMatchObject({ kind: "deployment", bytes: 30 * GIB, label: "b" });
    expect(segments[3].bytes).toBe(12 * GIB);
    expect(segments[4].pct).toBeCloseTo(25);
  });

  it("keeps a deployment's colour when others disappear (colour follows the entity)", () => {
    const both = memorySegments(budget([
      { deployment_id: 1, name: "a", state: "ready", reserved: 10 * GIB, actual: 0 },
      { deployment_id: 2, name: "b", state: "ready", reserved: 10 * GIB, actual: 0 },
    ])).segments;
    const onlyB = memorySegments(budget([{ deployment_id: 2, name: "b", state: "ready", reserved: 10 * GIB, actual: 0 }])).segments;
    expect(onlyB.find((s) => s.key === "dep-2")!.color).toBe(both.find((s) => s.key === "dep-2")!.color);
  });

  it("clamps free at zero and reports over-commit", () => {
    const r = memorySegments(budget([], -5 * GIB));
    expect(r.segments.find((s) => s.key === "free")!.bytes).toBe(0);
    expect(r.overcommitted).toBe(5 * GIB);
  });
});
