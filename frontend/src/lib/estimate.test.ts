import { describe, expect, it } from "vitest";
import { estimateParts } from "@/lib/estimate";

describe("estimateParts", () => {
  it("lists non-zero parts in fixed order with labels", () => {
    expect(estimateParts({ weights: 10, draft_weights: 0, kv_cache: 5, state: 0, activations: 1, runtime: 3, margin: 2 })).toEqual([
      { key: "weights", label: "Weights", bytes: 10 },
      { key: "kv_cache", label: "KV cache", bytes: 5 },
      { key: "activations", label: "Activations", bytes: 1 },
      { key: "runtime", label: "Runtime (CUDA, graphs)", bytes: 3 },
      { key: "margin", label: "Margin / calibration", bytes: 2 },
    ]);
  });
  it("handles an empty estimate (adopted/unknown deployments)", () => {
    expect(estimateParts({})).toEqual([]);
  });
});
