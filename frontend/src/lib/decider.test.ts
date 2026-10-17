import { describe, expect, it } from "vitest";
import { decisionChip, deciderHealth, fmtConfidence, kindLabel, parserValueLabel, sortPlans, sourceLabel, suggestionState } from "./decider";
import { DIAGNOSIS_CAUSES, causeLabel, featureLabel, splitAdvice, workloadLabel, workloadRefetchMs } from "./decider-labels";

describe("decider helpers", () => {
  it("summarises the decider's health: deployment states map to their own words, never a false Error", () => {
    const s = (over: Record<string, unknown>) => ({ enabled: true, healthy: false, state: null, image: { present: true, building: false, error: null }, ...over }) as any;
    expect(deciderHealth(s({ healthy: true, state: "ready" }))).toEqual({ state: "healthy", label: "Healthy" });
    for (const st of ["planned", "queued", "starting", "loading", "unknown"]) expect(deciderHealth(s({ state: st }))).toEqual({ state: "starting", label: "Starting" });
    expect(deciderHealth(s({ state: "stopping" }))).toEqual({ state: "stopping", label: "Stopping" });
    expect(deciderHealth(s({ state: "sleeping" }))).toEqual({ state: "sleeping", label: "Sleeping" });
    expect(deciderHealth(s({ state: "stopped" }))).toEqual({ state: "stopped", label: "Stopped" });
    const building = { present: false, building: true, error: null };
    expect(deciderHealth(s({ state: null, image: building }))).toEqual({ state: "building", label: "Building" });
    expect(deciderHealth(s({ enabled: false, image: building })).state).toBe("building");
    expect(deciderHealth(s({ healthy: true, state: "ready", image: building })).state).toBe("healthy"); // rebuild while serving
    expect(deciderHealth(s({ enabled: false, healthy: true, state: "ready" })).state).toBe("disabled");
    for (const st of ["failed", "ready", null]) expect(deciderHealth(s({ state: st })).state).toBe("error");
  });

  it("formats confidence as a whole percentage", () => {
    expect(fmtConfidence(0.923)).toBe("92%");
    expect(fmtConfidence(1)).toBe("100%");
    expect(fmtConfidence(null)).toBe("—");
    expect(fmtConfidence(undefined)).toBe("—");
  });

  it("labels sources", () => {
    expect(sourceLabel("rules")).toBe("Rules");
    expect(sourceLabel("model")).toBe("AI");
    expect(sourceLabel("none")).toBe("None");
  });

  it("builds a chip: confidence only for model decisions", () => {
    expect(decisionChip("model", 0.92)).toBe("92% · AI");
    expect(decisionChip("rules", 1)).toBe("Rules");
    expect(decisionChip("model", null)).toBe("AI");
  });

  it("keeps the diagnosis taxonomy in sync with the backend (15 causes)", () => {
    expect(Object.keys(DIAGNOSIS_CAUSES)).toEqual([
      "gpu_oom", "kv_cache_too_small", "context_too_long", "unsupported_architecture", "trust_remote_code_required",
      "auth_gated", "bad_parser_flag", "tokenizer_or_template", "port_in_use", "offline_missing_files", "driver_mismatch",
      "disk_full", "integrity_check_failed", "api_key_missing", "other",
    ]);
    expect(causeLabel("integrity_check_failed")).toBe("Decision model files fail the integrity check");
    expect(causeLabel("api_key_missing")).toBe("Decision model key not delivered");
    expect(causeLabel("context_too_long")).toBe("Context longer than the model supports");
    expect(causeLabel("gpu_oom")).toBe("Out of GPU memory");
    expect(causeLabel("something_new")).toBe("something new");
    expect(causeLabel("something_new", "Server label")).toBe("Server label");
  });

  it("labels workloads, kinds, features and parser values", () => {
    expect(workloadLabel("agent_tool_use")).toBe("Agent / tool use");
    expect(workloadLabel("custom_x")).toBe("custom x");
    expect(kindLabel("vision-llm")).toBe("Vision LLM");
    expect(kindLabel("decision")).toBe("Decision");
    expect(kindLabel("llm")).toBe("LLM");
    expect(featureLabel("parser_tool")).toBe("Tool-call parser");
    expect(featureLabel("diagnosis")).toBe("Failure diagnosis");
    expect(parserValueLabel(null)).toBe("none");
    expect(parserValueLabel("hermes")).toBe("hermes");
  });

  it("sorts the recommended plan first and keeps the others in order", () => {
    const p = (id: string, recommended = false) => ({ id, recommended }) as any;
    expect(sortPlans([p("a"), p("b"), p("c", true)]).map((x) => x.id)).toEqual(["c", "a", "b"]);
    expect(sortPlans([p("a"), p("b")]).map((x) => x.id)).toEqual(["a", "b"]);
  });

  describe("suggestionState", () => {
    const s = (over: Record<string, any> = {}) => ({ value: "hermes", source: "model", confidence: 0.95, auto_apply: true, alternatives: [], ...over }) as any;
    it("auto-fills an empty, untouched field when auto_apply", () => {
      expect(suggestionState(s(), undefined, { touched: false, autoFilled: false })).toBe("autofill");
    });
    it("reports filled once auto-filled and the value is still the suggestion", () => {
      expect(suggestionState(s(), "hermes", { touched: false, autoFilled: true })).toBe("filled");
    });
    it("never overwrites a user-typed value: offers Apply instead", () => {
      expect(suggestionState(s(), "llama3_json", { touched: true, autoFilled: false })).toBe("apply");
      expect(suggestionState(s(), undefined, { touched: true, autoFilled: false })).toBe("apply");
    });
    it("offers Apply when below the auto-apply bar", () => {
      expect(suggestionState(s({ auto_apply: false }), undefined, { touched: false, autoFilled: false })).toBe("apply");
    });
    it("is quiet when the field already has the suggested value", () => {
      expect(suggestionState(s({ auto_apply: false }), "hermes", { touched: true, autoFilled: false })).toBe("matches");
      expect(suggestionState(s({ value: null }), undefined, { touched: false, autoFilled: false })).toBe("matches");
    });
    it("offers to clear when the suggestion is none but a parser is set", () => {
      expect(suggestionState(s({ value: null }), "hermes", { touched: false, autoFilled: false })).toBe("apply");
    });
    it("does not auto-fill a non-empty field (e.g. the old model's parser)", () => {
      expect(suggestionState(s(), "qwen3", { touched: false, autoFilled: false })).toBe("apply");
    });
  });

  it("splits advice into fits-the-workload and the rest only when a workload is known", () => {
    const a = (id: string, workload_fit?: string | null, relevance?: number) => ({ id, workload_fit, relevance }) as any;
    expect(splitAdvice([a("x", "boost", 2.9), a("y", null, 3)], true)).toEqual({ top: [a("x", "boost", 2.9)], rest: [a("y", null, 3)] });
    expect(splitAdvice([a("x", "boost"), a("y", "demote")], false)).toEqual({ top: [], rest: [a("x", "boost"), a("y", "demote")] });
    expect(splitAdvice([a("x"), a("y")], true)).toEqual({ top: [], rest: [a("x"), a("y")] });
    expect(splitAdvice([a("x", "boost"), a("y", "boost")], true)).toEqual({ top: [], rest: [a("x", "boost"), a("y", "boost")] });
  });

  it("re-polls the workload quickly until a label is known", () => {
    expect(workloadRefetchMs(undefined)).toBe(30000);
    expect(workloadRefetchMs(null)).toBe(30000);
    expect(workloadRefetchMs({ label: null, source: "none", confidence: 0.4 } as any)).toBe(30000);
    expect(workloadRefetchMs({ label: "idle", source: "rules", confidence: 1 } as any)).toBe(300000);
  });
});
