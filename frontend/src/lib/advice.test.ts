import { describe, expect, it } from "vitest";
import type { ConfigAdvice, RemoteCodeCheck } from "@/types";
import { changedKeys, compatText, defaultGoal, fillUntouched, remoteCodeWarns, shownOptions } from "./advice";

const opt = (key: string) => ({ key, config: {}, summary: key, required_bytes: 1, headroom_bytes: 1, context: 1, seqs: 1, kv_cache_dtype: "auto", prefix_caching: true, kv_pool_tokens: 1 });

describe("advice helpers", () => {
  it("fills only untouched fields and never overwrites typed values", () => {
    const r = fillUntouched({ max_model_len: 4096, max_num_seqs: 3 }, { max_model_len: 32768, max_num_seqs: 128, kv_cache_dtype: "fp8" }, ["max_num_seqs"]);
    expect(r.config).toEqual({ max_model_len: 32768, max_num_seqs: 3, kv_cache_dtype: "fp8" });
    expect(r.changed).toEqual(["max_model_len", "kv_cache_dtype"]);
  });
  it("lists changed keys", () => {
    expect(changedKeys({ a: 1, b: 2 }, { a: 1, b: 3, c: 4 }).sort()).toEqual(["b", "c"]);
  });
  it("shows the recommended option, else the top three", () => {
    const base: ConfigAdvice = { goal: "chat", options: [opt("a"), opt("b"), opt("c"), opt("d")], top: ["c", "a", "d"], recommended: null, source: "model", confidence: 0.9, decision_id: 1, free_bytes: 1, reason: null, disabled: false };
    expect(shownOptions({ ...base, recommended: "b" }).recommended?.key).toBe("b");
    expect(shownOptions({ ...base, recommended: null }).top.map((o) => o.key)).toEqual(["c", "a", "d"]);
    expect(shownOptions(null)).toEqual({ recommended: null, top: [] });
  });
  it("templates compatibility text", () => {
    expect(compatText("vLLM", { verdict: "runs_with_flags", reasons: ["supported_architecture", "trust_remote_code_required"], flags: ["trust_remote_code"] }))
      .toBe("vLLM: runs with extra settings (needs trust_remote_code (custom code))");
    expect(compatText("llama.cpp", undefined)).toBe("llama.cpp: unknown");
  });
  it("defaults the goal from the tool parser", () => {
    expect(defaultGoal({ tool_call_parser: "hermes" })).toBe("agent_tool_use");
    expect(defaultGoal({})).toBe("chat");
  });
  it("warns on findings or an unusual label, never clears on routine", () => {
    const c: RemoteCodeCheck = { verdict: "no_findings", findings: [], source: "rules", confidence: null, model_label: null, decision_id: 1, scanned_files: 3, truncated: false, error: null, disabled: false, remote_code: true };
    expect(remoteCodeWarns(c)).toBe(false);
    expect(remoteCodeWarns({ ...c, findings: [{ file: "m.py", line: 1, kind: "network_import", detail: "socket" }] })).toBe(true);
    expect(remoteCodeWarns({ ...c, verdict: "unusual" })).toBe(true);
  });
});
