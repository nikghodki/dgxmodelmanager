import { describe, expect, it } from "vitest";
import { adviceApplicable, canRemove, favouritePath, finiteSeries } from "./logic";

describe("deployment actions", () => {
  it("offers Remove only for stopped or failed deployments (I5)", () => {
    expect(canRemove("stopped")).toBe(true);
    expect(canRemove("failed")).toBe(true);
    for (const s of ["ready", "sleeping", "starting", "loading", "stopping", "unknown", "planned"]) expect(canRemove(s)).toBe(false);
  });
});

describe("advice apply (M8)", () => {
  const vllm = { engine: "vllm" };
  it("hides Apply for a GGUF model_ref on a vLLM deployment", () => {
    expect(adviceApplicable({ model_ref: "/data/gguf/Qwen3-8B-Q4_K_M.gguf" }, vllm)).toBe(false);
    expect(adviceApplicable({ model_ref: "unsloth/Qwen3-8B-GGUF/Qwen3-8B-Q4.GGUF" }, vllm)).toBe(false);
  });
  it("keeps Apply for patches and same-engine variants", () => {
    expect(adviceApplicable({ patch: { kv_cache_dtype: "fp8" } }, vllm)).toBe(true);
    expect(adviceApplicable({ model_ref: "Qwen/Qwen3-8B-FP8" }, vllm)).toBe(true);
    expect(adviceApplicable({}, vllm)).toBe(false);
  });
});

describe("favourites (M9)", () => {
  it("URL-encodes the ref so GGUF absolute paths survive the proxy", () => {
    expect(favouritePath("/data/gguf/x.gguf")).toBe("/planner/favourites/%2Fdata%2Fgguf%2Fx.gguf");
    expect(favouritePath("Qwen/Qwen3-8B")).toBe("/planner/favourites/Qwen%2FQwen3-8B");
  });
});

describe("charts (M12)", () => {
  it("replaces non-finite values with null", () => {
    const out = finiteSeries([{ ts: 1, a: NaN, b: 2, c: Infinity, d: null }], ["a", "b", "c", "d"]);
    expect(out).toEqual([{ ts: 1, a: null, b: 2, c: null, d: null }]);
  });
});

import { deploymentActions } from "./logic";

describe("deploymentActions", () => {
  const d = (state: string, sleep_capable = true) => ({ state, sleep_capable });
  it("viewers can only open", () => {
    for (const s of ["ready", "sleeping", "failed", "stopped"]) expect(deploymentActions(d(s), false)).toEqual(["open"]);
  });
  it("admins get state-appropriate actions", () => {
    expect(deploymentActions(d("ready"), true)).toEqual(["open", "sleep", "switch", "reconfigure", "stop"]);
    expect(deploymentActions(d("ready", false), true)).toEqual(["open", "switch", "reconfigure", "stop"]);
    expect(deploymentActions(d("sleeping"), true)).toEqual(["open", "wake", "switch", "reconfigure", "stop"]);
    expect(deploymentActions(d("loading"), true)).toEqual(["open", "stop"]);
    expect(deploymentActions(d("failed"), true)).toEqual(["open", "reconfigure", "start", "remove"]);
    expect(deploymentActions(d("stopped"), true)).toEqual(["open", "reconfigure", "start", "remove"]);
    expect(deploymentActions(d("unknown"), true)).toEqual(["open", "stop"]);
  });
  it("a system deployment (the decision model) is managed on its page: no switch, reconfigure or start", () => {
    const sys = (state: string) => ({ state, sleep_capable: false, system: true });
    expect(deploymentActions(sys("ready"), true)).toEqual(["open", "manage", "stop"]);
    expect(deploymentActions(sys("stopped"), true)).toEqual(["open", "manage", "remove"]);
    expect(deploymentActions(sys("failed"), true)).toEqual(["open", "manage", "remove"]);
    expect(deploymentActions(sys("loading"), true)).toEqual(["open", "manage", "stop"]);
    expect(deploymentActions(sys("ready"), false)).toEqual(["open"]);
  });
  it("a queued deployment (waiting for a start-up slot) can only be stopped (cancelled)", () => {
    expect(deploymentActions(d("queued"), true)).toEqual(["open", "stop"]);
    expect(deploymentActions(d("queued"), false)).toEqual(["open"]);
  });
});

import { submitState } from "./logic";

describe("submitState", () => {
  const fits = { fit: { fits: true } } as any;
  const short = { fit: { fits: false } } as any;
  it("blocks while there is no preview or while pending", () => {
    expect(submitState(null, [], false, "deploy")).toEqual({ disabled: true, label: "Deploy" });
    expect(submitState(fits, [], true, "deploy")).toEqual({ disabled: true, label: "Deploying…" });
  });
  it("requires a plan when the model does not fit", () => {
    expect(submitState(short, [], false, "deploy")).toEqual({ disabled: true, label: "Choose a plan to fit" });
    expect(submitState(short, [{ type: "stop", deployment_id: 1 }], false, "deploy")).toEqual({ disabled: false, label: "Apply plan and deploy" });
  });
  it("labels switch and reconfigure", () => {
    expect(submitState(fits, [], false, "switch").label).toBe("Switch");
    expect(submitState(fits, [], true, "reconfigure").label).toBe("Working… this can take minutes");
  });
});
