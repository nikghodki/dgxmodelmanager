import { describe, expect, it } from "vitest";
import { actionLabel, insightsKey, insightsPath, openCount, parseAction, severityLabel, severityTone, statusLabel } from "./insights";

describe("insights helpers", () => {
  it.each([
    [{ type: "reconfigure", deployment_id: 3, patch: { kv_cache_dtype: "fp8" } }, true],
    [{ type: "reconfigure", deployment_id: 3, patch: {} }, false],
    // security review L6: a model switch or remote code is never an insight action (server FORBIDDEN_PATCH_KEYS)
    [{ type: "reconfigure", deployment_id: 3, patch: { model_ref: "evil/model" } }, false],
    [{ type: "reconfigure", deployment_id: 3, patch: { kv_cache_dtype: "fp8", trust_remote_code: true } }, false],
    [{ type: "reconfigure", deployment_id: "3", patch: { a: 1 } }, false],
    [{ type: "schedule", deployment_id: 2, proposal: { at_local: "22:00" } }, true],
    [{ type: "threshold", feature: "plan", value: 0.7 }, true],
    [{ type: "threshold", feature: "plan", value: 7 }, false],
    [{ type: "copy", text: "x" }, true],
    [{ type: "copy", text: "" }, false],
    [{ type: "open", path: "/litellm" }, true],
    [{ type: "open", path: "//evil.example" }, false],
    [{ type: "open", path: "https://evil.example" }, false],
    [{ type: "exec", cmd: "x" }, false],
    [null, false], ["reconfigure", false], [{ type: "copy", text: "x", label: "" }, false],
  ])("parseAction(%j) valid=%s", (a, ok) => expect(parseAction(a) !== null).toBe(ok));

  it("labels actions, severities and statuses", () => {
    expect(actionLabel({ type: "threshold", feature: "plan", value: 0.7 })).toBe("Set threshold…");
    expect(actionLabel({ type: "copy", text: "x", label: "Copy snippet" })).toBe("Copy snippet");
    expect(severityLabel("critical")).toBe("Critical");
    expect(severityTone("bogus")).toContain("neutral");
    expect(statusLabel("snoozed")).toBe("Snoozed");
  });

  it("builds paths and keys under one prefix", () => {
    expect(insightsPath()).toBe("/insights");
    expect(insightsPath({ status: "open,snoozed", subject: "deployment:4" })).toBe("/insights?status=open%2Csnoozed&subject=deployment%3A4");
    expect(insightsKey({ status: "open" })[0]).toBe("insights");
    expect(openCount([{ status: "open" }, { status: "snoozed" }, { status: "open" }] as any)).toBe(2);
    expect(openCount(undefined)).toBe(0);
  });
});
