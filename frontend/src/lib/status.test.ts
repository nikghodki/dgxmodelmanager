import { describe, expect, it } from "vitest";
import { statusInfo } from "@/lib/status";

describe("statusInfo", () => {
  it.each([
    ["ready", "good", "Ready", false], ["synced", "good", "Synced", false], ["done", "good", "Done", false],
    ["conflict", "warning", "Name conflict", false],
    ["starting", "warning", "Starting", true], ["loading", "warning", "Loading", true], ["stopping", "warning", "Stopping", true],
    ["running", "warning", "Downloading", true], ["pending", "warning", "Pending", false],
    ["sleeping", "sleeping", "Sleeping", false],
    ["failed", "critical", "Failed", false], ["error", "critical", "Error", false],
    ["stopped", "neutral", "Stopped", false], ["unknown", "neutral", "Unknown", false], ["none", "neutral", "Not registered", false],
    ["queued", "neutral", "Queued", false], ["cancelled", "neutral", "Cancelled", false], ["disabled", "neutral", "Disabled", false],
    ["planned", "neutral", "Planned", false],
  ])("%s → %s", (state, kind, label, spin) => {
    expect(statusInfo(state)).toEqual({ kind, label, spin });
  });
  it("falls back to neutral with a capitalised label", () => {
    expect(statusInfo("weird_state")).toEqual({ kind: "neutral", label: "Weird state", spin: false });
    expect(statusInfo("")).toEqual({ kind: "neutral", label: "—", spin: false });
  });
});
