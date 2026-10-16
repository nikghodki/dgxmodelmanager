import { describe, expect, it } from "vitest";
import { buildCommands } from "@/components/app/commands";

const dep = (id: number, name: string, state: string, sleep_capable = true) => ({ id, name, state, sleep_capable } as any);
const art = (ref: string, complete = true, kind = "llm", engines = ["vllm"]) => ({ ref, complete, kind, engines } as any);
const labels = (xs: { label: string }[]) => xs.map((x) => x.label);

describe("buildCommands", () => {
  const deployments = [dep(1, "qwen", "ready"), dep(2, "coder", "sleeping"), dep(3, "old", "stopped")];
  const artifacts = [art("Qwen/Qwen3-8B"), art("org/partial", false), art("openai/whisper", true, "other")];

  it("gives viewers navigation only — no admin pages or actions", () => {
    const c = buildCommands({ isAdmin: false, deployments, artifacts });
    expect(labels(c)).toContain("Open qwen");
    expect(c.find((x) => x.label === "Processes")!.action).toEqual({ type: "navigate", to: "/processes" });
    expect(labels(c)).not.toContain("Users");
    expect(labels(c)).not.toContain("Deploy a model");
    expect(c.some((x) => x.action.type === "deployment")).toBe(false);
    expect(c.some((x) => x.group === "Models")).toBe(false);
  });

  it("gives admins state-appropriate deployment actions and deployable models", () => {
    const c = buildCommands({ isAdmin: true, deployments, artifacts });
    const l = labels(c);
    expect(l).toEqual(expect.arrayContaining(["Users", "Audit log", "Deploy a model", "Sleep qwen", "Stop qwen", "Wake coder", "Stop coder"]));
    expect(l).not.toContain("Wake qwen");
    expect(l).not.toContain("Stop old");
    expect(c.find((x) => x.label === "Stop qwen")!.destructive).toBe(true);
    expect(l).toContain("Deploy Qwen/Qwen3-8B");
    expect(l).not.toContain("Deploy org/partial");
    expect(l).not.toContain("Deploy openai/whisper");
    expect(c.find((x) => x.label === "Deploy Qwen/Qwen3-8B")!.action).toEqual({ type: "navigate", to: "/deploy?model=Qwen%2FQwen3-8B" });
  });

  it("carries the deployment state so stop can require typed confirmation", () => {
    const c = buildCommands({ isAdmin: true, deployments, artifacts });
    expect(c.find((x) => x.label === "Stop qwen")!.action).toEqual({ type: "deployment", op: "stop", id: 1, name: "qwen", state: "ready" });
    expect(c.find((x) => x.label === "Stop coder")!.action).toEqual({ type: "deployment", op: "stop", id: 2, name: "coder", state: "sleeping" });
  });

  it("derives per-deployment actions from deploymentActions (no sleep without sleep support, no stop when failed)", () => {
    const c = buildCommands({ isAdmin: true, deployments: [dep(4, "nosleep", "ready", false), dep(5, "bad", "failed")], artifacts: [] });
    const l = labels(c);
    expect(l).toContain("Stop nosleep");
    expect(l).not.toContain("Sleep nosleep");
    expect(l).not.toContain("Stop bad");
  });
});
