import { afterEach, describe, expect, it, vi } from "vitest";
import { renderHook } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { useLiveUpdates } from "./stream";

class FakeES {
  static last: FakeES;
  listeners: Record<string, ((e: { data: string }) => void)[]> = {};
  constructor() { FakeES.last = this; }
  addEventListener(name: string, fn: (e: { data: string }) => void) { (this.listeners[name] ??= []).push(fn); }
  emit(name: string, data: unknown) { for (const fn of this.listeners[name] ?? []) fn({ data: JSON.stringify(data) }); }
  close() {}
}

afterEach(() => vi.unstubAllGlobals());

function setup() {
  vi.stubGlobal("EventSource", FakeES);
  const qc = new QueryClient();
  const spy = vi.spyOn(qc, "invalidateQueries");
  renderHook(() => useLiveUpdates(true), { wrapper: ({ children }) => <QueryClientProvider client={qc}>{children}</QueryClientProvider> });
  return { qc, spy };
}

describe("useLiveUpdates — decision model topics", () => {
  it("refreshes a deployment's diagnosis on the diagnosis topic", () => {
    const { spy } = setup();
    FakeES.last.emit("diagnosis", { deployment_id: 7 });
    expect(spy).toHaveBeenCalledWith({ queryKey: ["diagnosis", 7] });
  });

  it("keeps the image build progress and refreshes the decider status when it ends", () => {
    const { qc, spy } = setup();
    FakeES.last.emit("decider", { phase: "building", message: "Step 3/7" });
    expect(qc.getQueryData(["decider-build"])).toEqual({ phase: "building", message: "Step 3/7" });
    expect(spy).not.toHaveBeenCalledWith({ queryKey: ["decider-status"] });
    FakeES.last.emit("decider", { phase: "started", message: "building" }); // the chip switches to Building
    expect(spy).toHaveBeenCalledWith({ queryKey: ["decider-status"] });
    spy.mockClear();
    FakeES.last.emit("decider", { phase: "done", message: "built" });
    expect(spy).toHaveBeenCalledWith({ queryKey: ["decider-status"] });
  });

  it("refreshes the decider status when the decider deployment changes (and only then)", () => {
    const { spy } = setup();
    FakeES.last.emit("deployment", { id: 4, engine: "vllm", system: false, state: "ready" });
    expect(spy).not.toHaveBeenCalledWith({ queryKey: ["decider-status"] });
    FakeES.last.emit("deployment", { id: 9, engine: "decision", system: true, state: "loading" });
    expect(spy).toHaveBeenCalledWith({ queryKey: ["decider-status"] });
  });

  it("refreshes the decider status when a deployment is removed", () => {
    const { spy } = setup();
    FakeES.last.emit("deployment_removed", { id: 9 });
    expect(spy).toHaveBeenCalledWith({ queryKey: ["decider-status"] });
  });
});
