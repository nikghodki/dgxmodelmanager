import { afterEach, describe, expect, it, vi } from "vitest";
import { renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { useNodes } from "./use-nodes";
import * as apiModule from "@/api";

vi.mock("@/api", async () => {
  const actual = await vi.importActual<typeof import("@/api")>("@/api");
  return { ...actual, api: vi.fn() };
});

const node = (id: string) => ({ id, label: id, remote: id !== "local", reachable: true, stale: false, budget: null });
const calls = () => vi.mocked(apiModule.api).mock.calls.filter((c) => c[0] === "/nodes").length;

function run(nodes: any[]) {
  vi.mocked(apiModule.api).mockImplementation(() => Promise.resolve(nodes));
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return renderHook(() => useNodes({ pollWhenMulti: 30 }), { wrapper: ({ children }) => <QueryClientProvider client={qc}>{children}</QueryClientProvider> });
}

afterEach(() => vi.mocked(apiModule.api).mockReset());

describe("useNodes polling", () => {
  it("never polls a single-node install", async () => {
    const h = run([node("local")]);
    await waitFor(() => expect(h.result.current.isSuccess).toBe(true));
    await new Promise((r) => setTimeout(r, 150));
    expect(calls()).toBe(1);
  });

  it("polls once more than one node is configured", async () => {
    run([node("local"), node("dgx-2")]);
    await waitFor(() => expect(calls()).toBeGreaterThan(2));
  });
});
