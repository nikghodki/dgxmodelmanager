import { describe, expect, it, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { SwitchDialog } from "./switch-dialog";
import * as apiModule from "@/api";

vi.mock("@/api", async () => {
  const actual = await vi.importActual("@/api");
  return { ...actual, api: vi.fn() };
});

describe("SwitchDialog", () => {
  it("clears plan when config changes, sends empty actions", async () => {
    const api = vi.mocked(apiModule.api);
    const d = {
      id: 1, name: "test", engine: "vllm", model_ref: "foo/bar", config: { max_model_len: 4096 },
      state: "ready", priority: 0,
    } as any;

    let previewCalls = 0;
    api.mockImplementation((url: string) => {
      if (url === "/inventory") return Promise.resolve([{ ref: "foo/bar", complete: true, kind: "llm", engines: ["vllm"], quant: "Q4", weights_bytes: 1e9 }]);
      if (url === "/deployments/preview") {
        previewCalls++;
        if (previewCalls === 1) {
          return Promise.resolve({
            estimate: { total: 1e9, weights: 1e9, draft_weights: 0, kv_cache: 0, state: 0, activations: 0, runtime: 0, margin: 0, notes: [], confidence: "measured" },
            fit: { fits: false, free_for_new: 5e8, headroom: -5e8 },
            plans: [{ id: "p1", title: "Stop X", detail: "Frees Y", new_required: 1e9, frees: 5e8, fits: true, actions: [{ type: "stop", deployment_id: 2 }] }],
          } as any);
        }
        return Promise.resolve({
          estimate: { total: 1e9, weights: 1e9, draft_weights: 0, kv_cache: 0, state: 0, activations: 0, runtime: 0, margin: 0, notes: [], confidence: "measured" },
          fit: { fits: true, free_for_new: 1.5e9, headroom: 5e8 },
          plans: [],
        } as any);
      }
      if (url === "/deployments/1/reconfigure") return Promise.resolve({ id: 1 });
      return Promise.reject(new Error("unexpected"));
    });

    const onClose = vi.fn();
    const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(
      <QueryClientProvider client={qc}>
        <SwitchDialog mode="reconfigure" deployment={d} onClose={onClose} />
      </QueryClientProvider>
    );

    await waitFor(() => expect(screen.getByText("Does not fit")).toBeInTheDocument());
    await userEvent.click(screen.getByRole("radio", { name: /Stop X/ }));
    const ctx = screen.getByLabelText("Context length (tokens)");
    await userEvent.clear(ctx);
    await userEvent.type(ctx, "8192");

    await waitFor(() => expect(screen.getByText("Fits")).toBeInTheDocument());

    const btn = screen.getByRole("button", { name: /Apply and restart/ });
    await userEvent.click(btn);

    await waitFor(() => {
      const call = api.mock.calls.find((c) => c[0] === "/deployments/1/reconfigure");
      expect(call).toBeTruthy();
      if (call && call[1]) {
        expect((call[1] as any).body.actions).toEqual([]);
      }
    });
  });
});
