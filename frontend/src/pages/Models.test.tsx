import { describe, expect, it, vi } from "vitest";
import { render, screen, within } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter } from "react-router-dom";
import { Models } from "./Models";
import * as auth from "@/auth";
import * as apiModule from "@/api";

vi.mock("@/auth");
vi.mock("@/api", async () => {
  const actual = await vi.importActual<typeof import("@/api")>("@/api");
  return { ...actual, api: vi.fn() };
});

const art = (ref: string, complete: boolean) => ({
  ref, source: "hf", revision: null, format: "safetensors", weights_bytes: 1e9, size_bytes: 1e9, complete, kind: "llm",
  architecture: "Qwen", quant: null, problems: [], max_context: 4096, is_moe: false, num_layers: 1, confident: true, has_mtp: false, engines: ["vllm"],
});

describe("Models", () => {
  it("shows a single, consistent status label per model", async () => {
    vi.mocked(auth.useIsAdmin).mockReturnValue(false);
    vi.mocked(apiModule.api).mockResolvedValue([art("org/done", true), art("org/partial", false)]);
    const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(
      <QueryClientProvider client={qc}>
        <MemoryRouter future={{ v7_startTransition: true, v7_relativeSplatPath: true }}><Models /></MemoryRouter>
      </QueryClientProvider>,
    );
    const done = (await screen.findByText("org/done")).closest("tr")!;
    const partial = screen.getByText("org/partial").closest("tr")!;
    const statusOf = (tr: HTMLElement) => within(tr).getAllByRole("cell").at(-2)!.textContent;
    expect(statusOf(done)).toBe("Complete");
    expect(statusOf(partial)).toBe("Incomplete");
    expect(within(done).getAllByRole("cell").at(-2)!.querySelector(".text-status-good")).not.toBeNull();
    expect(within(partial).getAllByRole("cell").at(-2)!.querySelector(".text-status-warning")).not.toBeNull();
  });
});
