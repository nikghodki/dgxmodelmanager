import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter } from "react-router-dom";
import { Catalog } from "./Catalog";
import * as auth from "@/auth";
import * as apiModule from "@/api";

vi.mock("@/auth");
vi.mock("@/api", async () => {
  const actual = await vi.importActual<typeof import("@/api")>("@/api");
  return { ...actual, api: vi.fn() };
});

const result = {
  repo_id: "org/model-a", downloads: 10, est_bytes: 1e9, quant: null, gated: false, downloaded: false,
  fit: { badge: "fits_now" }, support: { vllm: false, llamacpp: null }, llmfit: null,
};

function renderCatalog() {
  vi.mocked(auth.useIsAdmin).mockReturnValue(false);
  vi.mocked(apiModule.api).mockImplementation((url: string) => {
    if (url.startsWith("/catalog/search")) return Promise.resolve({ results: [result], free_for_new: 5e9, banner: null });
    if (url.startsWith("/catalog/details")) return new Promise(() => undefined);
    return Promise.reject(new Error(`unexpected ${url}`));
  });
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={qc}>
      <MemoryRouter future={{ v7_startTransition: true, v7_relativeSplatPath: true }}><Catalog /></MemoryRouter>
    </QueryClientProvider>,
  );
}

describe("Catalog", () => {
  it("opens a model's details from the keyboard (Tab + Enter)", async () => {
    renderCatalog();
    const user = userEvent.setup();
    await screen.findByText("org/model-a");
    const focused = () => document.activeElement?.tagName === "BUTTON" && document.activeElement.textContent === "org/model-a";
    for (let i = 0; i < 30 && !focused(); i++) await user.tab();
    expect(focused()).toBe(true);
    await user.keyboard("{Enter}");
    expect(await screen.findByRole("dialog")).toHaveTextContent("org/model-a");
  });

  it("marks engine support with distinct symbols and screen-reader text", async () => {
    renderCatalog();
    await screen.findByText("org/model-a");
    expect(screen.getByText("✗")).toBeInTheDocument();
    expect(screen.getByText("not supported")).toHaveClass("sr-only");
    expect(screen.getByText("unknown")).toHaveClass("sr-only");
  });

  it("shows the compatibility check in the details and before downloading", async () => {
    vi.mocked(apiModule.api).mockImplementation((url: string) => {
      if (url.startsWith("/catalog/search")) return Promise.resolve({ results: [result], free_for_new: 5e9, banner: null });
      if (url.startsWith("/catalog/details")) return Promise.resolve({ repo_id: "org/model-a", gated: false, sha: "s", files: [], architecture: "Qwen3ForCausalLM", max_context: 32768, quant: null, estimate: { total: 1e9 }, fit: { badge: "fits_now" }, gguf_files: [], support: { vllm: true, llamacpp: false }, llmfit: null, banner: null, variants: [], downloaded: false });
      if (url.startsWith("/catalog/compat")) return Promise.resolve({ repo: "org/model-a", sha: "s", architecture: "Qwen3ForCausalLM", gated: false, vllm: { verdict: "runs_with_flags", reasons: ["supported_architecture", "trust_remote_code_required"], flags: ["trust_remote_code"] }, llamacpp: { verdict: "unsupported", reasons: ["no_gguf"], flags: [] }, source: "rules", confidence: 1, decision_id: 1, cached: false, error: null, disabled: false });
      return Promise.reject(new Error(`unexpected ${url}`));
    });
    vi.mocked(auth.useIsAdmin).mockReturnValue(true);
    const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(<QueryClientProvider client={qc}><MemoryRouter future={{ v7_startTransition: true, v7_relativeSplatPath: true }}><Catalog /></MemoryRouter></QueryClientProvider>);
    await userEvent.click(await screen.findByRole("button", { name: "org/model-a" }));
    expect(await screen.findByText(/vLLM: runs with extra settings \(needs trust_remote_code/)).toBeInTheDocument();
    expect(screen.getByText(/Before downloading: vLLM: runs with extra settings · llama.cpp: not supported/)).toBeInTheDocument();
  });
});
