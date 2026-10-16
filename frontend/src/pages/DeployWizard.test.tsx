import { afterEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter } from "react-router-dom";
import { DeployWizard } from "./DeployWizard";
import * as auth from "@/auth";
import * as apiModule from "@/api";

vi.mock("@/auth");
vi.mock("@/api", async () => {
  const actual = await vi.importActual<typeof import("@/api")>("@/api");
  return { ...actual, api: vi.fn() };
});

const est = { total: 1e9, weights: 1e9, draft_weights: 0, kv_cache: 0, state: 0, activations: 0, runtime: 0, margin: 0, notes: [], confidence: "measured" };
const fitsPreview = { estimate: est, fit: { fits: true, free_for_new: 2e9, headroom: 1e9 }, plans: [] };
const shortPreview = { estimate: est, fit: { fits: false, free_for_new: 5e8, headroom: -5e8 }, plans: [] };
const artifact = { ref: "org/model-a", complete: true, kind: "llm", engines: ["vllm"], quant: "Q4", weights_bytes: 1e9, architecture: "Qwen", max_context: 32768, problems: [] };
const engines = [{ name: "vllm", supports_sleep: true, defaults: { max_model_len: 4096 }, image: null }];

function renderWizard() {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } });
  return render(
    <QueryClientProvider client={qc}>
      <MemoryRouter initialEntries={["/deploy?model=org%2Fmodel-a"]} future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
        <DeployWizard />
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

afterEach(() => { vi.mocked(apiModule.api).mockReset(); });

describe("DeployWizard", () => {
  it("redirects viewers to home", () => {
    vi.mocked(auth.useIsAdmin).mockReturnValue(false);
    const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    const { container } = render(
      <QueryClientProvider client={qc}>
        <MemoryRouter future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
          <DeployWizard />
        </MemoryRouter>
      </QueryClientProvider>
    );
    expect(container.textContent).toBe("");
  });

  it("renders for admins", () => {
    vi.mocked(auth.useIsAdmin).mockReturnValue(true);
    vi.mocked(apiModule.api).mockImplementation(() => new Promise(() => undefined));
    renderWizard();
    expect(screen.getByText("Deploy a model")).toBeInTheDocument();
  });

  it("re-plans after an insufficient_memory 409 without looping, and a chosen plan sticks", async () => {
    vi.mocked(auth.useIsAdmin).mockReturnValue(true);
    const errors = vi.spyOn(console, "error");
    let previewCalls = 0;
    const plans = [{ id: "p1", title: "Stop other", detail: "Frees 1 GiB", new_required: 1e9, frees: 1e9, fits: true, actions: [{ type: "stop", deployment_id: 7 }], patch: null }];
    vi.mocked(apiModule.api).mockImplementation((url: string) => {
      if (url === "/inventory") return Promise.resolve([artifact]);
      if (url === "/engines") return Promise.resolve(engines);
      if (url === "/deployments/preview") { previewCalls++; return Promise.resolve(fitsPreview); }
      if (url === "/deployments") return Promise.reject(new apiModule.ApiError(409, {
        code: "insufficient_memory", message: "Not enough memory", fit: { fits: false, free_for_new: 5e8, headroom: -5e8 }, plans,
      }));
      return Promise.reject(new Error(`unexpected ${url}`));
    });
    renderWizard();
    await waitFor(() => expect(screen.getAllByText("Fits").length).toBeGreaterThan(0));
    const deployBtn = screen.getAllByRole("button", { name: "Deploy" })[0];
    await userEvent.click(deployBtn);
    await waitFor(() => expect(screen.getAllByRole("radio", { name: /Stop other/ }).length).toBeGreaterThan(0));
    const callsAfter409 = previewCalls;
    await userEvent.click(screen.getAllByRole("radio", { name: /Stop other/ })[0]);
    await new Promise((r) => setTimeout(r, 50));
    expect(screen.getAllByRole("radio", { name: /Stop other/ })[0]).toHaveAttribute("aria-checked", "true");
    expect(screen.getAllByRole("button", { name: "Apply plan and deploy" })[0]).toBeEnabled();
    expect(previewCalls).toBe(callsAfter409);
    expect(errors.mock.calls.map((c) => String(c[0])).join("\n")).not.toMatch(/Maximum update depth/);
    errors.mockRestore();
  });

  it("ignores a stale preview that resolves after a newer one", async () => {
    vi.mocked(auth.useIsAdmin).mockReturnValue(true);
    const resolvers: ((v: unknown) => void)[] = [];
    vi.mocked(apiModule.api).mockImplementation((url: string) => {
      if (url === "/inventory") return Promise.resolve([artifact]);
      if (url === "/engines") return Promise.resolve(engines);
      if (url === "/deployments/preview") return new Promise((res) => resolvers.push(res));
      return Promise.reject(new Error(`unexpected ${url}`));
    });
    renderWizard();
    await waitFor(() => expect(resolvers.length).toBe(1));
    const ctx = screen.getByLabelText("Context length (tokens)");
    fireEvent.change(ctx, { target: { value: "8192" } });
    await waitFor(() => expect(resolvers.length).toBe(2));
    resolvers[1](fitsPreview);
    await waitFor(() => expect(screen.getAllByText("Fits").length).toBeGreaterThan(0));
    resolvers[0](shortPreview);
    await new Promise((r) => setTimeout(r, 20));
    expect(screen.queryByText("Does not fit")).toBeNull();
    expect(screen.getAllByText("Fits").length).toBeGreaterThan(0);
  });

  it("labels the model and engine pickers with their visible label and current value", async () => {
    vi.mocked(auth.useIsAdmin).mockReturnValue(true);
    vi.mocked(apiModule.api).mockImplementation((url: string) => {
      if (url === "/inventory") return Promise.resolve([artifact]);
      if (url === "/engines") return Promise.resolve(engines);
      if (url === "/deployments/preview") return Promise.resolve(fitsPreview);
      return Promise.reject(new Error(`unexpected ${url}`));
    });
    renderWizard();
    expect(await screen.findByRole("combobox", { name: /^Model\s+org\/model-a$/ })).toBeInTheDocument();
    expect(await screen.findByRole("combobox", { name: /^Engine\s+vLLM$/ })).toBeInTheDocument();
  });

  it("sends a finite priority even when the number input holds a non-integer", async () => {
    vi.mocked(auth.useIsAdmin).mockReturnValue(true);
    const api = vi.mocked(apiModule.api);
    api.mockImplementation((url: string) => {
      if (url === "/inventory") return Promise.resolve([artifact]);
      if (url === "/engines") return Promise.resolve(engines);
      if (url === "/deployments/preview") return Promise.resolve(fitsPreview);
      if (url === "/deployments") return new Promise(() => undefined);
      return Promise.reject(new Error(`unexpected ${url}`));
    });
    renderWizard();
    await waitFor(() => expect(screen.getAllByText("Fits").length).toBeGreaterThan(0));
    fireEvent.change(screen.getByLabelText("Priority"), { target: { value: ".5" } });
    await userEvent.click(screen.getAllByRole("button", { name: "Deploy" })[0]);
    const call = api.mock.calls.find((c) => c[0] === "/deployments");
    expect((call![1] as any).body.priority).toBe(0);
  });

  it("names the LiteLLM model, explains the single endpoint and validates the name", async () => {
    vi.mocked(auth.useIsAdmin).mockReturnValue(true);
    vi.mocked(apiModule.api).mockImplementation((url: string) => {
      if (url === "/inventory") return Promise.resolve([artifact]);
      if (url === "/engines") return Promise.resolve(engines);
      if (url === "/deployments/preview") return Promise.resolve({ ...fitsPreview, served_name: "model-a" });
      if (url === "/integrations/litellm") return Promise.resolve({ endpoint: "https://llm.example.com/v1" });
      return Promise.reject(new Error(`unexpected ${url}`));
    });
    renderWizard();
    const input = await screen.findByLabelText("Model name (LiteLLM)");
    await waitFor(() => expect(screen.getAllByText("Fits").length).toBeGreaterThan(0));
    expect(screen.getByText(/Defaults to the served name/)).toBeInTheDocument();
    expect(await screen.findByText("https://llm.example.com/v1")).toBeInTheDocument();
    expect(screen.getByText("model-a")).toBeInTheDocument();
    await userEvent.type(input, "bad name");
    expect(screen.getByText(/no spaces/)).toBeInTheDocument();
    expect(screen.getAllByRole("button", { name: "Deploy" })[0]).toBeDisabled();
    await userEvent.clear(input);
    await userEvent.type(input, "team-chat");
    expect(screen.getByText("team-chat")).toBeInTheDocument();
    expect(screen.getAllByRole("button", { name: "Deploy" })[0]).toBeEnabled();
  });
});
