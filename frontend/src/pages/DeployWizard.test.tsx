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
      if (url === "/integrations/litellm?probe=false") return Promise.resolve({ endpoint: "https://llm.example.com/v1" });
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

  it("links to the deployment that already uses the model name", async () => {
    vi.mocked(auth.useIsAdmin).mockReturnValue(true);
    vi.mocked(apiModule.api).mockImplementation((url: string) => {
      if (url === "/inventory") return Promise.resolve([artifact]);
      if (url === "/engines") return Promise.resolve(engines);
      if (url === "/deployments/preview") return Promise.resolve({ ...fitsPreview, served_name: "model-a" });
      if (url === "/deployments") return Promise.reject(new apiModule.ApiError(409, {
        code: "model_name_taken", message: "the model name model-a is already used by deployment other (ready); choose a different model name (alias)",
        holder: { id: 7, name: "other", state: "ready" },
      }));
      return Promise.reject(new Error(`unexpected ${url}`));
    });
    renderWizard();
    await waitFor(() => expect(screen.getAllByText("Fits").length).toBeGreaterThan(0));
    await userEvent.click(screen.getAllByRole("button", { name: "Deploy" })[0]);
    expect((await screen.findAllByRole("link", { name: "Open other" }))[0]).toHaveAttribute("href", "/deployments/7");
    expect(screen.getAllByText(/load-balance/)).toHaveLength(1);
    await userEvent.click(screen.getAllByRole("button", { name: "Use model-a-model-a" })[0]);
    expect(screen.getByLabelText("Model name (LiteLLM)")).toHaveValue("model-a-model-a");
    expect(screen.getByLabelText("Model name (LiteLLM)")).toHaveFocus();
  });

  it("explains a model name LiteLLM already has from an entry DGXMM did not create", async () => {
    vi.mocked(auth.useIsAdmin).mockReturnValue(true);
    vi.mocked(apiModule.api).mockImplementation((url: string) => {
      if (url === "/inventory") return Promise.resolve([artifact]);
      if (url === "/engines") return Promise.resolve(engines);
      if (url === "/deployments/preview") return Promise.resolve({ ...fitsPreview, served_name: "model-a" });
      if (url === "/deployments") return Promise.reject(new apiModule.ApiError(409, {
        code: "model_name_taken",
        message: "LiteLLM already has a model named model-a that DGXMM didn't create (static config or another tool). Pick another name, or remove that entry from LiteLLM first.",
        holder: null, external: { host: "old-engine", static: true },
      }));
      return Promise.reject(new Error(`unexpected ${url}`));
    });
    renderWizard();
    await waitFor(() => expect(screen.getAllByText("Fits").length).toBeGreaterThan(0));
    await userEvent.click(screen.getAllByRole("button", { name: "Deploy" })[0]);
    expect((await screen.findAllByText(/LiteLLM already has a model named model-a that DGXMM didn't create/)).length).toBeGreaterThan(0);
    expect(screen.queryByRole("link", { name: /^Open / })).toBeNull();
    await userEvent.click(screen.getAllByRole("button", { name: "Use model-a-model-a" })[0]);
    expect(screen.getByLabelText("Model name (LiteLLM)")).toHaveValue("model-a-model-a");
  });
});

describe("DeployWizard — parser suggestions", () => {
  const sug = (over: Record<string, any> = {}) => ({ value: "hermes", source: "model", confidence: 0.92, auto_apply: true, alternatives: [], ...over });
  function mock(suggest: () => Promise<any>, previews: any[] = []) {
    vi.mocked(auth.useIsAdmin).mockReturnValue(true);
    vi.mocked(apiModule.api).mockImplementation((url: string, opts?: any) => {
      if (url === "/inventory") return Promise.resolve([artifact]);
      if (url === "/engines") return Promise.resolve(engines);
      if (url === "/deployments/preview") { previews.push(opts.body); return Promise.resolve(fitsPreview); }
      if (url === "/decider/suggest-config") return suggest();
      return new Promise(() => undefined);
    });
  }
  const sugCalls = () => vi.mocked(apiModule.api).mock.calls.filter(([u]) => u === "/decider/suggest-config");

  it("asks for suggestions for the chosen model and engine, auto-fills an empty field and can undo", async () => {
    const previews: any[] = [];
    mock(() => Promise.resolve({ tool_call_parser: sug(), reasoning_parser: null }), previews);
    renderWizard();
    expect(await screen.findByText("Suggested: hermes · 92% · AI")).toBeInTheDocument();
    expect(sugCalls()[0][1]).toEqual({ method: "POST", body: { model_ref: "org/model-a", engine: "vllm" } });
    expect(screen.getByLabelText("Tool-call parser")).toHaveValue("hermes");
    expect(screen.getByText(/Filled automatically/)).toBeInTheDocument();
    await waitFor(() => expect(previews.some((b) => b.config.tool_call_parser === "hermes")).toBe(true));
    // no reasoning-parser suggestion → no chip for it
    expect(screen.queryByLabelText("Reasoning parser")).toBeNull();
    await userEvent.click(screen.getByRole("button", { name: /Undo: clear the automatic Tool-call parser/ }));
    expect(screen.getByLabelText("Tool-call parser")).toHaveValue("");
    await new Promise((r) => setTimeout(r, 20));
    expect(screen.getByLabelText("Tool-call parser")).toHaveValue(""); // not re-filled after undo
    expect(screen.getByRole("button", { name: /Apply suggested Tool-call parser: hermes/ })).toBeInTheDocument();
  });

  it("offers Apply instead of auto-filling below the auto-apply bar", async () => {
    mock(() => Promise.resolve({ tool_call_parser: null, reasoning_parser: sug({ value: "qwen3", confidence: 0.72, auto_apply: false }) }));
    renderWizard();
    expect(await screen.findByText("Suggested: qwen3 · 72% · AI")).toBeInTheDocument();
    expect(screen.getByLabelText("Reasoning parser")).toHaveValue("");
    await userEvent.click(screen.getByRole("button", { name: /Apply suggested Reasoning parser: qwen3/ }));
    expect(screen.getByLabelText("Reasoning parser")).toHaveValue("qwen3");
    expect(screen.queryByRole("button", { name: /Apply suggested Reasoning parser/ })).toBeNull();
  });

  it("never overwrites a value the user typed", async () => {
    let resolve: (v: any) => void = () => undefined;
    mock(() => new Promise((r) => { resolve = r; }));
    renderWizard();
    await userEvent.click(await screen.findByRole("button", { name: "Show advanced settings" }));
    const input = await screen.findByLabelText("Tool-call parser");
    await userEvent.type(input, "llama3_json");
    resolve({ tool_call_parser: sug({ source: "rules", confidence: 1 }), reasoning_parser: null });
    expect(await screen.findByText("Suggested: hermes · Rules")).toBeInTheDocument();
    expect(screen.getByLabelText("Tool-call parser")).toHaveValue("llama3_json");
    expect(screen.getByRole("button", { name: /Apply suggested Tool-call parser: hermes/ })).toBeInTheDocument();
  });

  it("shows nothing when the endpoint errors or has no suggestion", async () => {
    mock(() => Promise.reject(new apiModule.ApiError(503, { code: "decider_unavailable", message: "off" })));
    const { unmount } = renderWizard();
    await waitFor(() => expect(sugCalls().length).toBe(1));
    await new Promise((r) => setTimeout(r, 20));
    expect(screen.queryByText(/Suggested:/)).toBeNull();
    expect(screen.queryByRole("alert")).toBeNull();
    unmount();
    mock(() => Promise.resolve(null));
    renderWizard();
    await waitFor(() => expect(sugCalls().length).toBe(2));
    await new Promise((r) => setTimeout(r, 20));
    expect(screen.queryByText(/Suggested:/)).toBeNull();
  });
});

describe("DeployWizard — parser suggestions (review fixes)", () => {
  const sug = (over: Record<string, any> = {}) => ({ value: "hermes", source: "model", confidence: 0.92, auto_apply: true, alternatives: [], ...over });
  it("announces an auto-fill politely and skips a 'none' suggestion that already matches", async () => {
    vi.mocked(auth.useIsAdmin).mockReturnValue(true);
    vi.mocked(apiModule.api).mockImplementation((url: string) => {
      if (url === "/inventory") return Promise.resolve([artifact]);
      if (url === "/engines") return Promise.resolve(engines);
      if (url === "/deployments/preview") return Promise.resolve(fitsPreview);
      if (url === "/decider/suggest-config") return Promise.resolve({ tool_call_parser: sug(), reasoning_parser: sug({ value: null, auto_apply: false }) });
      return new Promise(() => undefined);
    });
    renderWizard();
    expect(await screen.findByText(/Filled automatically/)).toBeInTheDocument();
    expect(screen.getByText(/Filled automatically/).closest("[aria-live='polite']")).not.toBeNull();
    expect(screen.queryByText(/Suggested: none/)).toBeNull();
    expect(screen.queryByLabelText("Reasoning parser")).toBeNull(); // stays hidden with advanced settings
  });

  it("keeps a chosen plan when a suggestion auto-fills later", async () => {
    vi.mocked(auth.useIsAdmin).mockReturnValue(true);
    let resolve: (v: any) => void = () => undefined;
    const plans = [{ id: "p1", title: "Stop other", detail: "Frees 1 GiB", new_required: 1e9, frees: 1e9, fits: true, actions: [{ type: "stop", deployment_id: 7 }], patch: null }];
    vi.mocked(apiModule.api).mockImplementation((url: string) => {
      if (url === "/inventory") return Promise.resolve([artifact]);
      if (url === "/engines") return Promise.resolve(engines);
      if (url === "/deployments/preview") return Promise.resolve({ ...shortPreview, plans });
      if (url === "/decider/suggest-config") return new Promise((r) => { resolve = r; });
      return new Promise(() => undefined);
    });
    renderWizard();
    const radio = (await screen.findAllByRole("radio", { name: /Stop other/ }))[0];
    await userEvent.click(radio);
    expect(screen.getAllByRole("radio", { name: /Stop other/ })[0]).toHaveAttribute("aria-checked", "true");
    resolve({ tool_call_parser: sug(), reasoning_parser: null });
    expect(await screen.findByText(/Filled automatically/)).toBeInTheDocument();
    await new Promise((r) => setTimeout(r, 400)); // the debounced preview re-runs
    expect(screen.getAllByRole("radio", { name: /Stop other/ })[0]).toHaveAttribute("aria-checked", "true");
  });
});

describe("DeployWizard preview priority", () => {
  it("sends the chosen priority with every preview so plan safety compares against it", async () => {
    vi.mocked(auth.useIsAdmin).mockReturnValue(true);
    const previews: any[] = [];
    vi.mocked(apiModule.api).mockImplementation((url: string, opts?: any) => {
      if (url === "/inventory") return Promise.resolve([artifact]);
      if (url === "/engines") return Promise.resolve(engines);
      if (url === "/deployments/preview") { previews.push(opts.body); return Promise.resolve(fitsPreview); }
      if (url === "/decider/suggest-config") return Promise.resolve({ tool_call_parser: null, reasoning_parser: null });
      return Promise.reject(new Error(`unexpected ${url}`));
    });
    renderWizard();
    await waitFor(() => expect(previews.length).toBeGreaterThan(0));
    expect(previews[0].priority).toBe(0);
    fireEvent.change(screen.getByLabelText("Priority"), { target: { value: "40" } });
    await waitFor(() => expect(previews.some((b) => b.priority === 40)).toBe(true));
  });

  it("config advice fills untouched fields only; a typed value is never overwritten", async () => {
    vi.mocked(auth.useIsAdmin).mockReturnValue(true);
    const api = vi.mocked(apiModule.api);
    const adv = { goal: "chat", options: [{ key: "ctx16k-seq32-auto-pc", config: { max_model_len: 16384, max_num_seqs: 32, kv_cache_dtype: "auto", enable_prefix_caching: true, kv_tokens: 131072 }, summary: "16k context · 32 concurrent", required_bytes: 1e9, headroom_bytes: 1e9, context: 16384, seqs: 32, kv_cache_dtype: "auto", prefix_caching: true, kv_pool_tokens: 131072 }],
      recommended: "ctx16k-seq32-auto-pc", top: ["ctx16k-seq32-auto-pc"], source: "model", confidence: 0.9, decision_id: 1, free_bytes: 1e10, reason: null, disabled: false };
    api.mockImplementation((url: string) => {
      if (url === "/inventory") return Promise.resolve([artifact]);
      if (url === "/engines") return Promise.resolve(engines);
      if (url === "/deployments/preview") return Promise.resolve(fitsPreview);
      if (url === "/decider/config-advice") return Promise.resolve(adv);
      return Promise.reject(new Error(`unexpected ${url}`));
    });
    renderWizard();
    await screen.findByText(/Suggested config/);
    fireEvent.change(screen.getByLabelText("Concurrent sequences"), { target: { value: "5" } });
    await userEvent.click(screen.getByRole("button", { name: /Fill untouched fields/ }));
    expect(screen.getByLabelText("Context length (tokens)")).toHaveValue(16384);
    expect(screen.getByLabelText("Concurrent sequences")).toHaveValue(5);
    await waitFor(() => {
      const bodies = api.mock.calls.filter((c) => c[0] === "/deployments/preview").map((c) => (c[1] as any).body.config);
      expect(bodies.at(-1)).toMatchObject({ max_model_len: 16384, max_num_seqs: 5, kv_tokens: 131072 });
    });
  });

  it("shows the code scan with trust_remote_code on, and a clean scan keeps the generic warning", async () => {
    vi.mocked(auth.useIsAdmin).mockReturnValue(true);
    const remote = { ...artifact, has_remote_code: true };
    vi.mocked(apiModule.api).mockImplementation((url: string) => {
      if (url === "/inventory") return Promise.resolve([remote]);
      if (url === "/engines") return Promise.resolve(engines);
      if (url === "/deployments/preview") return Promise.resolve(fitsPreview);
      if (url === "/decider/remote-code-check") return Promise.resolve({ verdict: "no_findings", findings: [], source: "rules", confidence: null, model_label: null, decision_id: 1, scanned_files: 4, truncated: false, error: null, disabled: false, remote_code: true });
      return Promise.reject(new Error(`unexpected ${url}`));
    });
    renderWizard();
    await screen.findByLabelText("Context length (tokens)");
    await userEvent.click(screen.getByRole("button", { name: "Show advanced settings" }));
    await userEvent.click(screen.getByRole("switch", { name: "Trust remote code" }));
    expect(await screen.findByText("No suspicious patterns found in 4 files (static scan only)")).toBeInTheDocument();
    expect(screen.getByText(/trust_remote_code runs Python code from the model repository/)).toBeInTheDocument();
  });

  it("summarises compatibility for a repo that is not downloaded yet", async () => {
    vi.mocked(auth.useIsAdmin).mockReturnValue(true);
    vi.mocked(apiModule.api).mockImplementation((url: string) => {
      if (url === "/inventory") return Promise.resolve([]);
      if (url === "/engines") return Promise.resolve(engines);
      if (url.startsWith("/catalog/compat")) return Promise.resolve({ repo: "org/model-a", sha: "s", architecture: "Qwen3ForCausalLM", gated: false, vllm: { verdict: "runs", reasons: ["supported_architecture"], flags: [] }, llamacpp: { verdict: "unsupported", reasons: ["no_gguf"], flags: [] }, source: "rules", confidence: 1, decision_id: 1, cached: false, error: null, disabled: false });
      return Promise.reject(new Error(`unexpected ${url}`));
    });
    renderWizard();
    expect(await screen.findByText(/org\/model-a is not downloaded yet/)).toBeInTheDocument();
    expect(await screen.findByText("vLLM: runs")).toBeInTheDocument();
  });

  it("the engine chip's Use keeps typed values that stay valid for the new model", async () => {
    vi.mocked(auth.useIsAdmin).mockReturnValue(true);
    const other = { ...artifact, ref: "org/model-a-fp8" };
    vi.mocked(apiModule.api).mockImplementation((url: string) => {
      if (url === "/inventory") return Promise.resolve([artifact, other]);
      if (url === "/engines") return Promise.resolve(engines);
      if (url === "/deployments/preview") return Promise.resolve(fitsPreview);
      if (url === "/decider/engine-suggest") return Promise.resolve({ engine: "vllm", model_ref: "org/model-a-fp8", source: "model", confidence: 0.9, reason: "model", decision_id: 1, alternatives: {}, disabled: false });
      return Promise.reject(new Error(`unexpected ${url}`));
    });
    renderWizard();
    await screen.findByLabelText("Concurrent sequences");
    fireEvent.change(screen.getByLabelText("Concurrent sequences"), { target: { value: "5" } });
    await userEvent.click(await screen.findByRole("button", { name: /Use vLLM with org\/model-a-fp8/ }));
    await waitFor(() => expect(screen.getByRole("combobox", { name: /^Model\s+org\/model-a-fp8$/ })).toBeInTheDocument());
    expect(screen.getByLabelText("Concurrent sequences")).toHaveValue(5);
    expect(screen.getByLabelText("Context length (tokens)")).toHaveValue(4096);
  });
});
