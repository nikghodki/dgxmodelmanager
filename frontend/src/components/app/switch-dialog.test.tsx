import { afterEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { SwitchDialog } from "./switch-dialog";
import * as apiModule from "@/api";

vi.mock("@/api", async () => {
  const actual = await vi.importActual("@/api");
  return { ...actual, api: vi.fn() };
});

const est = { total: 1e9, weights: 1e9, draft_weights: 0, kv_cache: 0, state: 0, activations: 0, runtime: 0, margin: 0, notes: [], confidence: "measured" };
const fitsPreview = { estimate: est, fit: { fits: true, free_for_new: 2e9, headroom: 1e9 }, plans: [] };
const shortPreview = { estimate: est, fit: { fits: false, free_for_new: 5e8, headroom: -5e8 }, plans: [] };
const dep = { id: 1, name: "test", engine: "vllm", model_ref: "foo/bar", config: { max_model_len: 4096 }, state: "ready", priority: 0 } as any;
const inventory = [{ ref: "foo/bar", complete: true, kind: "llm", engines: ["vllm"], quant: "Q4", weights_bytes: 1e9, problems: [] }];

function renderDialog(mode: "switch" | "reconfigure" = "reconfigure", initialPatch?: Record<string, any>) {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } });
  return render(<QueryClientProvider client={qc}><SwitchDialog mode={mode} deployment={dep} initialPatch={initialPatch} patchSource="the diagnosis" onClose={vi.fn()} /></QueryClientProvider>);
}

afterEach(() => { vi.mocked(apiModule.api).mockReset(); });

describe("SwitchDialog", () => {
  it("warns explicitly when a prefilled patch enables trust_remote_code", async () => {
    vi.mocked(apiModule.api).mockImplementation((url: string) => {
      if (url === "/inventory") return Promise.resolve(inventory);
      if (url === "/deployments/preview") return Promise.resolve(fitsPreview);
      return Promise.reject(new Error(`unexpected ${url}`));
    });
    renderDialog("reconfigure", { trust_remote_code: true });
    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent(/Security-sensitive change: this enables trust_remote_code/);
    expect(alert).toHaveTextContent(/only if you trust this repository/);
    expect(screen.getByText(/trust_remote_code: default → true/)).toBeInTheDocument();
  });

  it("shows no security warning for other prefilled patches", async () => {
    vi.mocked(apiModule.api).mockImplementation((url: string) => {
      if (url === "/inventory") return Promise.resolve(inventory);
      if (url === "/deployments/preview") return Promise.resolve(fitsPreview);
      return Promise.reject(new Error(`unexpected ${url}`));
    });
    renderDialog("reconfigure", { max_model_len: 2048 });
    await screen.findByText(/max_model_len: 4096 → 2048/);
    expect(screen.queryByText(/Security-sensitive change/)).not.toBeInTheDocument();
  });

  it("re-plans after an insufficient_memory 409 without looping, and a chosen plan sticks", async () => {
    const errors = vi.spyOn(console, "error");
    let previewCalls = 0;
    const plans = [{ id: "p1", title: "Stop other", detail: "Frees 1 GiB", new_required: 1e9, frees: 1e9, fits: true, actions: [{ type: "stop", deployment_id: 7 }], patch: null }];
    vi.mocked(apiModule.api).mockImplementation((url: string) => {
      if (url === "/inventory") return Promise.resolve(inventory);
      if (url === "/deployments/preview") { previewCalls++; return Promise.resolve(fitsPreview); }
      if (url === "/deployments/1/reconfigure") return Promise.reject(new apiModule.ApiError(409, {
        code: "insufficient_memory", message: "Not enough memory", fit: { fits: false, free_for_new: 5e8, headroom: -5e8 }, plans,
      }));
      return Promise.reject(new Error(`unexpected ${url}`));
    });
    renderDialog();
    await waitFor(() => expect(screen.getByText("Fits")).toBeInTheDocument());
    await userEvent.click(screen.getByRole("button", { name: "Apply and restart" }));
    const radio = await screen.findByRole("radio", { name: /Stop other/ });
    const callsAfter409 = previewCalls;
    await userEvent.click(radio);
    await new Promise((r) => setTimeout(r, 50));
    expect(screen.getByRole("radio", { name: /Stop other/ })).toHaveAttribute("aria-checked", "true");
    expect(screen.getByRole("button", { name: "Apply plan and apply and restart" })).toBeEnabled();
    expect(previewCalls).toBe(callsAfter409);
    expect(errors.mock.calls.map((c) => String(c[0])).join("\n")).not.toMatch(/Maximum update depth/);
    errors.mockRestore();
  });

  it("ignores a stale preview that resolves after a newer one", async () => {
    const resolvers: ((v: unknown) => void)[] = [];
    vi.mocked(apiModule.api).mockImplementation((url: string) => {
      if (url === "/inventory") return Promise.resolve(inventory);
      if (url === "/deployments/preview") return new Promise((res) => resolvers.push(res));
      return Promise.reject(new Error(`unexpected ${url}`));
    });
    renderDialog();
    await waitFor(() => expect(resolvers.length).toBe(1));
    fireEvent.change(screen.getByLabelText("Context length (tokens)"), { target: { value: "8192" } });
    await waitFor(() => expect(resolvers.length).toBe(2));
    resolvers[1](fitsPreview);
    await waitFor(() => expect(screen.getByText("Fits")).toBeInTheDocument());
    resolvers[0](shortPreview);
    await new Promise((r) => setTimeout(r, 20));
    expect(screen.queryByText("Does not fit")).toBeNull();
    expect(screen.getByText("Fits")).toBeInTheDocument();
  });

  it("names the model picker by its visible label and current value", async () => {
    vi.mocked(apiModule.api).mockImplementation((url: string) => {
      if (url === "/inventory") return Promise.resolve(inventory);
      if (url === "/deployments/preview") return Promise.resolve(fitsPreview);
      return Promise.reject(new Error(`unexpected ${url}`));
    });
    renderDialog("switch");
    expect(await screen.findByRole("combobox", { name: /^Model\s+foo\/bar$/ })).toBeInTheDocument();
  });

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

describe("SwitchDialog — parser suggestions", () => {
  const sug = { value: "qwen3_coder", source: "model", confidence: 0.95, auto_apply: true, alternatives: [] };
  const inv2 = [...inventory, { ...inventory[0], ref: "foo/new" }];
  function mock() {
    vi.mocked(apiModule.api).mockImplementation((url: string) => {
      if (url === "/inventory") return Promise.resolve(inv2);
      if (url === "/deployments/preview") return Promise.resolve(fitsPreview);
      if (url === "/decider/suggest-config") return Promise.resolve({ tool_call_parser: sug, reasoning_parser: null });
      return new Promise(() => undefined);
    });
  }
  const renderWith = (d: any, initialPatch?: Record<string, any>) => {
    const qc = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } });
    return render(<QueryClientProvider client={qc}><SwitchDialog mode="reconfigure" deployment={d} initialPatch={initialPatch} onClose={vi.fn()} /></QueryClientProvider>);
  };
  const sugCalls = () => vi.mocked(apiModule.api).mock.calls.filter(([u]) => u === "/decider/suggest-config");

  it("does not ask while the model is unchanged", async () => {
    mock();
    renderWith(dep);
    await waitFor(() => expect(screen.getByText("Fits")).toBeInTheDocument());
    expect(sugCalls()).toHaveLength(0);
  });

  it("suggests for the new model and auto-fills an empty parser", async () => {
    mock();
    renderWith(dep, { model_ref: "foo/new" });
    expect(await screen.findByText("Suggested: qwen3_coder · 95% · AI")).toBeInTheDocument();
    expect(sugCalls()[0][1]).toEqual({ method: "POST", body: { model_ref: "foo/new", engine: "vllm" } });
    expect(screen.getByLabelText("Tool-call parser")).toHaveValue("qwen3_coder");
  });

  it("keeps the configured parser and offers Apply instead", async () => {
    mock();
    renderWith({ ...dep, config: { max_model_len: 4096, tool_call_parser: "hermes" } }, { model_ref: "foo/new" });
    expect(await screen.findByText("Suggested: qwen3_coder · 95% · AI")).toBeInTheDocument();
    expect(screen.getByLabelText("Tool-call parser")).toHaveValue("hermes");
    await userEvent.click(screen.getByRole("button", { name: /Apply suggested Tool-call parser: qwen3_coder/ }));
    expect(screen.getByLabelText("Tool-call parser")).toHaveValue("qwen3_coder");
  });
});

describe("SwitchDialog — initial patch (Apply fix / Apply advice)", () => {
  it("shows every patched field even if advanced, summarises the changes, and removes keys patched to null", async () => {
    const bodies: any[] = [];
    vi.mocked(apiModule.api).mockImplementation((url: string, opts?: any) => {
      if (url === "/inventory") return Promise.resolve(inventory);
      if (url === "/deployments/preview") return Promise.resolve(fitsPreview);
      if (url === "/deployments/1/reconfigure") { bodies.push(opts.body); return Promise.resolve({}); }
      return new Promise(() => undefined);
    });
    const qc = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } });
    render(<QueryClientProvider client={qc}><SwitchDialog mode="reconfigure" patchSource="the diagnosis"
      deployment={{ ...dep, config: { max_model_len: 4096, tool_call_parser: "hermes" } }}
      initialPatch={{ trust_remote_code: true, tool_call_parser: null }} onClose={vi.fn()} /></QueryClientProvider>);
    // advanced settings are hidden, but the patched fields are shown
    expect(screen.getByRole("button", { name: "Show advanced settings" })).toBeInTheDocument();
    expect(screen.getByRole("switch", { name: "Trust remote code" })).toBeChecked();
    expect(screen.getByLabelText("Tool-call parser")).toHaveValue("");
    expect(screen.queryByLabelText("Served model name")).toBeNull();
    const summary = screen.getByRole("list", { name: "Changes from the diagnosis" });
    expect(summary).toHaveTextContent("trust_remote_code: default → true");
    expect(summary).toHaveTextContent("tool_call_parser: hermes → removed");
    await waitFor(() => expect(screen.getByText("Fits")).toBeInTheDocument());
    await userEvent.click(screen.getByRole("button", { name: "Apply and restart" }));
    await waitFor(() => expect(bodies).toHaveLength(1));
    expect(bodies[0].config).toEqual({ max_model_len: 4096, trust_remote_code: true });
    expect("tool_call_parser" in bodies[0].config).toBe(false);
  });
});

describe("SwitchDialog preview priority", () => {
  it("sends the deployment's priority with the preview", async () => {
    const previews: any[] = [];
    vi.mocked(apiModule.api).mockImplementation((url: string, opts?: any) => {
      if (url === "/inventory") return Promise.resolve(inventory);
      if (url === "/deployments/preview") { previews.push(opts.body); return Promise.resolve(fitsPreview); }
      return Promise.reject(new Error(`unexpected ${url}`));
    });
    const qc = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } });
    render(<QueryClientProvider client={qc}><SwitchDialog mode="reconfigure" deployment={{ ...dep, priority: 35 }} onClose={vi.fn()} /></QueryClientProvider>);
    await waitFor(() => expect(previews.length).toBeGreaterThan(0));
    expect(previews[0]).toMatchObject({ replacing: 1, priority: 35 });
  });
});
