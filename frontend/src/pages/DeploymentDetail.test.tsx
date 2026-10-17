import { describe, expect, it, vi } from "vitest";
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { DeploymentDetail } from "./DeploymentDetail";
import * as auth from "@/auth";
import * as apiModule from "@/api";

vi.mock("@/auth");
vi.mock("@/api", async () => {
  const actual = await vi.importActual<typeof import("@/api")>("@/api");
  return { ...actual, api: vi.fn() };
});

const dep = {
  id: 1, name: "qwen", engine: "vllm", model_ref: "Qwen/Qwen3-8B", state: "ready", priority: 0, alias: "chat", model_name: "chat",
  reserved_bytes: 1e9, actual_bytes: 1e9, estimate: {}, config: { max_model_len: 4096 }, litellm_status: "ok",
  last_error: null, exit_code: null, container_id: null, served_name: "qwen", endpoint: "http://dgxmm-qwen:8000/v1", host_endpoint: "http://127.0.0.1:8101/v1",
  sleep_capable: true, adopted_from: null, created_at: "", updated_at: "", events: [],
};

const key = {
  token: "b".repeat(64), key_name: "sk-...zz", key_alias: "team-a", models: ["chat"], spend: 0, max_budget: null, budget_duration: null,
  rpm_limit: null, tpm_limit: null, expires: null, created_at: null, blocked: false, metadata: {}, created_by_dgxmm: true,
};
const info = (over: Record<string, any> = {}) => ({
  deployment_id: 1, model_name: "chat", endpoint: "https://llm.example.com/v1", registered: true, status: "synced", enabled: true,
  error: null, keys: [key], keys_visible: true, keys_error: null, ...over,
});

function renderDetail(advice: any[] = [], { admin = true, litellm = info(), extra, path = "/deployments/1" }: { admin?: boolean; litellm?: any; extra?: (url: string, opts?: any) => any; path?: string } = {}) {
  vi.mocked(auth.useIsAdmin).mockReturnValue(admin);
  vi.mocked(apiModule.api).mockImplementation((url: string, opts?: any) => {
    const r = extra?.(url, opts);
    if (r !== undefined) return r;
    if (url === "/deployments/1" && !opts?.method) return Promise.resolve(dep);
    if (url === "/deployments/1/advice") return Promise.resolve(advice);
    if (url === "/deployments/1/litellm") return Promise.resolve(litellm);
    if (url === "/deployments") return Promise.resolve([dep, { ...dep, id: 2, alias: null, model_name: "coder", served_name: "coder" }]);
    return new Promise(() => undefined);
  });
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={qc}>
      <MemoryRouter initialEntries={[path]} future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
        <Routes><Route path="/deployments/:id" element={<DeploymentDetail />} /></Routes>
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

describe("DeploymentDetail", () => {
  it("disables header actions while one is in flight", async () => {
    renderDetail();
    const sleep = await screen.findByRole("button", { name: "Sleep" });
    await userEvent.click(sleep);
    expect(screen.getByRole("button", { name: "Sleep" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Reconfigure" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Sleep" }).querySelector(".animate-spin")).not.toBeNull();
  });

  it("colours warning advice as a warning, not as critical", async () => {
    renderDetail([{ id: "a1", severity: "warning", confidence: "high", title: "Raise sequences", rationale: "r", expected_effect: "e", patch: null }]);
    await userEvent.click(await screen.findByRole("tab", { name: /Advice/ }));
    const chip = await screen.findByText("warning");
    expect(chip.className).toMatch(/text-status-warning/);
    expect(chip.className).not.toMatch(/text-status-critical/);
  });

  it("shows API access as the single LiteLLM endpoint plus the model name, with snippets", async () => {
    renderDetail();
    const card = (await screen.findByRole("heading", { name: "API access" })).closest("[data-card]") as HTMLElement;
    expect(await within(card).findByText("https://llm.example.com/v1")).toBeInTheDocument();
    expect(within(card).getByText("chat")).toBeInTheDocument();
    expect(within(card).getByLabelText("curl example")).toHaveTextContent("https://llm.example.com/v1/chat/completions");
    expect(within(card).getByLabelText("curl example")).toHaveTextContent('"model": "chat"');
    expect(within(card).getByLabelText("Python example")).toHaveTextContent('base_url="https://llm.example.com/v1"');
    expect(within(card).getByRole("button", { name: "Copy Endpoint" })).toBeInTheDocument();
    expect(within(card).getByRole("button", { name: "Copy Model" })).toBeInTheDocument();
    expect(within(card).getByText("Synced")).toBeInTheDocument();
    const advanced = within(card).getByText("Direct engine endpoints (advanced)");
    expect(advanced.closest("details")).not.toHaveAttribute("open");
    expect(within(card).getByText("http://127.0.0.1:8101/v1")).toBeInTheDocument();
  });

  it("warns in API access when the model is not registered in LiteLLM", async () => {
    renderDetail([], { litellm: info({ registered: false, status: "pending" }) });
    expect(await screen.findByText(/not registered in LiteLLM/i)).toBeInTheDocument();
  });

  it("says a stopped deployment registers once ready, instead of warning that requests fail", async () => {
    renderDetail([], { litellm: info({ registered: false, status: "none" }), extra: (url, opts) =>
      url === "/deployments/1" && !opts?.method ? Promise.resolve({ ...dep, state: "stopped" }) : undefined });
    const card = (await screen.findByRole("heading", { name: "API access" })).closest("[data-card]") as HTMLElement;
    expect(await within(card).findByText(/Registers automatically when the deployment is ready/)).toBeInTheDocument();
    expect(within(card).queryByText(/requests will fail/)).toBeNull();
  });

  it("keeps the snippet's copy button clear of the first line", async () => {
    renderDetail();
    const pre = await screen.findByLabelText("curl example");
    expect(pre.className).toMatch(/\bpt-9\b/);   // the button sits in the top padding, above the code
  });

  const taken = () => Promise.reject(new apiModule.ApiError(409, {
    code: "model_name_taken", message: "the model name taken is already used by deployment other (ready); choose a different model name (alias)",
    holder: { id: 2, name: "other", state: "ready" },
  }));
  const patchHandler = (patches: any[], result: Record<string, any>) => (url: string, opts?: any) => {
    if (url === "/deployments/1" && opts?.method === "PATCH") {
      patches.push(opts.body);
      return opts.body.alias === "taken" ? taken() : Promise.resolve({ ...dep, alias: opts.body.alias, model_name: opts.body.alias, ...result });
    }
    return undefined;
  };

  it("lets admins rename the LiteLLM model with validation feedback and rewrites scoped keys", async () => {
    const patches: any[] = [];
    renderDetail([], { extra: patchHandler(patches, { affected_keys: [{ token: key.token, key_alias: "team-a", models: ["team-chat"] }], keys_rewritten: true, keys_error: null }) });
    await userEvent.click(await screen.findByRole("tab", { name: /LiteLLM/ }));
    const input = await screen.findByLabelText("Model name");
    expect(input).toHaveValue("chat");
    const save = screen.getByRole("button", { name: "Save model name" });
    expect(save).toBeDisabled();
    expect(screen.queryByRole("checkbox", { name: /Also update/ })).toBeNull();
    await userEvent.clear(input);
    await userEvent.type(input, "bad name");
    expect(screen.getByText(/no spaces/)).toBeInTheDocument();
    expect(save).toBeDisabled();
    await userEvent.clear(input);
    await userEvent.type(input, "taken");
    expect(screen.getByRole("checkbox", { name: "Also update 1 key scoped to chat" })).toBeChecked();
    await userEvent.click(save);
    expect(await screen.findByText(/already used by deployment other/)).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Open other" })).toHaveAttribute("href", "/deployments/2");
    await userEvent.clear(input);
    expect(screen.getByText(/served name \(qwen\)/)).toBeInTheDocument();
    await userEvent.type(input, "team-chat");
    await userEvent.click(save);
    await waitFor(() => expect(patches).toEqual([{ alias: "taken", rewrite_keys: true }, { alias: "team-chat", rewrite_keys: true }]));
    expect(await screen.findByText(/Updated 1 key to team-chat/)).toBeInTheDocument();
  });

  it("reports keys left on the old name when the rewrite is declined", async () => {
    const patches: any[] = [];
    renderDetail([], { extra: patchHandler(patches, { affected_keys: [{ token: key.token, key_alias: "team-a", models: ["chat"] }], keys_rewritten: false, keys_error: null }) });
    await userEvent.click(await screen.findByRole("tab", { name: /LiteLLM/ }));
    const input = await screen.findByLabelText("Model name");
    await userEvent.clear(input);
    await userEvent.type(input, "renamed");
    await userEvent.click(screen.getByRole("checkbox", { name: /Also update 1 key/ }));
    await userEvent.click(screen.getByRole("button", { name: "Save model name" }));
    await waitFor(() => expect(patches).toEqual([{ alias: "renamed", rewrite_keys: false }]));
    expect(await screen.findByText(/1 key still lists chat/)).toBeInTheDocument();
  });

  it("shows registration as unknown before the first sync", async () => {
    renderDetail([], { litellm: info({ registered: null, status: "pending" }) });
    await userEvent.click(await screen.findByRole("tab", { name: /LiteLLM/ }));
    expect(await screen.findByText("Unknown — not synced yet")).toBeInTheDocument();
  });

  it("explains a model name conflict found by the sync", async () => {
    const message = "model name chat is already registered for deployment other; give this deployment a different model name";
    renderDetail([], { litellm: info({ registered: false, status: "error", error: { code: "model_name_conflict", message } }) });
    expect((await screen.findAllByText(new RegExp(message))).length).toBeGreaterThan(0);
    await userEvent.click(await screen.findByRole("tab", { name: /LiteLLM/ }));
    expect(screen.getAllByText(/Model name conflict/).length).toBeGreaterThan(0);
  });

  it("links to the deployment holding the model name when start is refused", async () => {
    renderDetail([], { extra: (url, opts) => {
      if (url === "/deployments/1" && !opts?.method) return Promise.resolve({ ...dep, state: "stopped" });
      if (url === "/deployments/1/start") return taken();
      return undefined;
    } });
    await userEvent.click(await screen.findByRole("button", { name: "Start" }));
    expect(await screen.findByRole("link", { name: "Open other" })).toHaveAttribute("href", "/deployments/2");
    expect(screen.getByText(/load-balance/)).toBeInTheDocument();
    await userEvent.click(screen.getByRole("link", { name: /Set a model name for qwen/ }));
    expect(await screen.findByLabelText("Model name")).toHaveValue("qwen-qwen");
    expect(screen.getByLabelText("Model name")).toHaveFocus();
    expect(screen.queryByText(/load-balance/)).toBeNull(); // the start conflict notice clears on tab change
  });

  it("opens the LiteLLM tab with a suggested model name from the URL", async () => {
    renderDetail([], { path: "/deployments/1?tab=litellm&alias=qwen-qwen" });
    expect(await screen.findByLabelText("Model name")).toHaveValue("qwen-qwen");
    expect(screen.getByRole("button", { name: "Save model name" })).toBeEnabled();
  });

  it("reports a partial key rewrite", async () => {
    const patches: any[] = [];
    renderDetail([], { extra: patchHandler(patches, {
      affected_keys: [{ token: "t1", key_alias: "a", models: ["chat"] }, { token: "t2", key_alias: "b", models: ["chat"] }],
      keys_rewritten: 1, keys_error: { code: "litellm_unreachable", message: "could not reach LiteLLM" } }) });
    await userEvent.click(await screen.findByRole("tab", { name: /LiteLLM/ }));
    const input = await screen.findByLabelText("Model name");
    await userEvent.clear(input);
    await userEvent.type(input, "renamed");
    await userEvent.click(screen.getByRole("button", { name: "Save model name" }));
    expect(await screen.findByText(/Updated 1 of 2 keys/)).toBeInTheDocument();
    expect(screen.getByText(/could not reach LiteLLM/)).toBeInTheDocument();
  });

  it("lists keys with access and creates a key with this deployment's model preselected", async () => {
    renderDetail();
    await userEvent.click(await screen.findByRole("tab", { name: /LiteLLM/ }));
    expect(await screen.findByText("team-a")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Re-sync" })).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "Create key for this deployment" }));
    expect(await screen.findByRole("checkbox", { name: "chat" })).toBeChecked();
    expect(screen.getByRole("checkbox", { name: "coder" })).not.toBeChecked();
    expect(screen.getByRole("checkbox", { name: /All models/ })).not.toBeChecked();
  });

  it("gives viewers read-only API access and no key management", async () => {
    renderDetail([], { admin: false, litellm: info({ keys: [], keys_visible: false }) });
    expect(await screen.findByText("https://llm.example.com/v1")).toBeInTheDocument();
    await userEvent.click(await screen.findByRole("tab", { name: /LiteLLM/ }));
    expect(await screen.findByText("Registration")).toBeInTheDocument();
    expect(screen.queryByLabelText("Model name")).toBeNull();
    expect(screen.queryByRole("button", { name: /Create key/ })).toBeNull();
    expect(screen.queryByRole("button", { name: "Re-sync" })).toBeNull();
    expect(screen.queryByText("Keys with access")).toBeNull();
  });

  it("keeps the one-time key when the per-deployment refetch reports a key error", async () => {
    const full = "sk-TAB-KEY-0123456789";
    let loads = 0;
    renderDetail([], { extra: (url, opts) => {
      if (url === "/deployments/1/litellm") return Promise.resolve(loads++ === 0 ? info() : info({ keys: [], keys_error: { code: "litellm_unreachable", message: "could not reach LiteLLM" } }));
      if (url === "/integrations/litellm/keys" && opts?.method === "POST") return Promise.resolve({ ...key, key_alias: "tab-key", key: full });
      return undefined;
    } });
    await userEvent.click(await screen.findByRole("tab", { name: /LiteLLM/ }));
    await userEvent.click(await screen.findByRole("button", { name: "Create key for this deployment" }));
    await userEvent.type(screen.getByLabelText("Alias"), "tab-key");
    await userEvent.click(screen.getByRole("button", { name: "Create key" }));
    expect(await screen.findByDisplayValue(full)).toBeInTheDocument();
    await waitFor(() => expect(loads).toBeGreaterThan(1));
    await new Promise((r) => setTimeout(r, 20));
    expect(screen.getByDisplayValue(full)).toBeInTheDocument();
  });

  it("still offers (and defaults to) rewriting keys when the key list is unavailable", async () => {
    const patches: any[] = [];
    renderDetail([], { litellm: info({ keys: [], keys_error: { code: "litellm_no_db", message: "no db" } }),
      extra: patchHandler(patches, { affected_keys: [], keys_rewritten: 0, keys_error: null }) });
    await userEvent.click(await screen.findByRole("tab", { name: /LiteLLM/ }));
    const input = await screen.findByLabelText("Model name");
    await userEvent.clear(input);
    await userEvent.type(input, "renamed");
    expect(screen.getByRole("checkbox", { name: "Also update any keys scoped to chat" })).toBeChecked();
    await userEvent.click(screen.getByRole("button", { name: "Save model name" }));
    await waitFor(() => expect(patches).toEqual([{ alias: "renamed", rewrite_keys: true }]));
  });

  it("explains a disabled sync from the tab's Re-sync", async () => {
    renderDetail([], { extra: (url) => (url === "/integrations/litellm/sync" ? Promise.resolve({ ok: true, status: "disabled", registered: 0, removed: 0, error: null }) : undefined) });
    await userEvent.click(await screen.findByRole("tab", { name: /LiteLLM/ }));
    await userEvent.click(await screen.findByRole("button", { name: "Re-sync" }));
    expect(await screen.findByText(/disabled or has no master key/)).toBeInTheDocument();
  });

  it("falls back to the overview for an unknown tab", async () => {
    renderDetail([], { path: "/deployments/1?tab=bogus" });
    expect(await screen.findByRole("heading", { name: "API access" })).toBeInTheDocument();
  });
});

describe("DeploymentDetail — decision model", () => {
  const failed = { ...dep, state: "failed", last_error: "container exited (1)", exit_code: 1 };
  const diag = (over: Record<string, any> = {}) => ({
    cause: "gpu_oom", label: "Out of GPU memory", source: "model", confidence: 0.93, summary: "CUDA ran out of memory while allocating the KV cache.",
    fix_hint: "Halve the context length.", patch: { max_model_len: 2048 }, decision_id: 55, created_at: "2026-10-16T10:00:00Z", log_excerpt: "torch.OutOfMemoryError: CUDA out of memory", ...over,
  });
  const withDiag = (d: any, diagnosis: any, more?: (url: string, opts?: any) => any) => (url: string, opts?: any) => {
    const r = more?.(url, opts);
    if (r !== undefined) return r;
    if (url === "/deployments/1" && !opts?.method) return Promise.resolve(d);
    if (url === "/deployments/1/diagnosis") return Promise.resolve({ diagnosis });
    return undefined;
  };

  it("shows the diagnosis at the top of the overview for a failed deployment, and Apply fix opens Reconfigure with the patch", async () => {
    renderDetail([], { extra: withDiag(failed, diag()) });
    const card = (await screen.findByRole("heading", { name: "Diagnosis" })).closest("[data-card]") as HTMLElement;
    expect(within(card).getByText("Out of GPU memory")).toBeInTheDocument();
    expect(within(card).getByText("93% · AI")).toBeInTheDocument();
    expect(within(card).getByText(/CUDA ran out of memory/)).toBeInTheDocument();
    expect(within(card).getByText("Halve the context length.")).toBeInTheDocument();
    await userEvent.click(within(card).getByRole("button", { name: "Apply fix…" }));
    expect(await screen.findByText("Reconfigure qwen")).toBeInTheDocument();
    expect(screen.getByLabelText("Context length (tokens)")).toHaveValue(2048);
  });

  it("shows the feedback already given on the diagnosis after a reload", async () => {
    renderDetail([], { extra: withDiag(failed, diag({ feedback: "rejected" })) });
    const card = (await screen.findByRole("heading", { name: "Diagnosis" })).closest("[data-card]") as HTMLElement;
    expect(within(card).getByRole("button", { name: "Not helpful" })).toHaveAttribute("aria-pressed", "true");
    expect(within(card).getByRole("button", { name: "Helpful" })).toHaveAttribute("aria-pressed", "false");
  });

  it("records 👍/👎 feedback on the diagnosis", async () => {
    renderDetail([], { extra: withDiag(failed, diag(), (url) => (url === "/decider/decisions/55/feedback" ? Promise.resolve({ ok: true }) : undefined)) });
    const card = (await screen.findByRole("heading", { name: "Diagnosis" })).closest("[data-card]") as HTMLElement;
    await userEvent.click(within(card).getByRole("button", { name: "Helpful" }));
    await waitFor(() => expect(vi.mocked(apiModule.api)).toHaveBeenCalledWith("/decider/decisions/55/feedback", { method: "POST", body: { accepted: true } }));
    await waitFor(() => expect(within(card).getByRole("button", { name: "Helpful" })).toHaveAttribute("aria-pressed", "true"));
  });

  it("hides Apply fix and feedback from viewers", async () => {
    renderDetail([], { admin: false, extra: withDiag(failed, diag()) });
    const card = (await screen.findByRole("heading", { name: "Diagnosis" })).closest("[data-card]") as HTMLElement;
    expect(within(card).queryByRole("button", { name: "Apply fix…" })).toBeNull();
    expect(within(card).queryByRole("button", { name: "Helpful" })).toBeNull();
  });

  it("says so when the diagnosis is not confident, with the log excerpt", async () => {
    renderDetail([], { extra: withDiag(failed, diag({ cause: "other", source: "none", confidence: 0.41, patch: null, summary: "", fix_hint: "" })) });
    const card = (await screen.findByRole("heading", { name: "Diagnosis" })).closest("[data-card]") as HTMLElement;
    expect(within(card).getByText(/Couldn't diagnose confidently/)).toBeInTheDocument();
    expect(within(card).getByText("torch.OutOfMemoryError: CUDA out of memory")).toBeInTheDocument();
    expect(within(card).queryByRole("button", { name: "Apply fix…" })).toBeNull();
  });

  it("shows the diagnosis in Events, but not on the overview of a healthy deployment", async () => {
    renderDetail([], { extra: withDiag(dep, diag()) });
    expect(await screen.findByRole("heading", { name: "API access" })).toBeInTheDocument();
    await new Promise((r) => setTimeout(r, 20));
    expect(screen.queryByRole("heading", { name: "Diagnosis" })).toBeNull();
    await userEvent.click(screen.getByRole("tab", { name: "Events" }));
    expect(await screen.findByRole("heading", { name: "Diagnosis" })).toBeInTheDocument();
  });

  it("shows the workload badge on the Advice tab and a relevance divider", async () => {
    const advice = [
      { id: "a1", severity: "info", confidence: "high", title: "Enable speculative decoding", rationale: "r", expected_effect: "e", patch: null, relevance: 1.7, workload_fit: "boost" },
      { id: "a2", severity: "info", confidence: "high", title: "Raise sequences", rationale: "r", expected_effect: "e", patch: null, relevance: 1, workload_fit: null },
    ];
    renderDetail(advice, { extra: (url) => (url === "/deployments/1/workload" ? Promise.resolve({ label: "agent_tool_use", source: "model", confidence: 0.78, summary: null }) : undefined) });
    await userEvent.click(await screen.findByRole("tab", { name: /Advice/ }));
    expect(await screen.findByText("Workload: Agent / tool use · 78% · AI")).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Most relevant for this workload" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Other suggestions" })).toBeInTheDocument();
  });

  it("refetches the advice once the workload label is known so the order uses it", async () => {
    let adviceCalls = 0;
    renderDetail([], { extra: (url) => {
      if (url === "/deployments/1/advice") { adviceCalls++; return Promise.resolve([]); }
      if (url === "/deployments/1/workload") return new Promise((r) => setTimeout(() => r({ label: "idle", source: "rules", confidence: 1, summary: null }), 50));
      return undefined;
    } });
    await userEvent.click(await screen.findByRole("tab", { name: /Advice/ }));
    expect(await screen.findByText(/Workload: /)).toBeInTheDocument();
    await waitFor(() => expect(adviceCalls).toBeGreaterThanOrEqual(2));
  });

  it("shows advice without a workload badge when the workload is unknown", async () => {
    renderDetail([{ id: "a1", severity: "info", confidence: "high", title: "Raise sequences", rationale: "r", expected_effect: "e", patch: null }],
      { extra: (url) => (url === "/deployments/1/workload" ? Promise.resolve(null) : undefined) });
    await userEvent.click(await screen.findByRole("tab", { name: /Advice/ }));
    expect(await screen.findByText("Raise sequences")).toBeInTheDocument();
    expect(screen.queryByText(/^Workload:/)).toBeNull();
    expect(screen.queryByRole("heading", { name: "Most relevant for this workload" })).toBeNull();
  });

  it("Apply fix shows the change summary for the diagnosis", async () => {
    renderDetail([], { extra: withDiag(failed, diag({ patch: { trust_remote_code: true } })) });
    const card = (await screen.findByRole("heading", { name: "Diagnosis" })).closest("[data-card]") as HTMLElement;
    await userEvent.click(within(card).getByRole("button", { name: "Apply fix…" }));
    expect(await screen.findByRole("list", { name: "Changes from the diagnosis" })).toHaveTextContent("trust_remote_code: default → true");
    expect(screen.getByRole("switch", { name: "Trust remote code" })).toBeChecked();
  });

  it("announces a diagnosis politely", async () => {
    renderDetail([], { extra: withDiag(failed, diag()) });
    await screen.findByRole("heading", { name: "Diagnosis" });
    const live = document.querySelector("[aria-live='polite'][data-diagnosis-live]") as HTMLElement;
    expect(live).toHaveTextContent("Diagnosis: Out of GPU memory");
  });

  it("says Diagnosing… while a fresh failure has no diagnosis yet (or only an older one)", async () => {
    const now = new Date().toISOString();
    const recent = { ...failed, events: [{ ts: now, from: "starting", to: "failed", message: "exited", log_tail: null }] };
    renderDetail([], { extra: withDiag(recent, null) });
    expect(await screen.findByText("Diagnosing…")).toBeInTheDocument();
  });

  it("treats a diagnosis older than the latest failure as pending", async () => {
    const now = new Date().toISOString();
    const recent = { ...failed, events: [{ ts: now, from: "starting", to: "failed", message: "exited", log_tail: null }] };
    renderDetail([], { extra: withDiag(recent, diag({ created_at: "2026-01-01T00:00:00Z" })) });
    expect(await screen.findByText("Diagnosing…")).toBeInTheDocument();
    expect(screen.queryByRole("heading", { name: "Diagnosis" })).toBeNull();
  });

  it("does not claim to be diagnosing an old failure", async () => {
    const old = { ...failed, events: [{ ts: "2026-01-01T00:00:00Z", from: "starting", to: "failed", message: "exited", log_tail: null }] };
    renderDetail([], { extra: withDiag(old, null) });
    expect(await screen.findByRole("heading", { name: "API access" })).toBeInTheDocument();
    await new Promise((r) => setTimeout(r, 20));
    expect(screen.queryByText("Diagnosing…")).toBeNull();
  });

  it("marks a system deployment and explains that the decision model uses it", async () => {
    renderDetail([], { extra: (url, opts) => (url === "/deployments/1" && !opts?.method ? Promise.resolve({ ...dep, system: true }) : undefined) });
    expect(await screen.findByText("System")).toBeInTheDocument();
    const note = screen.getByText(/Used by the decision model/);
    expect(note).toHaveTextContent(/the watchdog stops it before any of your deployments/);
    expect(note).toHaveTextContent(/even with auto-protect off/);
    expect(note).not.toHaveTextContent(/never stops or sleeps/);
  });

  it("offers no LiteLLM endpoint, snippets or registration warning for a system deployment", async () => {
    renderDetail([], { litellm: info({ registered: false, model_name: "d3-mini" }),
      extra: (url, opts) => (url === "/deployments/1" && !opts?.method ? Promise.resolve({ ...dep, state: "ready", system: true, model_name: "d3-mini" }) : undefined) });
    const card = (await screen.findByRole("heading", { name: "API access" })).closest("[data-card]") as HTMLElement;
    expect(within(card).getByText(/System deployments aren't registered in LiteLLM/)).toBeInTheDocument();
    expect(within(card).queryByLabelText("curl example")).toBeNull();
    expect(within(card).queryByLabelText("Python example")).toBeNull();
    expect(screen.queryByText(/not registered in LiteLLM yet/)).toBeNull();
    await userEvent.click(screen.getByRole("tab", { name: /LiteLLM/ }));
    expect(await screen.findAllByText(/System deployments aren't registered in LiteLLM/)).not.toHaveLength(0);
    expect(screen.queryByLabelText("Model name")).toBeNull();
  });

  it("manages a system deployment on the Decision model page: no Switch, Reconfigure or Start", async () => {
    renderDetail([], { extra: (url, opts) => (url === "/deployments/1" && !opts?.method ? Promise.resolve({ ...dep, state: "stopped", system: true, sleep_capable: false }) : undefined) });
    expect(await screen.findByRole("link", { name: "Decision model page" })).toHaveAttribute("href", "/decision-model");
    expect(screen.queryByRole("button", { name: "Start" })).toBeNull();
    expect(screen.queryByRole("button", { name: "Reconfigure" })).toBeNull();
    await userEvent.click(screen.getByRole("button", { name: "More actions" }));
    expect(screen.queryByRole("menuitem", { name: /Switch model/ })).toBeNull();
    await userEvent.click(screen.getByRole("menuitem", { name: /Remove/ }));
    const dialog = await screen.findByRole("alertdialog");
    expect(within(dialog).getByText(/also turns the decision model off/)).toBeInTheDocument();
    expect(within(dialog).getByRole("button", { name: "Remove and turn off" })).toBeInTheDocument();
  });

  it("explains a queued start-up under the header", async () => {
    const queued = { ...dep, state: "queued", startup_queue: { position: 2, waiting_for: ["decider"], kind: "launch", since: 0, limit: 1,
      reason: "waiting for decider to finish starting before it can start (one engine starts at a time; 1 queued ahead)" } };
    renderDetail([], { extra: (url, opts) => (url === "/deployments/1" && !opts?.method ? Promise.resolve(queued) : undefined) });
    expect(await screen.findByText("Queued")).toBeInTheDocument();
    expect(screen.getByText(/1 queued ahead/)).toBeInTheDocument();
    expect(screen.getByText(/memory stays reserved/)).toBeInTheDocument();
  });
});
