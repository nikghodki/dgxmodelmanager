import { afterEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { LiteLLM } from "./LiteLLM";
import * as auth from "@/auth";
import * as apiModule from "@/api";

vi.mock("@/auth");
vi.mock("@/api", async () => {
  const actual = await vi.importActual<typeof import("@/api")>("@/api");
  return { ...actual, api: vi.fn() };
});

const okStatus = { reachable: true, version: "1.100.0", db: "connected", error: null, checked_at: "2026-10-16T16:56:43Z" };
const settings = (over: Record<string, any> = {}) => ({
  enabled: true, url: "http://litellm-proxy:4000", public_url: "https://llm.example.com", endpoint: "https://llm.example.com/v1",
  master_key_set: true, master_key_source: "secret_file", master_key_hint: null, default_url: "http://litellm-proxy:4000", status: okStatus, ...over,
});
const key = {
  token: "a".repeat(64), key_name: "sk-...HWqw", key_alias: "team-a", models: ["chat"], spend: 1.5, max_budget: 5, budget_duration: "30d",
  rpm_limit: 10, tpm_limit: null, expires: null, created_at: "2026-10-01T00:00:00Z", blocked: false, metadata: {}, created_by_dgxmm: true,
};
const models = [
  { id: "m1", model_name: "chat", model: "openai/qwen3", api_base: "http://dgxmm-a:8101/v1", managed_by_dgxmm: true, deployment: { id: 1, name: "qwen", state: "ready" }, db_model: true },
  { id: "m2", model_name: "hand-written", model: "openai/x", api_base: "http://x", managed_by_dgxmm: false, deployment: null, db_model: true },
];
const deployments = [{ id: 1, name: "qwen", model_name: "chat", alias: "chat", served_name: "qwen" }];

type Handler = (url: string, opts?: any) => any;
function setup(over: Record<string, any> = {}, extra?: Handler) {
  vi.mocked(auth.useIsAdmin).mockReturnValue(true);
  vi.mocked(apiModule.api).mockImplementation(async (url: string, opts?: any) => {
    const r = extra?.(url, opts);
    if (r !== undefined) return r;
    if (url === "/integrations/litellm?probe=false" && !opts?.method) return settings(over);
    if (url === "/integrations/litellm/status") return settings(over).status;
    if (url === "/integrations/litellm/models") return models;
    if (url === "/integrations/litellm/keys" && !opts?.method) return [key];
    if (url === "/deployments") return deployments;
    return new Promise(() => undefined);
  });
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } });
  render(
    <QueryClientProvider client={qc}>
      <MemoryRouter future={{ v7_startTransition: true, v7_relativeSplatPath: true }}><LiteLLM /></MemoryRouter>
    </QueryClientProvider>,
  );
  return qc;
}
const cached = (qc: QueryClient, secret: string) => qc.getMutationCache().getAll().some((m) => JSON.stringify(m.state).includes(secret));
const calls = (method: string, url: string) => vi.mocked(apiModule.api).mock.calls.filter(([u, o]) => u === url && (o as any)?.method === method);

afterEach(() => vi.mocked(apiModule.api).mockReset());

describe("LiteLLM page", () => {
  it("redirects viewers", () => {
    vi.mocked(auth.useIsAdmin).mockReturnValue(false);
    const qc = new QueryClient();
    render(<QueryClientProvider client={qc}><MemoryRouter initialEntries={["/litellm"]} future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
      <Routes><Route path="/litellm" element={<LiteLLM />} /><Route path="/" element={<p>home page</p>} /></Routes>
    </MemoryRouter></QueryClientProvider>);
    expect(screen.getByText("home page")).toBeInTheDocument();
    expect(vi.mocked(apiModule.api)).not.toHaveBeenCalled();
  });

  it("shows the connection, the single endpoint, models and keys", async () => {
    setup();
    expect(await screen.findByLabelText("Internal URL")).toHaveValue("http://litellm-proxy:4000");
    expect(screen.getByTestId("endpoint-preview")).toHaveTextContent("https://llm.example.com/v1");
    expect(screen.getByText(/Set \(from secret file\)/)).toBeInTheDocument();
    expect(await screen.findByText("hand-written")).toBeInTheDocument();
    expect(screen.getByText("external")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "qwen" })).toHaveAttribute("href", "/deployments/1");
    expect(await screen.findByText("team-a")).toBeInTheDocument();
    expect(screen.getByText("sk-...HWqw")).toBeInTheDocument();
    expect(screen.getByText("$1.50 / $5.00 per 30d")).toBeInTheDocument();
  });

  it("previews the public endpoint live and falls back to the internal URL", async () => {
    setup();
    const pub = await screen.findByLabelText("Public endpoint URL");
    await userEvent.clear(pub);
    expect(screen.getByTestId("endpoint-preview")).toHaveTextContent("http://litellm-proxy:4000/v1");
    await userEvent.type(pub, "http://198.51.100.5:4000/");
    expect(screen.getByTestId("endpoint-preview")).toHaveTextContent("http://198.51.100.5:4000/v1");
  });

  it("never renders the master key: a replaced key is sent once and the field is cleared", async () => {
    const qc = setup({}, (url, opts) => (url === "/integrations/litellm" && opts?.method === "PUT" ? settings({ master_key_source: "settings" }) : undefined));
    await userEvent.click(await screen.findByRole("button", { name: "Replace" }));
    await userEvent.type(screen.getByLabelText("New master key"), "sk-master-SECRET");
    await userEvent.click(screen.getByRole("button", { name: "Save" }));
    await waitFor(() => expect(calls("PUT", "/integrations/litellm")).toHaveLength(1));
    expect(calls("PUT", "/integrations/litellm")[0][1]).toEqual({ method: "PUT", body: { master_key: "sk-master-SECRET" } });
    expect(await screen.findByText(/Set \(from settings\)/)).toBeInTheDocument();
    expect(screen.queryByDisplayValue("sk-master-SECRET")).toBeNull();
    expect(document.body.innerHTML).not.toContain("sk-master-SECRET");
    await waitFor(() => expect(cached(qc, "sk-master-SECRET")).toBe(false));
  });

  it("prompts for the master key when the internal URL changes, without blocking Save", async () => {
    setup({}, (url, opts) => (url === "/integrations/litellm" && opts?.method === "PUT" ? settings({ url: "http://other:4000" }) : undefined));
    const url = await screen.findByLabelText("Internal URL");
    await userEvent.clear(url);
    await userEvent.type(url, "http://other:4000");
    expect(screen.getByText(/Enter the master key for the new URL/)).toBeInTheDocument();
    const save = screen.getByRole("button", { name: "Save" });
    expect(save).toBeEnabled();
    await userEvent.type(screen.getByLabelText("New master key"), "sk-again");
    await userEvent.click(save);
    await waitFor(() => expect(calls("PUT", "/integrations/litellm")).toHaveLength(1));
    expect(calls("PUT", "/integrations/litellm")[0][1]).toEqual({ method: "PUT", body: { url: "http://other:4000", master_key: "sk-again" } });
  });

  it("allows saving a URL change without a key (the old key just stops applying)", async () => {
    setup({}, (url, opts) => (url === "/integrations/litellm" && opts?.method === "PUT" ? settings({ url: "http://other:4000" }) : undefined));
    const url = await screen.findByLabelText("Internal URL");
    await userEvent.clear(url);
    await userEvent.type(url, "http://other:4000");
    await userEvent.click(screen.getByRole("button", { name: "Save" }));
    await waitFor(() => expect(calls("PUT", "/integrations/litellm")).toHaveLength(1));
    expect(calls("PUT", "/integrations/litellm")[0][1]).toEqual({ method: "PUT", body: { url: "http://other:4000" } });
  });

  it("shows the server's master key hint prominently and asks for the key", async () => {
    const hint = "the master key was entered for a different LiteLLM URL; re-enter it for this URL";
    setup({ master_key_set: false, master_key_source: "none", master_key_hint: hint });
    const shown = await screen.findAllByText(hint);
    expect(shown.length).toBeGreaterThan(0);
    expect(screen.getByTestId("master-key-hint")).toHaveTextContent(hint);
    expect(screen.getByTestId("master-key-hint").querySelector("svg")).not.toBeNull();
    expect(screen.getByLabelText("New master key")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Save" })).toBeDisabled();
  });

  it("shows test connection details and sync counts", async () => {
    setup({}, (url) => {
      if (url === "/integrations/litellm/test") return { ...okStatus, auth_ok: false, models: null, error: "LiteLLM rejected the master key; check it in the LiteLLM settings" };
      if (url === "/integrations/litellm/sync") return { ok: true, status: "synced", registered: 2, removed: 1, error: null };
      return undefined;
    });
    await userEvent.click(await screen.findByRole("button", { name: "Test connection" }));
    const result = await screen.findByRole("status", { name: "Connection test result" });
    expect(within(result).getByText(/rejected the master key/)).toBeInTheDocument();
    expect(within(result).getByText("1.100.0")).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "Sync now" }));
    expect(await screen.findByText(/2 registered, 1 removed/)).toBeInTheDocument();
  });

  it("explains an unreachable LiteLLM", async () => {
    setup({ status: { reachable: false, version: null, db: null, error: "could not reach LiteLLM at http://litellm-proxy:4000: ConnectError", checked_at: "" } },
      (url) => (url === "/integrations/litellm/models" || url === "/integrations/litellm/keys"
        ? Promise.reject(new apiModule.ApiError(502, { code: "litellm_unreachable", message: "could not reach LiteLLM" })) : undefined));
    expect((await screen.findAllByText("Cannot reach LiteLLM")).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/ConnectError/).length).toBeGreaterThan(0);
  });

  it("explains a LiteLLM without a database instead of listing keys", async () => {
    setup({ status: { ...okStatus, db: "Not connected" } });
    expect(await screen.findByText("LiteLLM has no database")).toBeInTheDocument();
    expect(screen.getByText(/DATABASE_URL/)).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Create key" })).toBeNull();
    expect(vi.mocked(apiModule.api).mock.calls.some(([u]) => u === "/integrations/litellm/keys")).toBe(false);
  });

  it("explains a missing master key", async () => {
    setup({ master_key_set: false, master_key_source: "none" });
    expect(await screen.findByText("Not set")).toBeInTheDocument();
    expect(screen.getAllByText("No master key set").length).toBeGreaterThan(0);
    expect(screen.getByRole("button", { name: "Set key" })).toBeInTheDocument();
    expect(vi.mocked(apiModule.api).mock.calls.some(([u]) => u === "/integrations/litellm/keys")).toBe(false);
  });

  it("shows a created key once, with a curl example, and forgets it after closing", async () => {
    const full = "sk-FULLKEY-0123456789";
    const qc = setup({}, (url, opts) => (url === "/integrations/litellm/keys" && opts?.method === "POST"
      ? { ...key, key_alias: "ci-bot", models: ["chat"], key: full } : undefined));
    await userEvent.click(await screen.findByRole("button", { name: "Create key" }));
    await userEvent.type(screen.getByLabelText("Alias"), "ci-bot");
    await userEvent.click(screen.getByLabelText("All models (including ones added later)"));
    await userEvent.click(screen.getByLabelText("chat"));
    await userEvent.type(screen.getByLabelText("Requests per minute"), "60");
    await userEvent.type(screen.getByLabelText("Expires after"), "30d");
    await userEvent.click(screen.getByRole("button", { name: "Create key" }));
    await waitFor(() => expect(calls("POST", "/integrations/litellm/keys")).toHaveLength(1));
    expect(calls("POST", "/integrations/litellm/keys")[0][1]).toEqual({ method: "POST", body: { key_alias: "ci-bot", models: ["chat"], rpm_limit: 60, duration: "30d" } });
    expect(await screen.findByDisplayValue(full)).toBeInTheDocument();
    expect(screen.getByLabelText("curl example")).toHaveTextContent(`Bearer ${full}`);
    expect(screen.getByLabelText("curl example")).toHaveTextContent("https://llm.example.com/v1/chat/completions");
    expect(screen.getByText(/only time the full key is shown/)).toBeInTheDocument();
    fireEvent.pointerDown(document.body); // an outside click must not dismiss the one-time key
    fireEvent.keyDown(document.activeElement ?? document.body, { key: "Escape" }); // nor Escape: it asks first
    expect(screen.getByDisplayValue(full)).toBeInTheDocument();
    expect(await screen.findByText(/Close without copying/)).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "Keep open" }));
    await userEvent.click(screen.getByRole("button", { name: "Done" }));
    expect(screen.getByText(/Close without copying/)).toBeInTheDocument();
    expect(screen.getByDisplayValue(full)).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "Close anyway" }));
    await waitFor(() => expect(screen.queryByDisplayValue(full)).toBeNull());
    expect(document.body.innerHTML).not.toContain(full);
    await waitFor(() => expect(cached(qc, full)).toBe(false));
  });

  it("revokes a key only after the alias is typed", async () => {
    setup({}, (_url, opts) => (opts?.method === "DELETE" ? null : undefined));
    await userEvent.click(await screen.findByRole("button", { name: "Actions for key team-a" }));
    await userEvent.click(await screen.findByRole("menuitem", { name: /Revoke/ }));
    const confirm = await screen.findByRole("button", { name: "Revoke key" });
    expect(confirm).toBeDisabled();
    await userEvent.type(screen.getByLabelText("Type team-a to confirm"), "team-");
    expect(confirm).toBeDisabled();
    await userEvent.type(screen.getByLabelText("Type team-a to confirm"), "a");
    expect(confirm).toBeEnabled();
    await userEvent.click(confirm);
    await waitFor(() => expect(calls("DELETE", `/integrations/litellm/keys/${key.token}`)).toHaveLength(1));
  });

  it("edits a key and sends only the changes", async () => {
    setup({}, (_url, opts) => (opts?.method === "PATCH" ? key : undefined));
    await userEvent.click(await screen.findByRole("button", { name: "Actions for key team-a" }));
    await userEvent.click(await screen.findByRole("menuitem", { name: /Edit/ }));
    const rpm = await screen.findByLabelText("Requests per minute");
    expect(rpm).toHaveValue("10");
    await userEvent.clear(rpm);
    expect(screen.getByText(/cannot remove a limit/)).toBeInTheDocument();
    await userEvent.type(screen.getByLabelText("Tokens per minute"), "5000");
    await userEvent.click(screen.getByRole("button", { name: "Save changes" }));
    await waitFor(() => expect(calls("PATCH", `/integrations/litellm/keys/${key.token}`)).toHaveLength(1));
    expect(calls("PATCH", `/integrations/litellm/keys/${key.token}`)[0][1]).toEqual({ method: "PATCH", body: { tpm_limit: 5000 } });
  });

  it("keeps the one-time key on screen when the key list refetch fails", async () => {
    const full = "sk-KEEP-ME-0123456789";
    let lists = 0;
    setup({}, (url, opts) => {
      if (url === "/integrations/litellm/keys" && opts?.method === "POST") return { ...key, key_alias: "ci-bot", key: full };
      if (url === "/integrations/litellm/keys" && !opts?.method && lists++ > 0) return Promise.reject(new apiModule.ApiError(502, { code: "litellm_unreachable", message: "could not reach LiteLLM" }));
      return undefined;
    });
    await userEvent.click(await screen.findByRole("button", { name: "Create key" }));
    await userEvent.type(screen.getByLabelText("Alias"), "ci-bot");
    await userEvent.click(screen.getByRole("button", { name: "Create key" }));
    expect(await screen.findByDisplayValue(full)).toBeInTheDocument();
    await waitFor(() => expect(lists).toBeGreaterThan(1));
    await new Promise((r) => setTimeout(r, 20));
    expect(screen.getByDisplayValue(full)).toBeInTheDocument();
  });

  it("closes the created-key dialog straight away once the key was copied", async () => {
    const full = "sk-COPIED-0123456789";
    Object.defineProperty(navigator, "clipboard", { configurable: true, value: { writeText: vi.fn().mockResolvedValue(undefined) } });
    setup({}, (url, opts) => (url === "/integrations/litellm/keys" && opts?.method === "POST" ? { ...key, key: full } : undefined));
    await userEvent.click(await screen.findByRole("button", { name: "Create key" }));
    await userEvent.type(screen.getByLabelText("Alias"), "x");
    await userEvent.click(screen.getByRole("button", { name: "Create key" }));
    await userEvent.click(await screen.findByRole("button", { name: "Copy key" }));
    await userEvent.click(screen.getByRole("button", { name: "Done" }));
    await waitFor(() => expect(screen.queryByDisplayValue(full)).toBeNull());
  });

  it("lets a key without an alias be edited", async () => {
    const plain = { ...key, key_alias: null };
    setup({}, (url, opts) => {
      if (url === "/integrations/litellm/keys" && !opts?.method) return [plain];
      if (opts?.method === "PATCH") return plain;
      return undefined;
    });
    await userEvent.click(await screen.findByRole("button", { name: "Actions for key sk-...HWqw" }));
    await userEvent.click(await screen.findByRole("menuitem", { name: /Edit/ }));
    await userEvent.type(await screen.findByLabelText("Tokens per minute"), "100");
    await userEvent.click(screen.getByRole("button", { name: "Save changes" }));
    await waitFor(() => expect(calls("PATCH", `/integrations/litellm/keys/${key.token}`)).toHaveLength(1));
    expect(calls("PATCH", `/integrations/litellm/keys/${key.token}`)[0][1]).toEqual({ method: "PATCH", body: { tpm_limit: 100 } });
  });

  it("shows no key hint when the internal URL is emptied back to the default it already is", async () => {
    setup();
    await userEvent.clear(await screen.findByLabelText("Internal URL"));
    expect(screen.queryByTestId("master-key-hint")).toBeNull();
    expect(screen.getByRole("button", { name: "Save" })).toBeDisabled();
  });

  it("renders the form, models and keys before the status probe answers", async () => {
    setup({ status: null }, (url) => (url === "/integrations/litellm/status" ? new Promise(() => undefined) : undefined));
    expect(await screen.findByLabelText("Internal URL")).toHaveValue("http://litellm-proxy:4000");
    expect(screen.getByText(/Checking LiteLLM/)).toBeInTheDocument();
    expect(await screen.findByText("hand-written")).toBeInTheDocument();
    expect(await screen.findByText("team-a")).toBeInTheDocument();
    expect(screen.queryByText("Cannot reach LiteLLM")).toBeNull();
    expect(calls(undefined as any, "/integrations/litellm")).toHaveLength(0);  // never the probing settings read
  });

  it("shows the LiteLLM version from the status probe", async () => {
    setup({ status: null }, (url) => (url === "/integrations/litellm/status" ? okStatus : undefined));
    expect(await screen.findByText(/LiteLLM 1\.100\.0/)).toBeInTheDocument();
    expect(screen.queryByText(/version unknown/)).toBeNull();
  });

  it("says turning the integration off removes DGXMM's models, and warns when some may remain", async () => {
    const user = userEvent.setup();
    const warning = "1 DGXMM model may remain registered in http://litellm-proxy:4000; remove them there, or re-enable the integration with that URL to clean up";
    setup({}, (url, opts) => (url === "/integrations/litellm" && opts?.method === "PUT"
      ? { ...settings({ enabled: false }), deregistered: 0, deregister_error: "could not reach LiteLLM", warning } : undefined));
    expect(await screen.findByText(/Turning it off \(or changing the internal URL\) first removes the models DGXMM registered there/)).toBeInTheDocument();
    await user.click(screen.getByRole("switch", { name: "Integration enabled" }));
    await user.click(screen.getByRole("button", { name: "Save" }));
    expect(await screen.findByText(warning)).toBeInTheDocument();
    expect(screen.getByText(/while the integration is off; turning it off removes the models DGXMM registered there/)).toBeInTheDocument();
  });

  it("shows routing advice per traffic class with copyable alias snippets, and changes nothing", async () => {
    const advice = {
      enabled: true, source: "model", computed_at: "2026-10-17T05:15:00Z",
      snippet: 'router_settings:\n  model_group_alias:\n    dgxmm-chat: "chat"',
      classes: [
        { class: "chat", label: "General chat and assistants", alias: "dgxmm-chat", deployment_id: 1, deployment: "qwen", model_name: "chat", model_ref: "Qwen/Qwen3-8B",
          source: "model", confidence: 0.91, reason: "ranked first by the decision model", routable: true, note: null, candidates: [{ deployment_id: 1, deployment: "qwen", model_name: "chat" }, { deployment_id: 2, deployment: "b", model_name: "b" }] },
        { class: "long_context", label: "Long documents", alias: "dgxmm-long-context", deployment_id: 2, deployment: "b", model_name: "b", model_ref: "org/b",
          source: "rules", confidence: null, reason: "the longest context", routable: false, note: "would suit; start b first (LiteLLM serves only ready deployments)", candidates: [{ deployment_id: 2, deployment: "b", model_name: "b" }] },
        { class: "vision", label: "Images in the prompt", alias: "dgxmm-vision", deployment_id: null, deployment: null, model_name: null, model_ref: null,
          source: "none", confidence: null, reason: "no deployment has this capability", routable: false, note: null, candidates: [] },
      ],
    };
    setup({}, (url) => (url === "/decider/routing-advice" ? advice : undefined));
    const card = await screen.findByLabelText("Routing advice");
    const chat = await within(card).findByText("dgxmm-chat");
    const row = chat.closest("tr") as HTMLElement;
    expect(within(row).getByText("Decision model · 91 %")).toBeInTheDocument();
    expect(within(row).getByText(/2 candidates/)).toBeInTheDocument();
    expect(within(row).getByRole("button", { name: "Copy dgxmm-chat snippet" })).toBeInTheDocument();
    const vision = within(card).getByText("dgxmm-vision").closest("tr") as HTMLElement;
    expect(within(vision).getByText("No suitable deployment")).toBeInTheDocument();
    expect(within(vision).queryByRole("button")).toBeNull();
    expect(within(card).getByLabelText("LiteLLM alias snippet")).toHaveTextContent('dgxmm-chat: "chat"');
    const long = within(card).getByText("dgxmm-long-context").closest("tr") as HTMLElement;
    expect(within(long).getByText(/start b first/)).toBeInTheDocument();
    expect(within(long).queryByRole("button")).toBeNull();
    expect(vi.mocked(apiModule.api).mock.calls.filter(([u, o]) => String(u).includes("routing") && (o as any)?.method)).toEqual([]);
  });
});
