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

function renderDetail(advice: any[] = [], { admin = true, litellm = info(), extra }: { admin?: boolean; litellm?: any; extra?: (url: string, opts?: any) => any } = {}) {
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
      <MemoryRouter initialEntries={["/deployments/1"]} future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
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

  it("lets admins rename the LiteLLM model with validation feedback", async () => {
    const patches: any[] = [];
    renderDetail([], { extra: (url, opts) => {
      if (url === "/deployments/1" && opts?.method === "PATCH") {
        patches.push(opts.body);
        return opts.body.alias === "taken" ? Promise.reject(new apiModule.ApiError(409, { code: "alias_in_use", message: "model name taken is already used by deployment other" }))
          : Promise.resolve({ ...dep, alias: opts.body.alias, model_name: opts.body.alias });
      }
      return undefined;
    } });
    await userEvent.click(await screen.findByRole("tab", { name: /LiteLLM/ }));
    const input = await screen.findByLabelText("Model name");
    expect(input).toHaveValue("chat");
    const save = screen.getByRole("button", { name: "Save model name" });
    expect(save).toBeDisabled();
    await userEvent.clear(input);
    await userEvent.type(input, "bad name");
    expect(screen.getByText(/no spaces/)).toBeInTheDocument();
    expect(save).toBeDisabled();
    await userEvent.clear(input);
    await userEvent.type(input, "taken");
    await userEvent.click(save);
    expect(await screen.findByText(/already used by deployment other/)).toBeInTheDocument();
    await userEvent.clear(input);
    expect(screen.getByText(/served name \(qwen\)/)).toBeInTheDocument();
    await userEvent.type(input, "team-chat");
    await userEvent.click(save);
    await waitFor(() => expect(patches).toEqual([{ alias: "taken" }, { alias: "team-chat" }]));
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
});
