import { afterEach, describe, expect, it, vi } from "vitest";
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter } from "react-router-dom";
import * as auth from "@/auth";
import * as apiModule from "@/api";
import { insight } from "@/test/insights";
import { InsightList } from "./insight-list";
import { InsightsBadge } from "./insights-badge";
import { DeploymentInsightsCard } from "./insights-card";

vi.mock("@/auth");
vi.mock("@/api", async () => {
  const actual = await vi.importActual<typeof import("@/api")>("@/api");
  return { ...actual, api: vi.fn() };
});
vi.mock("sonner", () => ({ toast: { success: vi.fn(), error: vi.fn() } }));


function mount(ui: React.ReactNode, { admin = true, handler }: { admin?: boolean; handler?: (url: string, opts?: any) => any } = {}) {
  vi.mocked(auth.useIsAdmin).mockReturnValue(admin);
  vi.mocked(apiModule.api).mockImplementation(async (url: string, opts?: any) => {
    const r = handler?.(url, opts);
    return r !== undefined ? r : new Promise(() => undefined);
  });
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } });
  render(<QueryClientProvider client={qc}><MemoryRouter future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>{ui}</MemoryRouter></QueryClientProvider>);
  return qc;
}
const calls = (method: string, url: string) => vi.mocked(apiModule.api).mock.calls.filter(([u, o]) => u === url && (o as any)?.method === method);

afterEach(() => vi.mocked(apiModule.api).mockReset());

describe("InsightList", () => {
  it("renders the body with its structure (lines, bullets, tables) instead of one run-together paragraph", () => {
    const body = "Routing advice:\n\n- a (chat): m1 on deployment d1 (rules)\n- b (batch): m2 on deployment d2 (rules)\n\nthreshold | answers\n--- | ---\n0.80 (current) | 12";
    mount(<InsightList insights={[insight({ body })]} />);
    const item = screen.getAllByRole("listitem")[0];
    expect(within(item).getByText("a (chat): m1 on deployment d1 (rules)").tagName).toBe("LI");
    expect(within(item).getByRole("table")).toBeInTheDocument();
    expect(within(item).getByRole("cell", { name: "0.80 (current)" })).toBeInTheDocument();
  });

  it("shows severity, title, body, source chip and a deployment link", () => {
    mount(<InsightList insights={[insight({ source: "model", confidence: 0.92 })]} />);
    const item = screen.getByRole("listitem");
    expect(within(item).getByText("Warning")).toBeInTheDocument();
    expect(within(item).getByText("KV cache pressure")).toBeInTheDocument();
    expect(within(item).getByText(/stayed above 95%/)).toBeInTheDocument();
    expect(within(item).getByText("92% · AI")).toBeInTheDocument();
    expect(within(item).getByRole("link", { name: "deployment:4" })).toHaveAttribute("href", "/deployments/4");
  });

  it("gives viewers no action, snooze or dismiss buttons", () => {
    mount(<InsightList insights={[insight()]} />, { admin: false });
    expect(screen.queryByRole("button")).toBeNull();
  });

  it("dismisses and snoozes for admins", async () => {
    mount(<InsightList insights={[insight()]} />, { handler: (url) => (url.startsWith("/insights/1/") ? insight({ status: "dismissed" }) : undefined) });
    await userEvent.click(screen.getByRole("button", { name: "Dismiss KV cache pressure" }));
    await waitFor(() => expect(calls("POST", "/insights/1/dismiss")).toHaveLength(1));
    await userEvent.click(screen.getByRole("button", { name: "Snooze KV cache pressure" }));
    await userEvent.click(await screen.findByRole("menuitem", { name: "1 day" }));
    await waitFor(() => expect(vi.mocked(apiModule.api)).toHaveBeenCalledWith("/insights/1/snooze", { method: "POST", body: { hours: 24 } }));
  });

  it("threshold action: confirm, PUT settings, then mark applied", async () => {
    const i = insight({ id: 7, kind: "tuning", feature: "tuning", subject: "feature:plan", action: { type: "threshold", feature: "plan", value: 0.7 } });
    mount(<InsightList insights={[i]} />, {
      handler: (url, opts) => {
        if (url === "/decider/status") return { features: { plan: { enabled: true, threshold: 0.6 } }, feature_specs: [{ name: "plan", label: "Plan recommendation" }] };
        if (url === "/decider/settings" && opts?.method === "PUT") return {};
        if (url === "/insights/7/applied") return { ...i, status: "applied" };
        return undefined;
      },
    });
    await userEvent.click(screen.getByRole("button", { name: "Set threshold…" }));
    const dlg = await screen.findByRole("alertdialog");
    await waitFor(() => expect(within(dlg).getByText(/Plan recommendation threshold to 70%/)).toBeInTheDocument());
    expect(within(dlg).getByText(/now 60%/)).toBeInTheDocument();
    await userEvent.click(within(dlg).getByRole("button", { name: "Set threshold" }));
    await waitFor(() => expect(calls("POST", "/insights/7/applied")).toHaveLength(1));
    expect(calls("PUT", "/decider/settings")[0][1]).toEqual({ method: "PUT", body: { features: { plan: { threshold: 0.7 } } } });
  });

  it("copy action shows the snippet; applying is the user's call", async () => {
    const i = insight({ id: 8, kind: "routing", feature: "routing", action: { type: "copy", text: "model_list: []", label: "Show snippet…" } });
    mount(<InsightList insights={[i]} />, { handler: (url) => (url === "/insights/8/applied" ? { ...i, status: "applied" } : undefined) });
    await userEvent.click(screen.getByRole("button", { name: "Show snippet…" }));
    expect(await screen.findByLabelText("Snippet")).toHaveTextContent("model_list: []");
    expect(calls("POST", "/insights/8/applied")).toHaveLength(0);
    await userEvent.click(screen.getByRole("button", { name: "Mark as applied" }));
    await waitFor(() => expect(calls("POST", "/insights/8/applied")).toHaveLength(1));
  });

  it("reconfigure action opens the deployment's reconfigure dialog prefilled", async () => {
    const dep = { id: 4, name: "chat", engine: "vllm", model_ref: "Qwen/Qwen3-8B", config: { kv_cache_dtype: "auto" }, priority: 0, state: "ready" };
    mount(<InsightList insights={[insight()]} />, { handler: (url) => (url === "/deployments/4" ? dep : url === "/inventory" ? [] : undefined) });
    await userEvent.click(screen.getByRole("button", { name: "Review change…" }));
    expect(await screen.findByText("Reconfigure chat")).toBeInTheDocument();
    expect(screen.getByText("Changes from the insight")).toBeInTheDocument();
    expect(screen.getByText("kv_cache_dtype: auto → fp8")).toBeInTheDocument();
  });

  it("no button for unknown or malformed actions, or closed insights", () => {
    mount(<InsightList insights={[
      insight({ id: 2, title: "A", action: { type: "exec" } as any }),
      insight({ id: 3, title: "B", action: { type: "open", path: "//evil" } }),
      insight({ id: 4, title: "C", status: "expired" }),
    ]} />);
    expect(screen.queryByRole("button", { name: /Review|Open|Set|Show/ })).toBeNull();
    expect(screen.getAllByRole("button", { name: /Dismiss/ })).toHaveLength(2); // the expired one has none
  });

  it("open action is a link", () => {
    mount(<InsightList insights={[insight({ action: { type: "open", path: "/litellm", label: "Open LiteLLM" } })]} />);
    expect(screen.getByRole("link", { name: /Open LiteLLM/ })).toHaveAttribute("href", "/litellm");
  });
});

describe("InsightsBadge", () => {
  it("counts open insights", async () => {
    mount(<InsightsBadge />, { handler: (url) => (url === "/insights?status=open" ? [insight(), insight({ id: 2 })] : undefined) });
    expect(await screen.findByRole("link", { name: "Insights: 2 open" })).toHaveAttribute("href", "/insights");
  });
  it("shows no count when there are none", async () => {
    mount(<InsightsBadge />, { handler: (url) => (url === "/insights?status=open" ? [] : undefined) });
    await waitFor(() => expect(screen.getByRole("link", { name: "Insights" })).toBeInTheDocument());
    expect(document.querySelector("[data-count]")).toBeNull();
  });
});

describe("DeploymentInsightsCard", () => {
  it("lists this deployment's live insights", async () => {
    mount(<DeploymentInsightsCard deploymentId={4} />, {
      handler: (url) => (url === "/insights?status=open%2Csnoozed&subject=deployment%3A4" ? [insight()] : undefined),
    });
    expect(await screen.findByText("KV cache pressure")).toBeInTheDocument();
  });
  it("loading, empty and error states", async () => {
    mount(<DeploymentInsightsCard deploymentId={4} />);
    expect(screen.getByLabelText("Loading insights")).toBeInTheDocument();
  });
  it("empty", async () => {
    mount(<DeploymentInsightsCard deploymentId={5} />, { handler: (url) => (url.startsWith("/insights") ? [] : undefined) });
    expect(await screen.findByText("Nothing to report.")).toBeInTheDocument();
  });
  it("error", async () => {
    mount(<DeploymentInsightsCard deploymentId={6} />, { handler: (url) => (url.startsWith("/insights") ? Promise.reject(new Error("boom")) : undefined) });
    expect(await screen.findByRole("alert")).toHaveTextContent("boom");
  });
});
