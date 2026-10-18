import { afterEach, describe, expect, it, vi } from "vitest";
import { render, screen, within } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter } from "react-router-dom";
import * as auth from "@/auth";
import * as apiModule from "@/api";
import { DecisionModel } from "./DecisionModel";

vi.mock("@/auth");
vi.mock("@/api", async () => {
  const actual = await vi.importActual<typeof import("@/api")>("@/api");
  return { ...actual, api: vi.fn() };
});
vi.mock("sonner", () => ({ toast: { success: vi.fn(), error: vi.fn() } }));

const spec = (name: string, label: string, extra: Record<string, unknown> = {}) => ({
  name, label, description: `${label} description.`, default_threshold: 0.8, background: false, needs_model: true, viewer_visible: true, members: [], ...extra,
});
const status = {
  enabled: true, healthy: true, deployment_id: 9, state: "ready", model: "vllm-sr/d3-mini", revision: "r", p50_ms: 70, decisions_24h: 1, accepted_rate: null,
  image: { present: true, building: false, error: null },
  features: { diagnosis: { enabled: true, threshold: 0.8 }, parser: { enabled: true, threshold: 0.8 }, health: { enabled: true, threshold: 0.8 }, command: { enabled: false, threshold: 0.7 }, tuning: { enabled: true, threshold: 0.8 } },
  feature_specs: [spec("diagnosis", "Failure diagnosis"), spec("parser", "Parser suggestions", { members: ["parser_tool", "parser_reasoning"] }),
    spec("health", "Health triage"), spec("command", "Natural-language commands"),
    spec("tuning", "Threshold tuning", { needs_model: false, background: true })],
  jobs: [
    { name: "health_check", feature: "health", schedule: "every 300 s", enabled: true, running: false, last_run_at: new Date(Date.now() - 120000).toISOString(),
      last_duration_ms: 41.2, last_result: "ok: 3 deployments", next_run_at: new Date(Date.now() + 180000).toISOString(), failures: 0, runs: 4, skipped: 0, budget_calls: 4, calls_used: 1, calls_refused: 0 },
    { name: "routing_advice", feature: "routing", schedule: "daily 03:00", enabled: true, running: false, last_run_at: null, last_duration_ms: null,
      last_result: "error: RuntimeError", next_run_at: null, failures: 3, runs: 3, skipped: 0, budget_calls: 10, calls_used: 0, calls_refused: 0 },
  ],
};

function mount() {
  vi.mocked(auth.useIsAdmin).mockReturnValue(true);
  vi.mocked(apiModule.api).mockImplementation(async (url: string) => {
    if (url === "/decider/status") return status;
    if (url.startsWith("/decider/decisions")) return [];
    return new Promise(() => undefined);
  });
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(<QueryClientProvider client={qc}><MemoryRouter future={{ v7_startTransition: true, v7_relativeSplatPath: true }}><DecisionModel /></MemoryRouter></QueryClientProvider>);
}

afterEach(() => vi.mocked(apiModule.api).mockReset());

describe("Decision model page — registry-driven", () => {
  it("renders the toggles and thresholds the server lists, with its labels", async () => {
    mount();
    expect(await screen.findByRole("switch", { name: "Health triage" })).toBeChecked();
    expect(screen.getByRole("switch", { name: "Natural-language commands" })).not.toBeChecked();
    expect(screen.getByText("Health triage description.")).toBeInTheDocument();
    expect(screen.getByRole("spinbutton", { name: "Natural-language commands threshold (%)" })).toHaveValue(70);
  });

  it("a rules-only feature keeps its toggle but has no threshold field", async () => {
    mount();
    expect(await screen.findByRole("switch", { name: "Threshold tuning" })).toBeChecked();
    expect(screen.queryByRole("spinbutton", { name: /Threshold tuning threshold/ })).toBeNull();
    const row = screen.getByRole("switch", { name: "Threshold tuning" }).closest("li")!;
    expect(within(row).getByText(/rules only/i)).toBeInTheDocument();
    expect(screen.getByRole("spinbutton", { name: "Health triage threshold (%)" })).toBeInTheDocument();
  });

  it("shows the background jobs with last result and next run", async () => {
    mount();
    const row = (await screen.findByText("health_check")).closest("tr")!;
    expect(within(row).getByText("every 300 s")).toBeInTheDocument();
    expect(within(row).getByText("ok: 3 deployments")).toBeInTheDocument();
    expect(within(row).getByText("in 3 min")).toBeInTheDocument();
    const bad = screen.getByText("routing_advice").closest("tr")!;
    expect(within(bad).getByText(/error: RuntimeError \(3 in a row\)/)).toBeInTheDocument();
  });
});
