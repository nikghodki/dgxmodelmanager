import { afterEach, describe, expect, it, vi } from "vitest";
import { render, screen, within } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter } from "react-router-dom";
import * as auth from "@/auth";
import * as apiModule from "@/api";
import { insight } from "@/test/insights";
import type { Insight } from "@/types";
import { InsightList } from "./insight-list";
import "./schedules-card"; // registers the schedule dialog, as main.tsx does

vi.mock("@/auth");
vi.mock("@/api", async () => {
  const actual = await vi.importActual<typeof import("@/api")>("@/api");
  return { ...actual, api: vi.fn(() => new Promise(() => undefined)) };
});
vi.mock("sonner", () => ({ toast: { success: vi.fn(), error: vi.fn() } }));

afterEach(() => vi.mocked(apiModule.api).mockClear());

// One insight per producing feature, with the action shapes the backend builds (dgxmm/decider/health.py, reading.py,
// tuning.py, routing.py, schedule.py, insight_engine.py).
const proposal = { sleep_action: "stop", sleep_at: "19:00", sleep_days: 31, wake_action: "start", wake_at: "07:45", wake_days: 31, timezone: "UTC" };
const KINDS: [Insight, string | null][] = [
  [insight({ id: 1, title: "Health: KV cache pressure", action: { type: "reconfigure", deployment_id: 4, patch: { kv_cache_dtype: "fp8" }, label: "Use fp8 KV cache" } }), "Use fp8 KV cache"],
  [insight({ id: 2, kind: "health", title: "Health: error creep", action: { type: "open", path: "/deployments/4", label: "Open deployment" } }), "Open deployment"],
  [insight({ id: 3, kind: "reading", feature: "reading", title: "Broken benchmark reading", action: { type: "open", path: "/deployments/4", label: "Open deployment" } }), "Open deployment"],
  [insight({ id: 4, kind: "tuning", feature: "tuning", subject: "feature:plan", severity: "info", title: "Plan threshold", action: { type: "threshold", feature: "plan", value: 0.7 } }), "Set threshold…"],
  [insight({ id: 5, kind: "routing", feature: "routing", subject: "litellm", severity: "info", title: "Routing advice", action: { type: "copy", text: "router_settings: {}", label: "Copy LiteLLM snippet" } }), "Copy LiteLLM snippet"],
  [insight({ id: 6, kind: "routing", feature: "routing", subject: "litellm", severity: "info", title: "Routing advice (none routable)", action: { type: "open", path: "/litellm", label: "Open LiteLLM" } }), "Open LiteLLM"],
  [insight({ id: 7, kind: "schedule_proposal", feature: "schedule", severity: "info", title: "Stop chat on a schedule?", action: { type: "schedule", deployment_id: 4, proposal } }), null],
  [insight({ id: 8, kind: "schedule_failed", feature: "schedule", title: "Scheduled sleep failed", action: { type: "schedule", deployment_id: 4, label: "Use stop / start", proposal } }), "Use stop / start"],
  [insight({ id: 9, kind: "degraded", feature: "health", subject: "job:health_check", title: "Health triage is degraded", action: null }), null],
];

describe("insights of every feature", () => {
  it("render with their action, and admins get one action control each", () => {
    vi.mocked(auth.useIsAdmin).mockReturnValue(true);
    const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(<QueryClientProvider client={qc}><MemoryRouter future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
      <InsightList insights={KINDS.map(([i]) => i)} showSubject />
    </MemoryRouter></QueryClientProvider>);
    const items = screen.getAllByRole("listitem");
    expect(items).toHaveLength(KINDS.length);
    KINDS.forEach(([i, label], n) => {
      const item = items[n];
      expect(within(item).getAllByText(i.title).length).toBeGreaterThan(0);
      if (label) expect(within(item).getByRole(i.action?.type === "open" ? "link" : "button", { name: new RegExp(label.replace("…", ".")) })).toBeInTheDocument();
      if (i.action?.type === "schedule") expect(within(item).getAllByRole("button").length).toBeGreaterThan(2); // action + snooze + dismiss
      if (!i.action) expect(within(item).queryByRole("link", { name: /Open|Review|Set|Copy/ })).toBeNull();
    });
  });

  it("viewers see every kind without action controls", () => {
    vi.mocked(auth.useIsAdmin).mockReturnValue(false);
    const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(<QueryClientProvider client={qc}><MemoryRouter future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
      <InsightList insights={KINDS.map(([i]) => ({ ...i, action: null }))} />
    </MemoryRouter></QueryClientProvider>);
    expect(screen.getAllByRole("listitem")).toHaveLength(KINDS.length);
    expect(screen.queryByRole("button")).toBeNull();
  });
});
