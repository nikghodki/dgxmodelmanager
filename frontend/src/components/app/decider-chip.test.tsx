import { afterEach, describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter } from "react-router-dom";
import { DeciderChip } from "./decider-chip";
import userEvent from "@testing-library/user-event";
import * as apiModule from "@/api";
import * as auth from "@/auth";

vi.mock("@/auth");

vi.mock("@/api", async () => {
  const actual = await vi.importActual<typeof import("@/api")>("@/api");
  return { ...actual, api: vi.fn() };
});

const features = { diagnosis: { enabled: true, threshold: 0.8 } };
const status = (over: Record<string, any> = {}) => ({
  enabled: true, healthy: true, deployment_id: 9, state: "ready", model: "vllm-sr/d3-mini", revision: "97dd054d",
  p50_ms: 75, decisions_24h: 4, accepted_rate: null, features, image: { present: true, building: false, error: null }, ...over,
});

afterEach(() => vi.clearAllMocks());

function setup(st: any, admin = true) {
  vi.mocked(auth.useIsAdmin).mockReturnValue(admin);
  vi.mocked(apiModule.api).mockImplementation(async (url: string) => {
    if (url === "/decider/status") {
      if (st instanceof Error) throw st;
      return st;
    }
    return new Promise(() => undefined);
  });
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(
    <QueryClientProvider client={qc}>
      <MemoryRouter future={{ v7_startTransition: true, v7_relativeSplatPath: true }}><DeciderChip /></MemoryRouter>
    </QueryClientProvider>,
  );
  return qc;
}

describe("DeciderChip", () => {
  it.each([
    [status(), "Healthy", "good"],
    [status({ healthy: false, state: "loading" }), "Starting", "warning"],
    [status({ healthy: false, state: "starting" }), "Starting", "warning"],
    [status({ enabled: false, healthy: false, deployment_id: null, state: null }), "Disabled", "neutral"],
    [status({ healthy: false, state: "failed" }), "Error", "critical"],
    [status({ healthy: false, state: "ready" }), "Error", "critical"], // ready but the circuit breaker is open
    [status({ healthy: false, state: "sleeping" }), "Sleeping", "sleeping"],
    [status({ healthy: false, state: "stopping" }), "Stopping", "warning"],
    [status({ healthy: false, state: "queued" }), "Starting", "warning"],
    [status({ healthy: false, state: "stopped" }), "Stopped", "neutral"],
    [status({ healthy: false, state: null, image: { present: false, building: true, error: null } }), "Building", "warning"],
  ])("shows the decider state %#", async (st, label, kind) => {
    setup(st);
    const link = await screen.findByRole("link", { name: `Decision model: ${label}` });
    expect(link).toHaveAttribute("href", "/decision-model");
    expect(link).toHaveAttribute("data-status", kind);
    expect(link).toHaveTextContent(label); // text, never colour only
  });

  it("renders nothing when the status cannot be read", async () => {
    setup(new Error("boom"));
    await new Promise((r) => setTimeout(r, 20));
    expect(screen.queryByRole("link")).toBeNull();
  });

  it("gives viewers a read-only status popover instead of a link to the admin page", async () => {
    setup(status(), false);
    const chip = await screen.findByRole("button", { name: "Decision model: Healthy" });
    expect(screen.queryByRole("link")).toBeNull();
    expect(chip).toHaveAttribute("data-status", "good");
    await userEvent.click(chip);
    expect(await screen.findByText("Only admins can change it.")).toBeInTheDocument();
    expect(screen.getByText("vllm-sr/d3-mini")).toBeInTheDocument();
  });

  it("tells viewers when the decision model is off", async () => {
    setup(status({ enabled: false, healthy: false, deployment_id: null, state: null }), false);
    await userEvent.click(await screen.findByRole("button", { name: "Decision model: Disabled" }));
    expect(await screen.findByText("Off: every feature uses DGXMM's rules.")).toBeInTheDocument();
  });

  it("shares the decision-model page's query so SSE invalidation updates both", async () => {
    const qc = setup(status());
    await screen.findByRole("link", { name: "Decision model: Healthy" });
    expect(qc.getQueryData(["decider-status"])).toMatchObject({ healthy: true });
  });
});
