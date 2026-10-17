import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter } from "react-router-dom";
import { TooltipProvider } from "@/components/ui/tooltip";
import { DeploymentsTable } from "@/components/app/deployments-table";

const longRef = "/data/hf/hub/models--unsloth--Qwen3-0.6B-GGUF/snapshots/50968a4468ef4233ed78cd7c3de230dd1d61a56b/Qwen3-0.6B-Q8_0.gguf";

describe("DeploymentsTable", () => {
  it("truncates long model refs but keeps the full value in a title", () => {
    const qc = new QueryClient();
    qc.setQueryData(["live", 1], []);
    render(
      <QueryClientProvider client={qc}><MemoryRouter future={{ v7_startTransition: true, v7_relativeSplatPath: true }}><TooltipProvider>
        <DeploymentsTable isAdmin={false} onSwitch={vi.fn()} onReconfigure={vi.fn()}
          deployments={[{ id: 1, name: "small", engine: "llamacpp", model_ref: longRef, state: "ready", reserved_bytes: 1, actual_bytes: 1, litellm_status: "synced", sleep_capable: false, last_error: null } as any]} />
      </TooltipProvider></MemoryRouter></QueryClientProvider>,
    );
    const cell = screen.getByTitle(longRef);
    expect(cell.className).toMatch(/truncate/);
    expect(screen.getByText("Ready")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /Actions for small/ })).toBeInTheDocument();
  });

  it("shows the LiteLLM model name and its registration status", () => {
    const qc = new QueryClient();
    qc.setQueryData(["live", 1], []);
    render(
      <QueryClientProvider client={qc}><MemoryRouter future={{ v7_startTransition: true, v7_relativeSplatPath: true }}><TooltipProvider>
        <DeploymentsTable isAdmin={false} onSwitch={vi.fn()} onReconfigure={vi.fn()}
          deployments={[{ id: 1, name: "small", engine: "vllm", model_ref: "org/m", state: "ready", reserved_bytes: 1, actual_bytes: 1, litellm_status: "pending", model_name: "team-chat", served_name: "small", alias: "team-chat", sleep_capable: false, last_error: null } as any]} />
      </TooltipProvider></MemoryRouter></QueryClientProvider>,
    );
    expect(screen.getByText("team-chat")).toBeInTheDocument();
    expect(screen.getByText("Pending")).toBeInTheDocument();
  });

  it("explains a LiteLLM registration error in a tooltip", async () => {
    const qc = new QueryClient();
    qc.setQueryData(["live", 1], []);
    render(
      <QueryClientProvider client={qc}><MemoryRouter future={{ v7_startTransition: true, v7_relativeSplatPath: true }}><TooltipProvider delayDuration={0}>
        <DeploymentsTable isAdmin={false} onSwitch={vi.fn()} onReconfigure={vi.fn()}
          deployments={[{ id: 1, name: "dup", engine: "vllm", model_ref: "org/m", state: "ready", reserved_bytes: 1, actual_bytes: 1, litellm_status: "error", model_name: "chat", served_name: "chat", alias: null, sleep_capable: false, last_error: null } as any]} />
      </TooltipProvider></MemoryRouter></QueryClientProvider>,
    );
    await userEvent.hover(screen.getByText("Error"));
    expect((await screen.findAllByText(/same model name/)).length).toBeGreaterThan(0);
  });

  it("shows no LiteLLM registration badge for a system deployment, and manages it on the Decision model page", async () => {
    const qc = new QueryClient();
    qc.setQueryData(["live", 1], []);
    render(
      <QueryClientProvider client={qc}><MemoryRouter future={{ v7_startTransition: true, v7_relativeSplatPath: true }}><TooltipProvider delayDuration={0}>
        <DeploymentsTable isAdmin onSwitch={vi.fn()} onReconfigure={vi.fn()}
          deployments={[{ id: 1, name: "decider", engine: "decision", model_ref: "vllm-sr/d3-mini", state: "ready", reserved_bytes: 1, actual_bytes: 1, litellm_status: "none", model_name: null, served_name: "d3-mini", alias: null, sleep_capable: false, last_error: null, system: true } as any]} />
      </TooltipProvider></MemoryRouter></QueryClientProvider>,
    );
    expect(screen.queryByText("Not registered")).toBeNull();
    expect(screen.queryByText("d3-mini")).toBeNull();
    expect(screen.getAllByTitle("System deployments aren't registered in LiteLLM").length).toBeGreaterThan(0);
    await userEvent.click(screen.getByRole("button", { name: "Actions for decider" }));
    expect(await screen.findByRole("menuitem", { name: /Decision model page/ })).toBeInTheDocument();
    for (const name of [/Switch model/, /Reconfigure/, /Start/]) expect(screen.queryByRole("menuitem", { name })).toBeNull();
  });

  function renderRow(row: Record<string, unknown>) {
    const qc = new QueryClient();
    qc.setQueryData(["live", 1], []);
    render(
      <QueryClientProvider client={qc}><MemoryRouter future={{ v7_startTransition: true, v7_relativeSplatPath: true }}><TooltipProvider delayDuration={0}>
        <DeploymentsTable isAdmin={false} onSwitch={vi.fn()} onReconfigure={vi.fn()}
          deployments={[{ id: 1, name: "dup", engine: "vllm", model_ref: "org/m", state: "ready", reserved_bytes: 1, actual_bytes: 1, model_name: "chat", served_name: "chat", alias: null, sleep_capable: false, last_error: null, ...row } as any]} />
      </TooltipProvider></MemoryRouter></QueryClientProvider>,
    );
  }

  it("shows the exact LiteLLM error reason from the deployment list", async () => {
    renderRow({ litellm_status: "error", litellm_error: { code: "model_name_conflict", message: "model name chat is already registered for deployment a; give this deployment a different model name" } });
    await userEvent.hover(screen.getByText("Error"));
    expect((await screen.findAllByText(/already registered for deployment a/)).length).toBeGreaterThan(0);
    expect(screen.queryAllByText(/usually because/)).toHaveLength(0);
  });

  it("explains a pending registration when the last sync failed", async () => {
    renderRow({ litellm_status: "pending", litellm_error: { code: "litellm_unreachable", message: "could not reach LiteLLM at http://litellm-proxy:4000" } });
    const badge = screen.getByText("Pending");
    await userEvent.hover(badge);
    expect((await screen.findAllByText(/could not reach LiteLLM/)).length).toBeGreaterThan(0);
  });

  it("keeps a plain badge without a LiteLLM error", () => {
    renderRow({ litellm_status: "synced", litellm_error: null });
    expect(screen.getByText("Synced").closest("[tabindex]")).toBeNull();
  });

  it("marks system deployments with a System badge", () => {
    const qc = new QueryClient();
    qc.setQueryData(["live", 1], []); qc.setQueryData(["live", 2], []);
    render(
      <QueryClientProvider client={qc}><MemoryRouter future={{ v7_startTransition: true, v7_relativeSplatPath: true }}><TooltipProvider>
        <DeploymentsTable isAdmin={false} onSwitch={vi.fn()} onReconfigure={vi.fn()}
          deployments={[
            { id: 1, name: "decider", engine: "decision", model_ref: "vllm-sr/d3-mini", state: "ready", reserved_bytes: 1, actual_bytes: 1, litellm_status: "disabled", sleep_capable: false, last_error: null, system: true } as any,
            { id: 2, name: "chat", engine: "vllm", model_ref: "org/m", state: "ready", reserved_bytes: 1, actual_bytes: 1, litellm_status: "synced", sleep_capable: false, last_error: null } as any,
          ]} />
      </TooltipProvider></MemoryRouter></QueryClientProvider>,
    );
    expect(screen.getByText("decider").closest("tr")!.textContent).toContain("System");
    expect(screen.getByText("chat").closest("tr")!.textContent).not.toContain("System");
  });

  it("shows why a queued deployment waits, as text and in a tooltip", async () => {
    renderRow({ state: "queued", startup_queue: { position: 1, waiting_for: ["decider"], kind: "launch", since: 0, limit: 1,
      reason: "waiting for decider to finish starting before it can start (one engine starts at a time)" } });
    expect(screen.getByText("Queued")).toBeInTheDocument();
    expect(screen.getByText(/Waiting for decider/)).toBeInTheDocument();
    await userEvent.hover(screen.getByText("Queued"));
    expect((await screen.findAllByText(/one engine starts at a time/)).length).toBeGreaterThan(0);
  });
});
