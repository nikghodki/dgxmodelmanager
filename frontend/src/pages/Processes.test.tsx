import { describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen, within } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter } from "react-router-dom";
import { Processes } from "./Processes";
import * as apiModule from "@/api";
import { TooltipProvider } from "@/components/ui/tooltip";
import type { ProcessesResponse, ProcessInfo } from "@/types";

vi.mock("@/api", async () => {
  const actual = await vi.importActual<typeof import("@/api")>("@/api");
  return { ...actual, api: vi.fn() };
});

const GIB = 1024 ** 3;
const P = (pid: number, over: Partial<ProcessInfo> = {}): ProcessInfo => ({
  pid, ppid: 1, uid: 0, user: "root", user_scope: "container", name: "python3", exe: "python3", command: `python3 app.py --id ${pid}`,
  command_truncated: false, container: null, managed: null, rss: GIB, gpu: 0, ...over,
});
const QWEN = { id: "aaaaaaaaaaaa", name: "dgxmm-qwen", image: "vllm/vllm-openai:v0.11" };
const LITE = { id: "bbbbbbbbbbbb", name: "litellm-proxy", image: "ghcr.io/berriai/litellm:main-latest" };
const body: ProcessesResponse = {
  scope: "all", total: 3, duration_ms: 12, containers: [], host: { rss: 0, gpu: 0, process_count: 0 },
  processes: [
    P(100, { container: QWEN, managed: { deployment_id: 7, name: "qwen" }, command: "vllm serve /m --api-key ***", gpu: 40 * GIB }),
    P(300, { user: "nikghodki-dgx", user_scope: "host", uid: 1000, command: "python train.py --hf-token=***", rss: 3 * GIB }),
    P(200, { container: LITE, command: "litellm --config /app/config.yaml --port 4000", user: "uid 1000", uid: 1000 }),
  ],
};

function renderPage() {
  vi.mocked(apiModule.api).mockResolvedValue(body);
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={qc}>
      <TooltipProvider>
        <MemoryRouter future={{ v7_startTransition: true, v7_relativeSplatPath: true }}><Processes /></MemoryRouter>
      </TooltipProvider>
    </QueryClientProvider>,
  );
}

describe("Processes", () => {
  it("fetches the full redacted list and groups by container by default", async () => {
    renderPage();
    expect(await screen.findByText("dgxmm-qwen")).toBeInTheDocument();
    expect(vi.mocked(apiModule.api).mock.calls[0][0]).toMatch(/^\/host\/processes\?scope=all&limit=\d+$/);
    expect(screen.getByText("litellm-proxy")).toBeInTheDocument();
    expect(screen.getByText("host")).toBeInTheDocument();
    expect(screen.getByText("vllm/vllm-openai:v0.11")).toBeInTheDocument();
    expect(screen.getAllByText("DGXMM").length).toBeGreaterThan(0); // managed is labelled in text, not colour only
  });

  it("by-process view shows command, container, user and filters by scope and search", async () => {
    renderPage();
    await screen.findByText("dgxmm-qwen");
    fireEvent.mouseDown(screen.getByRole("tab", { name: /by process/i }));
    fireEvent.click(screen.getByRole("tab", { name: /by process/i }));
    expect(await screen.findByText("vllm serve /m --api-key ***")).toBeInTheDocument();
    const row = screen.getByText("python train.py --hf-token=***").closest("tr")!;
    expect(within(row).getByText("host")).toBeInTheDocument();
    expect(within(row).getByText("nikghodki-dgx")).toBeInTheDocument();
    expect(within(row).getByText("300")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("radio", { name: /external/i }));
    expect(screen.queryByText("vllm serve /m --api-key ***")).toBeNull();
    expect(screen.getByText("python train.py --hf-token=***")).toBeInTheDocument();

    const lite = screen.getByText("litellm --config /app/config.yaml --port 4000").closest("tr")!;
    const u = within(lite).getByText("uid 1000 (in container)");
    expect(u).toHaveAttribute("title", "uid as seen inside the container; not a host account");

    fireEvent.change(screen.getByLabelText(/search/i), { target: { value: "litellm" } });
    expect(screen.queryByText("python train.py --hf-token=***")).toBeNull();
    expect(screen.getByText("litellm --config /app/config.yaml --port 4000")).toBeInTheDocument();
  });

  it("auto-refresh can be paused (fixed name, state in aria-pressed and in text)", async () => {
    renderPage();
    await screen.findByText("dgxmm-qwen");
    const btn = screen.getByRole("button", { name: "Pause auto-refresh" });
    expect(btn).toHaveAttribute("aria-pressed", "false");
    expect(screen.getByText("Refreshing every 5 s")).toBeInTheDocument();
    fireEvent.click(btn);
    expect(screen.getByRole("button", { name: "Pause auto-refresh" })).toHaveAttribute("aria-pressed", "true");
    expect(screen.getByText("Auto-refresh paused")).toBeInTheDocument();
  });

  it("scope radio group uses roving tabindex and arrow keys", async () => {
    renderPage();
    await screen.findByText("dgxmm-qwen");
    const radios = screen.getAllByRole("radio");
    expect(radios.map((r) => r.getAttribute("tabindex"))).toEqual(["0", "-1", "-1"]);
    fireEvent.keyDown(radios[0], { key: "ArrowRight" });
    expect(screen.getByRole("radio", { name: /external/i })).toHaveAttribute("aria-checked", "true");
    expect(document.activeElement).toBe(screen.getByRole("radio", { name: /external/i }));
    fireEvent.keyDown(document.activeElement!, { key: "ArrowLeft" });
    fireEvent.keyDown(document.activeElement!, { key: "ArrowLeft" });
    expect(screen.getByRole("radio", { name: /managed/i })).toHaveAttribute("aria-checked", "true"); // wraps
    expect(screen.getAllByRole("radio").map((r) => r.getAttribute("tabindex"))).toEqual(["-1", "-1", "0"]);
  });
});
