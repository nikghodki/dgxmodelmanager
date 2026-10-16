import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
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
  id: 1, name: "qwen", engine: "vllm", model_ref: "Qwen/Qwen3-8B", state: "ready", priority: 0, alias: null,
  reserved_bytes: 1e9, actual_bytes: 1e9, estimate: {}, config: { max_model_len: 4096 }, litellm_status: "ok",
  last_error: null, exit_code: null, container_id: null, served_name: "qwen", endpoint: null, host_endpoint: null,
  sleep_capable: true, adopted_from: null, created_at: "", updated_at: "", events: [],
};

function renderDetail(advice: any[] = []) {
  vi.mocked(auth.useIsAdmin).mockReturnValue(true);
  vi.mocked(apiModule.api).mockImplementation((url: string) => {
    if (url === "/deployments/1") return Promise.resolve(dep);
    if (url === "/deployments/1/advice") return Promise.resolve(advice);
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
});
