import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter, Route, Routes, useLocation } from "react-router-dom";
import { CommandPalette } from "@/components/app/command-palette";

let isAdminFlag = false;
vi.mock("@/auth", () => ({
  useAuth: () => ({
    user: { id: 1, username: "viewer", role: isAdminFlag ? "admin" : "viewer", must_change_pw: false },
    loading: false,
    login: vi.fn(),
    logout: vi.fn(),
    reload: vi.fn(),
  }),
  useIsAdmin: () => isAdminFlag,
}));

function Where() { return <div data-testid="where">{useLocation().pathname}</div>; }

describe("CommandPalette", () => {
  it("filters and navigates on Enter", async () => {
    const qc = new QueryClient();
    qc.setQueryData(["deployments"], []);
    qc.setQueryData(["inventory"], []);
    render(
      <QueryClientProvider client={qc}>
        <MemoryRouter initialEntries={["/"]} future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
          <Routes><Route path="*" element={<><CommandPalette open onOpenChange={() => undefined} /><Where /></>} /></Routes>
        </MemoryRouter>
      </QueryClientProvider>,
    );
    await userEvent.type(screen.getByPlaceholderText(/Search pages, deployments, models/), "planner");
    await userEvent.keyboard("{Enter}");
    expect(screen.getByTestId("where")).toHaveTextContent("/planner");
  });

  it("requires typing the name before stopping a ready deployment", async () => {
    isAdminFlag = true;
    const qc = new QueryClient();
    qc.setQueryData(["deployments"], [{ id: 1, name: "qwen", state: "ready", sleep_capable: true, model_ref: "Qwen/Qwen3-8B" }]);
    qc.setQueryData(["inventory"], []);
    render(
      <QueryClientProvider client={qc}>
        <MemoryRouter initialEntries={["/"]} future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
          <CommandPalette open onOpenChange={() => undefined} />
        </MemoryRouter>
      </QueryClientProvider>,
    );
    await userEvent.type(screen.getByPlaceholderText(/Search pages, deployments, models/), "Stop qwen");
    await userEvent.keyboard("{Enter}");
    expect(await screen.findByLabelText("Type qwen to confirm")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Stop" })).toBeDisabled();
    await userEvent.type(screen.getByLabelText("Type qwen to confirm"), "qwen");
    expect(screen.getByRole("button", { name: "Stop" })).toBeEnabled();
    isAdminFlag = false;
  });
});
