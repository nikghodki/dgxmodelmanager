import { describe, expect, it, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter, Outlet, useLocation } from "react-router-dom";
import { AuthProvider } from "@/auth";
import { AppRoutes } from "@/routes";
import * as apiModule from "@/api";

vi.mock("@/api", async () => {
  const actual = await vi.importActual<typeof import("@/api")>("@/api");
  return { ...actual, api: vi.fn() };
});
vi.mock("@/components/app/app-shell", () => ({
  AppShell: () => <div data-testid="app-shell"><Outlet /></div>,
}));

function Where() { return <div data-testid="where">{useLocation().pathname}</div>; }

function renderAt(path: string) {
  vi.mocked(apiModule.api).mockImplementation((url: string) => {
    if (url === "/auth/me") return Promise.resolve({ csrf_token: "t", user: { id: 1, username: "admin", role: "admin", must_change_pw: true } });
    return new Promise(() => undefined);
  });
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={qc}>
      <MemoryRouter initialEntries={[path]} future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
        <AuthProvider><AppRoutes /><Where /></AuthProvider>
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

describe("routes", () => {
  it("renders /change-password outside the app shell", async () => {
    renderAt("/change-password");
    expect(await screen.findByText("Set a new password")).toBeInTheDocument();
    expect(screen.queryByTestId("app-shell")).toBeNull();
  });

  it("still sends must-change-password users to /change-password", async () => {
    renderAt("/");
    expect(await screen.findByText("Set a new password")).toBeInTheDocument();
    expect(screen.getByTestId("where")).toHaveTextContent("/change-password");
  });

  it("serves the admin LiteLLM page at /litellm", async () => {
    vi.mocked(apiModule.api).mockImplementation((url: string) => {
      if (url === "/auth/me") return Promise.resolve({ csrf_token: "t", user: { id: 1, username: "admin", role: "admin", must_change_pw: false } });
      return new Promise(() => undefined);
    });
    const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(
      <QueryClientProvider client={qc}>
        <MemoryRouter initialEntries={["/litellm"]} future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
          <AuthProvider><AppRoutes /><Where /></AuthProvider>
        </MemoryRouter>
      </QueryClientProvider>,
    );
    await waitFor(() => expect(vi.mocked(apiModule.api).mock.calls.some(([u]) => u === "/integrations/litellm?probe=false")).toBe(true));
    expect(screen.getByTestId("where")).toHaveTextContent("/litellm");
  });
});
