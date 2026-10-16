import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter } from "react-router-dom";
import { Users } from "./Users";
import * as auth from "@/auth";

vi.mock("@/auth");

describe("Users", () => {
  it("redirects viewers to home", () => {
    vi.mocked(auth.useIsAdmin).mockReturnValue(false);
    vi.mocked(auth.useAuth).mockReturnValue({ user: null, loading: false, login: vi.fn(), logout: vi.fn(), reload: vi.fn() });
    const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    const { container } = render(
      <QueryClientProvider client={qc}>
        <MemoryRouter future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
          <Users />
        </MemoryRouter>
      </QueryClientProvider>
    );
    expect(container.textContent).toBe("");
  });

  it("renders for admins", () => {
    vi.mocked(auth.useIsAdmin).mockReturnValue(true);
    vi.mocked(auth.useAuth).mockReturnValue({ user: { id: 1, username: "admin", role: "admin", must_change_pw: false }, loading: false, login: vi.fn(), logout: vi.fn(), reload: vi.fn() });
    const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(
      <QueryClientProvider client={qc}>
        <MemoryRouter future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
          <Users />
        </MemoryRouter>
      </QueryClientProvider>
    );
    expect(screen.getByText("Users")).toBeInTheDocument();
  });
});
