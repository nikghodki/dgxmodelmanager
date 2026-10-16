import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter } from "react-router-dom";
import { Audit } from "./Audit";
import * as auth from "@/auth";

vi.mock("@/auth");

describe("Audit", () => {
  it("redirects viewers to home", () => {
    vi.mocked(auth.useIsAdmin).mockReturnValue(false);
    const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    const { container } = render(
      <QueryClientProvider client={qc}>
        <MemoryRouter future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
          <Audit />
        </MemoryRouter>
      </QueryClientProvider>
    );
    expect(container.textContent).toBe("");
  });

  it("renders for admins", () => {
    vi.mocked(auth.useIsAdmin).mockReturnValue(true);
    const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(
      <QueryClientProvider client={qc}>
        <MemoryRouter future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
          <Audit />
        </MemoryRouter>
      </QueryClientProvider>
    );
    expect(screen.getByText("Audit log")).toBeInTheDocument();
  });
});
