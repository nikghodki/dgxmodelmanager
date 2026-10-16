import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter } from "react-router-dom";
import { Audit } from "./Audit";
import * as auth from "@/auth";
import * as apiModule from "@/api";
import { TooltipProvider } from "@/components/ui/tooltip";

vi.mock("@/auth");
vi.mock("@/api", async () => {
  const actual = await vi.importActual<typeof import("@/api")>("@/api");
  return { ...actual, api: vi.fn(() => new Promise(() => undefined)) };
});

function renderAudit() {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={qc}>
      <TooltipProvider>
        <MemoryRouter future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
          <Audit />
        </MemoryRouter>
      </TooltipProvider>
    </QueryClientProvider>
  );
}

describe("Audit", () => {
  it("redirects viewers to home", () => {
    vi.mocked(auth.useIsAdmin).mockReturnValue(false);
    const { container } = renderAudit();
    expect(container.textContent).toBe("");
  });

  it("renders for admins", () => {
    vi.mocked(auth.useIsAdmin).mockReturnValue(true);
    renderAudit();
    expect(screen.getByText("Audit log")).toBeInTheDocument();
  });

  it("renders identical-looking entries without key collisions", async () => {
    vi.mocked(auth.useIsAdmin).mockReturnValue(true);
    const errors = vi.spyOn(console, "error").mockImplementation(() => undefined);
    const row = { ts: "2026-10-16T10:00:00", user: "admin", action: "deployment.sleep", target: "qwen", detail: {} };
    vi.mocked(apiModule.api).mockResolvedValueOnce([row, { ...row }]);
    renderAudit();
    expect(await screen.findAllByText("qwen")).toHaveLength(2);
    expect(errors.mock.calls.map((c) => String(c[0])).join("\n")).not.toMatch(/same key/);
    errors.mockRestore();
  });
});
