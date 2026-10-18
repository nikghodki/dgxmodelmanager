import { afterEach, describe, expect, it, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter } from "react-router-dom";
import * as auth from "@/auth";
import * as apiModule from "@/api";
import { insight } from "@/test/insights";
import { Insights } from "./Insights";

vi.mock("@/auth");
vi.mock("@/api", async () => {
  const actual = await vi.importActual<typeof import("@/api")>("@/api");
  return { ...actual, api: vi.fn() };
});
vi.mock("sonner", () => ({ toast: { success: vi.fn(), error: vi.fn() } }));

function mount(handler: (url: string) => any, admin = true) {
  vi.mocked(auth.useIsAdmin).mockReturnValue(admin);
  vi.mocked(apiModule.api).mockImplementation(async (url: string) => {
    const r = handler(url);
    return r !== undefined ? r : new Promise(() => undefined);
  });
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(<QueryClientProvider client={qc}><MemoryRouter future={{ v7_startTransition: true, v7_relativeSplatPath: true }}><Insights /></MemoryRouter></QueryClientProvider>);
}

afterEach(() => vi.mocked(apiModule.api).mockReset());

describe("Insights page", () => {
  it("loading state", () => {
    mount(() => undefined);
    expect(screen.getByLabelText("Loading insights")).toBeInTheDocument();
  });

  it("empty state", async () => {
    mount((url) => (url.startsWith("/insights") ? [] : undefined));
    expect(await screen.findByText("No open insights")).toBeInTheDocument();
  });

  it("error state", async () => {
    mount((url) => (url.startsWith("/insights") ? Promise.reject(new Error("server down")) : undefined));
    expect(await screen.findByRole("alert")).toHaveTextContent("server down");
  });

  it("lists live insights by default and filters by severity", async () => {
    mount((url) => (url === "/insights?status=open%2Csnoozed" ? [
      insight({ id: 1, severity: "critical", title: "Errors rising" }),
      insight({ id: 2, severity: "info", title: "Sleep at night?", kind: "proposal", feature: "schedule" }),
    ] : undefined));
    expect(await screen.findByText("Errors rising")).toBeInTheDocument();
    expect(screen.getByText("Sleep at night?")).toBeInTheDocument();
    await userEvent.click(screen.getByRole("combobox", { name: "Filter by severity" }));
    await userEvent.click(await screen.findByRole("option", { name: "Critical" }));
    await waitFor(() => expect(screen.queryByText("Sleep at night?")).toBeNull());
    expect(screen.getByText("Errors rising")).toBeInTheDocument();
  });

  it("status filter re-queries the server", async () => {
    mount((url) => (url.startsWith("/insights") ? [] : undefined));
    await screen.findByText("No open insights");
    await userEvent.click(screen.getByRole("combobox", { name: "Filter by status" }));
    await userEvent.click(await screen.findByRole("option", { name: "Dismissed" }));
    await waitFor(() => expect(vi.mocked(apiModule.api)).toHaveBeenCalledWith("/insights?status=dismissed"));
    expect(await screen.findByText("No insights match these filters")).toBeInTheDocument();
  });

  it("viewers see insights without action buttons", async () => {
    mount((url) => (url.startsWith("/insights") ? [insight({ action: null })] : undefined), false);
    expect(await screen.findByText("KV cache pressure")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /Dismiss|Snooze|Review/ })).toBeNull();
  });
});
