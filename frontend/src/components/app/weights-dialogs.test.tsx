import { afterEach, describe, expect, it, vi } from "vitest";
import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { CleanupIncompleteDialog, DeleteWeightsDialog } from "./weights-dialogs";
import * as apiModule from "@/api";
import type { Artifact, CleanupItem, CleanupPreview, DeletePreview } from "@/types";

vi.mock("@/api", async () => {
  const actual = await vi.importActual<typeof import("@/api")>("@/api");
  return { ...actual, api: vi.fn() };
});

const item = (ref: string, eligible = true): CleanupItem => ({
  ref, source: "hf", name: ref.split("/")[1], problems: [], paths: [`/h/${ref}`], bytes: 1024, notes: [],
  deployments_blocking: [], deployments_affected: [], download_active: false, recently_active: false, eligible, skip_reason: eligible ? null : "download_active",
});
const preview = (items: CleanupItem[]): CleanupPreview => ({ items, count: items.filter((i) => i.eligible).length, bytes: 0 });

describe("CleanupIncompleteDialog", () => {
  it("keeps the user's deselection when the preview refetches", async () => {
    const user = userEvent.setup();
    const qc = new QueryClient();
    const ui = (p: CleanupPreview, open = true) => (
      <QueryClientProvider client={qc}><CleanupIncompleteDialog open={open} onOpenChange={() => {}} preview={p} /></QueryClientProvider>
    );
    const { rerender } = render(ui(preview([item("org/a"), item("org/b")])));
    const dialog = await screen.findByRole("dialog");
    await user.click(within(dialog).getByRole("checkbox", { name: /org\/a/ }));
    expect(within(dialog).getByRole("checkbox", { name: /org\/a/ })).not.toBeChecked();
    rerender(ui(preview([item("org/a"), item("org/b"), item("org/c")])));   // new object, same + extra items
    expect(within(screen.getByRole("dialog")).getByRole("checkbox", { name: /org\/a/ })).not.toBeChecked();
    expect(within(screen.getByRole("dialog")).getByRole("checkbox", { name: /org\/b/ })).toBeChecked();
    rerender(ui(preview([item("org/a"), item("org/b", false)])));         // b became ineligible
    expect(within(screen.getByRole("dialog")).getByRole("checkbox", { name: /org\/b/ })).not.toBeChecked();
    expect(within(screen.getByRole("dialog")).getByRole("button", { name: /Clean up 0/ })).toBeDisabled();
  });

  it("re-selects all eligible items when reopened", async () => {
    const user = userEvent.setup();
    const qc = new QueryClient();
    const p = preview([item("org/a"), item("org/b")]);
    const ui = (open: boolean) => (
      <QueryClientProvider client={qc}><CleanupIncompleteDialog open={open} onOpenChange={() => {}} preview={p} /></QueryClientProvider>
    );
    const { rerender } = render(ui(true));
    await user.click(within(await screen.findByRole("dialog")).getByRole("checkbox", { name: /org\/a/ }));
    rerender(ui(false));
    rerender(ui(true));
    expect(within(await screen.findByRole("dialog")).getByRole("checkbox", { name: /org\/a/ })).toBeChecked();
  });
});

describe("DeleteWeightsDialog", () => {
  afterEach(() => vi.mocked(apiModule.api).mockReset());
  const dp = (over: Partial<DeletePreview> = {}): DeletePreview => ({
    ref: "org/fresh", source: "hf", name: "fresh", paths: ["/h/org/fresh"], bytes: 2048, deployments_blocking: [],
    deployments_affected: [], download_active: false, recently_active: false, blocked: false, notes: [], ...over,
  });
  const show = (p: DeletePreview, mode: "delete" | "cleanup") => {
    vi.mocked(apiModule.api).mockImplementation(async () => p);
    const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(<QueryClientProvider client={qc}><DeleteWeightsDialog artifact={{ ref: p.ref } as Artifact} mode={mode} onClose={() => {}} /></QueryClientProvider>);
  };

  it("blocks a single clean-up while partial files are recent, with the reason", async () => {
    show(dp({ recently_active: true }), "cleanup");
    expect(await screen.findByText(/written in the last 10 minutes/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Clean up" })).toBeDisabled();
  });

  it("allows a clean-up of old partial files", async () => {
    show(dp(), "cleanup");
    await screen.findByText(/Removes the partial download/);
    expect(screen.getByRole("button", { name: "Clean up" })).toBeEnabled();
  });
});
