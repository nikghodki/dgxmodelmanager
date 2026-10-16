import { describe, expect, it } from "vitest";
import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { CleanupIncompleteDialog } from "./weights-dialogs";
import type { CleanupItem, CleanupPreview } from "@/types";

const item = (ref: string, eligible = true): CleanupItem => ({
  ref, source: "hf", name: ref.split("/")[1], problems: [], paths: [`/h/${ref}`], bytes: 1024, notes: [],
  deployments_blocking: [], deployments_affected: [], download_active: false, eligible, skip_reason: eligible ? null : "download_active",
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
