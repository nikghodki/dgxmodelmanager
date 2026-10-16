import { describe, expect, it, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { StatusBadge } from "@/components/app/status-badge";
import { ConfirmDialog } from "@/components/app/confirm-dialog";
import { useActionMutation } from "@/hooks/use-action-mutation";
import { ApiError } from "@/api";

const toast = vi.hoisted(() => ({ success: vi.fn(), error: vi.fn() }));
vi.mock("sonner", async (orig) => ({ ...(await orig<typeof import("sonner")>()), toast }));

describe("StatusBadge", () => {
  it("shows icon + text, never colour alone", () => {
    render(<StatusBadge state="failed" />);
    const el = screen.getByText("Failed");
    expect(el.closest("[data-status]")).toHaveAttribute("data-status", "critical");
    expect(el.closest("[data-status]")!.querySelector("svg")).not.toBeNull();
  });
});

describe("ConfirmDialog", () => {
  it("requires typing the name before a destructive confirm", async () => {
    const onConfirm = vi.fn();
    render(<ConfirmDialog open onOpenChange={() => undefined} title="Stop qwen?" description="x" confirmLabel="Stop"
      destructive requireText="qwen" onConfirm={onConfirm} />);
    const btn = screen.getByRole("button", { name: "Stop" });
    expect(btn).toBeDisabled();
    await userEvent.type(screen.getByLabelText(/Type qwen to confirm/), "qwen");
    expect(btn).toBeEnabled();
    await userEvent.click(btn);
    expect(onConfirm).toHaveBeenCalledOnce();
  });
});

function Runner({ fn }: { fn: () => Promise<unknown> }) {
  const m = useActionMutation(fn, { success: "Done!" });
  return <button onClick={() => m.mutate(undefined)} disabled={m.isPending}>go</button>;
}

describe("useActionMutation", () => {
  const wrap = (ui: React.ReactNode) => render(<QueryClientProvider client={new QueryClient()}>{ui}</QueryClientProvider>);
  it("toasts success", async () => {
    wrap(<Runner fn={() => Promise.resolve(1)} />);
    await userEvent.click(screen.getByText("go"));
    await waitFor(() => expect(toast.success).toHaveBeenCalledWith("Done!"));
  });
  it("toasts the server detail message on error and re-enables", async () => {
    wrap(<Runner fn={() => Promise.reject(new ApiError(409, { code: "insufficient_memory", message: "needs 50.0 GiB but only 10.0 GiB is free" }))} />);
    await userEvent.click(screen.getByText("go"));
    await waitFor(() => expect(toast.error).toHaveBeenCalledWith("needs 50.0 GiB but only 10.0 GiB is free"));
    expect(screen.getByText("go")).toBeEnabled();
  });
});
