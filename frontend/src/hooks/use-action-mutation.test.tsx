import { describe, expect, it, vi } from "vitest";
import { act, renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { toast } from "sonner";
import { ApiError } from "@/api";
import { useActionMutation } from "./use-action-mutation";

vi.mock("sonner", () => ({ toast: { success: vi.fn(), error: vi.fn() } }));

const wrapper = ({ children }: { children: React.ReactNode }) => (
  <QueryClientProvider client={new QueryClient({ defaultOptions: { mutations: { retry: false } } })}>{children}</QueryClientProvider>
);

describe("useActionMutation", () => {
  it("explains a model-name conflict in the error toast, names the holder and offers the caller's fix", async () => {
    const err = new ApiError(409, { code: "model_name_taken", message: "the model name chat is already used", holder: { id: 2, name: "other", state: "ready" } });
    const fix = vi.fn();
    const { result } = renderHook(() => useActionMutation((_v: { id: number }) => Promise.reject(err), {
      errorAction: (_e, v) => ({ label: "Set model name", onClick: () => fix(v.id) }),
    }), { wrapper });
    act(() => result.current.mutate({ id: 5 }));
    await waitFor(() => expect(toast.error).toHaveBeenCalled());
    const [text, opts] = vi.mocked(toast.error).mock.calls[0] as [string, any];
    expect(text).toContain("other");
    expect(opts.description).toMatch(/load-balance/);
    expect(opts.action.label).toBe("Set model name");
    opts.action.onClick();
    expect(fix).toHaveBeenCalledWith(5);
  });
});
