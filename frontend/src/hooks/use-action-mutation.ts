import { useMutation, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";

interface Options<TData, TVars> {
  success?: string | ((data: TData, vars: TVars) => string);
  invalidate?: unknown[][];
  onSuccess?: (data: TData, vars: TVars) => void;
  /** How long a finished mutation (and its result) stays in the cache; 0 drops secrets as soon as it is reset. */
  gcTime?: number;
}

/** A react-query mutation that reports its outcome as a toast and refreshes the given queries. */
export function useActionMutation<TVars = void, TData = unknown>(fn: (vars: TVars) => Promise<TData>, opts: Options<TData, TVars> = {}) {
  const qc = useQueryClient();
  return useMutation<TData, Error, TVars>({
    mutationFn: fn,
    ...(opts.gcTime !== undefined && { gcTime: opts.gcTime }),
    onSuccess: (data, vars) => {
      if (opts.success) toast.success(typeof opts.success === "function" ? opts.success(data, vars) : opts.success);
      opts.onSuccess?.(data, vars);
    },
    onError: (err) => toast.error(err.message || "Request failed"),
    onSettled: () => {
      for (const key of opts.invalidate ?? []) qc.invalidateQueries({ queryKey: key });
    },
  });
}
