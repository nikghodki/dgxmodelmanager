import { useQuery } from "@tanstack/react-query";
import { api } from "@/api";
import { isMultiNode } from "@/lib/placement";
import type { NodeInfo } from "@/types";

/** GET /nodes (any signed-in user): the configured nodes with their memory budgets. A failed or pending request
 *  reads as a single-node install (callers that act on it check isPending / isError). ``pollWhenMulti``: refetch
 *  every that many ms, only once more than one node is configured (a single-node install never polls). */
export function useNodes(opts: { pollWhenMulti?: number } = {}) {
  return useQuery({
    queryKey: ["nodes"], queryFn: () => api<NodeInfo[]>("/nodes"), staleTime: 10000,
    refetchInterval: opts.pollWhenMulti ? (q) => (isMultiNode(q.state.data) ? opts.pollWhenMulti! : false) : undefined,
  });
}
