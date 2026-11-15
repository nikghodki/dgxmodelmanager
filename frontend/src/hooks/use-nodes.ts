import { useQuery } from "@tanstack/react-query";
import { api } from "@/api";
import type { NodeInfo } from "@/types";

/** GET /nodes (any signed-in user): the configured nodes with their memory budgets. A failed or pending request
 *  reads as a single-node install (the placement UI stays hidden). */
export function useNodes(opts: { refetchInterval?: number } = {}) {
  return useQuery({ queryKey: ["nodes"], queryFn: () => api<NodeInfo[]>("/nodes"), staleTime: 10000, ...opts });
}
