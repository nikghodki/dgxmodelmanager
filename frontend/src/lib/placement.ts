import type { Deployment, NodeInfo, Placement, PlacementOptionInfo } from "@/types";

export const LOCAL_NODE = "local";

/** A stable key for a placement (an option card, the chosen one). */
export const placementKey = (p: Placement) => `${p.kind}:${p.nodes.join(",")}`;

/** Runs somewhere other than the local node alone (a remote node, or a cluster). */
export const offLocal = (d: Pick<Deployment, "placement">) =>
  !!d.placement && (d.placement.kind === "cluster" || d.placement.nodes[0] !== LOCAL_NODE);

/** More than one node is configured (the placement UI is shown only then). */
export const isMultiNode = (nodes: NodeInfo[] | undefined) => (nodes?.length ?? 0) > 1;

/** The Runs on column: with more than one node configured, or when a deployment runs off the local node anyway
 *  (e.g. its node was removed from the configuration). */
export const showRunsOn = (nodes: NodeInfo[] | undefined, deployments: Pick<Deployment, "placement">[]) =>
  isMultiNode(nodes) || deployments.some(offLocal);

/** The Sleep action's label: a cluster's sleep_label (e.g. "Sleep (stops all nodes)"), else "Sleep". */
export const sleepLabel = (d: Pick<Deployment, "sleep_label">) =>
  d.sleep_label ? d.sleep_label[0].toUpperCase() + d.sleep_label.slice(1) : "Sleep";

/** A node's label (its id when it is not configured). */
export const nodeLabel = (nodes: NodeInfo[] | undefined, id: string | null | undefined) =>
  (id && nodes?.find((n) => n.id === id)?.label) || id || LOCAL_NODE;

export const PLACEMENT_BADGES: Record<string, { text: string; tone: "warning" | "critical" | "neutral" }> = {
  insufficient_memory: { text: "Does not fit", tone: "critical" },
  too_big_for_one_node: { text: "Too big for one DGX", tone: "warning" },
  node_stale: { text: "Node unavailable", tone: "neutral" },
};

/** Per node of an option: what it needs, what is free there, and the shortfall (0 when it fits there). */
export function optionNodes(o: PlacementOptionInfo) {
  return o.placement.nodes.map((n) => {
    const need = o.per_node[n] ?? 0;
    const free = o.free[n] ?? 0;
    return { node: n, need, free, short: Math.max(need - Math.max(free, 0), 0) };
  });
}

/** A rank container's Docker status (or the deployment state it falls back to) as a status badge. */
export function rankBadge(status: string): { state: string; label?: string } {
  if (status === "running") return { state: "ready", label: "Running" };
  if (status === "exited" || status === "dead") return { state: "failed", label: status === "dead" ? "Dead" : "Exited" };
  if (status === "missing") return { state: "failed", label: "Missing" };
  if (status === "created" || status === "restarting") return { state: "starting", label: status === "created" ? "Created" : "Restarting" };
  if (status === "paused") return { state: "unknown", label: "Paused" };
  return { state: status };
}
