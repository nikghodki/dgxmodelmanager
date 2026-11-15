import { Copy } from "lucide-react";
import { Link } from "react-router-dom";
import { ApiError, api } from "@/api";
import { Button } from "@/components/ui/button";
import { useActionMutation } from "@/hooks/use-action-mutation";
import { nodeLabel } from "@/lib/placement";
import type { NodeInfo } from "@/types";

export interface CopyWeightsActionInfo { type: "copy_weights"; node: string; ref: string; revision: string }

/** The copy_weights action of a weights_missing refusal (409), or null. */
export function copyWeightsAction(err: unknown): CopyWeightsActionInfo | null {
  if (!(err instanceof ApiError) || err.code !== "weights_missing") return null;
  const a = err.detail?.action;
  return a?.type === "copy_weights" && a.node && a.ref && a.revision ? a : null;
}

/** A deploy refused with weights_missing: copy the snapshot to the node (a Download row of kind "copy") and follow it
 *  on the Downloads page. Admin only, like the deploy wizard. */
export function CopyWeightsPrompt({ error, nodes }: { error: unknown; nodes: NodeInfo[] | undefined }) {
  const action = copyWeightsAction(error);
  const label = nodeLabel(nodes, action?.node);
  const copy = useActionMutation(() => api(`/nodes/${encodeURIComponent(action!.node)}/copy-weights`, {
    method: "POST", body: { ref: action!.ref, revision: action!.revision },
  }), { success: `Copying the weights to ${label}`, invalidate: [["downloads"]] });
  if (!action) return null;
  return (
    <div className="space-y-2 rounded-lg border border-status-warning/40 bg-status-warning-bg/40 p-3 text-xs">
      <p className="break-words">{(error as ApiError).message}</p>
      <div className="flex flex-wrap items-center gap-2">
        <Button size="sm" variant="outline" disabled={copy.isPending || copy.isSuccess} onClick={() => copy.mutate(undefined)}>
          <Copy aria-hidden className="mr-1 h-3.5 w-3.5" />Copy weights to {label}
        </Button>
        {copy.isSuccess
          ? <span>Copy started. Follow it on <Link to="/downloads" className="text-primary hover:underline">Downloads</Link>, then deploy again.</span>
          : <Link to="/downloads" className="text-primary hover:underline">Open Downloads</Link>}
      </div>
    </div>
  );
}
