import { useEffect } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { api, streamUrl } from "./api";

const LIVE_POINTS = 120;
/** SSE `decider` phases that end an image build (anything else is progress). */
export const DECIDER_BUILD_END = ["done", "complete", "completed", "ready", "error", "failed"];

export function useLiveUpdates(enabled: boolean) {
  const qc = useQueryClient();
  useEffect(() => {
    if (!enabled) return;
    const es = new EventSource(streamUrl, { withCredentials: true });
    const on = (name: string, fn: (d: any) => void) => es.addEventListener(name, (e) => fn(JSON.parse((e as MessageEvent).data)));
    on("budget", (d) => qc.setQueryData(["host"], d));
    on("deployment", (d) => {
      qc.invalidateQueries({ queryKey: ["deployments"] });
      qc.invalidateQueries({ queryKey: ["deployment", d.id] });
      // The decider's own deployment changed state: its status chip / page follow without polling.
      if (d?.engine === "decision" || d?.system) qc.invalidateQueries({ queryKey: ["decider-status"] });
    });
    on("deployment_removed", () => {
      qc.invalidateQueries({ queryKey: ["deployments"] });
      qc.invalidateQueries({ queryKey: ["decider-status"] });
    });
    on("download", () => qc.invalidateQueries({ queryKey: ["downloads"] }));
    on("inventory", () => qc.invalidateQueries({ queryKey: ["inventory"] }));
    on("metrics", (d: Record<string, any>) => {
      for (const [id, sample] of Object.entries(d)) {
        qc.setQueryData<any[]>(["live", Number(id)], (old = []) => [...old, sample].slice(-LIVE_POINTS));
      }
    });
    on("alert", (d) => qc.setQueryData(["alert"], d));
    // Decision model: a new failure diagnosis, and progress of the decider image build.
    on("diagnosis", (d) => qc.invalidateQueries({ queryKey: ["diagnosis", d.deployment_id] }));
    on("decider", (d) => {
      qc.setQueryData(["decider-build"], d);
      // A build starting or ending changes the status (the chip shows "Building" meanwhile).
      if (d?.phase === "started" || DECIDER_BUILD_END.includes(d?.phase)) qc.invalidateQueries({ queryKey: ["decider-status"] });
    });
    return () => es.close();
  }, [enabled, qc]);
}

export function useLive(id: number) {
  return useQuery<any[]>({ queryKey: ["live", id], queryFn: () => api<any[]>(`/deployments/${id}/metrics?window=live`), staleTime: Infinity });
}
