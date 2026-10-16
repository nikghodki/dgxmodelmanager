import { useEffect } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { api, streamUrl } from "./api";

const LIVE_POINTS = 120;

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
    });
    on("deployment_removed", () => qc.invalidateQueries({ queryKey: ["deployments"] }));
    on("download", () => qc.invalidateQueries({ queryKey: ["downloads"] }));
    on("inventory", () => qc.invalidateQueries({ queryKey: ["inventory"] }));
    on("metrics", (d: Record<string, any>) => {
      for (const [id, sample] of Object.entries(d)) {
        qc.setQueryData<any[]>(["live", Number(id)], (old = []) => [...old, sample].slice(-LIVE_POINTS));
      }
    });
    on("alert", (d) => qc.setQueryData(["alert"], d));
    return () => es.close();
  }, [enabled, qc]);
}

export function useLive(id: number) {
  return useQuery<any[]>({ queryKey: ["live", id], queryFn: () => api<any[]>(`/deployments/${id}/metrics?window=live`), staleTime: Infinity });
}
