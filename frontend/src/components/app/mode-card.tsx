import { useEffect, useState } from "react";
import { Loader2 } from "lucide-react";
import { ApiError, api } from "@/api";
import { KV } from "@/components/app/deployment-litellm";
import { ModeFields, idleInvalid, useIdleDefaultMinutes } from "@/components/app/mode-fields";
import { ErrorState } from "@/components/app/states";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { useActionMutation } from "@/hooks/use-action-mutation";
import { MODE_LABELS, deploymentMode, fmtMinutes, minutesToSeconds, modeSummary, parseIdleMinutes } from "@/lib/ondemand";
import type { Deployment, DeploymentMode } from "@/types";

const idleText = (s: number | null | undefined) => (s ? fmtMinutes(s / 60) : "");

/** Mode and idle timeout of a user deployment: editable by admins (PATCH /deployments/{id}/mode), read-only for viewers.
 *  System deployments are always on and get no card. */
export function DeploymentModeCard({ d, admin, className }: { d: Deployment; admin: boolean; className?: string }) {
  // a switch to always on that waits for the start shows as always on (with a note)
  const current = d.pending_mode ?? deploymentMode(d);
  const defaultMinutes = useIdleDefaultMinutes();
  const [mode, setMode] = useState<DeploymentMode>(current);
  const [idle, setIdle] = useState(idleText(d.idle_timeout_s));
  useEffect(() => { setMode(current); setIdle(idleText(d.idle_timeout_s)); }, [current, d.idle_timeout_s]);
  const save = useActionMutation((body: Record<string, unknown>) => api<Deployment>(`/deployments/${d.id}/mode`, { method: "PATCH", body }), {
    success: (_r, b) => `${d.name}: ${MODE_LABELS[b.mode as DeploymentMode].toLowerCase()}`, invalidate: [["deployment", d.id], ["deployments"]],
  });
  if (d.system) return null;
  if (!admin) return (
    <Card className={`min-w-0 shadow-card ${className ?? ""}`}>
      <CardHeader className="pb-2"><CardTitle className="text-sm">Mode</CardTitle></CardHeader>
      <CardContent><dl><KV label="Mode" value={modeSummary(d, defaultMinutes)} /></dl></CardContent>
    </Card>
  );
  const minutes = parseIdleMinutes(idle);
  const dirty = mode !== current || (mode === "on_demand" && idle !== idleText(d.idle_timeout_s));
  const body = mode === "on_demand" ? { mode, idle_timeout_s: minutes === null ? null : minutesToSeconds(minutes) } : { mode };
  const err = save.error;
  const noMemory = err instanceof ApiError && err.code === "insufficient_memory";
  return (
    <Card className={`min-w-0 shadow-card ${className ?? ""}`}>
      <CardHeader className="pb-2"><CardTitle className="text-sm">Mode</CardTitle>
        <CardDescription className="text-xs">On demand frees its memory while unused; the first request waits for it to start.</CardDescription></CardHeader>
      <CardContent className="space-y-3">
        <ModeFields id={`mode-${d.id}`} mode={mode} onMode={(m) => { setMode(m); save.reset(); }} idle={idle} onIdle={setIdle} allowEmpty defaultMinutes={defaultMinutes}
          hint={d.pending_mode === "always_on" ? "Switching to always on: it stays on demand until it is ready, so requests through LiteLLM keep working while it starts." : null} />
        {err && <ErrorState error={noMemory ? new Error(`Not enough memory to keep it always on: ${err.message}. ${d.name} stays on demand; free memory (stop or sleep another deployment) and try again.`) : err} />}
        <Button size="sm" disabled={!dirty || idleInvalid(mode, idle, true) || save.isPending} onClick={() => save.mutate(body)}>
          {save.isPending && <Loader2 aria-hidden className="mr-1 h-4 w-4 animate-spin" />}Save mode
        </Button>
      </CardContent>
    </Card>
  );
}
