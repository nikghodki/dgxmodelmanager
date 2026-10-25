import { useQuery } from "@tanstack/react-query";
import { api } from "@/api";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { IDLE_MAX_MINUTES, IDLE_MIN_MINUTES, IDLE_RANGE_TEXT, defaultIdleText, fmtMinutes, parseIdleMinutes } from "@/lib/ondemand";
import type { DeploymentMode, HostInfo } from "@/types";

export const ONDEMAND_NEEDS_LITELLM = "On demand needs LiteLLM (it starts models on requests that come through it).";

/** Whether requests through LiteLLM can start on-demand models (integration on, last sync good); null until loaded. */
export function useOnDemandReady(): boolean | null {
  const host = useQuery({ queryKey: ["host"], queryFn: () => api<HostInfo>("/host") });
  return host.data ? !!host.data.ondemand?.litellm_ready : null;
}

/** The global idle default in minutes, from the host payload (null until it is loaded). */
export function useIdleDefaultMinutes(): number | null {
  const host = useQuery({ queryKey: ["host"], queryFn: () => api<HostInfo>("/host") });
  const s = host.data?.ondemand?.idle_timeout_s;
  return s ? s / 60 : null;
}

/**
 * Mode radios ("On demand" / "Always on") plus the idle minutes for on demand. `allowEmpty`: an empty idle
 * field is valid and means the DGXMM default.
 */
export function ModeFields({ id, mode, onMode, idle, onIdle, allowEmpty = false, defaultMinutes, hint }: {
  id: string; mode: DeploymentMode; onMode: (m: DeploymentMode) => void; idle: string; onIdle: (v: string) => void; allowEmpty?: boolean;
  defaultMinutes?: number | null; /** a line under the radios */ hint?: string | null;
}) {
  const m = parseIdleMinutes(idle);
  const invalid = Number.isNaN(m) || (!allowEmpty && m === null);
  const n = m !== null && !Number.isNaN(m) ? `${fmtMinutes(m)} min` : allowEmpty && m === null ? defaultIdleText(defaultMinutes) : "N min";
  const opts: { value: DeploymentMode; label: string }[] = [
    { value: "on_demand", label: `On demand — start on the first request, sleep after ${n} idle` },
    { value: "always_on", label: "Always on" },
  ];
  return (
    <div className="space-y-3">
      <div role="radiogroup" aria-labelledby={`${id}-label`} className="space-y-2">
        <div id={`${id}-label`} className="text-sm font-medium leading-none">Mode</div>
        {opts.map((o) => (
          <label key={o.value} className="flex cursor-pointer items-start gap-2 text-sm">
            <input type="radio" name={id} value={o.value} checked={mode === o.value} onChange={() => onMode(o.value)}
              className="mt-0.5 h-4 w-4 shrink-0 accent-primary" />
            <span>{o.label}</span>
          </label>
        ))}
      </div>
      {hint && <p className="text-xs text-muted-foreground">{hint}</p>}
      {mode === "on_demand" && (
        <div className="max-w-xs space-y-1.5">
          <Label htmlFor={`${id}-idle`}>Idle minutes before sleeping</Label>
          <Input id={`${id}-idle`} type="number" inputMode="numeric" min={IDLE_MIN_MINUTES} max={IDLE_MAX_MINUTES} step={1}
            placeholder={allowEmpty ? defaultIdleText(defaultMinutes) : undefined} value={idle} aria-invalid={invalid} aria-describedby={`${id}-idle-help`}
            onChange={(e) => onIdle(e.target.value)} />
          <p id={`${id}-idle-help`} className={`text-xs ${invalid ? "text-status-critical" : "text-muted-foreground"}`}>
            {invalid ? IDLE_RANGE_TEXT : `Stopped after this long without requests; the next request starts it again.${allowEmpty ? " Empty uses the DGXMM default." : ""}`}
          </p>
        </div>
      )}
    </div>
  );
}

/** Whether the idle field blocks saving for this mode. */
export const idleInvalid = (mode: DeploymentMode, idle: string, allowEmpty = false) => {
  if (mode !== "on_demand") return false;
  const m = parseIdleMinutes(idle);
  return Number.isNaN(m) || (!allowEmpty && m === null);
};
