export type StatusKind = "good" | "warning" | "sleeping" | "critical" | "neutral";
export interface StatusInfo { kind: StatusKind; label: string; spin: boolean }

const MAP: Record<string, StatusInfo> = {
  ready: { kind: "good", label: "Ready", spin: false },
  synced: { kind: "good", label: "Synced", spin: false },
  conflict: { kind: "warning", label: "Name conflict", spin: false },
  done: { kind: "good", label: "Done", spin: false },
  starting: { kind: "warning", label: "Starting", spin: true },
  loading: { kind: "warning", label: "Loading", spin: true },
  stopping: { kind: "warning", label: "Stopping", spin: true },
  running: { kind: "warning", label: "Downloading", spin: true },
  pending: { kind: "warning", label: "Pending", spin: false },
  sleeping: { kind: "sleeping", label: "Sleeping", spin: false },
  failed: { kind: "critical", label: "Failed", spin: false },
  error: { kind: "critical", label: "Error", spin: false },
  stopped: { kind: "neutral", label: "Stopped", spin: false },
  unknown: { kind: "neutral", label: "Unknown", spin: false },
  none: { kind: "neutral", label: "Not registered", spin: false },
  queued: { kind: "neutral", label: "Queued", spin: false },
  cancelled: { kind: "neutral", label: "Cancelled", spin: false },
  disabled: { kind: "neutral", label: "Disabled", spin: false },
  planned: { kind: "neutral", label: "Planned", spin: false },
};

export function statusInfo(state: string): StatusInfo {
  if (MAP[state]) return MAP[state];
  if (!state) return { kind: "neutral", label: "—", spin: false };
  const words = state.replace(/[_-]+/g, " ");
  return { kind: "neutral", label: words.charAt(0).toUpperCase() + words.slice(1), spin: false };
}
