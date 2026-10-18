import { useQuery } from "@tanstack/react-query";
import { ShieldAlert } from "lucide-react";
import { api } from "@/api";
import { findingLabel, remoteCodeWarns } from "@/lib/advice";
import type { RemoteCodeCheck } from "@/types";
import { DecisionChip } from "./decision-chip";

/** The generic trust_remote_code warning: always shown while trust_remote_code is on (no check can remove it). */
export function RemoteCodeWarning() {
  return (
    <div role="alert" data-remote-code-generic className="flex items-start gap-2 rounded-md border border-status-warning/30 bg-status-warning-bg p-3 text-xs text-status-warning">
      <ShieldAlert aria-hidden className="mt-0.5 h-4 w-4 shrink-0" />
      <p><span className="font-medium">trust_remote_code runs Python code from the model repository</span> (as root in the engine container,
        with the Hugging Face cache mounted). Enable it only if you trust this repository&apos;s code.</p>
    </div>
  );
}

/** Remote-code risk (POST /decider/remote-code-check) for a model with custom code: findings with file:line. It only ever
 * adds warnings. A clean scan is reported neutrally ("no suspicious patterns found, static scan only") — never as
 * "safe" or "routine" — and the generic warning stays. A scan that failed or did not cover everything is a warning. */
export function RemoteCodeFindings({ modelRef, enabled }: { modelRef: string; enabled: boolean }) {
  const q = useQuery({
    queryKey: ["remote-code-check", modelRef],
    queryFn: () => api<RemoteCodeCheck>("/decider/remote-code-check", { method: "POST", body: { model_ref: modelRef } }),
    enabled: enabled && !!modelRef, retry: false, staleTime: 5 * 60_000, refetchOnWindowFocus: false,
  });
  const c = q.data && !q.error && Array.isArray(q.data.findings) ? q.data : null;
  if (!enabled) return null;
  if (q.isLoading) return <p className="text-xs text-muted-foreground">Scanning the repository&apos;s code…</p>;
  if (q.error) return <p role="alert" className="rounded-md border border-status-warning/30 bg-status-warning-bg p-3 text-xs text-status-warning">The code scan is unavailable: review the repository&apos;s code yourself before enabling this.</p>;
  if (!c || c.disabled || c.verdict === "no_custom_code") return null;
  const warns = remoteCodeWarns(c) || !!c.error || c.truncated;
  const n = c.findings.length;
  const plural = (k: number, w: string) => `${k} ${w}${k === 1 ? "" : "s"}`;
  const files = plural(c.scanned_files, "file");
  const flagged = new Set(c.findings.map((f) => f.file)).size;
  const title = c.error ? "Code scan: could not analyse the code — treat it as unusual"
    : n ? `Code scan: ${plural(n, "finding")} in ${plural(flagged, "file")} (${files} scanned)`
    : c.verdict === "unusual" ? "Code scan: the decision model flags this code as unusual"
    : c.truncated ? `Code scan: only part of the code was scanned (${files}) — treat the rest as unreviewed`
    : `No suspicious patterns found in ${files} (static scan only)`;
  return (
    <div role={warns ? "alert" : undefined} data-remote-code={c.verdict}
      className={warns ? "space-y-2 rounded-md border border-status-critical/30 bg-status-critical-bg p-3 text-xs text-status-critical" : "space-y-1 rounded-md border p-3 text-xs text-muted-foreground"}>
      <div className="flex flex-wrap items-center gap-2">
        <span className="font-medium">{title}</span>
        {c.model_label === "unusual" && <DecisionChip source="model" confidence={c.confidence} prefix="Unusual" title="Decision model's reading of the scan; it never removes a finding" />}
      </div>
      {n > 0 && (
        <ul className="max-h-48 space-y-0.5 overflow-y-auto font-mono">
          {c.findings.slice(0, 50).map((f, i) => <li key={`${f.file}:${f.line}:${f.kind}:${i}`} className="break-all">{f.file}:{f.line} — {findingLabel(f.kind)}</li>)}
          {n > 50 && <li>… and {n - 50} more</li>}
        </ul>
      )}
      {c.truncated && n > 0 && <p>Only part of the code was scanned (size limits).</p>}
    </div>
  );
}
