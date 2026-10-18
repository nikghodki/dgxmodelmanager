import { useEffect, useMemo, useState, type ReactNode } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Link, Navigate } from "react-router-dom";
import type { ColumnDef } from "@tanstack/react-table";
import { toast } from "sonner";
import { AlertTriangle, BookOpen, BrainCircuit, CheckCircle2, Circle, Hammer, KeyRound, Loader2, Play, Power, Save, XCircle } from "lucide-react";
import { ApiError, api } from "@/api";
import { useIsAdmin } from "@/auth";
import { ConfirmDialog } from "@/components/app/confirm-dialog";
import { CopyButton } from "@/components/app/copy-button";
import { DataTable } from "@/components/app/data-table";
import { DecisionChip } from "@/components/app/decision-chip";
import { FeedbackButtons } from "@/components/app/feedback-buttons";
import { EstimatePanel } from "@/components/app/estimate-panel";
import { PageHeader } from "@/components/app/page-header";
import { PlanChooser } from "@/components/app/plan-chooser";
import { ResponsiveDialog } from "@/components/app/responsive-dialog";
import { EmptyState, ErrorState, PageSkeleton } from "@/components/app/states";
import { StatusBadge } from "@/components/app/status-badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { fmtBytes, fmtRelative } from "@/format";
import { useActionMutation } from "@/hooks/use-action-mutation";
import { fmtConfidence } from "@/lib/decider";
import { DECIDER_DOCS_URL, DECIDER_FEATURES, deciderStartupHint, featureLabel } from "@/lib/decider-labels";
import { apiErrorText } from "@/lib/litellm";
import { DECIDER_BUILD_END } from "@/stream";
import { SchedulesPauseControl } from "@/components/app/schedules-card";
import type { DeciderBuildProgress, DeciderKeyRotation, DeciderSettingsPatch, DeciderStatus, Deployment, DecisionRow, Fit, PlanAction, PlanOption, Preview } from "@/types";

const ALL = "__all__";
const DECISION_FEATURES = ["diagnosis", "parser_tool", "parser_reasoning", "plan", "kind", "workload"];

function Row({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="flex items-center justify-between gap-3 border-b py-1.5 last:border-0">
      <dt className="text-muted-foreground">{label}</dt><dd className="min-w-0 break-all text-right">{children}</dd>
    </div>
  );
}

/** "ready"-like state of the decider for the StatusBadge (text label, never colour only). */
function health(s: DeciderStatus): { state: string; label: string } {
  if (!s.enabled) return { state: "disabled", label: "Disabled" };
  if (s.healthy) return { state: "ready", label: "Healthy" };
  if (s.state === "ready" && s.breaker?.open) return { state: "starting", label: "Paused (rules only)" };
  if (s.state && ["queued", "starting", "loading"].includes(s.state)) return { state: s.state === "queued" ? "starting" : s.state, label: s.state === "loading" ? "Loading" : s.state === "queued" ? "Queued" : "Starting" };
  return { state: "error", label: "Unhealthy" };
}

const HEALTH_STYLE: Record<string, string> = {
  ready: "bg-status-good-bg text-status-good", error: "bg-status-critical-bg text-status-critical",
  disabled: "bg-status-neutral-bg text-status-neutral", starting: "bg-status-warning-bg text-status-warning", loading: "bg-status-warning-bg text-status-warning",
};
function HealthBadge({ s }: { s: DeciderStatus }) {
  const h = health(s);
  const Icon = h.state === "ready" ? CheckCircle2 : h.state === "error" ? XCircle : h.state === "disabled" ? Circle : Loader2;
  return (
    <span className={`inline-flex items-center gap-1 whitespace-nowrap rounded-full px-2 py-0.5 text-xs font-medium ${HEALTH_STYLE[h.state]}`}>
      <Icon aria-hidden className={`h-3 w-3 ${Icon === Loader2 ? "animate-spin" : ""}`} />{h.label}
    </span>
  );
}

function StatusCard({ s, onRetry, retrying }: { s: DeciderStatus; onRetry: () => void; retrying: boolean }) {
  const dep = useQuery({ queryKey: ["deployment", s.deployment_id], queryFn: () => api<Deployment>(`/deployments/${s.deployment_id}`), enabled: s.deployment_id != null });
  return (
    <Card data-card className="min-w-0 shadow-card">
      <CardHeader className="pb-2">
        <CardTitle className="flex items-center gap-2 text-sm"><BrainCircuit aria-hidden className="h-4 w-4" /><h2>Status</h2></CardTitle>
        <CardDescription className="text-xs">A small local model that answers DGXMM's questions with calibrated confidence.</CardDescription>
      </CardHeader>
      <CardContent><dl className="text-sm">
        <Row label="Health"><HealthBadge s={s} /></Row>
        <Row label="Model"><span className="font-mono text-xs">{s.model}</span></Row>
        <Row label="Revision">{s.revision
          ? <span className="inline-flex items-center gap-1"><span className="font-mono text-xs" title={s.revision}>{s.revision.slice(0, 8)}</span><CopyButton value={s.revision} label="Revision" /></span>
          : "—"}</Row>
        <Row label="Deployment">{s.deployment_id != null
          ? <span className="inline-flex items-center gap-1.5"><Link className="hover:underline" to={`/deployments/${s.deployment_id}`}>{dep.data?.name ?? `#${s.deployment_id}`}</Link>{s.state && <StatusBadge state={s.state} />}</span>
          : "None"}</Row>
        <Row label="Memory">{dep.data ? <span className="tabular-nums">{`${fmtBytes(dep.data.actual_bytes)} / ${fmtBytes(dep.data.reserved_bytes)}`}</span> : "—"}</Row>
        <Row label="Latency (p50)"><span className="tabular-nums">{s.p50_ms != null ? `${Math.round(s.p50_ms)} ms` : "—"}</span></Row>
        <Row label="Decisions (24 h)"><span className="tabular-nums">{s.decisions_24h}</span></Row>
        <Row label="Acceptance rate"><span className="tabular-nums">{fmtConfidence(s.accepted_rate)}</span></Row>
        <Row label="Image">{s.image.building ? "Building…" : s.image.present ? "Built" : "Not built"}</Row>
        {s.breaker && <Row label="Circuit breaker">{s.breaker.open
          ? <span className="text-status-warning">{`Open — answering by rules for ${Math.round(s.breaker.retry_in_s)} s (${s.breaker.failures} failures)`}</span>
          : "Closed"}</Row>}
        {s.load && <Row label="Load"><span className="tabular-nums">{`${s.load.in_flight}/${s.load.max_in_flight} busy · ${s.load.queued} waiting · ${s.load.shed} answered by rules under load`}</span></Row>}
        {s.supervisor && <Row label="Auto-restart">{supervisorText(s.supervisor)}</Row>}
      </dl>
      {s.enabled && s.supervisor?.gave_up && (
        <div className="mt-2 flex flex-wrap items-center gap-2 text-xs">
          <span className="text-status-critical">Auto-restart stopped trying ({s.supervisor.restarts_last_hour} attempts in an hour).</span>
          <Button size="sm" variant="outline" disabled={retrying} onClick={onRetry}>{retrying ? <Loader2 className="mr-1 h-4 w-4 animate-spin" /> : <Play className="mr-1 h-4 w-4" />}Retry</Button>
        </div>
      )}
      {s.supervisor?.last_reason && !s.supervisor.gave_up && s.supervisor.last_action !== "probe_ok" && s.supervisor.last_action !== "idle" && (
        <p className="mt-2 break-words text-xs text-muted-foreground">Last event: {s.supervisor.last_reason}</p>
      )}
      {s.image.error && <p className="mt-2 break-words text-xs text-status-critical">Image build failed: {s.image.error}</p>}
      </CardContent>
    </Card>
  );
}

function supervisorText(v: NonNullable<DeciderStatus["supervisor"]>): string {
  if (v.gave_up) return `Gave up — ${v.last_reason ?? `${v.restarts_last_hour} restart attempts in an hour`}`;
  const restarts = v.restarts_last_hour ? ` · ${v.restarts_last_hour} restart${v.restarts_last_hour === 1 ? "" : "s"} in the last hour` : "";
  if (v.pending_restart) return `Restart pending${v.next_attempt_in_s != null ? ` (next try in ${Math.round(v.next_attempt_in_s)} s)` : ""}${restarts}`;
  if (v.last_action === "rotating") return "Paused while the API key is rotated";
  if (v.last_action === "rolled_back") return `Rolled back to the previous model${restarts}`;
  if (v.last_action === "memory_pressure") return `Waiting for memory${restarts}`;
  if (v.last_action === "backoff" && v.next_attempt_in_s != null) return `Retrying in ${Math.round(v.next_attempt_in_s)} s${restarts}`;
  if (v.probe) return `${v.probe.ok ? "Probe OK" : "Probe failed"}${v.probe.latency_ms != null ? ` (${Math.round(v.probe.latency_ms)} ms)` : ""}${restarts}`;
  return `On${restarts}`;
}

interface DownloadRow { id: number; repo_id: string; state: string; bytes_done: number; bytes_total: number; error: string | null }

const latestDownload = (rows: DownloadRow[] | undefined, repo: string) => (rows ?? []).filter((r) => r.repo_id === repo).sort((a, b) => b.id - a.id)[0];

/** A model that is not (completely) on disk: Download, or the progress / failure of its download. */
function DownloadState({ row, name, pending, onDownload }: { row?: DownloadRow; name: string; pending: boolean; onDownload: () => void }) {
  if (row && ["queued", "running"].includes(row.state)) {
    const pct = row.bytes_total ? Math.min(100, Math.floor((row.bytes_done / row.bytes_total) * 100)) : 0;
    return <Link to="/downloads" className="shrink-0 text-xs text-muted-foreground hover:underline" aria-label={`Downloading ${name}: ${pct}%`}>
      <Loader2 aria-hidden className="mr-1 inline h-3 w-3 animate-spin" />{row.bytes_total ? `Downloading ${pct}% · ${fmtBytes(row.bytes_done)} / ${fmtBytes(row.bytes_total)}` : "Downloading…"}</Link>;
  }
  return (
    <span className="flex shrink-0 flex-col items-end gap-1">
      <Button variant="outline" size="sm" disabled={pending} onClick={onDownload} aria-label={`Download ${name}`}>{row?.state === "failed" ? "Retry download" : "Download"}</Button>
      {row?.state === "failed" && row.error && <span title={row.error} className="line-clamp-3 max-w-[28ch] break-words text-right text-xs text-status-critical">{row.error}</span>}
    </span>
  );
}

/** Which pinned decision model to run; switching restarts the decider (same deployment). */
function ModelCard({ s }: { s: DeciderStatus }) {
  const models = s.models ?? [];
  const current = s.configured_model ?? s.model;
  const [choice, setChoice] = useState(current);
  useEffect(() => { setChoice(current); }, [current]);
  const save = useActionMutation(() => api("/decider/settings", { method: "PUT", body: { model: choice } }), { success: "Decision model saved", invalidate: [["decider-status"]] });
  const download = useActionMutation((repo: string) => api("/decider/models/download", { method: "POST", body: { model: repo } }), { success: "Download started — pinned files only, each checked against its hash", invalidate: [["decider-status"], ["downloads"]] });
  // Progress comes from the normal download rows (the SSE `download` topic refreshes ["downloads"]).
  const downloads = useQuery({ queryKey: ["downloads"], queryFn: () => api<DownloadRow[]>("/downloads"), enabled: models.some((m) => !m.downloaded) });
  const qc = useQueryClient();
  const finished = (downloads.data ?? []).filter((r) => r.state === "done").map((r) => r.id).join(",");
  useEffect(() => { if (finished) qc.invalidateQueries({ queryKey: ["decider-status"] }); }, [finished, qc]);
  const restart = useActionMutation(() => api("/decider/enable", { method: "POST", body: {} }), { success: "Restarting the decider with the selected model", invalidate: [["decider-status"], ["deployments"]] });
  if (!models.length) return null;
  const chosen = models.find((m) => m.repo === current);
  return (
    <Card data-card className="min-w-0 shadow-card">
      <CardHeader className="pb-2">
        <CardTitle className="text-sm"><h2>Model</h2></CardTitle>
        <CardDescription className="text-xs">Both are pinned to a revision whose files are checked against hashes built into the image. d3-mini is more accurate; d3-nano needs less memory.</CardDescription>
      </CardHeader>
      <CardContent className="space-y-3">
        <fieldset className="space-y-2">
          <legend className="sr-only">Decision model</legend>
          {models.map((m) => (
            <div key={m.repo} className="flex items-start justify-between gap-3 rounded-md border p-2">
              <label className="flex min-w-0 cursor-pointer items-start gap-2 text-sm">
                <input type="radio" name="decider-model" className="mt-1" value={m.repo} checked={choice === m.repo} onChange={() => setChoice(m.repo)} />
                <span className="min-w-0">
                  <span className="font-mono text-xs">{m.served_name}</span>{m.default && <span className="ml-1 text-xs text-muted-foreground">(default)</span>}
                  <span className="block text-xs text-muted-foreground">{`${m.params} · reserves ${fmtBytes(m.reservation_bytes)} · ${m.revision.slice(0, 8)}`}</span>
                </span>
              </label>
              {!m.downloaded && <DownloadState row={latestDownload(downloads.data, m.repo)} name={m.served_name} pending={download.isPending} onDownload={() => download.mutate(m.repo)} />}
            </div>
          ))}
        </fieldset>
        <div className="flex flex-wrap gap-2">
          <Button size="sm" disabled={choice === current || save.isPending} onClick={() => save.mutate(undefined)}>
            {save.isPending ? <Loader2 className="mr-1 h-4 w-4 animate-spin" /> : <Save className="mr-1 h-4 w-4" />}Save model</Button>
          {s.restart_required && s.enabled && chosen && (
            <Button size="sm" variant="outline" disabled={restart.isPending || !chosen.downloaded} onClick={() => restart.mutate(undefined)}>
              {restart.isPending ? <Loader2 className="mr-1 h-4 w-4 animate-spin" /> : <Play className="mr-1 h-4 w-4" />}{`Restart with ${chosen.served_name}`}</Button>
          )}
        </div>
        {s.restart_required && <p className="text-xs text-muted-foreground">{`${s.model} is running; ${current} takes over when the decider restarts.`}</p>}
      </CardContent>
    </Card>
  );
}

/** Label and description per settings group: from the server's feature registry, else the built-in five. */
/** ``needs_model: false`` (rules only, e.g. threshold tuning): never asks the model, so it has no threshold. */
const featureMetas = (s: DeciderStatus): { key: string; label: string; description: string; needs_model?: boolean }[] =>
  s.feature_specs?.length ? s.feature_specs.map((f) => ({ key: f.name, label: f.label, description: f.description, needs_model: f.needs_model })) : DECIDER_FEATURES;

const jobWhen = (iso: string | null) => (iso ? fmtRelative(iso) : "—");
function fmtIn(iso: string | null, now: number = Date.now()): string {
  if (!iso) return "—";
  const s = (Date.parse(iso) - now) / 1000;
  if (s <= 60) return "within a minute";
  if (s < 3600) return `in ${Math.round(s / 60)} min`;
  if (s < 86400) return `in ${Math.round(s / 3600)} h`;
  return `in ${Math.round(s / 86400)} d`;
}
/** Background insight jobs (admin part of the status): schedule, last run and result, next run. */
function JobsCard({ s }: { s: DeciderStatus }) {
  const jobs = s.jobs ?? [];
  if (!jobs.length) return null;
  return (
    <Card data-card className="min-w-0 shadow-card lg:col-span-2">
      <CardHeader className="pb-2">
        <CardTitle className="text-sm"><h2>Background jobs</h2></CardTitle>
        <CardDescription className="text-xs">Periodic checks that raise insights. They use a small budget of decision-model calls and always yield to interactive requests.</CardDescription>
      </CardHeader>
      <CardContent className="overflow-x-auto p-0">
        <table className="w-full text-sm">
          <thead><tr className="border-b text-left text-xs uppercase tracking-wider text-muted-foreground">
            {["Job", "Feature", "Schedule", "Last run", "Duration", "Result", "Next run"].map((h) => <th key={h} className="px-4 py-2 font-medium">{h}</th>)}
          </tr></thead>
          <tbody>
            {jobs.map((j) => (
              <tr key={j.name} data-job={j.name} className="border-b last:border-0">
                <td className="px-4 py-2 font-mono text-xs">{j.name}{j.running && <Loader2 aria-label="running" className="ml-1 inline h-3 w-3 animate-spin" />}</td>
                <td className="px-4 py-2">{featureLabel(j.feature)}{!j.enabled && <span className="ml-1 text-xs text-muted-foreground">(off)</span>}</td>
                <td className="whitespace-nowrap px-4 py-2 text-xs">{j.schedule}</td>
                <td className="whitespace-nowrap px-4 py-2 text-xs text-muted-foreground">{jobWhen(j.last_run_at)}</td>
                <td className="px-4 py-2 tabular-nums text-xs">{j.last_duration_ms != null ? `${Math.round(j.last_duration_ms)} ms` : "—"}</td>
                <td className={`px-4 py-2 text-xs ${j.last_result?.startsWith("error") ? "text-status-critical" : ""}`}>{j.last_result ?? "—"}{j.failures > 1 ? ` (${j.failures} in a row)` : ""}</td>
                <td className="whitespace-nowrap px-4 py-2 text-xs text-muted-foreground">{j.enabled ? fmtIn(j.next_run_at) : "—"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </CardContent>
    </Card>
  );
}

type Draft = Record<string, { enabled: boolean; pct: string }>;
const toDraft = (f: DeciderStatus["features"]): Draft =>
  Object.fromEntries(Object.entries(f).map(([k, v]) => [k, { enabled: v.enabled, pct: fmtPct(v.threshold) }]));
/** Server thresholds are 0..1; the form shows percent (one decimal). */
const fmtPct = (t: number) => String(Math.round(t * 1000) / 10);
const pctValid = (v: string) => v.trim() !== "" && Number.isFinite(Number(v)) && +v >= 0 && +v <= 100;
const clampPct = (v: string) => (v.trim() === "" || !Number.isFinite(Number(v)) ? v : String(Math.min(100, Math.max(0, +v))));

function FeaturesCard({ s }: { s: DeciderStatus }) {
  const [draft, setDraft] = useState<Draft>(() => toDraft(s.features));
  const serverKey = JSON.stringify(s.features);
  useEffect(() => { setDraft(toDraft(s.features)); }, [serverKey]); // eslint-disable-line react-hooks/exhaustive-deps
  // The server's registry order and labels (feature_specs); older servers only send `features`.
  const metas = featureMetas(s);
  const keys = [...metas.map((f) => f.key).filter((k) => k in s.features), ...Object.keys(s.features).filter((k) => !metas.some((f) => f.key === k))];
  const rulesOnly = (k: string) => metas.find((f) => f.key === k)?.needs_model === false;
  const patch: DeciderSettingsPatch = { features: {} };
  for (const k of keys) {
    const d = draft[k]; const cur = s.features[k];
    if (!d) continue;
    const p: Record<string, unknown> = {};
    if (d.enabled !== cur.enabled) p.enabled = d.enabled;
    if (!rulesOnly(k) && pctValid(d.pct) && fmtPct(cur.threshold) !== String(+d.pct)) p.threshold = Math.round(+d.pct * 10) / 1000;
    if (Object.keys(p).length) patch.features![k] = p;
  }
  const dirty = Object.keys(patch.features!).length > 0;
  const invalid = keys.some((k) => draft[k] && !rulesOnly(k) && !pctValid(draft[k].pct));
  const save = useActionMutation(() => api("/decider/settings", { method: "PUT", body: patch }), { success: "Decision model settings saved", invalidate: [["decider-status"]] });
  const set = (k: string, v: Partial<Draft[string]>) => setDraft((d) => ({ ...d, [k]: { ...d[k], ...v } }));
  return (
    <Card data-card className="min-w-0 shadow-card">
      <CardHeader className="pb-2">
        <CardTitle className="text-sm"><h2>Features</h2></CardTitle>
        <CardDescription className="text-xs">Rules always run first. The model's answer is used only at or above the threshold; below it DGXMM behaves as if the model were off.</CardDescription>
      </CardHeader>
      <CardContent>
        <form className="space-y-3" noValidate onSubmit={(e) => { e.preventDefault(); if (dirty && !invalid) save.mutate(undefined); }}>
          <ul className="divide-y">
            {keys.map((k) => {
              const meta = metas.find((f) => f.key === k);
              const label = meta?.label ?? featureLabel(k);
              const d = draft[k] ?? { enabled: false, pct: "" };
              const noThreshold = rulesOnly(k);
              const err = noThreshold || pctValid(d.pct) ? null : `${label}: enter a percentage from 0 to 100.`;
              return (
                <li key={k} className="flex flex-col gap-2 py-3 sm:flex-row sm:items-start sm:justify-between">
                  <div className="flex min-w-0 items-start gap-3">
                    <Switch id={`feat-${k}`} aria-label={label} checked={d.enabled} onCheckedChange={(c) => set(k, { enabled: c })} className="mt-0.5" />
                    <div className="min-w-0">
                      <Label htmlFor={`feat-${k}`} className="text-sm">{label}</Label>
                      {meta && <p className="text-xs text-muted-foreground">{meta.description}</p>}
                    </div>
                  </div>
                  {noThreshold ? (
                    <p className="shrink-0 pl-12 text-xs text-muted-foreground sm:pl-0" title="Never asks the decision model, so no threshold applies">Rules only — no threshold</p>
                  ) : (
                  <div className="shrink-0 space-y-1 pl-12 sm:pl-0">
                    <div className="flex items-center gap-2">
                      <span aria-hidden className="text-xs text-muted-foreground">Threshold</span>
                      <Input id={`thr-${k}`} aria-label={`${label} threshold (%)`} type="number" inputMode="decimal" min={0} max={100} step={1} className="h-8 w-20" value={d.pct}
                        aria-invalid={!!err} aria-describedby={err ? `thr-${k}-err` : undefined} onChange={(e) => set(k, { pct: e.target.value })} onBlur={(e) => set(k, { pct: clampPct(e.target.value) })} />
                      <span aria-hidden className="text-xs text-muted-foreground">%</span>
                    </div>
                    {err && <p id={`thr-${k}-err`} className="text-xs text-status-critical">{err}</p>}
                  </div>
                  )}
                </li>
              );
            })}
          </ul>
          <Button type="submit" size="sm" disabled={!dirty || invalid || save.isPending}>
            {save.isPending ? <Loader2 className="mr-1 h-4 w-4 animate-spin" /> : <Save className="mr-1 h-4 w-4" />}Save settings</Button>
        </form>
      </CardContent>
    </Card>
  );
}

function SubjectCell({ subject }: { subject: string }) {
  const m = /^deployment:(\d+)/.exec(subject);
  return m
    ? <Link to={`/deployments/${m[1]}`} className="block max-w-[28ch] truncate font-mono text-xs hover:underline" title={subject}>{subject}</Link>
    : <span className="block max-w-[28ch] truncate font-mono text-xs" title={subject}>{subject}</span>;
}

function DecisionsCard({ s }: { s: DeciderStatus }) {
  const [feature, setFeature] = useState(ALL);
  // Decision features as logged: each group's members (parser -> parser_tool, parser_reasoning), else the group.
  const decisionFeatures = s.feature_specs?.length ? s.feature_specs.flatMap((f) => (f.members.length ? f.members : [f.name])) : DECISION_FEATURES;
  const q = useQuery({
    queryKey: ["decider-decisions", feature],
    queryFn: () => api<DecisionRow[]>(`/decider/decisions?limit=100${feature === ALL ? "" : `&feature=${encodeURIComponent(feature)}`}`),
    refetchInterval: 30000,
  });
  const columns = useMemo<ColumnDef<DecisionRow, any>[]>(() => [
    { accessorKey: "created_at", header: "When", cell: ({ getValue }) => <span className="whitespace-nowrap text-xs text-muted-foreground" title={new Date(getValue()).toLocaleString()}>{fmtRelative(getValue())}</span> },
    { accessorKey: "feature", header: "Feature", cell: ({ getValue }) => featureLabel(getValue()) },
    { accessorKey: "subject", header: "Subject", meta: { hideOnMobile: true }, cell: ({ getValue }) => <SubjectCell subject={getValue()} /> },
    { accessorKey: "choice", header: "Choice", cell: ({ getValue }) => getValue() == null ? <span className="text-muted-foreground">Abstained</span> : <span className="font-mono text-xs">{getValue()}</span> },
    { accessorKey: "confidence", header: "Confidence", cell: ({ getValue }) => <span className="tabular-nums">{fmtConfidence(getValue())}</span> },
    { accessorKey: "source", header: "Source", cell: ({ getValue }) => <DecisionChip source={getValue()} confidence={null} /> },
    { accessorKey: "latency_ms", header: "Latency", meta: { hideOnMobile: true }, cell: ({ getValue }) => <span className="tabular-nums">{getValue() != null ? `${Math.round(getValue())} ms` : "—"}</span> },
    { id: "feedback", header: "Feedback", enableSorting: false, cell: ({ row }) => (row.original.choice == null ? null : <FeedbackButtons decisionId={row.original.id} feedback={row.original.feedback} />) },
  ], []);
  return (
    <Card data-card className="min-w-0 shadow-card lg:col-span-2">
      <CardHeader className="pb-2">
        <CardTitle className="text-sm"><h2>Recent decisions</h2></CardTitle>
        <CardDescription className="text-xs">Every question asked, with the answer used (rules, AI, or none). Feedback helps judge whether the thresholds are right.</CardDescription>
      </CardHeader>
      <CardContent className="space-y-3">
        <ErrorState error={q.error} onRetry={() => q.refetch()} />
        <DataTable columns={columns} data={q.data ?? []} loading={q.isLoading} getRowId={(r) => String(r.id)}
          toolbar={<Select value={feature} onValueChange={setFeature}><SelectTrigger className="h-9 w-52" aria-label="Filter by feature"><SelectValue /></SelectTrigger>
            <SelectContent><SelectItem value={ALL}>All features</SelectItem>{decisionFeatures.map((f) => <SelectItem key={f} value={f}>{featureLabel(f)}</SelectItem>)}</SelectContent></Select>}
          empty={<EmptyState icon={BrainCircuit} title="No decisions yet" description="Decisions appear here as DGXMM diagnoses failures, suggests parsers and recommends plans." />} />
      </CardContent>
    </Card>
  );
}

/** Plans to make room for the decider (409 insufficient_memory from POST /decider/enable). */
function PlanDialog({ fit, plans, pending, error, onSubmit, onClose }: {
  fit: Fit; plans: PlanOption[]; pending: boolean; error: string | null; onSubmit: (actions: PlanAction[]) => void; onClose: () => void;
}) {
  const [plan, setPlan] = useState<PlanOption | null>(null);
  useEffect(() => { setPlan(null); }, [plans]);
  // Only sleep/stop plans apply to the decider: it has a fixed config and model, so patch/model plans are hidden.
  const usable = plans.filter((p) => !p.patch && !p.model_ref && p.actions.every((a) => a.type === "sleep" || a.type === "stop"));
  const preview = { estimate: fit.estimate, fit, budget: undefined, plans: usable } as unknown as Preview;
  return (
    <ResponsiveDialog open onOpenChange={(o) => !o && !pending && onClose()} className="max-w-xl" title="Make room for the decision model"
      description={`The decision model needs about ${fit.estimate ? fmtBytes(fit.estimate.total) : "13 GiB"}. Choose what DGXMM may do to free it — nothing changes until you confirm.`}>
      <div className="space-y-4">
        {fit.estimate && <EstimatePanel preview={preview} />}
        <PlanChooser plans={usable} selectedId={plan?.id ?? null} onSelect={setPlan} />
        {error && <p className="text-xs text-status-critical">{error}</p>}
        <div className="flex justify-end gap-2">
          <Button variant="outline" disabled={pending} onClick={onClose}>Cancel</Button>
          <Button disabled={!plan || pending} onClick={() => plan && onSubmit(plan.actions)}>
            {pending && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}Apply plan and enable</Button>
        </div>
      </div>
    </ResponsiveDialog>
  );
}

/** Build the decider image (409 image_missing), following the SSE `decider` topic. */
function ImageDialog({ status, onEnable, enabling, onClose }: { status: DeciderStatus | undefined; onEnable: () => void; enabling: boolean; onClose: () => void }) {
  const progress = useQuery<DeciderBuildProgress | null>({ queryKey: ["decider-build"], queryFn: () => null, enabled: false, staleTime: Infinity });
  const [log, setLog] = useState<string[]>([]);
  const [started, setStarted] = useState(false);
  const p = progress.data;
  useEffect(() => { if (started && p?.message) setLog((l) => (l[l.length - 1] === p.message ? l : [...l, p.message].slice(-8))); }, [p, started]);
  const qc = useQueryClient();
  const build = useActionMutation(() => api("/decider/image/build", { method: "POST" }), { success: "Image build started", invalidate: [["decider-status"]] });
  // Forget the previous attempt before starting: its terminal phase must not end the new build.
  const start = () => {
    qc.setQueryData(["decider-build"], null); setLog([]); setStarted(true);
    build.mutate(undefined, { onError: () => setStarted(false) });
  };
  const ended = started && !!p && DECIDER_BUILD_END.includes(p.phase);
  const failed = ended && ["error", "failed"].includes(p!.phase);
  const done = (ended && !failed) || (!started && !!status?.image.present);
  const building = (started && !ended) || (!started && !!status?.image.building);
  return (
    <ResponsiveDialog open onOpenChange={(o) => !o && onClose()} className="max-w-xl" title="Build the decision model image"
      description="The decider runs from a local image (dgxmm-decider) built on this machine from a pinned vLLM base. The first build downloads several GB and takes a few minutes.">
      <div className="space-y-4 text-sm">
        {building && <p role="status" className="flex items-center gap-2"><Loader2 aria-hidden className="h-4 w-4 animate-spin" />Building{p?.phase && p.phase !== "building" ? ` (${p.phase})` : ""}…</p>}
        {done && <p role="status" className="flex items-center gap-2 text-status-good"><CheckCircle2 aria-hidden className="h-4 w-4" />Image ready</p>}
        {failed && <p role="alert" className="flex items-center gap-2 text-status-critical"><XCircle aria-hidden className="h-4 w-4" />Build failed{p?.message ? `: ${p.message}` : ""}</p>}
        {log.length > 0 && <ol aria-label="Build progress" className="max-h-48 space-y-0.5 overflow-auto rounded-md bg-zinc-950 p-2 font-mono text-xs text-zinc-100">{log.map((m, i) => <li key={i} className="break-all">{m}</li>)}</ol>}
        <div className="flex justify-end gap-2">
          <Button variant="outline" onClick={onClose}>Close</Button>
          {done ? (
            <Button disabled={enabling} onClick={onEnable}>{enabling && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}Enable now</Button>
          ) : (
            <Button disabled={building || build.isPending} onClick={start}>
              {build.isPending ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <Hammer className="mr-2 h-4 w-4" />}{failed ? "Retry build" : "Build image"}</Button>
          )}
        </div>
      </div>
    </ResponsiveDialog>
  );
}

function DecisionModelPage() {
  const qc = useQueryClient();
  const st = useQuery({ queryKey: ["decider-status"], queryFn: () => api<DeciderStatus>("/decider/status"), refetchInterval: 15000 });
  const [plans, setPlans] = useState<{ fit: Fit; plans: PlanOption[] } | null>(null);
  const [planErr, setPlanErr] = useState<string | null>(null);
  const [imageDialog, setImageDialog] = useState(false);
  const [confirmOff, setConfirmOff] = useState(false);
  const [confirmRotate, setConfirmRotate] = useState(false);
  // Not useActionMutation: the two 409s open dialogs instead of an error toast.
  const enable = useMutation<{ deployment_id: number }, Error, PlanAction[] | undefined>({
    mutationFn: (actions) => api("/decider/enable", { method: "POST", body: actions?.length ? { actions } : {} }),
    onSuccess: () => { setPlans(null); setImageDialog(false); toast.success(`Starting the decision model — ${deciderStartupHint(st.data?.configured_model ?? st.data?.model)}`); },
    onError: (e, actions) => {
      if (e instanceof ApiError && e.code === "insufficient_memory" && e.detail?.plans) {
        setImageDialog(false); // one dialog at a time
        setPlans({ fit: e.detail.fit, plans: e.detail.plans });
        setPlanErr(actions?.length ? "Memory changed since the plan was computed — choose a plan again." : null);
      } else if (e instanceof ApiError && e.code === "image_missing") {
        setPlans(null); setImageDialog(true);
      } else toast.error(apiErrorText(e));
    },
    onSettled: () => { qc.invalidateQueries({ queryKey: ["decider-status"] }); qc.invalidateQueries({ queryKey: ["deployments"] }); },
  });
  const disable = useActionMutation(() => api("/decider/disable", { method: "POST" }), {
    success: "Decision model disabled — features fall back to rules", invalidate: [["decider-status"], ["deployments"]], onSuccess: () => setConfirmOff(false),
  });

  // The key itself never reaches the browser: the response only says what happened.
  const rotate = useActionMutation(() => api<DeciderKeyRotation>("/decider/rotate-key", { method: "POST" }), {
    invalidate: [["decider-status"], ["deployments"]],
    onSuccess: (r) => {
      setConfirmRotate(false);
      if (r.error) toast.error(`API key ${r.rotated ? "rotated, but the decision model did not restart" : "not rotated"} (${r.error})`);
      else toast.success(r.restarted ? "API key rotated — the decision model is restarting with it" : "API key rotated");
    },
  });

  if (st.isLoading) return <PageSkeleton />;
  const s = st.data;
  const header = (
    <PageHeader title="Decision model"
      description="Advisory only: rules first, the model second, and it abstains when unsure. Nothing changes without your confirmation."
      actions={<>
        <Button asChild variant="ghost" size="sm"><a href={DECIDER_DOCS_URL} target="_blank" rel="noreferrer"><BookOpen className="mr-1 h-4 w-4" />Docs</a></Button>
        {s && s.enabled && s.state && ["stopped", "failed"].includes(s.state) && (
          <Button size="sm" disabled={enable.isPending} onClick={() => enable.mutate(undefined)}>{enable.isPending ? <Loader2 className="mr-1 h-4 w-4 animate-spin" /> : <Play className="mr-1 h-4 w-4" />}Start</Button>
        )}
        {s && <Button variant="outline" size="sm" disabled={rotate.isPending} onClick={() => setConfirmRotate(true)}><KeyRound className="mr-1 h-4 w-4" />Rotate key</Button>}
        {s && (s.enabled
          ? <Button variant="outline" size="sm" disabled={disable.isPending} onClick={() => setConfirmOff(true)}><Power className="mr-1 h-4 w-4" />Disable</Button>
          : <Button size="sm" disabled={enable.isPending} onClick={() => enable.mutate(undefined)}>{enable.isPending ? <Loader2 className="mr-1 h-4 w-4 animate-spin" /> : <Power className="mr-1 h-4 w-4" />}Enable</Button>)}
      </>} />
  );
  if (!s) return <div>{header}<ErrorState error={st.error ?? new Error("Decision model status unavailable")} onRetry={() => st.refetch()} /></div>;
  return (
    <div className="space-y-4">
      {header}
      {!s.enabled && (
        <p className="rounded-lg border bg-muted/50 px-4 py-3 text-sm text-muted-foreground">
          The decision model is off: every feature uses DGXMM's rules only. Enabling it deploys <span className="font-mono">{s.configured_model ?? s.model}</span> (about {fmtBytes((s.models ?? []).find((m) => m.repo === (s.configured_model ?? s.model))?.reservation_bytes ?? 13 * 1024 ** 3)}) as a system deployment.
          {!s.image.present && !s.image.building && <> Its image is not built yet. <Button variant="link" className="h-auto p-0 text-sm" onClick={() => setImageDialog(true)}>Build image…</Button></>}
        </p>
      )}
      {s.upgrade?.needed && (
        <div role="alert" aria-label="Decision model needs a restart" className="space-y-2 rounded-lg border border-status-warning/40 bg-status-warning-bg px-4 py-3 text-sm">
          <p className="flex items-center gap-2 font-medium"><AlertTriangle aria-hidden className="h-4 w-4 text-status-warning" />
            The running decision model predates this DGXMM version and needs {s.image.present ? "a restart" : "its new image built, then a restart"}:</p>
          <ul className="list-disc pl-6 text-muted-foreground">{s.upgrade.reasons.map((r) => <li key={r}>{r}</li>)}</ul>
          {s.image.present
            ? <Button size="sm" variant="outline" disabled={rotate.isPending} onClick={() => setConfirmRotate(true)}><KeyRound className="mr-1 h-4 w-4" />Restart with a new key</Button>
            : <Button size="sm" variant="outline" onClick={() => setImageDialog(true)}><Hammer className="mr-1 h-4 w-4" />Build image…</Button>}
        </div>
      )}
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <StatusCard s={s} onRetry={() => enable.mutate(undefined)} retrying={enable.isPending} />
        <FeaturesCard s={s} />
        <ModelCard s={s} />
        <JobsCard s={s} />
        <Card data-card className="min-w-0 shadow-card lg:col-span-2">
          <CardHeader className="pb-2"><CardTitle className="text-sm"><h2>Schedules</h2></CardTitle>
            <CardDescription className="text-xs">Sleep / wake schedules the admins own run whatever the toggles above say; this stops all of them at once.</CardDescription></CardHeader>
          <CardContent><SchedulesPauseControl compact /></CardContent>
        </Card>
        <DecisionsCard s={s} />
      </div>
      {plans && <PlanDialog fit={plans.fit} plans={plans.plans} pending={enable.isPending} error={planErr}
        onSubmit={(actions) => enable.mutate(actions)} onClose={() => setPlans(null)} />}
      {imageDialog && <ImageDialog status={s} enabling={enable.isPending} onEnable={() => enable.mutate(undefined)} onClose={() => setImageDialog(false)} />}
      <ConfirmDialog open={confirmOff} onOpenChange={setConfirmOff} title="Disable the decision model?"
        description="Its deployment is stopped and its memory released. Diagnosis, suggestions and recommendations fall back to rules." confirmLabel="Disable"
        destructive pending={disable.isPending} onConfirm={() => disable.mutate(undefined)} />
      <ConfirmDialog open={confirmRotate} onOpenChange={setConfirmRotate} title="Rotate the decision model's API key?"
        description="DGXMM generates a new key. A running decision model is stopped and restarted with it (a few minutes); features fall back to rules meanwhile. The key is never shown."
        confirmLabel="Rotate" pending={rotate.isPending} onConfirm={() => rotate.mutate(undefined)} />
    </div>
  );
}

export function DecisionModel() {
  return useIsAdmin() ? <DecisionModelPage /> : <Navigate to="/" replace />;
}
