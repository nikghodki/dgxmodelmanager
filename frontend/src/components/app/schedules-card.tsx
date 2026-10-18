import { useEffect, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { CalendarClock, Pause, Pencil, Play, Plus, Trash2 } from "lucide-react";
import { api } from "@/api";
import { useIsAdmin } from "@/auth";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { useActionMutation } from "@/hooks/use-action-mutation";
import {
  ALL_DAYS, DAY_NAMES, WEEKDAYS, actionLabel, daysLabel, isHHMM, nextRunPreview, pairOf, parseProposal, proposalRows,
  scheduleKey, toggleDay, validTimeZone,
} from "@/lib/schedules";
import type { Deployment, InsightAction, Schedule, ScheduleAction, ScheduleList, ScheduleSettings } from "@/types";
import { ConfirmDialog } from "./confirm-dialog";
import { type InsightActionDialogProps, registerInsightActionDialog } from "./insight-action";
import { ResponsiveDialog } from "./responsive-dialog";
import { ErrorState } from "./states";

const fmtWhen = (iso: string | Date | null) => (iso ? new Date(iso).toLocaleString(undefined, { weekday: "short", day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" }) : "—");

function useSchedules(deploymentId: number, enabled = true) {
  return useQuery({ queryKey: scheduleKey(deploymentId), queryFn: () => api<ScheduleList>(`/deployments/${deploymentId}/schedules`), enabled, refetchInterval: 60_000 });
}

const SETTINGS_KEY = ["schedule-settings"];

/** "Pause all schedules": while set, no schedule runs (audited). Shown on the Schedules section and the Decision model
 *  page; admins switch it, viewers see the state. */
export function SchedulesPauseControl({ compact = false }: { compact?: boolean }) {
  const admin = useIsAdmin();
  const q = useQuery({ queryKey: SETTINGS_KEY, queryFn: () => api<ScheduleSettings>("/schedules/settings") });
  const [confirm, setConfirm] = useState(false);
  const set = useActionMutation((paused: boolean) => api("/schedules/settings", { method: "PUT", body: { paused_all: paused } }), {
    success: (_d, p) => (p ? "All schedules paused" : "Schedules resumed"), invalidate: [SETTINGS_KEY, ["deployment"]],
    onSuccess: () => setConfirm(false),
  });
  const paused = !!q.data?.paused_all;
  if (!q.data) return null;
  return (
    <div className={`flex flex-wrap items-center gap-2 ${compact ? "" : "rounded-md border p-3"}`}>
      <p role="status" className="min-w-0 flex-1 text-sm">{paused
        ? <><span className="font-medium text-status-warning">All schedules are paused.</span> No deployment is put to sleep, stopped, woken or started by a schedule.</>
        : <span className="text-muted-foreground">Schedules run at their times (box time zone {q.data.timezone}).</span>}</p>
      {admin && (paused
        ? <Button size="sm" variant="outline" disabled={set.isPending} onClick={() => set.mutate(false)}><Play aria-hidden className="mr-1 h-3.5 w-3.5" />Resume all schedules</Button>
        : <Button size="sm" variant="outline" disabled={set.isPending} onClick={() => setConfirm(true)}><Pause aria-hidden className="mr-1 h-3.5 w-3.5" />Pause all schedules</Button>)}
      <ConfirmDialog open={confirm} onOpenChange={setConfirm} title="Pause all schedules?" confirmLabel="Pause all" pending={set.isPending}
        description="No schedule runs until you resume them: deployments stay as they are. Each schedule keeps its own settings." onConfirm={() => set.mutate(true)} />
    </div>
  );
}

function DayPicker({ value, onChange, idPrefix }: { value: number; onChange: (m: number) => void; idPrefix: string }) {
  return (
    <div role="group" aria-label="Days" className="flex flex-wrap gap-1" id={idPrefix}>
      {DAY_NAMES.map((d, i) => {
        const on = !!((value >> i) & 1);
        return (
          <button key={d} type="button" aria-pressed={on} onClick={() => onChange(toggleDay(value, i))}
            className={`h-8 min-w-[2.75rem] rounded-md border px-2 text-xs ${on ? "border-primary bg-primary text-primary-foreground" : "text-muted-foreground hover:bg-accent"}`}>{d}</button>
        );
      })}
      <button type="button" className="h-8 px-2 text-xs text-primary hover:underline" onClick={() => onChange(WEEKDAYS)}>Weekdays</button>
      <button type="button" className="h-8 px-2 text-xs text-primary hover:underline" onClick={() => onChange(ALL_DAYS)}>Every day</button>
    </div>
  );
}

interface Draft { action: ScheduleAction; days: number; at_local: string; timezone: string; pair: boolean; pair_at: string }

/** Add or edit one schedule (adding may also create the matching wake / start). Shows when it would run next. */
function ScheduleEditor({ deploymentId, list, editing, onClose }: { deploymentId: number; list: ScheduleList; editing: Schedule | null; onClose: () => void }) {
  const allowed = list.allowed_actions;
  const [d, setD] = useState<Draft>(() => editing
    ? { action: editing.action, days: editing.days, at_local: editing.at_local, timezone: editing.timezone, pair: false, pair_at: "" }
    : { action: allowed[0], days: WEEKDAYS, at_local: "19:00", timezone: list.timezone, pair: true, pair_at: "07:45" });
  const set = (p: Partial<Draft>) => setD((o) => ({ ...o, ...p }));
  const save = useActionMutation(() => editing
    ? api(`/schedules/${editing.id}`, { method: "PUT", body: { action: d.action, days: d.days, at_local: d.at_local, timezone: d.timezone } })
    : api(`/deployments/${deploymentId}/schedules`, { method: "POST", body: { schedules: [
      { action: d.action, days: d.days, at_local: d.at_local, timezone: d.timezone },
      ...(d.pair ? [{ action: pairOf(d.action), days: d.days, at_local: d.pair_at, timezone: d.timezone }] : []),
    ] } }), {
    success: editing ? "Schedule saved" : "Schedule added", invalidate: [[...scheduleKey(deploymentId)]], onSuccess: onClose,
  });
  const tzOk = validTimeZone(d.timezone);
  const problem = !d.days ? "Choose at least one day." : !isHHMM(d.at_local) || (d.pair && !isHHMM(d.pair_at)) ? "Times are HH:MM (24 h)."
    : !tzOk ? "Unknown time zone (an IANA name such as Europe/Berlin)." : null;
  const next = problem ? null : nextRunPreview(d.days, d.at_local, d.timezone);
  const nextPair = problem || !d.pair ? null : nextRunPreview(d.days, d.pair_at, d.timezone);
  return (
    <ResponsiveDialog open onOpenChange={(o) => !o && onClose()} title={editing ? "Edit schedule" : "Add a schedule"} className="max-w-lg">
      <form className="space-y-4" onSubmit={(e) => { e.preventDefault(); if (!problem) save.mutate(undefined); }}>
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
          <div className="space-y-1.5">
            <Label htmlFor="sched-action">Action</Label>
            <Select value={d.action} onValueChange={(v) => set({ action: v as ScheduleAction })}>
              <SelectTrigger id="sched-action" aria-label="Action"><SelectValue /></SelectTrigger>
              <SelectContent>{allowed.map((a) => <SelectItem key={a} value={a}>{actionLabel(a)}</SelectItem>)}</SelectContent>
            </Select>
          </div>
          <div className="space-y-1.5">
            <Label htmlFor="sched-at">At</Label>
            <Input id="sched-at" type="time" value={d.at_local} onChange={(e) => set({ at_local: e.target.value })} required />
          </div>
        </div>
        <div className="space-y-1.5"><Label htmlFor="sched-days">Days</Label><DayPicker idPrefix="sched-days" value={d.days} onChange={(days) => set({ days })} /></div>
        <div className="space-y-1.5">
          <Label htmlFor="sched-tz">Time zone</Label>
          <Input id="sched-tz" value={d.timezone} onChange={(e) => set({ timezone: e.target.value.trim() })} aria-invalid={!tzOk} />
        </div>
        {!editing && (
          <div className="space-y-2 rounded-md border p-3">
            <label className="flex items-center gap-2 text-sm"><input type="checkbox" checked={d.pair} onChange={(e) => set({ pair: e.target.checked })} />
              Also {pairOf(d.action)} it on the same days at</label>
            {d.pair && <Input aria-label={`${actionLabel(pairOf(d.action))} at`} type="time" value={d.pair_at} onChange={(e) => set({ pair_at: e.target.value })} className="w-32" />}
          </div>
        )}
        <div role="status" aria-live="polite" className="rounded-md bg-muted/50 p-3 text-xs text-muted-foreground">
          {problem ?? <>Next run: <span className="font-medium text-foreground">{actionLabel(d.action)} {fmtWhen(next)}</span>
            {nextPair && <> · {actionLabel(pairOf(d.action))} {fmtWhen(nextPair)}</>} (your browser's time)</>}
        </div>
        <ErrorState error={save.error} />
        <div className="flex justify-end gap-2">
          <Button type="button" variant="outline" size="sm" onClick={onClose}>Cancel</Button>
          <Button type="submit" size="sm" disabled={!!problem || save.isPending}>{editing ? "Save" : "Add"}</Button>
        </div>
      </form>
    </ResponsiveDialog>
  );
}

/** Schedules section of Deployment detail: list with the next run, add / edit / pause / delete (admins). */
export function DeploymentSchedulesCard({ deployment, className }: { deployment: Deployment; className?: string }) {
  const admin = useIsAdmin();
  const q = useSchedules(deployment.id, !deployment.system);
  const [editing, setEditing] = useState<Schedule | null | "new">(null);
  const [deleting, setDeleting] = useState<Schedule | null>(null);
  const key = [...scheduleKey(deployment.id)];
  const toggle = useActionMutation((s: Schedule) => api(`/schedules/${s.id}`, { method: "PUT", body: { enabled: !s.enabled } }), {
    success: (_d, s) => (s.enabled ? "Schedule paused" : "Schedule resumed"), invalidate: [key],
  });
  const del = useActionMutation((s: Schedule) => api(`/schedules/${s.id}`, { method: "DELETE" }), {
    success: "Schedule deleted", invalidate: [key], onSuccess: () => setDeleting(null),
  });
  if (deployment.system) return null;
  const list = q.data;
  const rows = list?.schedules ?? [];
  const canAdd = admin && !!list?.allowed_actions.length;
  return (
    <Card className={`min-w-0 shadow-card ${className ?? ""}`}>
      <CardHeader className="flex-row items-start justify-between gap-2 space-y-0 pb-2">
        <div className="min-w-0">
          <CardTitle className="flex items-center gap-1.5 text-sm"><CalendarClock aria-hidden className="h-4 w-4" />Schedules</CardTitle>
          <CardDescription className="text-xs">DGXMM puts it to sleep and wakes it at these times. It never does so while requests arrive, and never stops anything else to make room.</CardDescription>
        </div>
        {canAdd && <Button variant="outline" size="sm" onClick={() => setEditing("new")}><Plus aria-hidden className="mr-1 h-3.5 w-3.5" />Add</Button>}
      </CardHeader>
      <CardContent className="space-y-3">
        {list?.paused_all && <SchedulesPauseControl />}
        <ErrorState error={q.error} onRetry={() => q.refetch()} />
        {q.isLoading ? <p className="text-sm text-muted-foreground">Loading schedules…</p> : !rows.length ? (
          <p className="text-sm text-muted-foreground">No schedules.{admin ? " Add one, or accept a suggestion from Insights." : ""}</p>
        ) : (
          <ul className="divide-y" aria-label="Schedules">
            {rows.map((s) => (
              <li key={s.id} className="flex flex-col gap-2 py-2 text-sm sm:flex-row sm:items-center sm:justify-between">
                <div className="min-w-0">
                  <p className="flex flex-wrap items-center gap-2">
                    <span className="font-medium">{actionLabel(s.action)} at {s.at_local}</span>
                    <span className="text-muted-foreground">{s.days_label} · {s.timezone}</span>
                    {!s.enabled && <Badge variant="secondary">Paused</Badge>}
                    {s.enabled && s.deferred && <Badge variant="outline">Deferred</Badge>}
                    {s.enabled && s.pending && <Badge variant="outline">Pending</Badge>}
                  </p>
                  <p className="text-xs text-muted-foreground">Next run: {s.enabled ? fmtWhen(s.next_run_at) : "paused"}
                    {s.last_result && <> · Last: <span className="break-words">{s.last_result}</span>{s.last_run_at && <> ({fmtWhen(s.last_run_at)})</>}</>}</p>
                </div>
                {admin && (
                  <div className="flex shrink-0 gap-1">
                    <Button variant="ghost" size="icon" className="h-8 w-8" aria-label={`Edit ${s.action} at ${s.at_local}`} onClick={() => setEditing(s)}><Pencil className="h-4 w-4" /></Button>
                    <Button variant="ghost" size="icon" className="h-8 w-8" disabled={toggle.isPending} aria-label={`${s.enabled ? "Pause" : "Resume"} ${s.action} at ${s.at_local}`} onClick={() => toggle.mutate(s)}>
                      {s.enabled ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}</Button>
                    <Button variant="ghost" size="icon" className="h-8 w-8 text-status-critical" aria-label={`Delete ${s.action} at ${s.at_local}`} onClick={() => setDeleting(s)}><Trash2 className="h-4 w-4" /></Button>
                  </div>
                )}
              </li>
            ))}
          </ul>
        )}
      </CardContent>
      {editing && list && <ScheduleEditor deploymentId={deployment.id} list={list} editing={editing === "new" ? null : editing} onClose={() => setEditing(null)} />}
      <ConfirmDialog open={!!deleting} onOpenChange={(o) => !o && setDeleting(null)} title="Delete this schedule?"
        description={deleting ? `${actionLabel(deleting.action)} at ${deleting.at_local} (${daysLabel(deleting.days)}) will no longer run.` : ""}
        confirmLabel="Delete" destructive pending={del.isPending} onConfirm={() => deleting && del.mutate(deleting)} />
    </Card>
  );
}

/** The schedule insight's dialog: what would run when; "Create schedules" creates the sleep and wake rows. */
export function ScheduleProposalDialog({ insight, action, onClose, onApplied }: InsightActionDialogProps<Extract<InsightAction, { type: "schedule" }>>) {
  const p = parseProposal(action.proposal);
  const [now] = useState(() => Date.now());
  const create = useActionMutation(() => api(`/deployments/${action.deployment_id}/schedules`, { method: "POST", body: { schedules: proposalRows(p!) } }), {
    success: "Schedules created", invalidate: [[...scheduleKey(action.deployment_id)], ["insights"]], onSuccess: () => { onApplied(); onClose(); },
  });
  useEffect(() => { create.reset(); }, [action.deployment_id]); // eslint-disable-line react-hooks/exhaustive-deps
  return (
    <ResponsiveDialog open onOpenChange={(o) => !o && onClose()} title={insight.title} className="max-w-lg">
      {!p ? (
        <div className="space-y-3 text-sm"><p className="text-muted-foreground">This suggestion cannot be shown by this version of DGXMM.</p>
          <div className="flex justify-end"><Button variant="outline" size="sm" onClick={onClose}>Close</Button></div></div>
      ) : (
        <div className="space-y-3 text-sm">
          <p className="text-muted-foreground">{insight.body}</p>
          <table className="w-full text-left text-sm">
            <caption className="sr-only">What will run when</caption>
            <thead className="text-xs text-muted-foreground"><tr><th className="py-1 font-medium">Action</th><th className="font-medium">When</th><th className="font-medium">First run</th></tr></thead>
            <tbody>
              {proposalRows(p).map((r) => (
                <tr key={r.action} className="border-t">
                  <td className="py-1.5 font-medium">{actionLabel(r.action)}</td>
                  <td>{r.at_local} · {daysLabel(r.days)} · {r.timezone}</td>
                  <td className="tabular-nums">{fmtWhen(nextRunPreview(r.days, r.at_local, r.timezone, now))}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="text-xs text-muted-foreground">You own these schedules: pause, edit or delete them on the deployment's page. A run is skipped while requests arrive, and a wake-up that does not fit is skipped rather than stopping anything else.</p>
          <ErrorState error={create.error} />
          <div className="flex justify-end gap-2">
            <Button variant="outline" size="sm" onClick={onClose}>Cancel</Button>
            <Button size="sm" disabled={create.isPending} onClick={() => create.mutate(undefined)}>Create schedules</Button>
          </div>
        </div>
      )}
    </ResponsiveDialog>
  );
}

registerInsightActionDialog("schedule", ScheduleProposalDialog);
