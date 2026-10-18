import type { ScheduleAction, ScheduleProposal } from "@/types";

export const DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"] as const;
export const ALL_DAYS = 0b1111111;
export const WEEKDAYS = 0b0011111;
const HHMM = /^([01]\d|2[0-3]):[0-5]\d$/;

/** "Mon–Fri", "Sat, Sun", "every day" (the server's describe_days). */
export function daysLabel(mask: number): string {
  const days = DAY_NAMES.map((_, i) => i).filter((i) => (mask >> i) & 1);
  if (days.length === 7) return "every day";
  if (!days.length) return "no day";
  const runs: number[][] = [];
  for (const d of days) {
    const last = runs[runs.length - 1];
    if (last && d === last[last.length - 1] + 1) last.push(d); else runs.push([d]);
  }
  return runs.map((r) => (r.length >= 3 ? `${DAY_NAMES[r[0]]}–${DAY_NAMES[r[r.length - 1]]}` : r.map((x) => DAY_NAMES[x]).join(", "))).join(", ");
}

export const toggleDay = (mask: number, day: number) => mask ^ (1 << day);
export const isHHMM = (s: string) => HHMM.test(s);
export const pairOf = (a: ScheduleAction): ScheduleAction => ({ sleep: "wake", wake: "sleep", stop: "start", start: "stop" } as const)[a];
export const actionLabel = (a: ScheduleAction) => a[0].toUpperCase() + a.slice(1);
export const scheduleKey = (deploymentId: number) => ["deployment", deploymentId, "schedules"] as const;

const isMask = (v: unknown): v is number => typeof v === "number" && Number.isInteger(v) && v >= 1 && v <= ALL_DAYS;
const isTime = (v: unknown): v is string => typeof v === "string" && HHMM.test(v);

/** The proposal of a schedule insight, or null when it is not one this version understands. */
export function parseProposal(raw: unknown): ScheduleProposal | null {
  if (!raw || typeof raw !== "object") return null;
  const p = raw as Record<string, unknown>;
  const down = p.sleep_action, up = p.wake_action;
  if (!((down === "sleep" && up === "wake") || (down === "stop" && up === "start"))) return null;
  if (!isTime(p.sleep_at) || !isTime(p.wake_at) || !isMask(p.sleep_days) || !isMask(p.wake_days)) return null;
  if (typeof p.timezone !== "string" || !p.timezone || p.timezone.length > 64) return null;
  return p as unknown as ScheduleProposal;
}

/** The two schedules a proposal creates (POST /deployments/{id}/schedules). */
export function proposalRows(p: ScheduleProposal) {
  return [
    { action: p.sleep_action, days: p.sleep_days, at_local: p.sleep_at, timezone: p.timezone },
    { action: p.wake_action, days: p.wake_days, at_local: p.wake_at, timezone: p.timezone },
  ];
}

/** Milliseconds `tz` is ahead of UTC at instant `t`. */
function tzOffset(t: number, tz: string): number {
  const parts = new Intl.DateTimeFormat("en-US", { timeZone: tz, hourCycle: "h23", year: "numeric", month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit", second: "2-digit" }).formatToParts(new Date(t));
  const get = (k: string) => Number(parts.find((p) => p.type === k)?.value);
  return Date.UTC(get("year"), get("month") - 1, get("day"), get("hour") % 24, get("minute"), get("second")) - Math.floor(t / 1000) * 1000;
}

/** The UTC instant of wall time y-m-d hh:mm in `tz` (a time inside a spring-forward gap lands just after it). */
export function zonedTime(y: number, m: number, d: number, hh: number, mm: number, tz: string): number {
  const wall = Date.UTC(y, m, d, hh, mm);
  const before = tzOffset(wall - 12 * 3600_000, tz), after = tzOffset(wall + 12 * 3600_000, tz);
  // the instants whose wall time is exactly this one: two in an autumn overlap (the first wins), none in a gap
  const exact = [wall - before, wall - after].filter((t) => t + tzOffset(t, tz) === wall);
  return exact.length ? Math.min(...exact) : wall - before; // gap: the pre-transition offset lands after it
}

/** Next run of a schedule after `now` (preview before saving; the server's next_run_at is authoritative). */
export function nextRunPreview(days: number, at: string, tz: string, now: number = Date.now()): Date | null {
  if (!days || !isHHMM(at)) return null;
  let ymd: { y: number; m: number; d: number };
  try {
    const parts = new Intl.DateTimeFormat("en-US", { timeZone: tz, year: "numeric", month: "2-digit", day: "2-digit" }).formatToParts(new Date(now));
    const get = (k: string) => Number(parts.find((p) => p.type === k)?.value);
    ymd = { y: get("year"), m: get("month") - 1, d: get("day") };
  } catch {
    return null; // unknown time zone
  }
  const [hh, mm] = at.split(":").map(Number);
  for (let i = 0; i < 8; i++) {
    const date = new Date(Date.UTC(ymd.y, ymd.m, ymd.d + i));
    const weekday = (date.getUTCDay() + 6) % 7; // 0 = Monday
    if (!((days >> weekday) & 1)) continue;
    const t = zonedTime(date.getUTCFullYear(), date.getUTCMonth(), date.getUTCDate(), hh, mm, tz);
    if (t > now) return new Date(t);
  }
  return null;
}

export function validTimeZone(tz: string): boolean {
  try {
    new Intl.DateTimeFormat("en-US", { timeZone: tz });
    return /^[A-Za-z0-9_+-]+(\/[A-Za-z0-9_+-]+)*$/.test(tz);
  } catch {
    return false;
  }
}
