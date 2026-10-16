export const GIB = 1024 ** 3;
export type Tone = "green" | "amber" | "red" | "slate" | "blue" | "violet";

export function fmtBytes(n: number | null | undefined, digits = 1): string {
  if (n === null || n === undefined || Number.isNaN(n)) return "—";
  const units = ["B", "KiB", "MiB", "GiB", "TiB"];
  let v = Math.abs(n);
  let i = 0;
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024;
    i++;
  }
  return `${n < 0 ? "-" : ""}${v.toFixed(i === 0 ? 0 : digits)} ${units[i]}`;
}
export const fmtPct = (x?: number | null) => (x == null ? "—" : `${Math.round(x * 100)}%`);
export const fmtNum = (x?: number | null, d = 1) => (x == null ? "—" : x.toFixed(d));
export const fmtMs = (s?: number | null) => (s == null ? "—" : `${Math.round(s * 1000)} ms`);

export function stateTone(state: string): Tone {
  if (state === "ready") return "green";
  if (state === "failed") return "red";
  if (["starting", "loading", "stopping", "planned"].includes(state)) return "amber";
  if (state === "sleeping") return "violet";
  if (state === "unknown") return "blue";
  return "slate";
}
