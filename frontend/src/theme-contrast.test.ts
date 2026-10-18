// WCAG AA text contrast (>= 4.5:1) of the theme's token pairs, in both themes, computed from src/index.css.
// Tinted backgrounds (`bg-primary/15` and friends) are composited over the surfaces they sit on.
import { readFileSync } from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";

const css = readFileSync(path.resolve(__dirname, "index.css"), "utf8");

function tokens(selector: string): Record<string, string> {
  const start = css.indexOf(`${selector} {`);
  const body = css.slice(start, css.indexOf("}", start));
  const out: Record<string, string> = {};
  for (const m of body.matchAll(/--([\w-]+):\s*([^;]+);/g)) out[m[1]] = m[2].trim();
  return out;
}

type RGB = [number, number, number];

function rgb(v: string): RGB {
  if (v.startsWith("#")) return [1, 3, 5].map((i) => parseInt(v.slice(i, i + 2), 16) / 255) as RGB;
  const [h, s, l] = v.split(/\s+/).map((x) => parseFloat(x));
  const S = s / 100, L = l / 100;
  const a = S * Math.min(L, 1 - L);
  const f = (n: number) => { const k = (n + h / 30) % 12; return L - a * Math.max(-1, Math.min(k - 3, 9 - k, 1)); };
  return [f(0), f(8), f(4)];
}

const lum = (c: RGB) => {
  const [r, g, b] = c.map((x) => (x <= 0.04045 ? x / 12.92 : ((x + 0.055) / 1.055) ** 2.4));
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
};
const ratio = (a: RGB, b: RGB) => { const [x, y] = [lum(a), lum(b)].sort((p, q) => q - p); return (x + 0.05) / (y + 0.05); };
const over = (fg: RGB, bg: RGB, alpha: number) => fg.map((x, i) => alpha * x + (1 - alpha) * bg[i]) as RGB;

// [text, surface, optional tint over the surface and its alpha]
const PAIRS: [string, string, string?, number?][] = [
  ["primary-foreground", "primary"],
  ["primary-foreground", "card", "primary", 0.9], // hover:bg-primary/90 buttons and badges
  ["destructive-foreground", "destructive"],
  ["destructive-foreground", "card", "destructive", 0.9], // hover:bg-destructive/90
  ["foreground", "background"], ["card-foreground", "card"], ["secondary-foreground", "secondary"],
  ["accent-foreground", "accent"], ["sidebar-foreground", "sidebar"],
  ["muted-foreground", "card"], // also chart axis ticks and legend labels (metric-chart-legend.test.tsx)
  ["muted-foreground", "background"], ["muted-foreground", "muted"],
  ["muted-foreground", "sidebar"], ["muted-foreground", "card", "muted", 0.5],
  ...["card", "background", "sidebar", "popover"].flatMap((s) => [
    [ "primary", s ] as [string, string],                                // links, active nav, <summary>
    ...[0.1, 0.15, 0.2].map((a) => ["primary", s, "primary", a] as [string, string, string, number]), // tinted badges, avatar
  ]),
  ...["good", "warning", "sleeping", "critical", "neutral"].flatMap((s) => [
    [`status-${s}`, `status-${s}-bg`] as [string, string], [`status-${s}`, "card"] as [string, string],
  ]),
];

describe.each([["light", ":root"], ["dark", ".dark"]])("%s theme text contrast", (_name, selector) => {
  const t = tokens(selector);
  it.each(PAIRS.map((p) => [`${p[0]} on ${p[1]}${p[2] ? ` + ${p[2]}/${Math.round((p[3] ?? 0) * 100)}` : ""}`, p]))(
    "%s is at least 4.5:1", (_label, [fg, surface, tint, alpha]) => {
      const bg = tint ? over(rgb(t[tint]), rgb(t[surface]), alpha ?? 1) : rgb(t[surface]);
      expect(ratio(rgb(t[fg]), bg)).toBeGreaterThanOrEqual(4.5);
    });
});
