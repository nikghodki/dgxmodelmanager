// Chart legends: the label text is drawn in the theme's muted-foreground (>= 4.5:1 on cards in both themes, see
// theme-contrast.test.ts), not in the series colour (several series colours are below 4.5:1 on a light card:
// e.g. #1baf7a 2.8, #eb6834 3.2); the series colour stays on the line icon next to it.
import { cloneElement, type ReactElement } from "react";
import { describe, expect, it, vi } from "vitest";
import { render } from "@testing-library/react";
import { MetricChart } from "@/components/app/metric-chart";
import { readChartColors } from "@/lib/chart-colors";

vi.mock("recharts", async (importOriginal) => {
  const mod = await importOriginal<typeof import("recharts")>();
  // jsdom has no layout: give the chart a fixed size so its legend renders
  const ResponsiveContainer = ({ children }: { children: ReactElement }) => cloneElement(children, { width: 600, height: 200 } as any);
  return { ...mod, ResponsiveContainer };
});

describe("chart legend", () => {
  it("draws the labels in the muted text colour, the series colours only on the icons", () => {
    const series = [{ key: "used", label: "Used", slot: 1 }, { key: "external", label: "External", slot: 2 },
      { key: "free_for_new", label: "Free for new", slot: 3 }];
    const { container } = render(<MetricChart data={[{ ts: 1792134000, used: 1, external: 2, free_for_new: 3 }]} series={series} />);
    const c = readChartColors();
    const labels = Array.from(container.querySelectorAll<HTMLElement>(".recharts-legend-item-text"));
    expect(labels.map((l) => l.textContent)).toEqual(["Used", "External", "Free for new"]);
    const norm = (v: string) => { const d = document.createElement("div"); d.style.color = v; return d.style.color; };
    // the colour the label text is drawn in: the nearest inline colour above its text node
    const drawn = (l: HTMLElement) => {
      let el: HTMLElement | null = document.createTreeWalker(l, NodeFilter.SHOW_TEXT).nextNode()?.parentElement ?? l;
      while (el && !el.style.color) el = el.parentElement;
      return norm(el?.style.color ?? "");
    };
    for (const l of labels) {
      expect(drawn(l)).toBe(norm(c.axis));
      expect(c.series.slice(0, 3).map(norm)).not.toContain(drawn(l));
    }
    const icons = Array.from(container.querySelectorAll(".recharts-legend-item path, .recharts-legend-item line"));
    expect(icons.length).toBeGreaterThanOrEqual(3);
  });
});
