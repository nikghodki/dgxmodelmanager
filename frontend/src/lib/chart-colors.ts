import { useMemo } from "react";
import { useTheme } from "@/components/app/theme";

const css = (name: string) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const hsl = (name: string) => { const v = css(name); return v ? `hsl(${v})` : ""; };

export function readChartColors() {
  return {
    series: [1, 2, 3, 4, 5, 6, 7, 8].map((i) => css(`--chart-${i}`) || "#2a78d6"),
    grid: hsl("--border") || "#e5e9f0",
    axis: hsl("--muted-foreground") || "#475569",
    text: hsl("--foreground") || "#0f172a",
    tooltipBg: hsl("--popover") || "#ffffff",
    tooltipBorder: hsl("--border") || "#e5e9f0",
  };
}

export function useChartColors() {
  const { resolved } = useTheme();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  return useMemo(() => readChartColors(), [resolved]);
}
