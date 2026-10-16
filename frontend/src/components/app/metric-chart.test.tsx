import { afterEach, describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { ChartCard, MetricChart } from "@/components/app/metric-chart";
import { readChartColors, useChartColors } from "@/lib/chart-colors";
import { ThemeProvider, useTheme } from "@/components/app/theme";

describe("charts", () => {
  afterEach(() => {
    document.documentElement.removeAttribute("style");
    document.documentElement.classList.remove("dark");
    localStorage.clear();
  });

  it("reads series colours from CSS variables", () => {
    document.documentElement.style.setProperty("--chart-1", "#123456");
    document.documentElement.style.setProperty("--chart-2", "#abcdef");
    const c = readChartColors();
    expect(c.series[0]).toBe("#123456");
    expect(c.series[1]).toBe("#abcdef");
  });

  it("re-reads colours when the resolved theme changes", async () => {
    // Set different colours for light and dark modes via a style element
    const style = document.createElement("style");
    style.textContent = `
      :root { --chart-1: #aaaaaa; }
      .dark { --chart-1: #bbbbbb; }
    `;
    document.head.appendChild(style);

    function Probe() {
      const c = useChartColors();
      const { setTheme } = useTheme();
      return (
        <>
          <div data-testid="color">{c.series[0]}</div>
          <button onClick={() => setTheme("light")}>Light</button>
          <button onClick={() => setTheme("dark")}>Dark</button>
        </>
      );
    }

    render(
      <ThemeProvider><Probe /></ThemeProvider>
    );

    // Start in light mode (default)
    expect(screen.getByTestId("color")).toHaveTextContent("#aaaaaa");

    // Switch to dark
    await userEvent.click(screen.getByRole("button", { name: "Dark" }));
    expect(screen.getByTestId("color")).toHaveTextContent("#bbbbbb");

    document.head.removeChild(style);
  });

  it("shows an empty state instead of NaN when there is no data", () => {
    render(<MetricChart data={[]} series={[{ key: "gen_tps", label: "generation", slot: 1 }]} />);
    expect(screen.getByText("No data yet")).toBeInTheDocument();
  });

  it("offers a table view with formatted values and '—' for gaps", async () => {
    render(<ChartCard title="Throughput" data={[{ ts: 1792134000, gen_tps: 12.345 }, { ts: 1792134005, gen_tps: NaN }]}
      series={[{ key: "gen_tps", label: "generation", slot: 1 }]} fmt={(v) => v.toFixed(1)} />);
    await userEvent.click(screen.getByRole("button", { name: /Table/ }));
    expect(screen.getByRole("columnheader", { name: "generation" })).toBeInTheDocument();
    expect(screen.getByText("12.3")).toBeInTheDocument();
    expect(screen.getAllByText("—").length).toBeGreaterThan(0);
  });
});
