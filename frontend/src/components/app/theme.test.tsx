import { beforeEach, describe, expect, it, vi } from "vitest";
import { act, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { ThemeProvider, ThemeToggle, useTheme, THEME_KEY } from "@/components/app/theme";

function mockMedia(dark: boolean) {
  const listeners: ((e: { matches: boolean }) => void)[] = [];
  const mq = { matches: dark, media: "", addEventListener: (_: string, l: any) => listeners.push(l), removeEventListener: vi.fn() };
  window.matchMedia = vi.fn().mockReturnValue(mq) as any;
  return { set(v: boolean) { mq.matches = v; listeners.forEach((l) => l({ matches: v })); } };
}

function Probe() {
  const { resolved, theme } = useTheme();
  return <span data-testid="probe">{theme}:{resolved}</span>;
}

describe("theme", () => {
  beforeEach(() => { localStorage.clear(); document.documentElement.className = ""; });

  it("defaults to system and follows the OS live", () => {
    const media = mockMedia(false);
    render(<ThemeProvider><Probe /></ThemeProvider>);
    expect(screen.getByTestId("probe")).toHaveTextContent("system:light");
    act(() => media.set(true));
    expect(document.documentElement.classList.contains("dark")).toBe(true);
    expect(screen.getByTestId("probe")).toHaveTextContent("system:dark");
  });

  it("persists an explicit choice and applies the class", async () => {
    mockMedia(false);
    render(<ThemeProvider><ThemeToggle /><Probe /></ThemeProvider>);
    await userEvent.click(screen.getByRole("button", { name: "Change theme" }));
    await userEvent.click(await screen.findByRole("menuitemradio", { name: /Dark/ }));
    expect(localStorage.getItem(THEME_KEY)).toBe("dark");
    expect(document.documentElement.classList.contains("dark")).toBe(true);
  });

  it("restores the saved theme on mount (no flash path)", () => {
    mockMedia(true);
    localStorage.setItem(THEME_KEY, "light");
    render(<ThemeProvider><Probe /></ThemeProvider>);
    expect(document.documentElement.classList.contains("dark")).toBe(false);
    expect(screen.getByTestId("probe")).toHaveTextContent("light:light");
  });
});
