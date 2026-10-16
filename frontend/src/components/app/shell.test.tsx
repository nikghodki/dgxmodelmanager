import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { TooltipProvider } from "@/components/ui/tooltip";
import { Sidebar } from "@/components/app/sidebar";
import { AlertBanner } from "@/components/app/alert-banner";

const wrap = (ui: React.ReactNode) => render(<MemoryRouter><TooltipProvider>{ui}</TooltipProvider></MemoryRouter>);

describe("shell", () => {
  it("sidebar shows grouped nav and hides admin links for viewers", () => {
    wrap(<Sidebar isAdmin={false} collapsed={false} onToggle={vi.fn()} />);
    expect(screen.getByText("Operate")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: /Deployments/ })).toHaveAttribute("href", "/deployments");
    expect(screen.queryByRole("link", { name: /Users/ })).toBeNull();
  });
  it("collapsed sidebar keeps accessible names", () => {
    wrap(<Sidebar isAdmin collapsed onToggle={vi.fn()} />);
    expect(screen.getByRole("link", { name: "Audit log" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Expand sidebar" })).toBeInTheDocument();
  });
  it("alert banner renders for warning/critical only, with icon text", () => {
    const { rerender } = wrap(<AlertBanner alert={{ level: "ok" }} />);
    expect(screen.queryByRole("alert")).toBeNull();
    rerender(<MemoryRouter><TooltipProvider><AlertBanner alert={{ level: "critical", message: "only 5.0 GiB available" }} /></TooltipProvider></MemoryRouter>);
    expect(screen.getByRole("alert")).toHaveTextContent("Memory critical");
    expect(screen.getByRole("alert")).toHaveTextContent("only 5.0 GiB available");
  });
});
