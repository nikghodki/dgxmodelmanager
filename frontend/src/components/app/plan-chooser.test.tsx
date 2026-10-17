import { describe, expect, it, vi } from "vitest";
import { render, screen, within } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { PlanChooser } from "./plan-chooser";

const plan = (id: string, over: Record<string, any> = {}) => ({
  id, title: `Plan ${id}`, detail: "d", new_required: 1e9, frees: 2e9, fits: true, actions: [], patch: null, ...over,
});

describe("PlanChooser", () => {
  it("puts the recommended plan first with badge, reason, confidence and source", () => {
    render(<PlanChooser selectedId={null} onSelect={vi.fn()} plans={[
      plan("a"),
      plan("b", { recommended: true, recommendation: { source: "model", confidence: 0.87, reason: "frees 48 GiB; qwen idle 3 days" } }),
    ]} />);
    const radios = screen.getAllByRole("radio");
    expect(radios[0]).toHaveAccessibleName(/Plan b/);
    const rec = within(radios[0]);
    expect(rec.getByText("Recommended")).toBeInTheDocument();
    expect(rec.getByText("frees 48 GiB; qwen idle 3 days")).toBeInTheDocument();
    expect(rec.getByText("87% · AI")).toBeInTheDocument();
    expect(within(radios[1]).queryByText("Recommended")).toBeNull();
  });

  it("shows notes that are not plans, such as the decision model's memory, with a link to its page", () => {
    const note = "The decision model holds 13 GiB — disable it on the Decision model page to free it";
    render(<MemoryRouter future={{ v7_startTransition: true, v7_relativeSplatPath: true }}><PlanChooser selectedId={null} onSelect={vi.fn()} notes={[note]} plans={[plan("a")]} /></MemoryRouter>);
    expect(screen.getByText(note, { exact: false })).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Open it" })).toHaveAttribute("href", "/decision-model");
    expect(screen.getAllByRole("radio")).toHaveLength(1); // a note, never a plan action
  });

  it("shows the notes when no plan fits either", () => {
    render(<MemoryRouter future={{ v7_startTransition: true, v7_relativeSplatPath: true }}><PlanChooser selectedId={null} onSelect={vi.fn()} notes={["The decision model holds 9 GiB — disable it on the Decision model page to free it"]} plans={[]} /></MemoryRouter>);
    expect(screen.getByText(/No automatic plan frees enough memory/)).toBeInTheDocument();
    expect(screen.getByText(/holds 9 GiB/)).toBeInTheDocument();
  });

  it("annotates unsafe plans with a text note (icon + text), still selectable", () => {
    const onSelect = vi.fn();
    render(<PlanChooser selectedId={null} onSelect={onSelect} plans={[plan("a", { unsafe_reason: "stops an active deployment" })]} />);
    const radio = screen.getByRole("radio");
    expect(within(radio).getByText("Unsafe: stops an active deployment")).toBeInTheDocument();
    expect(radio).not.toBeDisabled();
    radio.click();
    expect(onSelect).toHaveBeenCalled();
  });

  it("shows rules recommendations without a percentage", () => {
    render(<PlanChooser selectedId={null} onSelect={vi.fn()} plans={[plan("a", { recommended: true, recommendation: { source: "rules", confidence: 1, reason: "only safe plan" } })]} />);
    expect(screen.getByText("Rules")).toBeInTheDocument();
  });
});
