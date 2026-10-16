import { describe, expect, it, vi } from "vitest";
import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type { ColumnDef } from "@tanstack/react-table";
import { DataTable } from "@/components/app/data-table";

type Row = { name: string; size: number; ref: string };
const rows: Row[] = [
  { name: "beta", size: 2, ref: "org/beta" },
  { name: "alpha", size: 9, ref: "/data/hf/hub/models--unsloth--Qwen3-0.6B-GGUF/snapshots/50968a4468ef4233ed78cd7c3de230dd1d61a56b/Qwen3-0.6B-Q8_0.gguf" },
];
const columns: ColumnDef<Row, any>[] = [
  { accessorKey: "name", header: "Name" },
  { accessorKey: "size", header: "Size", meta: { label: "Size" } },
  { accessorKey: "ref", header: "Ref", meta: { mono: true } },
];

function setMobile(on: boolean) {
  window.matchMedia = vi.fn().mockImplementation((q: string) => ({
    matches: on && q.includes("max-width"), media: q, addEventListener: vi.fn(), removeEventListener: vi.fn(),
  })) as any;
}

describe("DataTable", () => {
  it("sorts by a column header and filters by search", async () => {
    setMobile(false);
    render(<DataTable columns={columns} data={rows} searchPlaceholder="Search" searchText={(r) => r.name} />);
    const body = () => screen.getAllByRole("row").slice(1).map((r) => within(r).getAllByRole("cell")[0].textContent);
    expect(body()).toEqual(["beta", "alpha"]);
    await userEvent.click(screen.getByRole("button", { name: /Name/ }));
    expect(body()).toEqual(["alpha", "beta"]);
    await userEvent.type(screen.getByPlaceholderText("Search"), "bet");
    expect(body()).toEqual(["beta"]);
  });

  it("renders stacked cards on mobile with labels and wrapping long values", () => {
    setMobile(true);
    render(<DataTable columns={columns} data={rows} mobileTitle={(r) => r.name} />);
    expect(screen.queryByRole("table")).toBeNull();
    const cards = screen.getAllByRole("listitem");
    expect(cards).toHaveLength(2);
    expect(within(cards[1]).getByText("Size")).toBeInTheDocument();
    const longVal = within(cards[1]).getByText(/Qwen3-0.6B-Q8_0.gguf/);
    expect(longVal.className).toMatch(/break-all/);
  });

  it("shows the empty slot when there is no data", () => {
    setMobile(false);
    render(<DataTable columns={columns} data={[]} empty={<p>Nothing here</p>} />);
    expect(screen.getByText("Nothing here")).toBeInTheDocument();
  });
});
