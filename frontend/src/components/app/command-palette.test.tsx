import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter, Route, Routes, useLocation } from "react-router-dom";
import { CommandPalette } from "@/components/app/command-palette";

function Where() { return <div data-testid="where">{useLocation().pathname}</div>; }

describe("CommandPalette", () => {
  it("filters and navigates on Enter", async () => {
    const qc = new QueryClient();
    qc.setQueryData(["deployments"], []);
    qc.setQueryData(["inventory"], []);
    render(
      <QueryClientProvider client={qc}>
        <MemoryRouter initialEntries={["/"]}>
          <Routes><Route path="*" element={<><CommandPalette open onOpenChange={() => undefined} /><Where /></>} /></Routes>
        </MemoryRouter>
      </QueryClientProvider>,
    );
    await userEvent.type(screen.getByPlaceholderText(/Search pages, deployments, models/), "planner");
    await userEvent.keyboard("{Enter}");
    expect(screen.getByTestId("where")).toHaveTextContent("/planner");
  });
});
