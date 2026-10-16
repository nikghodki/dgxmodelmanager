import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter } from "react-router-dom";
import { TooltipProvider } from "@/components/ui/tooltip";
import { DeploymentsTable } from "@/components/app/deployments-table";

const longRef = "/data/hf/hub/models--unsloth--Qwen3-0.6B-GGUF/snapshots/50968a4468ef4233ed78cd7c3de230dd1d61a56b/Qwen3-0.6B-Q8_0.gguf";

describe("DeploymentsTable", () => {
  it("truncates long model refs but keeps the full value in a title", () => {
    const qc = new QueryClient();
    qc.setQueryData(["live", 1], []);
    render(
      <QueryClientProvider client={qc}><MemoryRouter future={{ v7_startTransition: true, v7_relativeSplatPath: true }}><TooltipProvider>
        <DeploymentsTable isAdmin={false} onSwitch={vi.fn()} onReconfigure={vi.fn()}
          deployments={[{ id: 1, name: "small", engine: "llamacpp", model_ref: longRef, state: "ready", reserved_bytes: 1, actual_bytes: 1, litellm_status: "synced", sleep_capable: false, last_error: null } as any]} />
      </TooltipProvider></MemoryRouter></QueryClientProvider>,
    );
    const cell = screen.getByTitle(longRef);
    expect(cell.className).toMatch(/truncate/);
    expect(screen.getByText("Ready")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /Actions for small/ })).toBeInTheDocument();
  });
});
