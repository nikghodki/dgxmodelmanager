import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { TooltipProvider } from "@/components/ui/tooltip";
import type { Deployment } from "@/types";
import { ApiAccessCard } from "./deployment-litellm";

const dep = (over: Partial<Deployment> = {}) => ({
  id: 1, name: "q", engine: "vllm", model_ref: "org/q", state: "ready", served_name: "q", model_name: "q", alias: null,
  endpoint: "http://dgxmm-q:8000/v1", host_endpoint: "http://127.0.0.1:8000/v1", litellm_status: "synced", system: false,
  ...over,
}) as unknown as Deployment;

const show = (d: Deployment) => render(<TooltipProvider><ApiAccessCard d={d} loading={false} error={null} /></TooltipProvider>);

describe("ApiAccessCard direct engine endpoints (final review I9)", () => {
  it("does not offer a keyed engine URL as if it worked without the key", () => {
    show(dep({ engine_key: true }));
    expect(screen.getByText(/needs the engine key; call it through LiteLLM/)).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Copy On the DGX" })).toBeNull();
    expect(screen.queryByRole("button", { name: "Copy Gateway network" })).toBeNull();
  });

  it("keeps the copy buttons for a deployment without an engine key", () => {
    show(dep({ engine_key: false }));
    expect(screen.getByRole("button", { name: "Copy On the DGX" })).toBeInTheDocument();
    expect(screen.queryByText(/needs the engine key/)).toBeNull();
  });
});
