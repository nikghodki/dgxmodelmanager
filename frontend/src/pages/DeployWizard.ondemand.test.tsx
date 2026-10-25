import { afterEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter } from "react-router-dom";
import { DeployWizard } from "./DeployWizard";
import * as auth from "@/auth";
import * as apiModule from "@/api";

vi.mock("@/auth");
vi.mock("@/api", async () => {
  const actual = await vi.importActual<typeof import("@/api")>("@/api");
  return { ...actual, api: vi.fn() };
});

const est = { total: 1e9, weights: 1e9, draft_weights: 0, kv_cache: 0, state: 0, activations: 0, runtime: 0, margin: 0, notes: [], confidence: "measured" };
const fitsPreview = { estimate: est, fit: { fits: true, free_for_new: 2e9, headroom: 1e9 }, plans: [] };
const artifact = { ref: "org/model-a", complete: true, kind: "llm", engines: ["vllm"], quant: "Q4", weights_bytes: 1e9, architecture: "Qwen", max_context: 32768, problems: [] };
const engines = [{ name: "vllm", supports_sleep: true, defaults: { max_model_len: 4096 }, image: null }];

function setup(idleDefault = 300, litellmReady = true) {
  vi.mocked(auth.useIsAdmin).mockReturnValue(true);
  const api = vi.mocked(apiModule.api);
  api.mockImplementation((url: string) => {
    if (url === "/host") return Promise.resolve({ ondemand: { idle_timeout_s: idleDefault, queue_timeout_s: 900, litellm_ready: litellmReady } });
    if (url === "/inventory") return Promise.resolve([artifact]);
    if (url === "/engines") return Promise.resolve(engines);
    if (url === "/deployments/preview") return Promise.resolve(fitsPreview);
    if (url === "/deployments") return new Promise(() => undefined);
    return Promise.reject(new Error(`unexpected ${url}`));
  });
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } });
  render(
    <QueryClientProvider client={qc}>
      <MemoryRouter initialEntries={["/deploy?model=org%2Fmodel-a"]} future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
        <DeployWizard />
      </MemoryRouter>
    </QueryClientProvider>,
  );
  return api;
}

const deployBody = (api: ReturnType<typeof setup>) => (api.mock.calls.find((c) => c[0] === "/deployments")![1] as any).body;

afterEach(() => { vi.mocked(apiModule.api).mockReset(); });

describe("DeployWizard on-demand mode", () => {
  it("pre-fills the global idle default and sends null while it is unchanged", async () => {
    const api = setup(600);
    await waitFor(() => expect(screen.getAllByText("Fits").length).toBeGreaterThan(0));
    await waitFor(() => expect(screen.getByLabelText("Idle minutes before sleeping")).toHaveValue(10));
    expect(screen.getByRole("radio", { name: /sleep after 10 min idle$/ })).toBeChecked();
    await userEvent.click(screen.getAllByRole("button", { name: "Deploy" })[0]);
    expect(deployBody(api)).toMatchObject({ mode: "on_demand", idle_timeout_s: null });
  });

  it("defaults to on demand with a 5 minute idle timeout and sends a changed idle_timeout_s", async () => {
    const api = setup();
    await waitFor(() => expect(screen.getAllByText("Fits").length).toBeGreaterThan(0));
    await waitFor(() => expect(screen.getByLabelText("Idle minutes before sleeping")).toHaveValue(5));
    const onDemand = screen.getByRole("radio", { name: /^On demand — start on the first request, sleep after 5 min idle$/ });
    expect(onDemand).toBeChecked();
    expect(screen.getByRole("radio", { name: "Always on" })).not.toBeChecked();
    const idle = screen.getByLabelText("Idle minutes before sleeping");
    expect(idle).toHaveValue(5);
    expect(idle).toHaveAttribute("min", "1");
    expect(idle).toHaveAttribute("max", "1440");
    fireEvent.change(idle, { target: { value: "10" } });
    expect(screen.getByRole("radio", { name: /sleep after 10 min idle$/ })).toBeChecked();
    await userEvent.click(screen.getAllByRole("button", { name: "Deploy" })[0]);
    expect(deployBody(api)).toMatchObject({ mode: "on_demand", idle_timeout_s: 600 });
  });

  it("defaults to always on with a hint when LiteLLM is not set up or its last sync failed", async () => {
    const api = setup(300, false);
    await waitFor(() => expect(screen.getAllByText("Fits").length).toBeGreaterThan(0));
    await waitFor(() => expect(screen.getByText("On demand needs LiteLLM (it starts models on requests that come through it).")).toBeInTheDocument());
    expect(screen.getByRole("radio", { name: "Always on" })).toBeChecked();
    expect(screen.queryByLabelText("Idle minutes before sleeping")).toBeNull();
    await userEvent.click(screen.getAllByRole("button", { name: "Deploy" })[0]);
    expect(deployBody(api)).toMatchObject({ mode: "always_on", idle_timeout_s: null });
  });

  it("still lets you choose on demand without LiteLLM", async () => {
    const api = setup(300, false);
    await waitFor(() => expect(screen.getAllByText("Fits").length).toBeGreaterThan(0));
    await userEvent.click(screen.getByRole("radio", { name: /^On demand/ }));
    await userEvent.click(screen.getAllByRole("button", { name: "Deploy" })[0]);
    expect(deployBody(api)).toMatchObject({ mode: "on_demand", idle_timeout_s: null });
  });

  it("shows no LiteLLM hint when on demand can work", async () => {
    setup();
    await waitFor(() => expect(screen.getByRole("radio", { name: /^On demand/ })).toBeChecked());
    expect(screen.queryByText(/On demand needs LiteLLM/)).toBeNull();
  });

  it("hides the idle input for always on and sends no idle timeout", async () => {
    const api = setup();
    await waitFor(() => expect(screen.getAllByText("Fits").length).toBeGreaterThan(0));
    await userEvent.click(screen.getByRole("radio", { name: "Always on" }));
    expect(screen.queryByLabelText("Idle minutes before sleeping")).toBeNull();
    await userEvent.click(screen.getAllByRole("button", { name: "Deploy" })[0]);
    expect(deployBody(api)).toMatchObject({ mode: "always_on", idle_timeout_s: null });
  });

  it("blocks deploying with an idle timeout outside 1–1440 minutes", async () => {
    setup();
    await waitFor(() => expect(screen.getAllByText("Fits").length).toBeGreaterThan(0));
    const idle = screen.getByLabelText("Idle minutes before sleeping");
    fireEvent.change(idle, { target: { value: "1441" } });
    expect(idle).toHaveAttribute("aria-invalid", "true");
    expect(screen.getByText("Enter 1 to 1440 minutes.")).toBeInTheDocument();
    for (const b of screen.getAllByRole("button", { name: "Deploy" })) expect(b).toBeDisabled();
    fireEvent.change(idle, { target: { value: "0" } });
    for (const b of screen.getAllByRole("button", { name: "Deploy" })) expect(b).toBeDisabled();
  });
});
