import { afterEach, describe, expect, it, vi } from "vitest";
import { act, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter } from "react-router-dom";
import { DecisionModel } from "./DecisionModel";
import * as auth from "@/auth";
import * as apiModule from "@/api";

vi.mock("@/auth");
vi.mock("@/api", async () => {
  const actual = await vi.importActual<typeof import("@/api")>("@/api");
  return { ...actual, api: vi.fn() };
});
vi.mock("sonner", () => ({ toast: { success: vi.fn(), error: vi.fn() } }));

const GIB = 1024 ** 3;
const features = { diagnosis: { enabled: true, threshold: 0.8 }, parser: { enabled: true, threshold: 0.9 }, plan: { enabled: true, threshold: 0.6 }, kind: { enabled: true, threshold: 0.8 }, workload: { enabled: true, threshold: 0.7 } };
const status = (over: Record<string, any> = {}) => ({
  enabled: true, healthy: true, deployment_id: 9, state: "ready", model: "vllm-sr/d3-mini", revision: "97dd054df4b3e0b3beb6af23ab8c82b7c0052e0e",
  p50_ms: 75.4, decisions_24h: 42, accepted_rate: 0.9, features, image: { present: true, building: false, error: null }, ...over,
});
const decider = { id: 9, name: "decider", engine: "decision", model_ref: "vllm-sr/d3-mini", state: "ready", reserved_bytes: 13 * GIB, actual_bytes: 12.5 * GIB, system: true };
const rows = [
  { id: 101, created_at: "2026-10-16T10:00:00Z", feature: "diagnosis", subject: "deployment:12", source: "model", choice: "gpu_oom", confidence: 0.93, latency_ms: 81, feedback: null },
  { id: 102, created_at: "2026-10-16T09:00:00Z", feature: "parser_tool", subject: "artifact:Qwen/Qwen3-8B", source: "rules", choice: "hermes", confidence: 1, latency_ms: null, feedback: "accepted" },
  { id: 103, created_at: "2026-10-16T08:00:00Z", feature: "plan", subject: "preview:abc", source: "none", choice: null, confidence: 0.41, latency_ms: 70, feedback: null },
];
const est = { total: 13 * GIB, weights: 9 * GIB, draft_weights: 0, kv_cache: 0, state: 0, activations: 0, runtime: 0, margin: 0, notes: [], confidence: "fixed" };
const shortFit = { fits: false, required: 13 * GIB, free_for_new: 5 * GIB, headroom: -8 * GIB, estimate: est };
const plans = [
  { id: "p1", title: "Stop chat", detail: "Frees 20 GiB", new_required: 13 * GIB, frees: 20 * GIB, fits: true, actions: [{ type: "stop", deployment_id: 3 }], patch: null },
  { id: "p2", title: "Shorter context", detail: "Halve max_model_len", new_required: 9 * GIB, frees: 0, fits: true, actions: [{ type: "patch", patch: { max_model_len: 2048 } }], patch: { max_model_len: 2048 } },
  { id: "p3", title: "Smaller model", detail: "Use a smaller model", new_required: 5 * GIB, frees: 0, fits: true, actions: [], patch: null, model_ref: "other/small" },
];

type Handler = (url: string, opts?: any) => any;
function setup({ st = status(), extra, admin = true }: { st?: any; extra?: Handler; admin?: boolean } = {}) {
  vi.mocked(auth.useIsAdmin).mockReturnValue(admin);
  vi.mocked(apiModule.api).mockImplementation(async (url: string, opts?: any) => {
    const r = extra?.(url, opts);
    if (r !== undefined) return r;
    if (url === "/decider/status") return st;
    if (url === "/deployments/9") return decider;
    if (url.startsWith("/decider/decisions?")) return rows;
    return new Promise(() => undefined);
  });
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } });
  const view = render(
    <QueryClientProvider client={qc}>
      <MemoryRouter future={{ v7_startTransition: true, v7_relativeSplatPath: true }}><DecisionModel /></MemoryRouter>
    </QueryClientProvider>,
  );
  return Object.assign(qc, { unmount: view.unmount });
}
const renderPage = setup;
const calls = (method: string, url: string) => vi.mocked(apiModule.api).mock.calls.filter(([u, o]) => u === url && (o as any)?.method === method);

afterEach(() => vi.mocked(apiModule.api).mockReset());

describe("Decision model page", () => {
  it("redirects viewers", () => {
    vi.mocked(auth.useIsAdmin).mockReturnValue(false);
    vi.mocked(apiModule.api).mockImplementation(() => new Promise(() => undefined));
    const { container } = render(<QueryClientProvider client={new QueryClient()}><MemoryRouter future={{ v7_startTransition: true, v7_relativeSplatPath: true }}><DecisionModel /></MemoryRouter></QueryClientProvider>);
    expect(container.textContent).toBe("");
  });

  it("shows the status: model, revision, health, memory, latency, volume and acceptance", async () => {
    setup();
    const card = (await screen.findByRole("heading", { name: "Status" })).closest("[data-card]") as HTMLElement;
    expect(within(card).getByText("vllm-sr/d3-mini")).toBeInTheDocument();
    expect(within(card).getByText("97dd054d")).toBeInTheDocument();
    expect(within(card).getByText("Healthy")).toBeInTheDocument();
    expect(await within(card).findByText("12.5 GiB / 13.0 GiB")).toBeInTheDocument();
    expect(within(card).getByText("75 ms")).toBeInTheDocument();
    expect(within(card).getByText("42")).toBeInTheDocument();
    expect(within(card).getByText("90%")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Disable" })).toBeInTheDocument();
  });

  it("saves per-feature toggles and thresholds (only what changed)", async () => {
    setup({ extra: (url, opts) => (url === "/decider/settings" && opts?.method === "PUT" ? { ok: true } : undefined) });
    await userEvent.click(await screen.findByRole("switch", { name: "Plan recommendation" }));
    const thr = screen.getByLabelText("Failure diagnosis threshold (%)");
    await userEvent.clear(thr);
    await userEvent.type(thr, "85");
    await userEvent.click(screen.getByRole("button", { name: "Save settings" }));
    await waitFor(() => expect(calls("PUT", "/decider/settings")).toHaveLength(1));
    expect((calls("PUT", "/decider/settings")[0][1] as any).body).toEqual({ features: { plan: { enabled: false }, diagnosis: { threshold: 0.85 } } });
  });

  it("clamps thresholds to 0–100 % and names the feature when a value is invalid", async () => {
    setup({ extra: (url, opts) => (url === "/decider/settings" && opts?.method === "PUT" ? { ok: true } : undefined) });
    const thr = await screen.findByLabelText("Failure diagnosis threshold (%)");
    await userEvent.clear(thr);
    await userEvent.type(thr, "150");
    await userEvent.tab();
    expect(thr).toHaveValue(100);
    await userEvent.clear(thr);
    expect(screen.getByText("Failure diagnosis: enter a percentage from 0 to 100.")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Save settings" })).toBeDisabled();
    await userEvent.type(thr, "5");
    await userEvent.click(screen.getByRole("button", { name: "Save settings" }));
    await waitFor(() => expect(calls("PUT", "/decider/settings")).toHaveLength(1));
    expect((calls("PUT", "/decider/settings")[0][1] as any).body).toEqual({ features: { diagnosis: { threshold: 0.05 } } });
  });

  it("enables via a plan when the decider does not fit", async () => {
    let n = 0;
    setup({ st: status({ enabled: false, healthy: false, deployment_id: null, state: null }), extra: (url, opts) => {
      if (url === "/decider/enable" && opts?.method === "POST") {
        n++;
        if (n === 1) throw new apiModule.ApiError(409, { code: "insufficient_memory", message: "Not enough memory", fit: shortFit, plans });
        return { deployment_id: 9 };
      }
      return undefined;
    } });
    await userEvent.click(await screen.findByRole("button", { name: "Enable" }));
    const dialog = await screen.findByRole("dialog");
    expect(within(dialog).getByText("Does not fit")).toBeInTheDocument();
    // plans that patch the config or swap the model cannot apply to the decider
    expect(within(dialog).getAllByRole("radio")).toHaveLength(1);
    expect(within(dialog).queryByText("Shorter context")).toBeNull();
    expect(within(dialog).queryByText("Smaller model")).toBeNull();
    await userEvent.click(within(dialog).getByRole("radio", { name: /Stop chat/ }));
    await userEvent.click(within(dialog).getByRole("button", { name: "Apply plan and enable" }));
    await waitFor(() => expect(calls("POST", "/decider/enable")).toHaveLength(2));
    expect((calls("POST", "/decider/enable")[1][1] as any).body).toEqual({ actions: [{ type: "stop", deployment_id: 3 }] });
    await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());
  });

  it("offers to build the image when it is missing, with live progress", async () => {
    const qc = setup({ st: status({ enabled: false, healthy: false, deployment_id: null, state: null, image: { present: false, building: false, error: null } }), extra: (url, opts) => {
      if (url === "/decider/enable" && opts?.method === "POST") throw new apiModule.ApiError(409, { code: "image_missing", message: "image dgxmm-decider:0.1 is not built", build_url: "/api/decider/image/build" });
      if (url === "/decider/image/build" && opts?.method === "POST") return {};
      return undefined;
    } });
    await userEvent.click(await screen.findByRole("button", { name: "Enable" }));
    const dialog = await screen.findByRole("dialog");
    await userEvent.click(within(dialog).getByRole("button", { name: "Build image" }));
    await waitFor(() => expect(calls("POST", "/decider/image/build")).toHaveLength(1));
    act(() => { qc.setQueryData(["decider-build"], { phase: "building", message: "Step 2/5 : RUN pip install" }); });
    expect(await within(dialog).findByText("Step 2/5 : RUN pip install")).toBeInTheDocument();
    act(() => { qc.setQueryData(["decider-build"], { phase: "done", message: "Successfully built" }); });
    expect(await within(dialog).findByRole("button", { name: "Enable now" })).toBeInTheDocument();
  });

  it("starts a fresh build on Retry: no stale failure from the last attempt", async () => {
    const qc = setup({ st: status({ enabled: false, healthy: false, deployment_id: null, state: null, image: { present: false, building: false, error: null } }), extra: (url, opts) => {
      if (url === "/decider/enable" && opts?.method === "POST") throw new apiModule.ApiError(409, { code: "image_missing", message: "not built" });
      if (url === "/decider/image/build" && opts?.method === "POST") return {};
      return undefined;
    } });
    await userEvent.click(await screen.findByRole("button", { name: "Enable" }));
    const dialog = await screen.findByRole("dialog");
    await userEvent.click(within(dialog).getByRole("button", { name: "Build image" }));
    act(() => { qc.setQueryData(["decider-build"], { phase: "error", message: "pull failed" }); });
    await userEvent.click(await within(dialog).findByRole("button", { name: "Retry build" }));
    await waitFor(() => expect(calls("POST", "/decider/image/build")).toHaveLength(2));
    expect(qc.getQueryData(["decider-build"])).toBeNull();
    expect(within(dialog).queryByText(/pull failed/)).toBeNull();
    expect(within(dialog).queryByRole("button", { name: "Enable now" })).toBeNull();
    expect(within(dialog).getByText(/Building/)).toBeInTheDocument();
  });

  it("swaps the image dialog for the plan dialog when enabling then needs memory", async () => {
    const qc = setup({ st: status({ enabled: false, healthy: false, deployment_id: null, state: null, image: { present: false, building: false, error: null } }), extra: (url, opts) => {
      if (url === "/decider/enable" && opts?.method === "POST") {
        if (calls("POST", "/decider/enable").length === 1) throw new apiModule.ApiError(409, { code: "image_missing", message: "not built" });
        throw new apiModule.ApiError(409, { code: "insufficient_memory", message: "Not enough memory", fit: shortFit, plans });
      }
      if (url === "/decider/image/build" && opts?.method === "POST") return {};
      return undefined;
    } });
    await userEvent.click(await screen.findByRole("button", { name: "Enable" }));
    const dialog = await screen.findByRole("dialog");
    await userEvent.click(within(dialog).getByRole("button", { name: "Build image" }));
    act(() => { qc.setQueryData(["decider-build"], { phase: "done", message: "built" }); });
    await userEvent.click(await within(dialog).findByRole("button", { name: "Enable now" }));
    await waitFor(() => expect(screen.getByText("Make room for the decision model")).toBeInTheDocument());
    expect(screen.getAllByRole("dialog")).toHaveLength(1);
    expect(screen.queryByText("Build the decision model image")).toBeNull();
  });

  it("offers Start when enabled but the decider deployment is stopped", async () => {
    setup({ st: status({ healthy: false, state: "stopped" }), extra: (url, opts) => (url === "/decider/enable" && opts?.method === "POST" ? { deployment_id: 9 } : undefined) });
    await userEvent.click(await screen.findByRole("button", { name: "Start" }));
    await waitFor(() => expect(calls("POST", "/decider/enable")).toHaveLength(1));
    expect(screen.getByRole("button", { name: "Disable" })).toBeInTheDocument();
  });

  it("rotates the API key only after confirmation and reports the restart", async () => {
    const { toast } = await import("sonner");
    setup({ extra: (url, opts) => (url === "/decider/rotate-key" && opts?.method === "POST"
      ? { rotated: true, restarted: true, deployment_id: 9, state: "starting", error: null } : undefined) });
    await userEvent.click(await screen.findByRole("button", { name: "Rotate key" }));
    expect(calls("POST", "/decider/rotate-key")).toHaveLength(0);
    const dialog = await screen.findByRole("alertdialog");
    expect(within(dialog).getByText(/restarted with it/)).toBeInTheDocument();
    await userEvent.click(within(dialog).getByRole("button", { name: "Rotate" }));
    await waitFor(() => expect(calls("POST", "/decider/rotate-key")).toHaveLength(1));
    await waitFor(() => expect(toast.success).toHaveBeenCalledWith("API key rotated — the decision model is restarting with it"));
  });

  it("reports a rotation whose restart failed", async () => {
    const { toast } = await import("sonner");
    setup({ extra: (url, opts) => (url === "/decider/rotate-key" && opts?.method === "POST"
      ? { rotated: true, restarted: false, deployment_id: 9, state: "stopped", error: "insufficient_memory" } : undefined) });
    await userEvent.click(await screen.findByRole("button", { name: "Rotate key" }));
    await userEvent.click(within(await screen.findByRole("alertdialog")).getByRole("button", { name: "Rotate" }));
    await waitFor(() => expect(toast.error).toHaveBeenCalledWith("API key rotated, but the decision model did not restart (insufficient_memory)"));
  });

  it("warns when the running decider predates the current hardening and offers the restart", async () => {
    setup({ st: status({ upgrade: { needed: true, reasons: ["runs image dgxmm-decider:poc, expected dgxmm-decider:3", "runs as root"] } }) });
    const alert = await screen.findByRole("alert", { name: "Decision model needs a restart" });
    expect(within(alert).getByText(/dgxmm-decider:poc/)).toBeInTheDocument();
    expect(within(alert).getByText(/runs as root/)).toBeInTheDocument();
    await userEvent.click(within(alert).getByRole("button", { name: "Restart with a new key" }));
    expect(await screen.findByRole("alertdialog")).toBeInTheDocument();
  });

  it("asks for the image build first when the new image is missing", async () => {
    setup({ st: status({ upgrade: { needed: true, reasons: ["runs image dgxmm-decider:poc, expected dgxmm-decider:3"] }, image: { present: false, building: false, error: null } }) });
    const alert = await screen.findByRole("alert", { name: "Decision model needs a restart" });
    expect(within(alert).queryByRole("button", { name: "Restart with a new key" })).toBeNull();
    expect(within(alert).getByRole("button", { name: "Build image…" })).toBeInTheDocument();
  });

  it("shows no upgrade warning for an up-to-date decider", async () => {
    setup({ st: status({ upgrade: { needed: false, reasons: [] } }) });
    await screen.findByRole("heading", { name: "Status" });
    expect(screen.queryByRole("alert", { name: "Decision model needs a restart" })).toBeNull();
  });

  it("links the docs to the decision model section of the usage guide", async () => {
    setup();
    expect(await screen.findByRole("link", { name: "Docs" })).toHaveAttribute("href", "https://github.com/nikghodki/dgxmodelmanager/blob/main/docs/usage.md#decision-model");
  });

  it("disables after confirmation", async () => {
    setup({ extra: (url, opts) => (url === "/decider/disable" && opts?.method === "POST" ? { ok: true } : undefined) });
    await userEvent.click(await screen.findByRole("button", { name: "Disable" }));
    const dialog = await screen.findByRole("alertdialog");
    await userEvent.click(within(dialog).getByRole("button", { name: "Disable" }));
    await waitFor(() => expect(calls("POST", "/decider/disable")).toHaveLength(1));
  });

  it("lists recent decisions and records feedback", async () => {
    setup({ extra: (url, opts) => (url === "/decider/decisions/101/feedback" && opts?.method === "POST" ? { ok: true } : undefined) });
    const row = (await screen.findByText("gpu_oom")).closest("tr")!;
    expect(within(row).getByText("Failure diagnosis")).toBeInTheDocument();
    expect(within(row).getByText("93%")).toBeInTheDocument();
    expect(within(row).getByText("AI")).toBeInTheDocument();
    expect(within(row).getByText("81 ms")).toBeInTheDocument();
    expect(within(row).getByRole("link", { name: "deployment:12" })).toHaveAttribute("href", "/deployments/12");
    await userEvent.click(within(row).getByRole("button", { name: "Helpful" }));
    await waitFor(() => expect((calls("POST", "/decider/decisions/101/feedback")[0]?.[1] as any)?.body).toEqual({ accepted: true }));
    const other = screen.getByText("hermes").closest("tr")!;
    expect(within(other).getByRole("button", { name: "Helpful" })).toHaveAttribute("aria-pressed", "true");
    // an abstention has nothing to judge
    const abstained = screen.getByText("Abstained").closest("tr")!;
    expect(within(abstained).queryByRole("button", { name: "Helpful" })).toBeNull();
    expect(within(abstained).getByText("None").closest("[data-decision-source]")!.className).toMatch(/border-dashed/);
  });

  const models = [
    { repo: "vllm-sr/d3-mini", revision: "97dd054df4b3e0b3beb6af23ab8c82b7c0052e0e", served_name: "d3-mini", params: "4.5B", reservation_bytes: 13 * GIB, default: true, downloaded: true },
    { repo: "vllm-sr/d3-nano", revision: "aeb70d4dd53ed25b855c3b0a66394b8fe265db7f", served_name: "d3-nano", params: "2.2B", reservation_bytes: 8 * GIB, default: false, downloaded: false },
  ];
  const lifecycle = {
    configured_model: "vllm-sr/d3-mini", restart_required: false, models,
    breaker: { open: false, failures: 0, retry_in_s: 0 }, load: { max_in_flight: 2, in_flight: 0, queued: 0, shed: 3, service_ms: 140 },
    supervisor: { last_action: "probe_ok", last_reason: null, probe: { ok: true, latency_ms: 40, at: 0 }, failed_probes: 0, restarts_last_hour: 0, max_restarts_per_hour: 5, backoff_s: 30, gave_up: false, next_attempt_in_s: null },
  };

  it("shows the circuit breaker, back-pressure and auto-restart state", async () => {
    setup({ st: status({ ...lifecycle, healthy: false, breaker: { open: true, failures: 3, retry_in_s: 42 },
      supervisor: { ...lifecycle.supervisor, last_action: "gave_up", gave_up: true, restarts_last_hour: 5, last_reason: "crashed: container exited with code 1" } }) });
    const card = (await screen.findByRole("heading", { name: "Status" })).closest("[data-card]") as HTMLElement;
    expect(within(card).getByText(/Open — answering by rules for 42 s/)).toBeInTheDocument();
    expect(within(card).getByText(/3 answered by rules under load/)).toBeInTheDocument();
    expect(within(card).getByText("Gave up — crashed: container exited with code 1")).toBeInTheDocument();
  });

  it("names why auto-restart gave up and retries by enabling again", async () => {
    setup({ st: status({ ...lifecycle, healthy: false, state: "failed", supervisor: { ...lifecycle.supervisor, last_action: "gave_up", gave_up: true, restarts_last_hour: 5,
      last_reason: "crashed: exited; restart failed: could not start container: image dgxmm-decider:3 not found" } }),
      extra: (url, opts) => (url === "/decider/enable" && opts?.method === "POST" ? { deployment_id: 9, state: "starting" } : undefined) });
    const card = (await screen.findByRole("heading", { name: "Status" })).closest("[data-card]") as HTMLElement;
    expect(within(card).getByText("Gave up — crashed: exited; restart failed: could not start container: image dgxmm-decider:3 not found")).toBeInTheDocument();
    await userEvent.click(within(card).getByRole("button", { name: "Retry" }));
    await waitFor(() => expect(calls("POST", "/decider/enable")).toHaveLength(1));
  });

  it("shows the progress of a decision-model download", async () => {
    const rows = [{ id: 7, repo_id: "vllm-sr/d3-nano", state: "running", bytes_done: 2 * GIB, bytes_total: 5 * GIB, error: null }];
    setup({ st: status(lifecycle), extra: (url) => (url === "/downloads" ? rows : undefined) });
    const card = (await screen.findByRole("heading", { name: "Model" })).closest("[data-card]") as HTMLElement;
    expect(await within(card).findByRole("link", { name: "Downloading d3-nano: 40%" })).toHaveAttribute("href", "/downloads");
    expect(within(card).queryByRole("button", { name: "Download d3-nano" })).toBeNull();
  });

  it("shows why a decision-model download failed and offers to retry it", async () => {
    const rows = [{ id: 7, repo_id: "vllm-sr/d3-nano", state: "failed", bytes_done: 2 * GIB, bytes_total: 5 * GIB, error: "model.safetensors: SHA-256 differs" }];
    setup({ st: status(lifecycle), extra: (url, opts) => (url === "/downloads" ? rows : url === "/decider/models/download" && opts?.method === "POST" ? { id: 8, state: "queued" } : undefined) });
    const card = (await screen.findByRole("heading", { name: "Model" })).closest("[data-card]") as HTMLElement;
    expect(await within(card).findByText("model.safetensors: SHA-256 differs")).toBeInTheDocument();
    await userEvent.click(within(card).getByRole("button", { name: "Download d3-nano" }));
    await waitFor(() => expect(calls("POST", "/decider/models/download")).toHaveLength(1));
  });

  it("says how long the selected model usually takes to start", async () => {
    const { toast } = await import("sonner");
    for (const [model, hint] of [["vllm-sr/d3-mini", "usually ready in 2–4 minutes"], ["vllm-sr/d3-nano", "usually ready in about 90 seconds"]]) {
      vi.mocked(toast.success).mockClear();
      const { unmount } = renderPage({ st: status({ enabled: false, healthy: false, deployment_id: null, state: null, configured_model: model }),
        extra: (url, opts) => (url === "/decider/enable" && opts?.method === "POST" ? { deployment_id: 9 } : undefined) });
      await userEvent.click(await screen.findByRole("button", { name: "Enable" }));
      await waitFor(() => expect(toast.success).toHaveBeenCalledWith(`Starting the decision model — ${hint}`));
      unmount();
    }
  });

  it("says auto-restart is paused while the API key is being rotated", async () => {
    setup({ st: status({ ...lifecycle, supervisor: { ...lifecycle.supervisor, last_action: "rotating", last_reason: "the API key is being rotated; the rotation restarts the decider" } }) });
    const card = (await screen.findByRole("heading", { name: "Status" })).closest("[data-card]") as HTMLElement;
    expect(within(card).getByText("Paused while the API key is rotated")).toBeInTheDocument();
  });

  it("selects the decision model and switches a running decider to it", async () => {
    const onDisk = { ...lifecycle, models: models.map((m) => ({ ...m, downloaded: true })) };
    let st = status(onDisk);
    setup({ extra: (url, opts) => {
      if (url === "/decider/status") return st;
      if (url === "/decider/settings" && opts?.method === "PUT") { st = status({ ...onDisk, configured_model: "vllm-sr/d3-nano", restart_required: true }); return st; }
      if (url === "/decider/enable" && opts?.method === "POST") return { deployment_id: 9 };
      return undefined;
    } });
    const card = (await screen.findByRole("heading", { name: "Model" })).closest("[data-card]") as HTMLElement;
    expect(within(card).getByText(/13.0 GiB/)).toBeInTheDocument();
    await userEvent.click(within(card).getByRole("radio", { name: /d3-nano/ }));
    await userEvent.click(within(card).getByRole("button", { name: "Save model" }));
    await waitFor(() => expect(calls("PUT", "/decider/settings")).toHaveLength(1));
    expect((calls("PUT", "/decider/settings")[0][1] as any).body).toEqual({ model: "vllm-sr/d3-nano" });
    await userEvent.click(await within(card).findByRole("button", { name: /Restart with d3-nano/ }));
    await waitFor(() => expect(calls("POST", "/decider/enable")).toHaveLength(1));
  });

  it("offers to download a model that is not on disk", async () => {
    setup({ st: status(lifecycle), extra: (url, opts) => (url === "/decider/models/download" && opts?.method === "POST" ? { id: 4, state: "queued" } : undefined) });
    const card = (await screen.findByRole("heading", { name: "Model" })).closest("[data-card]") as HTMLElement;
    await userEvent.click(within(card).getByRole("button", { name: "Download d3-nano" }));
    await waitFor(() => expect(calls("POST", "/decider/models/download")).toHaveLength(1));
    expect((calls("POST", "/decider/models/download")[0][1] as any).body).toEqual({ model: "vllm-sr/d3-nano" });
  });

  it("shows a pending restart and a model rollback", async () => {
    setup({ st: status({ ...lifecycle, supervisor: { ...lifecycle.supervisor, last_action: "memory_pressure", pending_restart: "not answering health probes (3 in a row)", next_attempt_in_s: 60, last_reason: "not enough memory to restart" } }) });
    const card = (await screen.findByRole("heading", { name: "Status" })).closest("[data-card]") as HTMLElement;
    expect(within(card).getByText(/Restart pending \(next try in 60 s\)/)).toBeInTheDocument();
  });

  it("explains a rollback after a failed model switch", async () => {
    setup({ st: status({ ...lifecycle, supervisor: { ...lifecycle.supervisor, last_action: "rolled_back", last_reason: "vllm-sr/d3-nano failed to start (exit 3); rolled back to vllm-sr/d3-mini" } }) });
    const card = (await screen.findByRole("heading", { name: "Status" })).closest("[data-card]") as HTMLElement;
    expect(within(card).getByText(/Rolled back to the previous model/)).toBeInTheDocument();
    expect(within(card).getByText(/rolled back to vllm-sr\/d3-mini/)).toBeInTheDocument();
  });
});
