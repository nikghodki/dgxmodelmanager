import { afterEach, describe, expect, it, vi } from "vitest";
import { render, screen, waitFor, within } from "@testing-library/react";
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

const GIB = 1024 ** 3;
const est = { total: 160 * GIB, weights: 150 * GIB, draft_weights: 0, kv_cache: 0, state: 0, activations: 0, runtime: 0, margin: 0, notes: [], confidence: "measured" };
const fitsPreview = { estimate: est, fit: { fits: true, free_for_new: 200 * GIB, headroom: 1e9 }, plans: [] };
const BIG = "deepseek-ai/DeepSeek-V4-Flash-0731";
const artifactFor = (ref: string) => ({ ref, complete: true, kind: "llm", engines: ["vllm"], quant: "FP8", weights_bytes: 150 * GIB, architecture: "DeepseekV4", max_context: 32768, problems: [] });
const engines = [{ name: "vllm", supports_sleep: true, defaults: { max_model_len: 4096 }, image: null }];
const budget = (free: number) => ({ total: 121 * GIB, available: 100 * GIB, margin: 4 * GIB, os_reserve: 4 * GIB, usable: 113 * GIB, committed: 0, managed_actual: 0, external: 0, free_for_new: free, deployments: [] });
const TWO_NODES = [
  { id: "local", label: "dgx-1", remote: false, reachable: true, stale: false, budget: budget(100 * GIB) },
  { id: "dgx-2", label: "dgx-2", remote: true, reachable: true, stale: false, budget: budget(100 * GIB) },
];
const LABELS: Record<string, string> = { local: "dgx-1", "dgx-2": "dgx-2" };

function opt(kind: "single" | "cluster", nodes: string[], over: Record<string, any> = {}) {
  const label = kind === "single" ? LABELS[nodes[0]] : `${nodes.map((n) => LABELS[n]).join(" + ")} (cluster)`;
  const per = kind === "single" ? 160 * GIB : 80 * GIB;
  return {
    placement: { kind, nodes }, label, per_node: Object.fromEntries(nodes.map((n) => [n, per])),
    free: Object.fromEntries(nodes.map((n) => [n, 100 * GIB])), fits: true, recommended: false, reason: "fits", ...over,
  };
}

let options: any[] = [];
let nodes: any[] = TWO_NODES;
let deployResult: () => Promise<any> = () => new Promise(() => undefined);
const mockPlacementOptions = (o: any[]) => { options = o; };
const mockNodes = (n: any[]) => { nodes = n; };
const api = () => vi.mocked(apiModule.api);
const lastPost = (url: string) => {
  const calls = api().mock.calls.filter((c) => c[0] === url && (c[1] as any)?.method === "POST");
  return (calls[calls.length - 1]![1] as any).body;
};

function setup(ref = BIG) {
  vi.mocked(auth.useIsAdmin).mockReturnValue(true);
  api().mockImplementation((url: string, opts?: any) => {
    if (url === "/inventory") return Promise.resolve([artifactFor(ref)]);
    if (url === "/engines") return Promise.resolve(engines);
    if (url === "/nodes") return Promise.resolve(nodes);
    if (url === "/deployments/placement-options") return Promise.resolve(options);
    if (url === "/deployments/preview") return Promise.resolve(fitsPreview);
    if (url === "/deployments" && opts?.method === "POST") return deployResult();
    return new Promise(() => undefined);
  });
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } });
  return render(
    <QueryClientProvider client={qc}>
      <MemoryRouter initialEntries={[`/deploy?model=${encodeURIComponent(ref)}`]} future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
        <DeployWizard />
      </MemoryRouter>
    </QueryClientProvider>,
  );
}
const placementRadios = async () => within(await screen.findByRole("radiogroup", { name: "Where to run" })).getAllByRole("radio");
const pickModel = async (ref: string) => {
  await screen.findByRole("combobox", { name: new RegExp(`^Model\\s+${ref.replace(/[.*+?^${}()|[\]\\/]/g, "\\$&")}$`) });
};

afterEach(() => { api().mockReset(); options = []; nodes = TWO_NODES; deployResult = () => new Promise(() => undefined); });

describe("DeployWizard: where to run", () => {
  it("pre-selects the cluster for a model too big for one DGX", async () => {
    const user = userEvent.setup();
    mockPlacementOptions([opt("single", ["local"], { fits: false, reason: "too_big_for_one_node" }),
                          opt("single", ["dgx-2"], { fits: false, reason: "too_big_for_one_node" }),
                          opt("cluster", ["local", "dgx-2"], { fits: true, recommended: true })]);
    setup(); await pickModel(BIG);
    expect(await screen.findByRole("radio", { name: /dgx-1 \+ dgx-2/ })).toBeChecked();
    expect(screen.getAllByText("Too big for one DGX")).toHaveLength(2);
    expect(screen.getByText("Recommended")).toBeInTheDocument();
    await waitFor(() => expect(lastPost("/deployments/preview").placement).toEqual({ kind: "cluster", nodes: ["local", "dgx-2"] }));
    await waitFor(() => expect(screen.getAllByRole("button", { name: "Deploy" })[0]).toBeEnabled());
    await user.click(screen.getAllByRole("button", { name: "Deploy" })[0]);
    expect(lastPost("/deployments").placement).toEqual({ kind: "cluster", nodes: ["local", "dgx-2"] });
  });

  it("shows the memory each node needs against what it has free", async () => {
    mockPlacementOptions([opt("single", ["local"], { recommended: true }), opt("single", ["dgx-2"]), opt("cluster", ["local", "dgx-2"])]);
    setup(); await pickModel(BIG);
    const cluster = await screen.findByRole("radio", { name: /dgx-1 \+ dgx-2/ });
    expect(cluster.textContent).toMatch(/dgx-1\D*80(\.0)? GiB\D*100(\.0)? GiB free/);
    expect(cluster.textContent).toMatch(/dgx-2\D*80(\.0)? GiB\D*100(\.0)? GiB free/);
    expect(screen.getByRole("radio", { name: /^dgx-1(?! \+)/ })).toBeChecked();
  });

  it("the user can choose another placement and it is sent", async () => {
    const user = userEvent.setup();
    mockPlacementOptions([opt("single", ["local"], { recommended: true }), opt("single", ["dgx-2"]), opt("cluster", ["local", "dgx-2"])]);
    setup(); await pickModel(BIG);
    const two = (await placementRadios()).find((r) => /^dgx-2/.test(r.textContent ?? ""))!;
    await user.click(two);
    expect(two).toBeChecked();
    await waitFor(() => expect(lastPost("/deployments/preview").placement).toEqual({ kind: "single", nodes: ["dgx-2"] }));
    await waitFor(() => expect(screen.getAllByRole("button", { name: "Deploy" })[0]).toBeEnabled());
    await user.click(screen.getAllByRole("button", { name: "Deploy" })[0]);
    expect(lastPost("/deployments").placement).toEqual({ kind: "single", nodes: ["dgx-2"] });
  });

  it("with nothing recommended shows the shortfall and keeps Deploy disabled", async () => {
    mockPlacementOptions([opt("single", ["local"], { fits: false, reason: "too_big_for_one_node", free: { local: 50 * GIB } }),
                          opt("single", ["dgx-2"], { fits: false, reason: "node_stale", free: { "dgx-2": 0 } }),
                          opt("cluster", ["local", "dgx-2"], { fits: false, reason: "node_stale" })]);
    setup(); await pickModel(BIG);
    await screen.findByText("Nothing fits right now");
    expect((await placementRadios()).filter((r) => r.getAttribute("aria-checked") === "true")).toEqual([]);
    expect(screen.getAllByText("Node unavailable")).toHaveLength(2);
    expect(screen.getByRole("radio", { name: /^dgx-1(?! \+)/ }).textContent).toMatch(/short by 110(\.0)? GiB/);
    expect(screen.getAllByRole("button", { name: "Choose where to run" })[0]).toBeDisabled();
  });

  it("disables Deploy for a placement that does not fit", async () => {
    const user = userEvent.setup();
    mockPlacementOptions([opt("single", ["local"], { recommended: true }), opt("single", ["dgx-2"], { fits: false, reason: "insufficient_memory" })]);
    setup(); await pickModel(BIG);
    expect(await screen.findByText("Does not fit")).toBeInTheDocument();
    const two = (await placementRadios()).find((r) => /^dgx-2/.test(r.textContent ?? ""))!;
    await user.click(two);
    await waitFor(() => expect(screen.getAllByRole("button", { name: "Choose a plan to fit" })[0]).toBeDisabled());
  });

  it("hides the step with a single node", async () => {
    mockNodes([{ id: "local", label: "dgx-1" }]); setup("org/model-a"); await pickModel("org/model-a");
    await waitFor(() => expect(api().mock.calls.some((c) => c[0] === "/deployments/preview")).toBe(true));
    expect(screen.queryByText("Where to run")).toBeNull();
    expect(api().mock.calls.some((c) => c[0] === "/deployments/placement-options")).toBe(false);
    expect("placement" in lastPost("/deployments/preview")).toBe(false);
  });

  it("offers to copy missing weights to the node and links to Downloads", async () => {
    const user = userEvent.setup();
    mockPlacementOptions([opt("single", ["local"]), opt("single", ["dgx-2"], { recommended: true })]);
    deployResult = () => Promise.reject(new apiModule.ApiError(409, {
      code: "weights_missing", message: "not in the HF cache of node dgx-2",
      action: { type: "copy_weights", node: "dgx-2", ref: BIG, revision: "abc123" },
    }));
    setup(); await pickModel(BIG);
    await waitFor(() => expect(screen.getAllByRole("button", { name: "Deploy" })[0]).toBeEnabled());
    await user.click(screen.getAllByRole("button", { name: "Deploy" })[0]);
    const copy = (await screen.findAllByRole("button", { name: "Copy weights to dgx-2" }))[0];
    api().mockImplementationOnce(() => Promise.resolve({ id: 5, kind: "copy", node: "dgx-2", state: "queued" }));
    await user.click(copy);
    await waitFor(() => expect(api()).toHaveBeenCalledWith("/nodes/dgx-2/copy-weights", { method: "POST", body: { ref: BIG, revision: "abc123" } }));
    expect((await screen.findAllByRole("link", { name: /Downloads/ }))[0]).toHaveAttribute("href", "/downloads");
  });
});
