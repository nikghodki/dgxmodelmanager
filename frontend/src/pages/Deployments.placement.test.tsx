import { afterEach, describe, expect, it, vi } from "vitest";
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { TooltipProvider } from "@/components/ui/tooltip";
import { Deployments } from "./Deployments";
import { DeploymentDetail } from "./DeploymentDetail";
import { Dashboard } from "./Dashboard";
import { Downloads } from "./Downloads";
import * as auth from "@/auth";
import * as apiModule from "@/api";

vi.mock("@/auth");
vi.mock("@/api", async () => {
  const actual = await vi.importActual<typeof import("@/api")>("@/api");
  return { ...actual, api: vi.fn() };
});

const GIB = 1024 ** 3;
const PLACEMENTS: Record<string, any> = {
  "dgx-1": { kind: "single", nodes: ["local"] }, "dgx-2": { kind: "single", nodes: ["dgx-2"] },
  "dgx-1 + dgx-2 (cluster)": { kind: "cluster", nodes: ["local", "dgx-2"] }, local: { kind: "single", nodes: ["local"] },
};
let nextId = 1;
const dep = (name: string, label: string, over: Record<string, any> = {}) => ({
  id: nextId++, name, engine: "vllm", model_ref: `org/${name}`, state: "ready", priority: 0, alias: null, model_name: name,
  reserved_bytes: GIB, actual_bytes: GIB, estimate: {}, config: {}, litellm_status: "synced", last_error: null, exit_code: null,
  container_id: null, served_name: name, endpoint: null, host_endpoint: null, sleep_capable: false, adopted_from: null,
  created_at: "", updated_at: "", events: [], placement: PLACEMENTS[label], placement_label: label, rank_summary: [], ...over,
});
const budget = (free: number, deployments: any[] = []) => ({ total: 121 * GIB, available: 100 * GIB, margin: 4 * GIB, os_reserve: 4 * GIB, usable: 113 * GIB, committed: 0, managed_actual: 0, external: 0, free_for_new: free, deployments });
const TWO_NODES = [
  { id: "local", label: "dgx-1", remote: false, reachable: true, stale: false, budget: budget(90 * GIB) },
  { id: "dgx-2", label: "dgx-2", remote: true, reachable: true, stale: false, budget: budget(70 * GIB) },
];

let deployments: any[] = [];
let nodes: any[] | null = null;  // null: GET /nodes never answers
let extra: (url: string, opts?: any) => any = () => undefined;
const mockDeployments = (d: any[]) => { deployments = d; };

function setup(ui: React.ReactNode, path = "/") {
  vi.mocked(auth.useIsAdmin).mockReturnValue(true);
  vi.mocked(apiModule.api).mockImplementation((url: string, opts?: any) => {
    const r = extra(url, opts);
    if (r !== undefined) return r;
    if (url === "/deployments") return Promise.resolve(deployments);
    const m = url.match(/^\/deployments\/(\d+)$/);
    if (m && !opts?.method) return Promise.resolve(deployments.find((d) => d.id === Number(m[1])));
    if (url === "/nodes") return nodes ? Promise.resolve(nodes) : new Promise(() => undefined);
    return new Promise(() => undefined);
  });
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={qc}><TooltipProvider>
      <MemoryRouter initialEntries={[path]} future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
        <Routes><Route path="/" element={ui} /><Route path="/deployments/:id" element={<DeploymentDetail />} /></Routes>
      </MemoryRouter>
    </TooltipProvider></QueryClientProvider>,
  );
}

afterEach(() => { vi.mocked(apiModule.api).mockReset(); deployments = []; nodes = null; extra = () => undefined; });

describe("where each deployment runs", () => {
  it("shows where each deployment runs", async () => {
    mockDeployments([dep("q", "dgx-2"), dep("ds", "dgx-1 + dgx-2 (cluster)")]); setup(<Deployments />);
    expect(await screen.findByText("dgx-1 + dgx-2 (cluster)")).toBeInTheDocument();
    expect(screen.getByText("dgx-2")).toBeInTheDocument();
    expect(screen.getByRole("columnheader", { name: /Runs on/ })).toBeInTheDocument();
  });

  it("labels local deployments too once a second node is configured", async () => {
    nodes = TWO_NODES;
    mockDeployments([dep("q", "dgx-1")]); setup(<Deployments />);
    expect(await screen.findByRole("columnheader", { name: /Runs on/ })).toBeInTheDocument();
    expect(screen.getByText("dgx-1")).toBeInTheDocument();
  });

  it("has no Runs on column on a single-node install", async () => {
    nodes = [{ id: "local", label: "local", remote: false, reachable: true, stale: false, budget: budget(90 * GIB) }];
    mockDeployments([dep("q", "local")]); setup(<Deployments />);
    await screen.findByRole("link", { name: "q" });
    await waitFor(() => expect(vi.mocked(apiModule.api).mock.calls.some((c) => c[0] === "/nodes")).toBe(true));
    expect(screen.queryByRole("columnheader", { name: /Runs on/ })).toBeNull();
  });

  it("names the cluster sleep after what it does, and a slept cluster offers Start", async () => {
    const user = userEvent.setup();
    mockDeployments([dep("ds", "dgx-1 + dgx-2 (cluster)", { sleep_label: "sleep (stops all nodes)" }),
                     dep("ds2", "dgx-1 + dgx-2 (cluster)", { state: "stopped", sleep_label: "sleep (stops all nodes)" })]);
    setup(<Deployments />);
    await user.click(await screen.findByRole("button", { name: "Actions for ds" }));
    expect(await screen.findByRole("menuitem", { name: "Sleep (stops all nodes)" })).toBeInTheDocument();
    await user.keyboard("{Escape}");
    await user.click(screen.getByRole("button", { name: "Actions for ds2" }));
    expect(await screen.findByRole("menuitem", { name: "Start" })).toBeInTheDocument();
    expect(screen.queryByRole("menuitem", { name: "Wake" })).toBeNull();
  });
});

describe("DeploymentDetail on a cluster", () => {
  it("lists each rank with its node and state, and the cluster sleep label", async () => {
    const d = dep("ds", "dgx-1 + dgx-2 (cluster)", {
      sleep_label: "sleep (stops all nodes)",
      rank_summary: [{ rank: 0, node: "local", node_label: "dgx-1", role: "head", status: "running" },
                     { rank: 1, node: "dgx-2", node_label: "dgx-2", role: "worker", status: "exited" }],
    });
    mockDeployments([d]); setup(<div />, `/deployments/${d.id}`);
    const ranks = await screen.findByRole("table", { name: "Ranks" });
    const rows = within(ranks).getAllByRole("row").slice(1);
    expect(rows.map((r) => r.textContent)).toEqual([expect.stringMatching(/0.*dgx-1.*head.*Running/i), expect.stringMatching(/1.*dgx-2.*worker.*Exited/i)]);
    expect(screen.getByRole("button", { name: "Sleep (stops all nodes)" })).toBeInTheDocument();
    expect(screen.getByText(/dgx-1 \+ dgx-2 \(cluster\)/)).toBeInTheDocument();
  });
});

describe("Dashboard with two nodes", () => {
  it("shows one memory bar per node and a cluster share on both", async () => {
    const ds = dep("ds", "dgx-1 + dgx-2 (cluster)");
    const share = (node: string) => ({ deployment_id: ds.id, name: "ds", state: "ready", reserved: 40 * GIB, actual: 0, container_id: null, node });
    nodes = [{ ...TWO_NODES[0], budget: budget(50 * GIB, [share("local")]) }, { ...TWO_NODES[1], budget: budget(30 * GIB, [share("dgx-2")]) }];
    mockDeployments([ds]);
    extra = (url) => (url === "/host" ? Promise.resolve({ budget: budget(50 * GIB, [share("local")]), external_top: [], gpu: {}, alert: { level: "ok" }, disk: null }) : undefined);
    setup(<Dashboard />);
    const bars = await screen.findAllByRole("img", { name: /Unified memory usage on/ });
    expect(bars.map((b) => b.getAttribute("aria-label"))).toEqual(["Unified memory usage on dgx-1", "Unified memory usage on dgx-2"]);
    for (const name of ["dgx-1", "dgx-2"]) {
      const region = screen.getByRole("region", { name: `Memory on ${name}` });
      expect(within(region).getByText("ds")).toBeInTheDocument();
    }
  });

  it("keeps the single memory bar on a single-node install", async () => {
    nodes = [{ id: "local", label: "local", remote: false, reachable: true, stale: false, budget: budget(90 * GIB) }];
    extra = (url) => (url === "/host" ? Promise.resolve({ budget: budget(90 * GIB), external_top: [], gpu: {}, alert: { level: "ok" }, disk: null }) : undefined);
    setup(<Dashboard />);
    expect(await screen.findByRole("img", { name: "Unified memory usage" })).toBeInTheDocument();
    await waitFor(() => expect(vi.mocked(apiModule.api).mock.calls.some((c) => c[0] === "/nodes")).toBe(true));
    expect(screen.queryByRole("img", { name: /Unified memory usage on/ })).toBeNull();
  });
});

describe("Downloads with weight copies", () => {
  it("labels a copy row with its target node", async () => {
    nodes = TWO_NODES;
    extra = (url) => (url === "/downloads" ? Promise.resolve([
      { id: 1, repo_id: "org/big", files: ["a"], bytes_done: 1, bytes_total: 2, state: "running", error: null, kind: "copy", node: "dgx-2", created_at: "" },
      { id: 2, repo_id: "org/small", files: ["a"], bytes_done: 2, bytes_total: 2, state: "done", error: null, kind: "download", node: null, created_at: "" },
    ]) : undefined);
    setup(<Downloads />);
    expect(await screen.findByText(/Copy to dgx-2/)).toBeInTheDocument();
    expect(screen.getAllByText(/Download/).length).toBeGreaterThan(0);
  });
});
