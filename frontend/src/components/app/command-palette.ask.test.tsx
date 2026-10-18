import { afterEach, describe, expect, it, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter, Route, Routes, useLocation } from "react-router-dom";
import * as apiModule from "@/api";
import type { CommandInterpretation, CommandResult } from "@/types";
import { CommandPalette } from "./command-palette";

let isAdminFlag = true;
vi.mock("@/auth", () => ({
  useAuth: () => ({ user: { id: 1, username: "u", role: isAdminFlag ? "admin" : "viewer", must_change_pw: false }, loading: false, login: vi.fn(), logout: vi.fn(), reload: vi.fn() }),
  useIsAdmin: () => isAdminFlag,
}));
vi.mock("@/api", async () => {
  const actual = await vi.importActual<typeof import("@/api")>("@/api");
  return { ...actual, api: vi.fn() };
});
vi.mock("sonner", () => ({ toast: { success: vi.fn(), error: vi.fn() } }));

const DEPS = [{ id: 1, name: "qwen", state: "ready", sleep_capable: true, model_ref: "Qwen/Qwen3-8B" }];
const interp = (over: Partial<CommandInterpretation>): CommandInterpretation => ({
  action: "stop", target: { kind: "deployment", name: "qwen", id: 1, state: "ready", engine: "vllm" }, params: {},
  confidence: 0.92, label: "Stop qwen", source: "model", ...over,
});

function Where() { return <div data-testid="where">{useLocation().pathname + useLocation().search}</div>; }

function mount(result: CommandResult | ((text: string) => CommandResult), extra: (url: string, o?: any) => any = () => undefined) {
  vi.mocked(apiModule.api).mockImplementation(async (url: string, o?: any) => {
    if (url === "/decider/command") return typeof result === "function" ? result(o.body.text) : result;
    if (url === "/deployments") return DEPS;
    if (url === "/inventory") return [];
    const r = extra(url, o);
    return r !== undefined ? r : {};
  });
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  qc.setQueryData(["deployments"], DEPS);
  qc.setQueryData(["inventory"], []);
  render(
    <QueryClientProvider client={qc}>
      <MemoryRouter initialEntries={["/"]} future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
        <Routes><Route path="*" element={<><CommandPalette open onOpenChange={() => undefined} /><Where /></>} /></Routes>
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

const writes = () => vi.mocked(apiModule.api).mock.calls.filter(([u, o]) => (o as any)?.method === "POST" && u !== "/decider/command");
const input = () => screen.getByPlaceholderText(/type a command/);

afterEach(() => { vi.mocked(apiModule.api).mockReset(); isAdminFlag = true; });

describe("⌘K Ask mode", () => {
  it("offers Interpret only when no entry matches, and lists the interpretations with confidence", async () => {
    mount({ interpretations: [interp({}), interp({ action: "sleep", label: "Sleep qwen", confidence: 0.4 })], source: "model", message: null, refused: false });
    await userEvent.type(input(), "planner");
    expect(screen.queryByText(/Interpret:/)).toBeNull();
    await userEvent.clear(input());
    await userEvent.type(input(), "turn qwen off for tonight");
    expect(await screen.findByRole("option", { name: /Interpret: turn qwen off for tonight/ })).toBeInTheDocument();
    await userEvent.keyboard("{Enter}");
    expect(await screen.findByRole("option", { name: "Stop qwen, confidence 92%" })).toBeInTheDocument();
    expect(screen.getByRole("option", { name: "Sleep qwen, confidence 40%" })).toBeInTheDocument();
    expect(vi.mocked(apiModule.api)).toHaveBeenCalledWith("/decider/command", { method: "POST", body: { text: "turn qwen off for tonight" } });
    expect(document.querySelector("[aria-live=polite]")).toHaveTextContent("2 interpretations: Stop qwen, 92%; Sleep qwen, 40%");
    expect(writes()).toEqual([]); // nothing runs on interpretation
  });

  it("choosing an interpretation opens the normal confirmation; only confirming runs it", async () => {
    mount({ interpretations: [interp({})], source: "model", message: null, refused: false });
    await userEvent.type(input(), "turn qwen off");
    await userEvent.keyboard("{Enter}");
    await userEvent.click(await screen.findByRole("option", { name: /Stop qwen/ }));
    expect(await screen.findByLabelText("Type qwen to confirm")).toBeInTheDocument();
    expect(writes()).toEqual([]);
    await userEvent.type(screen.getByLabelText("Type qwen to confirm"), "qwen");
    await userEvent.click(screen.getByRole("button", { name: "Stop" }));
    await waitFor(() => expect(writes()).toEqual([["/deployments/1/stop", { method: "POST" }]]));
  });

  it("cancelling the confirmation runs nothing", async () => {
    mount({ interpretations: [interp({ action: "wake", label: "Wake qwen" })], source: "rules", message: null, refused: false });
    await userEvent.type(input(), "bring qwen back");
    await userEvent.keyboard("{Enter}");
    await userEvent.click(await screen.findByRole("option", { name: /Wake qwen/ }));
    await userEvent.click(await screen.findByRole("button", { name: "Cancel" }));
    expect(writes()).toEqual([]);
  });

  it("keyboard: arrows move between interpretations, Enter picks", async () => {
    mount({ interpretations: [interp({ confidence: 0.5 }), interp({ action: "navigate", label: "Open deployment qwen", target: { kind: "deployment", name: "qwen", id: 1 }, confidence: 0.4 })], source: "model", message: null, refused: false });
    await userEvent.type(input(), "qwen thing");
    await userEvent.keyboard("{Enter}");
    await screen.findByRole("option", { name: /Open deployment qwen/ });
    await userEvent.keyboard("{ArrowDown}{Enter}");
    expect(screen.getByTestId("where")).toHaveTextContent("/deployments/1");
    expect(writes()).toEqual([]);
  });

  it("deploy opens the wizard prefilled; free memory opens the plan chooser", async () => {
    mount((text) => ({ interpretations: [text.startsWith("deploy")
      ? interp({ action: "deploy", target: { kind: "model", name: "Qwen/Qwen3-8B", ref: "Qwen/Qwen3-8B" }, params: { model_ref: "Qwen/Qwen3-8B", context: 32768 }, label: "Deploy Qwen/Qwen3-8B" })
      : interp({ action: "free_memory", target: null, params: { bytes: 20 * 1024 ** 3 }, label: "Free 20 GiB (choose a plan)" })], source: "rules", message: null, refused: false }),
    (url) => (url.startsWith("/decider/free-plans") ? { bytes: 20 * 1024 ** 3, free_for_new: 5 * 1024 ** 3, shortfall: 15 * 1024 ** 3, plans: [
      { id: "sleep:1", title: "Put qwen to sleep", detail: "Frees about 16 GiB.", new_required: 1, frees: 16 * 1024 ** 3, fits: true, actions: [{ type: "sleep", deployment_id: 1 }], patch: null, recommended: false, recommendation: null, unsafe_reason: null },
    ] } : undefined));
    await userEvent.type(input(), "deploy qwen3 8b at 32k");
    await userEvent.keyboard("{Enter}");
    await userEvent.click(await screen.findByRole("option", { name: /Deploy Qwen/ }));
    expect(screen.getByTestId("where")).toHaveTextContent("/deploy?model=Qwen%2FQwen3-8B&ctx=32768");
  });

  it("free memory: plans are applied only after choosing one and confirming", async () => {
    mount({ interpretations: [interp({ action: "free_memory", target: null, params: { bytes: 20 * 1024 ** 3 }, label: "Free 20 GiB (choose a plan)" })], source: "rules", message: null, refused: false },
      (url) => (url.startsWith("/decider/free-plans") ? { bytes: 20 * 1024 ** 3, free_for_new: 5 * 1024 ** 3, shortfall: 15 * 1024 ** 3, plans: [
        { id: "sleep:1", title: "Put qwen to sleep", detail: "Frees about 16 GiB.", new_required: 1, frees: 16 * 1024 ** 3, fits: true, actions: [{ type: "sleep", deployment_id: 1 }], patch: null, recommended: false, recommendation: null, unsafe_reason: null },
      ] } : undefined));
    await userEvent.type(input(), "need 20 gigs");
    await userEvent.keyboard("{Enter}");
    await userEvent.click(await screen.findByRole("option", { name: /Free 20 GiB/ }));
    expect(await screen.findByLabelText("GiB to free")).toHaveValue(20);
    const apply = screen.getByRole("button", { name: "Apply plan" });
    expect(apply).toBeDisabled();
    await userEvent.click(await screen.findByRole("radio", { name: /Put qwen to sleep/ }));
    expect(writes()).toEqual([]);
    await userEvent.click(apply);
    await waitFor(() => expect(writes()).toEqual([["/decider/free-memory", { method: "POST", body: { gib: 20, plan_id: "sleep:1", acknowledge_unsafe: null } }]]));
  });

  it("free memory: an unsafe plan needs an explicit acknowledgement", async () => {
    mount({ interpretations: [interp({ action: "free_memory", target: null, params: { bytes: 20 * 1024 ** 3 }, label: "Free 20 GiB (choose a plan)" })], source: "rules", message: null, refused: false },
      (url) => (url.startsWith("/decider/free-plans") ? { bytes: 20 * 1024 ** 3, free_for_new: 5 * 1024 ** 3, shortfall: 15 * 1024 ** 3, plans: [
        { id: "stop:1", title: "Stop qwen", detail: "Frees about 16 GiB.", new_required: 1, frees: 16 * 1024 ** 3, fits: true, actions: [{ type: "stop", deployment_id: 1 }], patch: null, recommended: false, recommendation: null, unsafe_reason: "would stop qwen: served requests 2 min ago", unsafe_fingerprint: "fp-seen" },
      ] } : undefined));
    await userEvent.type(input(), "need 20 gigs");
    await userEvent.keyboard("{Enter}");
    await userEvent.click(await screen.findByRole("option", { name: /Free 20 GiB/ }));
    await userEvent.click(await screen.findByRole("radio", { name: /Stop qwen/ }));
    const apply = screen.getByRole("button", { name: "Apply plan" });
    expect(apply).toBeDisabled();
    await userEvent.click(screen.getByRole("checkbox", { name: /Apply it anyway, although it is unsafe: would stop qwen/ }));
    await userEvent.click(apply);
    await waitFor(() => expect(writes()).toEqual([["/decider/free-memory", { method: "POST", body: { gib: 20, plan_id: "stop:1", acknowledge_unsafe: "fp-seen" } }]]));
  });

  it("free memory: a plan that became unsafe can be acknowledged inline and applied again", async () => {
    const plan = { id: "stop:1", title: "Stop qwen", detail: "Frees about 16 GiB.", new_required: 1, frees: 16 * 1024 ** 3, fits: true, actions: [{ type: "stop", deployment_id: 1 }], patch: null, recommended: false, recommendation: null, unsafe_reason: null };
    let posts = 0;
    mount({ interpretations: [interp({ action: "free_memory", target: null, params: { bytes: 20 * 1024 ** 3 }, label: "Free 20 GiB (choose a plan)" })], source: "rules", message: null, refused: false },
      (url, o) => {
        if (url.startsWith("/decider/free-plans")) return { bytes: 20 * 1024 ** 3, free_for_new: 5 * 1024 ** 3, shortfall: 15 * 1024 ** 3, plans: [plan] };
        if (url === "/decider/free-memory" && o?.method === "POST") {
          posts += 1;
          if (posts === 1) return Promise.reject(new apiModule.ApiError(409, { code: "plan_unsafe", message: "this plan is not safe: would stop qwen: served requests 1 min ago", plan: { ...plan, unsafe_reason: "would stop qwen: served requests 1 min ago", unsafe_fingerprint: "fp-new" } }));
          return { applied: [{ type: "stop", deployment_id: 1 }], plan };
        }
        return undefined;
      });
    await userEvent.type(input(), "need 20 gigs");
    await userEvent.keyboard("{Enter}");
    await userEvent.click(await screen.findByRole("option", { name: /Free 20 GiB/ }));
    await userEvent.click(await screen.findByRole("radio", { name: /Stop qwen/ }));
    await userEvent.click(screen.getByRole("button", { name: "Apply plan" }));
    const box = await screen.findByRole("checkbox", { name: /Apply it anyway, although it is unsafe: would stop qwen: served requests 1 min ago/ });
    expect(screen.getByRole("button", { name: "Apply plan" })).toBeDisabled();
    await userEvent.click(box);
    await userEvent.click(screen.getByRole("button", { name: "Apply plan" }));
    await waitFor(() => expect(writes().at(-1)).toEqual(["/decider/free-memory", { method: "POST", body: { gib: 20, plan_id: "stop:1", acknowledge_unsafe: "fp-new" } }]));
  });

  it("restart says it is two steps", async () => {
    mount({ interpretations: [interp({ action: "restart", label: "Restart qwen" })], source: "rules", message: null, refused: false });
    await userEvent.type(input(), "bounce the qwen thing");
    await userEvent.keyboard("{Enter}");
    await userEvent.click(await screen.findByRole("option", { name: /Restart qwen/ }));
    expect(await screen.findByText(/Two steps: it is stopped, then started again/)).toBeInTheDocument();
    expect(writes()).toEqual([]);
  });

  it("couldn't interpret / refusals are shown, with nothing to pick", async () => {
    mount((text) => text.startsWith("delete")
      ? { interpretations: [], source: "rules", message: "Typed commands cannot remove deployments, delete downloaded weights or manage users.", refused: true }
      : { interpretations: [], source: "model", message: "Couldn't interpret that.", refused: false });
    await userEvent.type(input(), "make it go brrr");
    await userEvent.keyboard("{Enter}");
    expect((await screen.findAllByText("Couldn't interpret that.")).length).toBeGreaterThan(0);
    await userEvent.clear(input());
    await userEvent.type(input(), "delete everything now");
    await userEvent.keyboard("{Enter}");
    expect((await screen.findAllByText(/cannot remove deployments/)).length).toBeGreaterThan(0);
    expect(screen.queryAllByRole("option", { name: /confidence/ })).toEqual([]);
    expect(writes()).toEqual([]);
  });

  it("viewers can ask too (the server returns navigation only)", async () => {
    isAdminFlag = false;
    mount({ interpretations: [interp({ action: "navigate", label: "Open deployment qwen", target: { kind: "deployment", name: "qwen", id: 1 } })], source: "rules", message: null, refused: false });
    await userEvent.type(input(), "where is qwen at");
    await userEvent.keyboard("{Enter}");
    await userEvent.click(await screen.findByRole("option", { name: /Open deployment qwen/ }));
    expect(screen.getByTestId("where")).toHaveTextContent("/deployments/1");
  });
});
