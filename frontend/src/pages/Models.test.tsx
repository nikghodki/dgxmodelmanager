import { beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter } from "react-router-dom";
import { Models } from "./Models";
import * as auth from "@/auth";
import * as apiModule from "@/api";

vi.mock("@/auth");
vi.mock("@/api", async () => {
  const actual = await vi.importActual<typeof import("@/api")>("@/api");
  return { ...actual, api: vi.fn() };
});

const art = (ref: string, complete: boolean, extra: Record<string, unknown> = {}) => ({
  ref, source: "hf", revision: null, format: "safetensors", weights_bytes: 1e9, size_bytes: 1e9, complete, kind: "llm",
  architecture: "Qwen", quant: null, problems: complete ? [] : ["download in progress or interrupted"], max_context: 4096,
  is_moe: false, num_layers: 1, confident: true, has_mtp: false, engines: ["vllm"], ...extra,
});

const GIB = 1024 ** 3;
const preview = (extra: Record<string, unknown> = {}) => ({
  ref: "org/done", source: "hf", name: "done", paths: ["/home/u/.cache/huggingface/hub/models--org--done"], bytes: 2 * GIB,
  deployments_blocking: [], deployments_affected: [], download_active: false, blocked: false, notes: [], ...extra,
});
const cleanupPreview = {
  items: [
    { ref: "org/partial", source: "hf", name: "partial", problems: ["download in progress or interrupted"], paths: ["/h/a"], bytes: 3 * GIB,
      deployments_blocking: [], deployments_affected: [], download_active: false, eligible: true, skip_reason: null },
    { ref: "org/busy", source: "hf", name: "busy", problems: ["no snapshot"], paths: ["/h/b"], bytes: 1 * GIB,
      deployments_blocking: [], deployments_affected: [], download_active: true, eligible: false, skip_reason: "download_active" },
  ],
  count: 1, bytes: 3 * GIB,
};

type Handler = (path: string, opts?: { method?: string; body?: unknown }) => unknown;
function mockApi(overrides: Handler = () => undefined) {
  const api = vi.mocked(apiModule.api);
  api.mockImplementation(async (path: string, opts?: { method?: string; body?: unknown }) => {
    const o = overrides(path, opts);
    if (o !== undefined) return o as any;
    if (path === "/inventory") return [art("org/done", true), art("org/partial", false), art("org/busy", false)] as any;
    if (path.startsWith("/inventory/delete-preview")) return preview() as any;
    if (path === "/inventory/cleanup-incomplete/preview") return cleanupPreview as any;
    if (path.startsWith("/inventory?ref=")) return { ref: "org/done", freed_bytes: 2 * GIB, paths: [], method: "helper", warnings: [] } as any;
    if (path === "/inventory/cleanup-incomplete") return { deleted: [{ ref: "org/partial", freed_bytes: 3 * GIB }], skipped: [], failed: [], freed_bytes: 3 * GIB } as any;
    throw new Error(`unexpected ${path}`);
  });
  return api;
}

function renderModels() {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={qc}>
      <MemoryRouter future={{ v7_startTransition: true, v7_relativeSplatPath: true }}><Models /></MemoryRouter>
    </QueryClientProvider>,
  );
}

const row = async (name: string) => (await screen.findByText(name)).closest("tr")!;

describe("Models", () => {
  beforeEach(() => { vi.mocked(apiModule.api).mockReset(); });

  it("shows a single, consistent status label per model", async () => {
    vi.mocked(auth.useIsAdmin).mockReturnValue(false);
    mockApi();
    renderModels();
    const done = await row("org/done");
    const partial = await row("org/partial");
    const status = (tr: HTMLElement) => within(tr).getAllByRole("cell").find((c) => /Complete|Incomplete/.test(c.textContent ?? ""))!;
    expect(status(done).textContent).toBe("Complete");
    expect(status(partial).textContent).toBe("Incomplete");
    expect(status(done).querySelector(".text-status-good")).not.toBeNull();
    expect(status(partial).querySelector(".text-status-warning")).not.toBeNull();
  });

  it("hides delete and clean-up actions from viewers", async () => {
    vi.mocked(auth.useIsAdmin).mockReturnValue(false);
    const api = mockApi();
    renderModels();
    await row("org/done");
    expect(screen.queryByRole("button", { name: /Delete/ })).toBeNull();
    expect(screen.queryByRole("button", { name: /Clean up/ })).toBeNull();
    expect(api.mock.calls.some(([p]) => p.includes("cleanup-incomplete"))).toBe(false);
  });

  it("incomplete rows offer Clean up instead of Deploy", async () => {
    vi.mocked(auth.useIsAdmin).mockReturnValue(true);
    mockApi();
    renderModels();
    const done = await row("org/done");
    const partial = await row("org/partial");
    expect(within(done).getByRole("link", { name: /Deploy/ })).toBeInTheDocument();
    expect(within(done).getByRole("button", { name: "Delete done" })).toBeInTheDocument();
    expect(within(partial).queryByRole("link", { name: /Deploy/ })).toBeNull();
    expect(within(partial).getByRole("button", { name: "Clean up partial" })).toBeInTheDocument();
  });

  it("delete dialog shows the preview and requires the typed model name", async () => {
    vi.mocked(auth.useIsAdmin).mockReturnValue(true);
    const user = userEvent.setup();
    const api = mockApi((p) => p.startsWith("/inventory/delete-preview")
      ? preview({ deployments_affected: [{ id: 4, name: "old-dep", state: "stopped" }] }) : undefined);
    renderModels();
    await user.click(within(await row("org/done")).getByRole("button", { name: "Delete done" }));
    const dialog = await screen.findByRole("alertdialog");
    expect(await within(dialog).findByText("2.0 GiB")).toBeInTheDocument();
    expect(within(dialog).getByText("/home/u/.cache/huggingface/hub/models--org--done")).toBeInTheDocument();
    expect(within(dialog).getByText(/old-dep/)).toBeInTheDocument();
    expect(within(dialog).getByText(/will fail to start/)).toBeInTheDocument();
    const confirm = within(dialog).getByRole("button", { name: "Delete weights" });
    expect(confirm).toBeDisabled();
    await user.type(within(dialog).getByLabelText(/Type done to confirm/), "done");
    expect(confirm).toBeEnabled();
    await user.click(confirm);
    await waitFor(() => expect(api).toHaveBeenCalledWith("/inventory?ref=org%2Fdone", { method: "DELETE" }));
  });

  it("delete is blocked while a deployment uses the model", async () => {
    vi.mocked(auth.useIsAdmin).mockReturnValue(true);
    const user = userEvent.setup();
    const api = mockApi((p) => p.startsWith("/inventory/delete-preview")
      ? preview({ blocked: true, deployments_blocking: [{ id: 7, name: "live-dep", state: "ready" }] }) : undefined);
    renderModels();
    await user.click(within(await row("org/done")).getByRole("button", { name: "Delete done" }));
    const dialog = await screen.findByRole("alertdialog");
    expect(await within(dialog).findByText(/live-dep/)).toBeInTheDocument();
    expect(within(dialog).getByText(/Stop it first|Stop them first/)).toBeInTheDocument();
    await user.type(within(dialog).getByLabelText(/Type done to confirm/), "done");
    expect(within(dialog).getByRole("button", { name: "Delete weights" })).toBeDisabled();
    expect(api.mock.calls.some(([, o]) => o?.method === "DELETE")).toBe(false);
  });

  it("bulk clean-up dialog lists incomplete models with checkboxes and explains skips", async () => {
    vi.mocked(auth.useIsAdmin).mockReturnValue(true);
    const user = userEvent.setup();
    const api = mockApi();
    renderModels();
    await user.click(await screen.findByRole("button", { name: /Clean up incomplete \(1 · 3\.0 GiB\)/ }));
    const dialog = await screen.findByRole("dialog");
    const ok = within(dialog).getByRole("checkbox", { name: /org\/partial/ });
    const busy = within(dialog).getByRole("checkbox", { name: /org\/busy/ });
    expect(ok).toBeChecked();
    expect(busy).not.toBeChecked();
    expect(busy).toBeDisabled();
    expect(within(dialog).getByText(/download is still running/i)).toBeInTheDocument();
    await user.click(within(dialog).getByRole("button", { name: /Clean up 1 · 3\.0 GiB/ }));
    await waitFor(() => expect(api).toHaveBeenCalledWith("/inventory/cleanup-incomplete", { method: "POST", body: { refs: ["org/partial"] } }));
  });

  it("row Clean up removes just that incomplete model through the cleanup endpoint", async () => {
    vi.mocked(auth.useIsAdmin).mockReturnValue(true);
    const user = userEvent.setup();
    const api = mockApi((p) => p.startsWith("/inventory/delete-preview")
      ? preview({ ref: "org/partial", name: "partial", bytes: 3 * GIB }) : undefined);
    renderModels();
    await user.click(within(await row("org/partial")).getByRole("button", { name: "Clean up partial" }));
    const dialog = await screen.findByRole("alertdialog");
    expect(await within(dialog).findByText("3.0 GiB")).toBeInTheDocument();
    await user.click(within(dialog).getByRole("button", { name: "Clean up" }));
    await waitFor(() => expect(api).toHaveBeenCalledWith("/inventory/cleanup-incomplete", { method: "POST", body: { refs: ["org/partial"] } }));
  });

  it("shows preview notes (e.g. shared blobs)", async () => {
    vi.mocked(auth.useIsAdmin).mockReturnValue(true);
    const user = userEvent.setup();
    mockApi((p) => p.startsWith("/inventory/delete-preview") ? preview({ bytes: 0, notes: ["Q8.gguf: its data is shared with other files"] }) : undefined);
    renderModels();
    await user.click(within(await row("org/done")).getByRole("button", { name: "Delete done" }));
    expect(await within(await screen.findByRole("alertdialog")).findByText(/shared with other files/)).toBeInTheDocument();
  });

  it("refetches the preview when the delete is refused", async () => {
    vi.mocked(auth.useIsAdmin).mockReturnValue(true);
    const user = userEvent.setup();
    const api = mockApi((_p, o) => {
      if (o?.method === "DELETE") return Promise.reject(new apiModule.ApiError(409, { code: "model_in_use", message: "in use" }));
      return undefined;
    });
    renderModels();
    await user.click(within(await row("org/done")).getByRole("button", { name: "Delete done" }));
    const dialog = await screen.findByRole("alertdialog");
    await user.type(await within(dialog).findByLabelText(/Type done to confirm/), "done");
    await user.click(within(dialog).getByRole("button", { name: "Delete weights" }));
    const previews = () => api.mock.calls.filter(([p]) => p.startsWith("/inventory/delete-preview")).length;
    await waitFor(() => expect(previews()).toBeGreaterThanOrEqual(2));
  });
  it("labels kinds as LLM / Draft / Other and hides low-value columns below 2xl", async () => {
    vi.mocked(auth.useIsAdmin).mockReturnValue(false);
    mockApi((path) => path === "/inventory" ? [art("org/done", true), art("org/draft", true, { kind: "draft" }), art("org/x", true, { kind: "other" })] : undefined);
    renderModels();
    expect(within(await row("org/done")).getByText("LLM")).toBeInTheDocument();
    expect(within(await row("org/draft")).getByText("Draft")).toBeInTheDocument();
    expect(within(await row("org/x")).getByText("Other")).toBeInTheDocument();
    expect(screen.queryByText("llm")).toBeNull();
    for (const h of ["Architecture", "Context"]) {
      expect(screen.getByText(h).closest("th")!.className).toMatch(/\bhidden\b.*\b2xl:table-cell\b/);
    }
  });
});
