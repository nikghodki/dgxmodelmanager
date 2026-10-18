import { afterEach, describe, expect, it, vi } from "vitest";
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter } from "react-router-dom";
import * as auth from "@/auth";
import * as apiModule from "@/api";
import { insight } from "@/test/insights";
import type { Deployment, Schedule } from "@/types";
import { InsightActionButton } from "./insight-action";
import { DeploymentSchedulesCard, SchedulesPauseControl } from "./schedules-card";

vi.mock("@/auth");
vi.mock("@/api", async () => {
  const actual = await vi.importActual<typeof import("@/api")>("@/api");
  return { ...actual, api: vi.fn() };
});
vi.mock("sonner", () => ({ toast: { success: vi.fn(), error: vi.fn() } }));
vi.mock("@/hooks/use-insights", () => ({ useInsightActions: () => ({ applied: { mutate: appliedMutate } }) }));
const appliedMutate = vi.fn();

const dep = { id: 4, name: "qwen", state: "ready", system: false, sleep_capable: true } as unknown as Deployment;
const sched = (over: Partial<Schedule> = {}): Schedule => ({
  id: 7, deployment_id: 4, action: "sleep", days: 31, days_label: "Mon–Fri", at_local: "19:00", timezone: "Europe/Berlin",
  enabled: true, created_by: 1, created_at: null, updated_at: null, last_run_at: null, last_result: null,
  next_run_at: "2026-10-19T17:00:00+00:00", ...over,
});
const LIST = { deployment_id: 4, schedules: [sched(), sched({ id: 8, action: "wake", at_local: "07:45", last_result: "ok: woken up" })], allowed_actions: ["sleep", "wake"], timezone: "Europe/Berlin" };

function mount(ui: React.ReactNode, admin = true, handler: (url: string, opts?: any) => any = () => LIST) {
  vi.mocked(auth.useIsAdmin).mockReturnValue(admin);
  vi.mocked(apiModule.api).mockImplementation(async (url: string, opts?: any) => handler(url, opts));
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(<QueryClientProvider client={qc}><MemoryRouter future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>{ui}</MemoryRouter></QueryClientProvider>);
}

const writes = () => vi.mocked(apiModule.api).mock.calls.filter(([, o]) => (o as any)?.method && (o as any).method !== "GET");

afterEach(() => { vi.mocked(apiModule.api).mockReset(); appliedMutate.mockReset(); });

describe("DeploymentSchedulesCard", () => {
  it("lists schedules with the next run and last result", async () => {
    mount(<DeploymentSchedulesCard deployment={dep} />);
    const list = await screen.findByRole("list", { name: "Schedules" });
    expect(within(list).getByText("Sleep at 19:00")).toBeInTheDocument();
    expect(within(list).getByText(/ok: woken up/)).toBeInTheDocument();
    expect(within(list).getAllByText(/Next run:/)).toHaveLength(2);
  });

  it("viewers get a read-only list", async () => {
    mount(<DeploymentSchedulesCard deployment={dep} />, false);
    await screen.findByText("Sleep at 19:00");
    expect(screen.queryByRole("button", { name: /Add|Edit|Pause|Delete/ })).toBeNull();
  });

  it("pauses and deletes (with a confirmation)", async () => {
    mount(<DeploymentSchedulesCard deployment={dep} />, true, (_url, o) => (o?.method ? {} : LIST));
    await userEvent.click(await screen.findByRole("button", { name: "Pause sleep at 19:00" }));
    await waitFor(() => expect(writes()).toEqual([["/schedules/7", { method: "PUT", body: { enabled: false } }]]));
    await userEvent.click(screen.getByRole("button", { name: "Delete wake at 07:45" }));
    expect(writes()).toHaveLength(1); // nothing deleted before the confirmation
    await userEvent.click(await screen.findByRole("button", { name: "Delete" }));
    await waitFor(() => expect(writes()[1]).toEqual(["/schedules/8", { method: "DELETE" }]));
  });

  it("adds a sleep with its wake and previews the next run", async () => {
    mount(<DeploymentSchedulesCard deployment={dep} />, true, (_url, o) => (o?.method ? { schedules: [] } : { ...LIST, schedules: [] }));
    await userEvent.click(await screen.findByRole("button", { name: "Add" }));
    expect(await screen.findByRole("status")).toHaveTextContent(/Next run: Sleep .* · Wake/);
    await userEvent.click(screen.getByRole("button", { name: "Sat" }));
    expect(screen.getByRole("button", { name: "Sat" })).toHaveAttribute("aria-pressed", "true");
    await userEvent.click(screen.getByRole("button", { name: "Add" }));
    await waitFor(() => expect(writes()[0]).toEqual(["/deployments/4/schedules", { method: "POST", body: { schedules: [
      { action: "sleep", days: 63, at_local: "19:00", timezone: "Europe/Berlin" },
      { action: "wake", days: 63, at_local: "07:45", timezone: "Europe/Berlin" },
    ] } }]));
  });

  it("refuses an unknown time zone before sending", async () => {
    mount(<DeploymentSchedulesCard deployment={dep} />, true, () => ({ ...LIST, schedules: [] }));
    await userEvent.click(await screen.findByRole("button", { name: "Add" }));
    const tz = screen.getByLabelText("Time zone");
    await userEvent.clear(tz);
    await userEvent.type(tz, "Mars/Olympus");
    expect(screen.getByRole("status")).toHaveTextContent("Unknown time zone");
    expect(screen.getAllByRole("button", { name: "Add" }).at(-1)).toBeDisabled();
  });

  it("shows that all schedules are paused and lets admins resume them", async () => {
    mount(<DeploymentSchedulesCard deployment={dep} />, true, (url, o) => (o?.method ? { paused_all: false, timezone: "UTC" }
      : url === "/schedules/settings" ? { paused_all: true, timezone: "UTC" } : { ...LIST, paused_all: true }));
    expect(await screen.findByText("All schedules are paused.")).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "Resume all schedules" }));
    await waitFor(() => expect(writes()).toEqual([["/schedules/settings", { method: "PUT", body: { paused_all: false } }]]));
  });

  it("pausing all asks for a confirmation; viewers only see the state", async () => {
    mount(<SchedulesPauseControl />, true, (_url, o) => (o?.method ? { paused_all: true, timezone: "UTC" } : { paused_all: false, timezone: "UTC" }));
    await userEvent.click(await screen.findByRole("button", { name: "Pause all schedules" }));
    expect(writes()).toEqual([]);
    await userEvent.click(await screen.findByRole("button", { name: "Pause all" }));
    await waitFor(() => expect(writes()).toEqual([["/schedules/settings", { method: "PUT", body: { paused_all: true } }]]));
  });

  it("viewers see the paused state without a button", async () => {
    mount(<SchedulesPauseControl />, false, () => ({ paused_all: true, timezone: "UTC" }));
    expect(await screen.findByText("All schedules are paused.")).toBeInTheDocument();
    expect(screen.queryByRole("button")).toBeNull();
  });

  it("marks deferred and pending runs", async () => {
    mount(<DeploymentSchedulesCard deployment={dep} />, true, () => ({ ...LIST, schedules: [sched({ deferred: true, last_result: "deferred: a request arrived 2 min ago (retrying until 20:00)" }), sched({ id: 8, action: "wake", at_local: "07:45", pending: true })] }));
    expect(await screen.findByText("Deferred")).toBeInTheDocument();
    expect(screen.getByText("Pending")).toBeInTheDocument();
  });

  it("renders nothing for a system deployment", () => {
    mount(<DeploymentSchedulesCard deployment={{ ...dep, system: true }} />);
    expect(screen.queryByText("Schedules")).toBeNull();
  });
});

describe("schedule insight dialog", () => {
  const proposal = { sleep_action: "sleep", sleep_at: "19:00", sleep_days: 31, wake_action: "wake", wake_at: "07:45", wake_days: 31, timezone: "Europe/Berlin", weeks: 3 };
  const ins = insight({ id: 9, kind: "schedule_proposal", feature: "schedule", title: "Sleep qwen on a schedule?", body: "qwen was idle every night.",
    action: { type: "schedule", deployment_id: 4, proposal } });

  it("shows what will run when and creates the two schedules only on confirmation", async () => {
    mount(<InsightActionButton insight={ins} />, true, (_url, o) => (o?.method ? { schedules: [] } : undefined));
    await userEvent.click(screen.getByRole("button", { name: "Review schedule…" }));
    const table = await screen.findByRole("table", { name: "What will run when" });
    expect(within(table).getByText(/19:00 · Mon–Fri · Europe\/Berlin/)).toBeInTheDocument();
    expect(within(table).getByText(/07:45 · Mon–Fri · Europe\/Berlin/)).toBeInTheDocument();
    expect(writes()).toEqual([]);
    await userEvent.click(screen.getByRole("button", { name: "Create schedules" }));
    await waitFor(() => expect(writes()).toEqual([["/deployments/4/schedules", { method: "POST", body: { schedules: [
      { action: "sleep", days: 31, at_local: "19:00", timezone: "Europe/Berlin" },
      { action: "wake", days: 31, at_local: "07:45", timezone: "Europe/Berlin" },
    ] } }]]));
    await waitFor(() => expect(appliedMutate).toHaveBeenCalledWith(9));
  });

  it("cancel creates nothing; a malformed proposal is not offered", async () => {
    mount(<InsightActionButton insight={ins} />);
    await userEvent.click(screen.getByRole("button", { name: "Review schedule…" }));
    await userEvent.click(await screen.findByRole("button", { name: "Cancel" }));
    expect(writes()).toEqual([]);
    expect(appliedMutate).not.toHaveBeenCalled();
  });

  it("an unknown proposal shape says so and cannot be applied", async () => {
    const bad = insight({ ...ins, action: { type: "schedule", deployment_id: 4, proposal: { sleep_action: "remove" } } });
    mount(<InsightActionButton insight={bad} />);
    await userEvent.click(screen.getByRole("button", { name: "Review schedule…" }));
    expect(await screen.findByText(/cannot be shown by this version/)).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Create schedules" })).toBeNull();
  });
});
