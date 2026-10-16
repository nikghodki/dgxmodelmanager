import { describe, expect, it } from "vitest";
import { containerLabel, filterScope, groupByContainer, processSearchText, userLabel } from "@/lib/processes";
import type { ProcessInfo } from "@/types";

const P = (pid: number, over: Partial<ProcessInfo> = {}): ProcessInfo => ({
  pid, ppid: 1, uid: 0, user: "root", user_scope: "host", name: "python3", exe: "python3", command: `python3 app.py --id ${pid}`,
  command_truncated: false, container: null, managed: null, rss: 100, gpu: 0, ...over,
});
const LITE = { id: "bbbbbbbbbbbb", name: "litellm-proxy", image: "ghcr.io/berriai/litellm:main-latest" };
const QWEN = { id: "aaaaaaaaaaaa", name: "dgxmm-qwen", image: "vllm/vllm-openai:v0.11" };
const procs = [
  P(100, { container: QWEN, managed: { deployment_id: 7, name: "qwen" }, rss: 2000, gpu: 40000 }),
  P(101, { container: QWEN, managed: { deployment_id: 7, name: "qwen" }, rss: 500 }),
  P(300, { user: "nik", rss: 3000, gpu: 1000 }),
  P(200, { container: LITE, rss: 1000 }),
];

describe("processes", () => {
  it("filters by scope", () => {
    expect(filterScope(procs, "all").map((p) => p.pid)).toEqual([100, 101, 300, 200]);
    expect(filterScope(procs, "external").map((p) => p.pid)).toEqual([300, 200]);
    expect(filterScope(procs, "managed").map((p) => p.pid)).toEqual([100, 101]);
  });

  it("groups by container with host as its own group, biggest first", () => {
    const g = groupByContainer(procs);
    expect(g.map((x) => x.key)).toEqual(["aaaaaaaaaaaa", "host", "bbbbbbbbbbbb"]);
    expect(g[0]).toMatchObject({ name: "dgxmm-qwen", image: "vllm/vllm-openai:v0.11", rss: 2500, gpu: 40000, managed: { deployment_id: 7, name: "qwen" } });
    expect(g[0].processes.map((p) => p.pid)).toEqual([100, 101]);
    expect(g[1]).toMatchObject({ name: "host", image: null, rss: 3000, gpu: 1000, managed: null });
  });

  it("labels containers, including unnamed ones", () => {
    expect(containerLabel(null)).toBe("host");
    expect(containerLabel(LITE)).toBe("litellm-proxy");
    expect(containerLabel({ id: "cccccccccccc", name: null, image: null })).toBe("cccccccccccc");
  });

  it("search text covers command, container, image, user and pid", () => {
    const t = processSearchText(procs[3]);
    for (const s of ["--id 200", "litellm-proxy", "berriai", "root", "200"]) expect(t).toContain(s);
    expect(processSearchText(procs[2])).toContain("host");
  });

  it("labels container uids as not host accounts", () => {
    expect(userLabel("nik", "host")).toEqual({ text: "nik", title: undefined });
    expect(userLabel("uid 1000", "container")).toEqual({ text: "uid 1000 (in container)", title: "uid as seen inside the container; not a host account" });
    expect(userLabel("root", "container").text).toBe("root (in container)");
    expect(userLabel("x", undefined)).toEqual({ text: "x", title: undefined }); // older servers
  });
});
