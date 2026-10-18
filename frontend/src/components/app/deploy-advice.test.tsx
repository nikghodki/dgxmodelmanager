import { afterEach, describe, expect, it, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { ReactNode } from "react";
import * as apiModule from "@/api";
import { CompatLine } from "./compat-line";
import { ConfigAdviceCard } from "./config-advice";
import { EngineChip } from "./engine-chip";
import { RemoteCodeFindings, RemoteCodeWarning } from "./remote-code-panel";
import { SpeculativeField } from "./speculative-field";

vi.mock("@/api", async () => {
  const actual = await vi.importActual<typeof import("@/api")>("@/api");
  return { ...actual, api: vi.fn() };
});
const api = vi.mocked(apiModule.api);
afterEach(() => api.mockReset());

function wrap(ui: ReactNode) {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(<QueryClientProvider client={qc}>{ui}</QueryClientProvider>);
}
const opt = (key: string, config: Record<string, any>, summary = key) => ({ key, config, summary, required_bytes: 1e9, headroom_bytes: 1e9, context: 32768, seqs: 32, kv_cache_dtype: "auto", prefix_caching: true, kv_pool_tokens: 131072 });
const advice = (over: Record<string, any> = {}) => ({ goal: "chat", options: [opt("a", { max_model_len: 32768, max_num_seqs: 32 }, "32k context · 32 concurrent"), opt("b", { max_model_len: 65536, max_num_seqs: 8 }, "64k context · 8 concurrent"), opt("c", { max_model_len: 8192, max_num_seqs: 256 }, "8k context · 256 concurrent")], recommended: "a", top: ["a", "b", "c"], source: "model", confidence: 0.91, decision_id: 3, free_bytes: 5e10, reason: null, disabled: false, ...over });

describe("ConfigAdviceCard", () => {
  it("fills only untouched fields from the suggested config", async () => {
    api.mockResolvedValue(advice());
    const onFill = vi.fn();
    wrap(<ConfigAdviceCard modelRef="org/m" engine="vllm" goal="chat" config={{ max_model_len: 4096, max_num_seqs: 3 }} touched={["max_num_seqs"]} onFill={onFill} />);
    expect(await screen.findByText(/Suggested config/)).toBeInTheDocument();
    expect(screen.getByText("91% · AI")).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: /Fill untouched fields/ }));
    expect(onFill).toHaveBeenCalledWith({ max_model_len: 32768, max_num_seqs: 3 }, ["max_model_len"]);
    expect(screen.getByRole("status")).toHaveTextContent("Filled 1 field");
    expect(api).toHaveBeenCalledWith("/decider/config-advice", { method: "POST", body: { model_ref: "org/m", engine: "vllm", goal: "chat" } });
  });

  it("a fill mixed with typed values is reported with the memory re-check", async () => {
    api.mockResolvedValue(advice());
    wrap(<ConfigAdviceCard modelRef="org/m" engine="vllm" goal="chat" config={{ max_model_len: 4096, max_num_seqs: 3 }} touched={["max_num_seqs"]} onFill={vi.fn()} fits={false} />);
    await userEvent.click(await screen.findByRole("button", { name: /Fill untouched fields/ }));
    expect(screen.getByRole("status")).toHaveTextContent("kept your max_num_seqs — re-checked: it does not fit the free memory");
  });

  it("lists the top three with their trade-offs below the threshold", async () => {
    api.mockResolvedValue(advice({ recommended: null, source: "none", top: ["b", "c", "a"] }));
    wrap(<ConfigAdviceCard modelRef="org/m" engine="vllm" goal="chat" config={{}} touched={[]} onFill={vi.fn()} />);
    expect(await screen.findByText(/Configs that fit/)).toBeInTheDocument();
    const items = screen.getAllByRole("listitem").map((li) => li.textContent);
    expect(items[0]).toMatch(/^64k context/);
    expect(items).toHaveLength(3);
  });

  it("says when nothing fits and stays silent on errors", async () => {
    api.mockResolvedValue(advice({ options: [], recommended: null, top: [], reason: "nothing_fits" }));
    const { unmount } = wrap(<ConfigAdviceCard modelRef="org/m" engine="vllm" goal="chat" config={{}} touched={[]} onFill={vi.fn()} />);
    expect(await screen.findByText(/no sizing of this model fits/)).toBeInTheDocument();
    unmount();
    api.mockRejectedValue(new Error("boom"));
    const { container } = wrap(<ConfigAdviceCard modelRef="org/m" engine="vllm" goal="chat" config={{}} touched={[]} onFill={vi.fn()} />);
    await waitFor(() => expect(api).toHaveBeenCalledTimes(2));
    expect(container.textContent).toBe("");
  });
});

describe("EngineChip", () => {
  it("offers the other format's engine and model", async () => {
    api.mockResolvedValue({ engine: "llamacpp", model_ref: "/models/m-Q4_K_M.gguf", source: "model", confidence: 0.88, reason: "model", decision_id: 1, alternatives: {}, disabled: false });
    const onUse = vi.fn();
    wrap(<EngineChip modelRef="org/m" goal="chat" engine="vllm" onUse={onUse} />);
    await userEvent.click(await screen.findByRole("button", { name: /Use llama\.cpp with/ }));
    expect(onUse).toHaveBeenCalledWith("llamacpp", "/models/m-Q4_K_M.gguf");
  });
  it("shows a matching rules suggestion without a button and hides abstentions", async () => {
    api.mockResolvedValue({ engine: "vllm", model_ref: "org/m", source: "rules", confidence: 1, reason: "safetensors_only", decision_id: 1, alternatives: {}, disabled: false });
    const { unmount } = wrap(<EngineChip modelRef="org/m" goal="chat" engine="vllm" onUse={vi.fn()} />);
    expect(await screen.findByText(/Suggested: vLLM/)).toBeInTheDocument();
    expect(screen.queryByRole("button")).toBeNull();
    unmount();
    api.mockResolvedValue({ engine: null, model_ref: null, source: "none", confidence: 0.6, reason: "both_formats", decision_id: 2, alternatives: {}, disabled: false });
    const { container } = wrap(<EngineChip modelRef="org/x" goal="chat" engine="vllm" onUse={vi.fn()} />);
    await waitFor(() => expect(api).toHaveBeenCalledTimes(2));
    expect(container.textContent).toBe("");
  });
});

describe("SpeculativeField", () => {
  const drafts = [{ ref: "Qwen/Qwen3-0.6B", complete: true, kind: "llm" } as any];
  it("shows the suggested draft and applies it only on click", async () => {
    api.mockResolvedValue({ method: "draft_model", candidates: [{ ref: "Qwen/Qwen3-0.6B", weights_bytes: 1e9, reason: "same_tokenizer", summary: "", p: 1 }], suggested: "Qwen/Qwen3-0.6B", source: "rules", confidence: 1, decision_id: 1, reason: null, disabled: false });
    const onChange = vi.fn();
    wrap(<SpeculativeField modelRef="Qwen/Qwen3-8B" value={{ method: "draft_model", model: null }} onChange={onChange} drafts={drafts} />);
    expect(await screen.findByText(/Suggested draft: Qwen\/Qwen3-0\.6B/)).toBeInTheDocument();
    expect(onChange).not.toHaveBeenCalled();
    await userEvent.click(screen.getByRole("button", { name: /Apply suggested draft/ }));
    expect(onChange).toHaveBeenCalledWith({ num_speculative_tokens: 3, method: "draft_model", model: "Qwen/Qwen3-0.6B" });
  });
  it("a method that needs a draft model reaches the config only with a model", async () => {
    api.mockResolvedValue({ method: "draft_model", candidates: [], suggested: null, source: "none", confidence: null, decision_id: null, reason: "no_compatible_draft", disabled: false });
    const onChange = vi.fn();
    wrap(<SpeculativeField modelRef="Qwen/Qwen3-8B" value={undefined} onChange={onChange} drafts={drafts} />);
    await userEvent.click(screen.getByRole("combobox", { name: "Speculative decoding" }));
    await userEvent.click(await screen.findByRole("option", { name: "draft_model" }));
    expect(onChange).toHaveBeenLastCalledWith(undefined);
    expect(screen.getByRole("combobox", { name: "Draft model" })).toBeInTheDocument();
  });

  it("asks nothing while speculative decoding is off", () => {
    wrap(<SpeculativeField modelRef="Qwen/Qwen3-8B" value={undefined} onChange={vi.fn()} drafts={drafts} />);
    expect(api).not.toHaveBeenCalled();
  });
});

describe("RemoteCodeFindings", () => {
  it("lists findings with file:line", async () => {
    api.mockResolvedValue({ verdict: "unusual", findings: [{ file: "modeling_x.py", line: 12, kind: "network_import", detail: "socket" }], source: "rules", confidence: 1, model_label: null, decision_id: 1, scanned_files: 2, truncated: false, error: null, disabled: false, remote_code: true });
    wrap(<RemoteCodeFindings modelRef="org/m" enabled />);
    expect(await screen.findByText("modeling_x.py:12 — imports a network module")).toBeInTheDocument();
    expect(screen.getByRole("alert")).toHaveTextContent("1 finding in 1 file");
    expect(screen.getByRole("alert")).toHaveTextContent("2 files scanned");
  });
  it("counts the files with findings, not the files scanned", async () => {
    const f = (file: string, line: number) => ({ file, line, kind: "network_import", detail: "socket" });
    api.mockResolvedValue({ verdict: "unusual", findings: [f("a.py", 1), f("a.py", 9), f("b.py", 3)], source: "rules", confidence: 1, model_label: null, decision_id: 1, scanned_files: 14, truncated: false, error: null, disabled: false, remote_code: true });
    wrap(<RemoteCodeFindings modelRef="org/m" enabled />);
    expect(await screen.findByText(/3 findings in 2 files/)).toBeInTheDocument();
    expect(screen.getByRole("alert")).toHaveTextContent("14 files scanned");
    expect(screen.getByRole("alert")).not.toHaveTextContent("in 14 files");
  });
  it("a routine answer never removes the generic warning and is reported neutrally", async () => {
    api.mockResolvedValue({ verdict: "no_findings", findings: [], source: "rules", confidence: null, model_label: null, decision_id: 1, scanned_files: 3, truncated: false, error: null, disabled: false, remote_code: true });
    wrap(<><RemoteCodeWarning /><RemoteCodeFindings modelRef="org/m" enabled /></>);
    expect(await screen.findByText("No suspicious patterns found in 3 files (static scan only)")).toBeInTheDocument();
    expect(screen.queryByText(/routine/i)).toBeNull();
    expect(screen.getAllByRole("alert")).toHaveLength(1);
    expect(screen.getByRole("alert")).toHaveTextContent("trust_remote_code runs Python code");
  });
  it("names the execution and loading patterns of the security review (M1)", async () => {
    api.mockResolvedValue({ verdict: "unusual", source: "rules", confidence: 1, model_label: null, decision_id: 1, scanned_files: 1, truncated: false, error: null, disabled: false, remote_code: true,
      findings: ["code_object", "object_walk", "native_load", "remote_fetch"].map((kind, i) => ({ file: "m.py", line: i + 1, kind, detail: kind })) });
    wrap(<RemoteCodeFindings modelRef="org/m" enabled />);
    expect(await screen.findByText("m.py:1 — builds or patches a code object")).toBeInTheDocument();
    expect(screen.getByText("m.py:2 — walks the object graph (__subclasses__ / __globals__)")).toBeInTheDocument();
    expect(screen.getByText("m.py:3 — loads native code at run time")).toBeInTheDocument();
    expect(screen.getByText("m.py:4 — downloads files or code from the Hub at run time")).toBeInTheDocument();
  });
  it("a failed or partial scan is a warning", async () => {
    api.mockResolvedValue({ verdict: "unusual", findings: [{ file: "*", line: 0, kind: "unanalysable", detail: "scan_timeout" }], source: "rules", confidence: 1, model_label: null, decision_id: 1, scanned_files: 0, truncated: true, error: "scan_timeout", disabled: false, remote_code: true });
    const { unmount } = wrap(<RemoteCodeFindings modelRef="org/m" enabled />);
    expect(await screen.findByRole("alert")).toHaveTextContent("could not analyse the code — treat it as unusual");
    unmount();
    api.mockResolvedValue({ verdict: "unknown", findings: [], source: "none", confidence: null, model_label: null, decision_id: 1, scanned_files: 60, truncated: true, error: null, disabled: false, remote_code: true });
    wrap(<RemoteCodeFindings modelRef="org/x" enabled />);
    expect(await screen.findByRole("alert")).toHaveTextContent("only part of the code was scanned");
  });
});

describe("CompatLine", () => {
  it("shows both engines with templated reasons", async () => {
    api.mockResolvedValue({ repo: "org/m", sha: "s", architecture: "X", gated: true, vllm: { verdict: "runs_with_flags", reasons: ["supported_architecture", "trust_remote_code_required", "auth_gated"], flags: ["trust_remote_code", "hf_token"] }, llamacpp: { verdict: "unsupported", reasons: ["no_gguf"], flags: [] }, source: "rules", confidence: 1, decision_id: 1, cached: false, error: null, disabled: false });
    wrap(<CompatLine repo="org/m" />);
    expect(await screen.findByText(/vLLM: runs with extra settings/)).toBeInTheDocument();
    expect(screen.getByText("llama.cpp: not supported (no GGUF files in this repo)")).toBeInTheDocument();
    expect(screen.getByText("Deploying it needs: trust_remote_code, HF token")).toBeInTheDocument();
    expect(api).toHaveBeenCalledWith("/catalog/compat?repo=org%2Fm");
  });
  it("says when the Hub is unreachable", async () => {
    api.mockResolvedValue({ repo: "org/m", sha: null, architecture: null, gated: false, vllm: { verdict: "unknown", reasons: [], flags: [] }, llamacpp: { verdict: "unknown", reasons: [], flags: [] }, source: "none", confidence: null, decision_id: null, cached: false, error: "hub_unavailable", disabled: false });
    wrap(<CompatLine repo="org/m" />);
    expect(await screen.findByText(/unavailable \(Hugging Face not reachable\)/)).toBeInTheDocument();
  });
});
