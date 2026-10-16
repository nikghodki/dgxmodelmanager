import { useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { api } from "../api";
import { useIsAdmin } from "../auth";
import { Badge, Button, Card, ErrorBox, Input, Modal, Table } from "../components/ui";
import { fmtBytes } from "../format";

const FIT_TONE: Record<string, "green" | "amber" | "red" | "slate"> = { fits_now: "green", fits_after_evict: "amber", too_big: "red", unknown: "slate" };
const FIT_LABEL: Record<string, string> = { fits_now: "fits now", fits_after_evict: "fits if you evict", too_big: "too big", unknown: "size unknown" };

export function Catalog() {
  const admin = useIsAdmin();
  const [q, setQ] = useState("");
  const [term, setTerm] = useState("qwen");
  const [gguf, setGguf] = useState(false);
  const [ctx, setCtx] = useState(32768);
  const [open, setOpen] = useState<string | null>(null);
  const res = useQuery({ queryKey: ["catalog", term, gguf, ctx], queryFn: () => api<any>(`/catalog/search?q=${encodeURIComponent(term)}&gguf=${gguf}&ctx=${ctx}`) });
  return (
    <div className="space-y-4">
      <Card title="Hugging Face catalog">
        <form className="flex flex-wrap items-end gap-3" onSubmit={(e) => { e.preventDefault(); setTerm(q); }}>
          <div className="w-72"><Input placeholder="Search models (e.g. qwen3 nvfp4)" value={q} onChange={(e) => setQ(e.target.value)} /></div>
          <label className="text-sm"><input type="checkbox" checked={gguf} onChange={(e) => setGguf(e.target.checked)} /> GGUF (llama.cpp)</label>
          <label className="text-sm">Context <Input className="!w-28" type="number" value={ctx} onChange={(e) => setCtx(parseInt(e.target.value || "32768", 10))} /></label>
          <Button variant="primary">Search</Button>
        </form>
        {res.data?.banner && <p className="mt-2 text-xs text-amber-700">{res.data.banner}</p>}
        <ErrorBox error={res.error} />
      </Card>
      <Card title={res.data ? `${res.data.results.length} results · ${fmtBytes(res.data.free_for_new)} free now` : "Results"}>
        <Table head={["Model", "Downloads", "Approx size", "Fit", "Engines", "llmfit", ""]}>
          {(res.data?.results ?? []).map((m: any) => (
            <tr key={m.repo_id}>
              <td className="td break-all font-medium">{m.repo_id} {m.quant && <Badge>{m.quant}</Badge>} {m.gated && <Badge tone="amber">gated</Badge>} {m.downloaded && <Badge tone="green">on disk</Badge>}</td>
              <td className="td tabular-nums">{m.downloads.toLocaleString()}</td>
              <td className="td tabular-nums">{fmtBytes(m.est_bytes)}</td>
              <td className="td"><Badge tone={FIT_TONE[m.fit.badge]}>{FIT_LABEL[m.fit.badge]}</Badge></td>
              <td className="td text-xs">vLLM {m.support.vllm === null ? "?" : m.support.vllm ? "✓" : "✗"} · llama.cpp {m.support.llamacpp ? "✓" : "✗"}</td>
              <td className="td text-xs">{m.llmfit ? <>{m.llmfit.fit_level ?? ""} {m.llmfit.estimated_tps ? `~${m.llmfit.estimated_tps} tok/s` : ""}{m.llmfit.mismatch && <span className="text-amber-700"> ⚠</span>}</> : "—"}</td>
              <td className="td"><Button onClick={() => setOpen(m.repo_id)}>Details</Button></td>
            </tr>
          ))}
        </Table>
      </Card>
      {open && <DetailsModal repo={open} ctx={ctx} admin={admin} onClose={() => setOpen(null)} />}
    </div>
  );
}

function DetailsModal({ repo, ctx, admin, onClose }: { repo: string; ctx: number; admin: boolean; onClose: () => void }) {
  const d = useQuery({ queryKey: ["catalog-details", repo, ctx], queryFn: () => api<any>(`/catalog/details?repo=${encodeURIComponent(repo)}&ctx=${ctx}`) });
  const dl = useMutation({ mutationFn: (include?: string[]) => api("/downloads", { method: "POST", body: { repo_id: repo, include } }) });
  const x = d.data;
  return (
    <Modal open wide title={repo} onClose={onClose}>
      <ErrorBox error={d.error || dl.error} />
      {!x ? <p className="text-sm text-slate-500">Loading…</p> : (
        <div className="space-y-3 text-sm">
          <p>{x.architecture ?? "unknown architecture"} · {x.quant ?? "?"} · max context {x.max_context?.toLocaleString() ?? "?"} {x.gated && <Badge tone="amber">gated: needs an HF token</Badge>}</p>
          {x.estimate && (
            <p>
              vLLM estimate at {ctx.toLocaleString()} context × 8 sequences: <b>{fmtBytes(x.estimate.total)}</b>{" "}
              <Badge tone={FIT_TONE[x.fit.badge]}>{FIT_LABEL[x.fit.badge]}</Badge>
              {x.llmfit && <> · llmfit {x.llmfit.memory_required_gb ?? "?"} GB, {x.llmfit.estimated_tps ?? "?"} tok/s{x.llmfit.mismatch && <span className="text-amber-700"> (estimates differ by over 20%)</span>}</>}
            </p>
          )}
          <p>Engines: vLLM {x.support.vllm === null ? "unknown (registry not loaded yet)" : x.support.vllm ? "supported" : "not supported"} · llama.cpp {x.support.llamacpp ? "GGUF files available" : "no GGUF in this repo"}</p>
          {admin && x.estimate && !x.downloaded && <Button variant="primary" disabled={dl.isPending || dl.isSuccess} onClick={() => dl.mutate(undefined)}>{dl.isSuccess ? "Download queued" : "Download safetensors"}</Button>}
          {x.gguf_files.length > 0 && (
            <Table head={["GGUF file", "Size", "Quant", "Fit", ""]}>
              {x.gguf_files.map((f: any) => (
                <tr key={f.name}>
                  <td className="td break-all font-mono text-xs">{f.name}</td><td className="td">{fmtBytes(f.size)}</td><td className="td">{f.quant}</td>
                  <td className="td"><Badge tone={FIT_TONE[f.fit.badge]}>{FIT_LABEL[f.fit.badge]}</Badge></td>
                  <td className="td">{admin && <Button onClick={() => dl.mutate([f.name])}>Download</Button>}</td>
                </tr>
              ))}
            </Table>
          )}
          {x.variants.length > 0 && (
            <div><h4 className="font-semibold">Other variants</h4><ul className="list-disc pl-5">{x.variants.map((v: any) => <li key={v.repo_id}>{v.repo_id} {v.quant && <Badge>{v.quant}</Badge>} {v.est_bytes && fmtBytes(v.est_bytes)}</li>)}</ul></div>
          )}
        </div>
      )}
    </Modal>
  );
}
