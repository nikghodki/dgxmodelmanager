import type { Preview } from "../types";
import { fmtBytes } from "../format";
import { Badge } from "./ui";

export function EstimatePanel({ preview }: { preview: Preview }) {
  const e = preview.estimate;
  const rows: [string, number][] = [
    ["Weights", e.weights], ["Draft weights", e.draft_weights], ["KV cache", e.kv_cache], ["Recurrent state", e.state],
    ["Activations", e.activations], ["Runtime (CUDA/graphs)", e.runtime], ["Margin / calibration", e.margin],
  ];
  const f = preview.fit;
  return (
    <div className="space-y-3 text-sm">
      <div className="flex items-center gap-2">
        {f.fits ? <Badge tone="green">Fits</Badge> : <Badge tone="red">Does not fit</Badge>}
        <span>
          needs <b>{fmtBytes(e.total)}</b>, free <b>{fmtBytes(f.free_for_new)}</b>,{" "}
          {f.fits ? <>headroom {fmtBytes(f.headroom)}</> : <>short by {fmtBytes(-f.headroom)}</>}
        </span>
        <Badge tone={e.confidence === "measured" ? "green" : e.confidence === "low" ? "amber" : "slate"}>{e.confidence}</Badge>
      </div>
      <table className="w-full text-xs">
        <tbody>
          {rows.filter(([, v]) => v !== 0).map(([k, v]) => (
            <tr key={k}><td className="py-0.5 text-slate-500">{k}</td><td className="py-0.5 text-right tabular-nums">{fmtBytes(v)}</td></tr>
          ))}
          {preview.kv_pool_tokens != null && (
            <tr><td className="py-0.5 text-slate-500">KV pool tokens (shared by all sequences)</td><td className="py-0.5 text-right tabular-nums">{preview.kv_pool_tokens.toLocaleString()}</td></tr>
          )}
          {preview.gpu_memory_utilization !== undefined && (
            <tr><td className="py-0.5 text-slate-500">vLLM --gpu-memory-utilization</td><td className="py-0.5 text-right tabular-nums">{preview.gpu_memory_utilization.toFixed(2)}</td></tr>
          )}
        </tbody>
      </table>
      {e.notes.map((n) => <p key={n} className="text-xs text-amber-700">{n}</p>)}
      {preview.llmfit && (
        <p className="text-xs text-slate-600">
          llmfit: {preview.llmfit.fit_level ?? "?"} · est. {preview.llmfit.estimated_tps ?? "?"} tok/s · memory{" "}
          {preview.llmfit.memory_required_gb ?? "?"} GB
          {preview.llmfit.mismatch && <span className="ml-1 text-amber-700">(differs from DGXMM by over 20%)</span>}
        </p>
      )}
    </div>
  );
}
