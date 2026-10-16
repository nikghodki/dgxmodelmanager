import { AlertTriangle, CheckCircle2, XCircle } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { fmtBytes } from "@/format";
import { estimateParts } from "@/lib/estimate";
import { cn } from "@/lib/utils";
import type { Preview } from "@/types";

export function EstimatePanel({ preview }: { preview: Preview }) {
  const e = preview.estimate;
  const f = preview.fit;
  const parts = estimateParts(e);
  const total = e.total || 1;
  return (
    <div className="space-y-4 text-sm">
      <div className={cn("flex items-start gap-2 rounded-md px-3 py-2", f.fits ? "bg-status-good-bg text-status-good" : "bg-status-critical-bg text-status-critical")}>
        {f.fits ? <CheckCircle2 className="mt-0.5 h-4 w-4 shrink-0" /> : <XCircle className="mt-0.5 h-4 w-4 shrink-0" />}
        <div>
          <div className="font-semibold">{f.fits ? "Fits" : "Does not fit"}</div>
          <div className="text-xs tabular-nums">needs {fmtBytes(e.total)} of {fmtBytes(f.free_for_new)} free · {f.fits ? `${fmtBytes(f.headroom)} headroom` : `short by ${fmtBytes(-f.headroom)}`}</div>
        </div>
      </div>
      <div>
        <div className="flex h-2.5 w-full gap-[2px]" role="img" aria-label="Estimate breakdown">
          {parts.filter((p) => p.bytes > 0).map((p, i) => (
            <div key={p.key} className="h-full min-w-[2px] rounded-[3px]" style={{ width: `${(p.bytes / total) * 100}%`, background: `var(--chart-${i + 1})` }} title={`${p.label}: ${fmtBytes(p.bytes)}`} />
          ))}
        </div>
        <dl className="mt-3 space-y-1">
          {parts.map((p, i) => (
            <div key={p.key} className="flex items-center justify-between gap-2 text-xs">
              <dt className="flex items-center gap-1.5 text-muted-foreground"><span aria-hidden className="h-2 w-2 rounded-[2px]" style={{ background: `var(--chart-${i + 1})` }} />{p.label}</dt>
              <dd className="tabular-nums">{fmtBytes(p.bytes)}</dd>
            </div>
          ))}
          {preview.kv_pool_tokens != null && <div className="flex justify-between text-xs"><dt className="text-muted-foreground">KV pool (shared by all sequences)</dt><dd className="tabular-nums">{preview.kv_pool_tokens.toLocaleString()} tokens</dd></div>}
          {preview.gpu_memory_utilization !== undefined && <div className="flex justify-between text-xs"><dt className="text-muted-foreground">vLLM memory share</dt><dd className="font-mono">{preview.gpu_memory_utilization.toFixed(2)}</dd></div>}
          <div className="flex justify-between text-xs"><dt className="text-muted-foreground">Confidence</dt><dd><Badge variant="outline" className="text-[10px]">{e.confidence}</Badge></dd></div>
        </dl>
      </div>
      {e.notes.map((n) => <p key={n} className="flex gap-1.5 text-xs text-status-warning"><AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0" />{n}</p>)}
      {preview.llmfit && (
        <p className="text-xs text-muted-foreground">
          llmfit: {preview.llmfit.fit_level ?? "?"} · ~{preview.llmfit.estimated_tps ?? "?"} tok/s · {preview.llmfit.memory_required_gb ?? "?"} GB
          {preview.llmfit.mismatch && <span className="text-status-warning"> · differs from DGXMM by over 20%</span>}
        </p>
      )}
    </div>
  );
}
