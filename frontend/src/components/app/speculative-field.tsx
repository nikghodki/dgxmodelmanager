import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { api } from "@/api";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { DRAFT_METHODS, DRAFT_REASONS } from "@/lib/advice";
import type { Artifact, DraftSuggestion } from "@/types";
import { DecisionChip } from "./decision-chip";

const OFF = "__off__";
const NEEDS_MODEL = ["draft_model", "eagle", "eagle3", "dflash"];
type Spec = { method: string; model?: string | null; num_speculative_tokens?: number } | undefined;

/** vLLM speculative decoding: method, draft model and tokens, with the draft pairing suggestion
 * (POST /decider/draft-suggest): "Suggested draft: X" + Apply; never filled without the click. A method that needs a
 * draft model reaches the config only once a model is chosen (an incomplete setting would fail validation). */
export function SpeculativeField({ modelRef, value, onChange, drafts }: {
  modelRef: string; value: Spec; onChange: (v: Spec) => void; drafts: Artifact[];
}) {
  const [pending, setPending] = useState<string | null>(null); // chosen method still waiting for its draft model
  const method = value?.method ?? pending ?? OFF;
  const needs = NEEDS_MODEL.includes(method);
  const q = useQuery({
    queryKey: ["draft-suggest", modelRef, method],
    queryFn: () => api<DraftSuggestion>("/decider/draft-suggest", { method: "POST", body: { target_ref: modelRef, method } }),
    enabled: !!modelRef && needs, retry: false, staleTime: 60_000, refetchOnWindowFocus: false,
  });
  const s = needs && q.data && !q.error && Array.isArray(q.data.candidates) ? q.data : null;
  const choices = drafts.filter((a) => a.ref !== modelRef && a.complete && (a.kind === "draft" || a.kind === "llm"));
  const set = (patch: Partial<NonNullable<Spec>>) => {
    const next = { num_speculative_tokens: 3, ...(value ?? { method }), ...patch } as NonNullable<Spec>;
    if (NEEDS_MODEL.includes(next.method) && !next.model) {
      setPending(next.method);
      onChange(undefined);
      return;
    }
    setPending(null);
    onChange(next);
  };
  return (
    <div className="space-y-3 rounded-md border p-3" data-speculative>
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
        <div className="space-y-1.5"><Label id="spec-method-label">Speculative decoding</Label>
          <Select value={method} onValueChange={(m) => { if (m === OFF) { setPending(null); onChange(undefined); } else set({ method: m, model: NEEDS_MODEL.includes(m) ? value?.model ?? null : null }); }}>
            <SelectTrigger aria-labelledby="spec-method-label"><SelectValue /></SelectTrigger>
            <SelectContent><SelectItem value={OFF}>Off</SelectItem>{DRAFT_METHODS.map((m) => <SelectItem key={m} value={m}>{m}</SelectItem>)}</SelectContent>
          </Select></div>
        {needs && (
          <div className="space-y-1.5 sm:col-span-2"><Label id="spec-model-label">Draft model</Label>
            <Select value={value?.model ?? ""} onValueChange={(m) => set({ model: m })}>
              <SelectTrigger aria-labelledby="spec-model-label"><SelectValue placeholder="Choose a downloaded model" /></SelectTrigger>
              <SelectContent>{choices.map((a) => <SelectItem key={a.ref} value={a.ref}>{a.ref}</SelectItem>)}</SelectContent>
            </Select></div>
        )}
        {method !== OFF && (
          <div className="space-y-1.5"><Label htmlFor="spec-tokens">Speculative tokens</Label>
            <Input id="spec-tokens" type="number" value={value?.num_speculative_tokens ?? 3} onChange={(e) => { const n = parseInt(e.target.value, 10); set({ num_speculative_tokens: Number.isFinite(n) ? n : 3 }); }} /></div>
        )}
      </div>
      {s && !s.disabled && s.suggested && (
        <div className="flex flex-wrap items-center gap-2 text-xs" data-draft-suggestion={s.suggested}>
          <DecisionChip source={s.source} confidence={s.confidence} prefix={`Suggested draft: ${s.suggested}`}
            title={DRAFT_REASONS[s.candidates.find((c) => c.ref === s.suggested)?.reason ?? ""]} />
          {value?.model !== s.suggested && <Button type="button" variant="outline" size="sm" className="h-6 px-2 text-xs" aria-label={`Apply suggested draft ${s.suggested}`} onClick={() => set({ model: s.suggested })}>Apply</Button>}
        </div>
      )}
      {s && !s.disabled && !s.suggested && s.candidates.length > 0 && (
        <p className="text-xs text-muted-foreground">Compatible drafts on disk: {s.candidates.map((c) => `${c.ref} (${DRAFT_REASONS[c.reason] ?? c.reason})`).join(", ")}</p>
      )}
      {s && !s.disabled && s.reason === "no_compatible_draft" && <p className="text-xs text-muted-foreground">No downloaded model shares this model&apos;s tokenizer or matches this method.</p>}
    </div>
  );
}
