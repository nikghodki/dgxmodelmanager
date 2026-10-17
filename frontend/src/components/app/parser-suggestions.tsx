import { useEffect, useState, type Dispatch, type ReactNode, type SetStateAction } from "react";
import { useQuery } from "@tanstack/react-query";
import { api } from "@/api";
import { Button } from "@/components/ui/button";
import { ENGINE_FIELDS } from "@/engineFields";
import { parserValueLabel, suggestionState } from "@/lib/decider";
import type { ConfigSuggestions, Suggestion } from "@/types";
import { DecisionChip } from "./decision-chip";

export const PARSER_KEYS = ["tool_call_parser", "reasoning_parser"] as const;
type ParserKey = (typeof PARSER_KEYS)[number];
type Config = Record<string, any>;

interface Track { key: string; touched: string[]; filled: Record<string, string>; dismissed: string[] }
const fresh = (key: string): Track => ({ key, touched: [], filled: {}, dismissed: [] });

/**
 * Decision-model parser suggestions for a model + engine (POST /decider/suggest-config).
 * - auto-fills an empty field the user has not touched when the server says `auto_apply` (rules, or model ≥ threshold);
 * - otherwise offers an explicit Apply; never overwrites a value the user typed;
 * - silent (no hints) when the endpoint errors, returns null, or is disabled.
 * Call `noteChange(prev, next)` from the form's onChange so user edits count as "touched".
 */
export function useParserSuggestions({ modelRef, engine, enabled = true, config, setConfig, onUserEdit }: {
  modelRef: string; engine: string; enabled?: boolean; config: Config; setConfig: Dispatch<SetStateAction<Config>>;
  /** Called after an explicit Apply/undo (e.g. to drop a chosen plan); not for auto-fills, which keep the plan. */
  onUserEdit?: () => void;
}): { hints: Record<string, ReactNode>; noteChange: (prev: Config, next: Config) => void } {
  const fields = (ENGINE_FIELDS[engine] ?? []).filter((f) => (PARSER_KEYS as readonly string[]).includes(f.key));
  const on = enabled && !!modelRef && !!engine && fields.length > 0;
  const q = useQuery({
    queryKey: ["suggest-config", modelRef, engine],
    queryFn: () => api<ConfigSuggestions | null>("/decider/suggest-config", { method: "POST", body: { model_ref: modelRef, engine } }),
    enabled: on, retry: false, staleTime: 5 * 60_000, refetchOnWindowFocus: false,
  });
  const scope = `${modelRef}|${engine}`;
  const [raw, setTrack] = useState<Track>(() => fresh(scope));
  const track = raw.key === scope ? raw : fresh(scope); // a new model/engine starts clean
  const update = (fn: (t: Track) => Track) => setTrack((t) => fn(t.key === scope ? t : fresh(scope)));
  const data = on && !q.error ? q.data ?? null : null;
  const sugg = (k: ParserKey): Suggestion | null => data?.[k] ?? null;
  // "autoFilled" only while the field still holds our value: if something else (e.g. the wizard re-applying engine
  // defaults) cleared it without a user edit, it may be filled again.
  const flags = (k: string) => ({ touched: track.touched.includes(k) || track.dismissed.includes(k), autoFilled: k in track.filled && config[k] === track.filled[k] });

  // Auto-fill (functional update: the field must still be empty when it lands).
  useEffect(() => {
    if (!data) return;
    for (const f of fields) {
      const s = sugg(f.key as ParserKey);
      if (!s || s.value === null || suggestionState(s, config[f.key], flags(f.key)) !== "autofill") continue;
      const value = s.value;
      setConfig((c) => (c[f.key] == null || c[f.key] === "" ? { ...c, [f.key]: value } : c));
      update((t) => ({ ...t, filled: { ...t.filled, [f.key]: value } }));
    }
  }); // eslint-disable-line react-hooks/exhaustive-deps -- cheap; guarded by suggestionState

  const setValue = (k: string, v: string | null) => {
    setConfig((c) => {
      const next = { ...c };
      if (v === null) delete next[k]; else next[k] = v;
      return next;
    });
    onUserEdit?.();
  };

  const hints: Record<string, ReactNode> = {};
  for (const f of fields) {
    const s = sugg(f.key as ParserKey);
    if (!s || s.source === "none") continue;
    const st = suggestionState(s, config[f.key], flags(f.key));
    if (st === "matches" && s.value === null) continue; // "Suggested: none" on an empty field says nothing (and must not un-hide it)
    const shown = parserValueLabel(s.value);
    hints[f.key] = (
      <div className="flex flex-wrap items-center gap-x-2 gap-y-1 text-xs" data-suggestion={f.key}>
        <DecisionChip source={s.source} confidence={s.confidence} prefix={`Suggested: ${shown}`}
          title={s.source === "model" ? "Suggested by the decision model" : "From DGXMM's known model families"} />
        {st === "filled" && (
          <span role="status" aria-live="polite" className="text-muted-foreground">Filled automatically —{" "}
            <Button type="button" variant="link" className="h-auto p-0 text-xs" aria-label={`Undo: clear the automatic ${f.label}`}
              onClick={() => { setValue(f.key, null); update((t) => { const { [f.key]: _gone, ...filled } = t.filled; return { ...t, dismissed: [...t.dismissed, f.key], filled }; }); }}>undo</Button>
          </span>
        )}
        {st === "apply" && (
          <Button type="button" variant="outline" size="sm" className="h-6 px-2 text-xs" aria-label={`Apply suggested ${f.label}: ${shown}`}
            onClick={() => { setValue(f.key, s.value); update((t) => ({ ...t, touched: [...t.touched, f.key] })); }}>Apply</Button>
        )}
      </div>
    );
  }

  const noteChange = (prev: Config, next: Config) => {
    const changed = PARSER_KEYS.filter((k) => (prev[k] ?? "") !== (next[k] ?? ""));
    if (changed.length) update((t) => ({ ...t, touched: [...new Set([...t.touched, ...changed])] }));
  };
  return { hints, noteChange };
}
