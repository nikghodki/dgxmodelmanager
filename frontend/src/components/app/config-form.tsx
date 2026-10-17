import type { ReactNode } from "react";
import { ENGINE_FIELDS } from "@/engineFields";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";

export const DEFAULT_SENTINEL = "__default__";

export function ConfigForm({ engine, value, onChange, showAdvanced, hints, alwaysShow }: {
  engine: string; value: Record<string, any>; onChange: (v: Record<string, any>) => void; showAdvanced: boolean;
  /** Extra content under a field (e.g. a decision-model suggestion); a hinted advanced field is shown anyway. */
  hints?: Record<string, ReactNode>;
  /** Fields shown even when advanced settings are hidden (e.g. ones an applied fix changes). */
  alwaysShow?: string[];
}) {
  const set = (k: string, v: unknown) => {
    const next = { ...value };
    if (v === "" || v === null || v === DEFAULT_SENTINEL || (typeof v === "number" && Number.isNaN(v))) delete next[k];
    else next[k] = v;
    onChange(next);
  };
  const fields = (ENGINE_FIELDS[engine] ?? []).filter((f) => showAdvanced || !f.advanced || !!hints?.[f.key] || !!alwaysShow?.includes(f.key));
  return (
    <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
      {fields.map((f) => {
        const id = `cfg-${f.key}`;
        if (f.type === "bool") {
          return (
            <div key={f.key} className="flex items-start justify-between gap-3 rounded-md border p-3">
              <div className="space-y-0.5"><Label htmlFor={id}>{f.label}</Label>{f.help && <p className="text-xs text-muted-foreground">{f.help}</p>}</div>
              <Switch id={id} aria-label={f.label} checked={!!value[f.key]} onCheckedChange={(c) => set(f.key, c)} />
            </div>
          );
        }
        return (
          <div key={f.key} className="space-y-1.5">
            <Label htmlFor={id}>{f.label}</Label>
            {f.type === "select" ? (
              <Select value={value[f.key] ? String(value[f.key]) : DEFAULT_SENTINEL} onValueChange={(v) => set(f.key, v)}>
                <SelectTrigger id={id} aria-label={f.label}><SelectValue /></SelectTrigger>
                <SelectContent>
                  {f.options!.map((o) => <SelectItem key={o || DEFAULT_SENTINEL} value={o || DEFAULT_SENTINEL}>{o || "Default"}</SelectItem>)}
                </SelectContent>
              </Select>
            ) : f.type === "int" ? (
              <Input id={id} type="number" inputMode="numeric" value={value[f.key] ?? ""} onChange={(e) => set(f.key, e.target.value === "" ? "" : parseInt(e.target.value, 10))} />
            ) : (
              <Input id={id} value={value[f.key] ?? ""} onChange={(e) => set(f.key, e.target.value)} />
            )}
            {f.help && <p className="text-xs text-muted-foreground">{f.help}</p>}
            {hints?.[f.key]}
          </div>
        );
      })}
    </div>
  );
}
