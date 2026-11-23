import type { ReactNode } from "react";
import { ENGINE_FIELDS } from "@/engineFields";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import type { EngineInfo } from "@/types";

export const DEFAULT_SENTINEL = "__default__";

/** An engine's image choice: its default image and the images config.yaml allows instead (images.<engine>_allowed). */
export interface ImageChoice { default: string | null; allowed: string[] }

export function imageChoice(engines: EngineInfo[] | undefined, engine: string): ImageChoice | undefined {
  const e = engines?.find((x) => x.name === engine);
  return e ? { default: e.image, allowed: e.allowed_images ?? [] } : undefined;
}

export function ConfigForm({ engine, value, onChange, showAdvanced, hints, alwaysShow, images }: {
  engine: string; value: Record<string, any>; onChange: (v: Record<string, any>) => void; showAdvanced: boolean;
  /** The engine's image choice: an "Engine image" select when config.yaml allows images besides the default. */
  images?: ImageChoice;
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
  const chosen = typeof value.image === "string" && value.image ? value.image : null;
  // a stored image the allowlist no longer holds stays visible (DGXMM refuses to start it until it is changed)
  const imageOptions = [...(images?.allowed ?? []), ...(chosen && !images?.allowed.includes(chosen) ? [chosen] : [])];
  return (
    <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
      {imageOptions.length > 0 && (
        <div className="space-y-1.5 sm:col-span-2">
          <Label htmlFor="cfg-image">Engine image</Label>
          <Select value={chosen ?? DEFAULT_SENTINEL} onValueChange={(v) => set("image", v)}>
            <SelectTrigger id="cfg-image" aria-label="Engine image"><SelectValue /></SelectTrigger>
            <SelectContent>
              <SelectItem value={DEFAULT_SENTINEL}>{images?.default ? `Default (${images.default})` : "Default"}</SelectItem>
              {imageOptions.map((i) => (
                <SelectItem key={i} value={i}><span className="break-all font-mono text-xs">{i}{images?.allowed.includes(i) ? "" : " (no longer allowed)"}</span></SelectItem>
              ))}
            </SelectContent>
          </Select>
          <p className="text-xs text-muted-foreground">Only images the host allows in config.yaml (images.{engine}_allowed); single-node deployments only.</p>
        </div>
      )}
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
