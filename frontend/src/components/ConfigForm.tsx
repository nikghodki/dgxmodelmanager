import { ENGINE_FIELDS } from "../engineFields";
import { Field, Input, Select } from "./ui";

export function ConfigForm({ engine, value, onChange, showAdvanced }: {
  engine: string; value: Record<string, any>; onChange: (v: Record<string, any>) => void; showAdvanced: boolean;
}) {
  const set = (k: string, v: unknown) => {
    const next = { ...value };
    if (v === "" || v === null || (typeof v === "number" && Number.isNaN(v))) delete next[k];
    else next[k] = v;
    onChange(next);
  };
  return (
    <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
      {(ENGINE_FIELDS[engine] ?? []).filter((f) => showAdvanced || !f.advanced).map((f) => (
        <Field key={f.key} label={f.label} help={f.help}>
          {f.type === "bool" ? (
            <input type="checkbox" checked={!!value[f.key]} onChange={(e) => set(f.key, e.target.checked)} />
          ) : f.type === "select" ? (
            <Select value={value[f.key] ?? ""} onChange={(e) => set(f.key, e.target.value)}>
              {f.options!.map((o) => <option key={o} value={o}>{o || "(default)"}</option>)}
            </Select>
          ) : f.type === "int" ? (
            <Input type="number" value={value[f.key] ?? ""} onChange={(e) => set(f.key, e.target.value === "" ? "" : parseInt(e.target.value, 10))} />
          ) : (
            <Input value={value[f.key] ?? ""} onChange={(e) => set(f.key, e.target.value)} />
          )}
        </Field>
      ))}
    </div>
  );
}
