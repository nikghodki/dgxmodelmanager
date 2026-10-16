import { useEffect, useMemo, useState, type ReactNode } from "react";
import type { ColumnDef } from "@tanstack/react-table";
import { AlertTriangle, KeyRound, Loader2, MoreHorizontal, Pencil, Plus, Trash2 } from "lucide-react";
import { api } from "@/api";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from "@/components/ui/dropdown-menu";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { useActionMutation } from "@/hooks/use-action-mutation";
import {
  buildKeyCreateBody, buildKeyPatchBody, clearedLimits, curlSnippet, emptyKeyForm, fmtDate, fmtLimits, fmtSpend, isAllModels,
  keyFormFromKey, keyFormProblems, litellmProblem, type KeyForm,
} from "@/lib/litellm";
import type { LiteLLMCreatedKey, LiteLLMKey } from "@/types";
import { ConfirmDialog } from "./confirm-dialog";
import { CopyButton } from "./copy-button";
import { DataTable } from "./data-table";
import { EmptyState } from "./states";
import { ResponsiveDialog } from "./responsive-dialog";

/** A read-only code sample with a copy button. */
export function CodeBlock({ code, label }: { code: string; label: string }) {
  return (
    <div className="relative min-w-0">
      {/* pt-9: the copy button sits in the top padding, so it never covers the first (often long) line */}
      <pre aria-label={label} className="overflow-x-auto whitespace-pre rounded-md border bg-muted/50 p-3 pt-9 font-mono text-xs leading-relaxed">{code}</pre>
      <div className="absolute right-1.5 top-1.5"><CopyButton value={code} label={`Copy ${label}`} /></div>
    </div>
  );
}

export const keyLabel = (k: LiteLLMKey) => k.key_alias || k.key_name || k.token.slice(0, 12);

export function ModelBadges({ models }: { models: string[] }) {
  if (isAllModels(models)) return <Badge variant="secondary">All models</Badge>;
  return <span className="flex flex-wrap gap-1">{models.map((m) => <Badge key={m} variant="outline" className="font-mono text-[11px] font-normal">{m}</Badge>)}</span>;
}

function Field({ id, label, help, error, children }: { id: string; label: string; help?: ReactNode; error?: string; children: ReactNode }) {
  return (
    <div className="space-y-1.5">
      <Label htmlFor={id}>{label}</Label>
      {children}
      {error ? <p id={`${id}-err`} className="text-xs text-status-critical">{error}</p> : help && <p className="text-xs text-muted-foreground">{help}</p>}
    </div>
  );
}

function ModelPicker({ options, form, setForm, error }: { options: string[]; form: KeyForm; setForm: (f: KeyForm) => void; error?: string }) {
  const all = [...new Set([...options, ...form.models])].sort();
  const toggle = (m: string, on: boolean) => setForm({ ...form, models: on ? [...form.models, m] : form.models.filter((x) => x !== m) });
  return (
    <fieldset className="space-y-2" aria-describedby={error ? "key-models-err" : undefined}>
      <legend className="text-sm font-medium leading-none">Models</legend>
      <div className="flex items-center gap-2">
        <Checkbox id="key-all-models" checked={form.allModels} onCheckedChange={(v) => setForm({ ...form, allModels: v === true })} />
        <Label htmlFor="key-all-models" className="font-normal">All models (including ones added later)</Label>
      </div>
      <div className="max-h-44 space-y-1.5 overflow-y-auto rounded-md border p-2">
        {all.length === 0 && <p className="text-xs text-muted-foreground">No DGXMM model names yet.</p>}
        {all.map((m, i) => (
          <div key={m} className="flex items-center gap-2">
            <Checkbox id={`key-model-${i}`} disabled={form.allModels} checked={form.allModels || form.models.includes(m)}
              onCheckedChange={(v) => toggle(m, v === true)} />
            <Label htmlFor={`key-model-${i}`} className="break-all font-mono text-xs font-normal">{m}</Label>
          </div>
        ))}
      </div>
      {error && <p id="key-models-err" className="text-xs text-status-critical">{error}</p>}
    </fieldset>
  );
}

/** Create or edit a virtual key. Validation mirrors the backend so most mistakes never reach LiteLLM. */
export function KeyFormDialog({ open, onOpenChange, editing, modelOptions, preselect, pending, error, onSubmit }: {
  open: boolean; onOpenChange: (o: boolean) => void; editing: LiteLLMKey | null; modelOptions: string[]; preselect?: string[];
  pending: boolean; error: unknown; onSubmit: (body: Record<string, unknown>) => void;
}) {
  const mode = editing ? "edit" : "create";
  const [form, setForm] = useState<KeyForm>(emptyKeyForm(preselect));
  const [touched, setTouched] = useState(false);
  useEffect(() => {
    if (open) { setForm(editing ? keyFormFromKey(editing) : emptyKeyForm(preselect)); setTouched(false); }
  }, [open, editing]); // eslint-disable-line react-hooks/exhaustive-deps
  const problems = keyFormProblems(form, mode, editing);
  const show = (k: keyof KeyForm) => (touched || (form[k] as unknown) !== "" ? problems[k] : undefined);
  const patch = editing ? buildKeyPatchBody(editing, form) : null;
  const cleared = editing ? clearedLimits(editing, form) : [];
  const invalid = Object.keys(problems).length > 0;
  const set = (k: keyof KeyForm) => (e: React.ChangeEvent<HTMLInputElement>) => setForm({ ...form, [k]: e.target.value });
  const submit = (e: React.FormEvent) => {
    e.preventDefault();
    setTouched(true);
    if (invalid) return;
    onSubmit(editing ? patch! : buildKeyCreateBody(form));
  };
  return (
    <ResponsiveDialog open={open} onOpenChange={onOpenChange} className="max-w-2xl"
      title={editing ? `Edit key ${keyLabel(editing)}` : "Create virtual key"}
      description={editing ? "Changes apply immediately to clients using this key." : "The full key is shown once after creation."}>
      <form className="space-y-4" onSubmit={submit} noValidate>
        <Field id="key-alias" label="Alias" help="A name to recognise the key by, e.g. team-a or ci-bot." error={show("alias")}>
          <Input id="key-alias" autoComplete="off" value={form.alias} onChange={set("alias")} aria-invalid={!!show("alias")} />
        </Field>
        <ModelPicker options={modelOptions} form={form} setForm={setForm} error={touched ? problems.models : undefined} />
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
          <Field id="key-budget" label="Max budget (USD)" help="Empty: no budget." error={show("maxBudget")}>
            <Input id="key-budget" inputMode="decimal" value={form.maxBudget} onChange={set("maxBudget")} aria-invalid={!!show("maxBudget")} />
          </Field>
          <Field id="key-budget-duration" label="Budget resets every" help="e.g. 30d, 1w, 1mo. Empty: never resets." error={show("budgetDuration")}>
            <Input id="key-budget-duration" autoComplete="off" value={form.budgetDuration} onChange={set("budgetDuration")} aria-invalid={!!show("budgetDuration")} />
          </Field>
          <Field id="key-rpm" label="Requests per minute" help="Empty: unlimited." error={show("rpm")}>
            <Input id="key-rpm" inputMode="numeric" value={form.rpm} onChange={set("rpm")} aria-invalid={!!show("rpm")} />
          </Field>
          <Field id="key-tpm" label="Tokens per minute" help="Empty: unlimited." error={show("tpm")}>
            <Input id="key-tpm" inputMode="numeric" value={form.tpm} onChange={set("tpm")} aria-invalid={!!show("tpm")} />
          </Field>
          {!editing && (
            <Field id="key-duration" label="Expires after" help="e.g. 30d or 12h. Empty: never expires." error={show("duration")}>
              <Input id="key-duration" autoComplete="off" value={form.duration} onChange={set("duration")} aria-invalid={!!show("duration")} />
            </Field>
          )}
        </div>
        {editing && (
          <div className="flex items-center justify-between gap-3 rounded-md border p-3">
            <div><Label htmlFor="key-blocked">Blocked</Label><p className="text-xs text-muted-foreground">A blocked key is rejected by LiteLLM but kept.</p></div>
            <Switch id="key-blocked" checked={form.blocked} onCheckedChange={(v) => setForm({ ...form, blocked: v })} />
          </div>
        )}
        {cleared.length > 0 && (
          <p className="flex items-start gap-2 text-xs text-status-warning"><AlertTriangle aria-hidden className="mt-0.5 h-3.5 w-3.5 shrink-0" />
            LiteLLM cannot remove a limit once set, so the {cleared.join(", ")} will stay as before. Enter a new value to change it.</p>
        )}
        {!!error && <p role="alert" className="text-xs text-status-critical">{litellmProblem(error).description}</p>}
        <div className="flex flex-col-reverse gap-2 sm:flex-row sm:justify-end">
          <Button type="button" variant="outline" onClick={() => onOpenChange(false)}>Cancel</Button>
          <Button type="submit" disabled={pending || (touched && invalid) || (!!editing && Object.keys(patch!).length === 0)}>
            {pending && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}{editing ? "Save changes" : "Create key"}
          </Button>
        </div>
      </form>
    </ResponsiveDialog>
  );
}

/** Shows a newly created key exactly once. The parent drops the key from memory when this closes. */
export function CreatedKeyDialog({ created, endpoint, onClose }: { created: LiteLLMCreatedKey | null; endpoint: string; onClose: () => void }) {
  const model = created && !isAllModels(created.models) ? created.models[0] : "<model name>";
  const [copied, setCopied] = useState(false);
  const [confirming, setConfirming] = useState(false);
  useEffect(() => { setCopied(false); setConfirming(false); }, [created]);
  // Closing loses the key for good: only Done / the close button close it, and only after a copy or a confirmation.
  const requestClose = () => (copied ? onClose() : setConfirming(true));
  return (
    <ResponsiveDialog open={!!created} onOpenChange={(o) => !o && requestClose()} onDismissAttempt={requestClose} className="max-w-2xl"
      title={`Key created${created?.key_alias ? `: ${created.key_alias}` : ""}`} description="Copy the key now and store it somewhere safe.">
      {created && (
        <div className="min-w-0 space-y-4">
          <p role="alert" className="flex items-start gap-2 rounded-md border border-status-warning/30 bg-status-warning-bg px-3 py-2 text-sm text-status-warning">
            <AlertTriangle aria-hidden className="mt-0.5 h-4 w-4 shrink-0" />This is the only time the full key is shown. Neither DGXMM nor LiteLLM can show it again.
          </p>
          <div className="space-y-1.5">
            <Label htmlFor="created-key">Virtual key</Label>
            <div className="flex items-center gap-1">
              <Input id="created-key" readOnly value={created.key} className="font-mono text-xs" onFocus={(e) => e.target.select()} />
              <CopyButton value={created.key} label="Copy key" onCopied={() => { setCopied(true); setConfirming(false); }} />
            </div>
          </div>
          <div className="space-y-1.5">
            <p className="text-sm font-medium">Try it</p>
            <CodeBlock code={curlSnippet(endpoint, model, created.key)} label="curl example" />
          </div>
          {confirming ? (
            <div role="alertdialog" aria-label="Close without copying?" className="flex flex-col gap-2 rounded-md border border-status-warning/30 p-3 sm:flex-row sm:items-center sm:justify-between">
              <p className="text-sm">Close without copying? The key cannot be shown again.</p>
              <div className="flex gap-2">
                <Button variant="outline" size="sm" onClick={() => setConfirming(false)}>Keep open</Button>
                <Button variant="destructive" size="sm" onClick={onClose}>Close anyway</Button>
              </div>
            </div>
          ) : <div className="flex justify-end"><Button onClick={requestClose}>Done</Button></div>}
        </div>
      )}
    </ResponsiveDialog>
  );
}

export function KeysTable({ keys, loading, empty, onEdit, onRevoke }: {
  keys: LiteLLMKey[]; loading?: boolean; empty?: ReactNode; onEdit?: (k: LiteLLMKey) => void; onRevoke?: (k: LiteLLMKey) => void;
}) {
  const columns = useMemo<ColumnDef<LiteLLMKey, any>[]>(() => [
    { id: "alias", header: "Alias", accessorFn: (k) => keyLabel(k), cell: ({ row }) => (
      <span className="flex flex-wrap items-center gap-1.5 font-medium">{row.original.key_alias ?? <span className="text-muted-foreground">(no alias)</span>}
        {row.original.blocked && <Badge variant="outline" className="text-status-critical">blocked</Badge>}</span>) },
    { id: "key", header: "Key", accessorFn: (k) => k.key_name ?? "", meta: { mono: true }, enableSorting: false },
    { id: "models", header: "Models", enableSorting: false, cell: ({ row }) => <ModelBadges models={row.original.models} /> },
    { id: "spend", header: "Spend / budget", accessorFn: (k) => k.spend ?? 0, cell: ({ row }) => <span className="whitespace-nowrap">{fmtSpend(row.original)}</span> },
    { id: "limits", header: "Rate limits", enableSorting: false, meta: { hideOnMobile: true }, cell: ({ row }) => <span className="whitespace-nowrap">{fmtLimits(row.original)}</span> },
    { id: "expires", header: "Expires", accessorFn: (k) => k.expires ?? "", cell: ({ row }) => (row.original.expires ? fmtDate(row.original.expires) : "Never") },
    { id: "created", header: "Created", accessorFn: (k) => k.created_at ?? "", meta: { hideOnMobile: true }, cell: ({ row }) => fmtDate(row.original.created_at) },
  ], []);
  const actions = onEdit || onRevoke ? (k: LiteLLMKey) => (
    <DropdownMenu>
      <DropdownMenuTrigger asChild><Button variant="ghost" size="icon" className="h-8 w-8" aria-label={`Actions for key ${keyLabel(k)}`}><MoreHorizontal className="h-4 w-4" /></Button></DropdownMenuTrigger>
      <DropdownMenuContent align="end">
        {onEdit && <DropdownMenuItem onSelect={() => onEdit(k)}><Pencil className="mr-2 h-4 w-4" />Edit</DropdownMenuItem>}
        {onRevoke && <DropdownMenuItem className="text-status-critical focus:text-status-critical" onSelect={() => onRevoke(k)}><Trash2 className="mr-2 h-4 w-4" />Revoke</DropdownMenuItem>}
      </DropdownMenuContent>
    </DropdownMenu>
  ) : undefined;
  return (
    <DataTable columns={columns} data={keys} loading={loading} getRowId={(k) => k.token} rowActions={actions}
      searchPlaceholder={keys.length > 5 ? "Search keys" : undefined}
      searchText={(k) => `${k.key_alias ?? ""} ${k.key_name ?? ""} ${k.models.join(" ")}`}
      empty={empty ?? <EmptyState icon={KeyRound} title="No virtual keys" description="Create a key to give an app or a person access through LiteLLM." />} />
  );
}

/**
 * Keys table plus the create / created / edit / revoke dialogs. Admin-only: callers render it only for admins.
 * The full key from a create lives only in `created` (and the mutation result, reset on close).
 */
export function KeyManager({ keys, loading, empty, modelOptions, endpoint, preselect, createLabel = "Create key", problem }: {
  keys: LiteLLMKey[]; loading?: boolean; empty?: ReactNode; modelOptions: string[]; endpoint: string; preselect?: string[]; createLabel?: string;
  /** Shown instead of the table (and the create button) when keys cannot be listed; the dialogs stay mounted. */
  problem?: ReactNode;
}) {
  const [formOpen, setFormOpen] = useState(false);
  const [editing, setEditing] = useState<LiteLLMKey | null>(null);
  const [created, setCreated] = useState<LiteLLMCreatedKey | null>(null);
  const [revoking, setRevoking] = useState<LiteLLMKey | null>(null);
  const inv = [["litellm-keys"], ["deployment-litellm"]];
  const create = useActionMutation((body: Record<string, unknown>) => api<LiteLLMCreatedKey>("/integrations/litellm/keys", { method: "POST", body }), {
    success: "Key created", invalidate: inv, gcTime: 0, onSuccess: (k) => { setFormOpen(false); setCreated(k); },
  });
  const update = useActionMutation(({ token, body }: { token: string; body: Record<string, unknown> }) =>
    api<LiteLLMKey>(`/integrations/litellm/keys/${token}`, { method: "PATCH", body }), {
    success: "Key updated", invalidate: inv, onSuccess: () => { setFormOpen(false); setEditing(null); },
  });
  const revoke = useActionMutation((token: string) => api(`/integrations/litellm/keys/${token}`, { method: "DELETE" }), {
    success: "Key revoked", invalidate: inv, onSuccess: () => setRevoking(null),
  });
  const openForm = (k: LiteLLMKey | null) => { create.reset(); update.reset(); setEditing(k); setFormOpen(true); };
  const closeCreated = () => { setCreated(null); create.reset(); };
  return (
    <div className="space-y-3">
      {problem ?? (
        <>
          <div className="flex justify-end"><Button size="sm" onClick={() => openForm(null)}><Plus className="mr-1 h-4 w-4" />{createLabel}</Button></div>
          <KeysTable keys={keys} loading={loading} empty={empty} onEdit={(k) => openForm(k)} onRevoke={setRevoking} />
        </>
      )}
      <KeyFormDialog open={formOpen} onOpenChange={(o) => { setFormOpen(o); if (!o) setEditing(null); }} editing={editing}
        modelOptions={modelOptions} preselect={preselect} pending={create.isPending || update.isPending} error={editing ? update.error : create.error}
        onSubmit={(body) => (editing ? update.mutate({ token: editing.token, body }) : create.mutate(body))} />
      <CreatedKeyDialog created={created} endpoint={endpoint} onClose={closeCreated} />
      <ConfirmDialog open={!!revoking} onOpenChange={(o) => !o && setRevoking(null)} title={`Revoke key ${revoking ? keyLabel(revoking) : ""}?`}
        description="Apps using this key stop working immediately. This cannot be undone." confirmLabel="Revoke key" destructive
        requireText={revoking ? keyLabel(revoking) : undefined} pending={revoke.isPending} onConfirm={() => revoking && revoke.mutate(revoking.token)} />
    </div>
  );
}
