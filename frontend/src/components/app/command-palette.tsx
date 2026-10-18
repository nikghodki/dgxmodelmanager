import { useEffect, useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { useNavigate } from "react-router-dom";
import { useCommandState } from "cmdk";
import { ArrowRight, Box, Loader2, LogOut, Moon, Palette, Power, Server, Sparkles } from "lucide-react";
import { api } from "@/api";
import { useAuth, useIsAdmin } from "@/auth";
import { CommandDialog, CommandGroup, CommandInput, CommandItem, CommandList } from "@/components/ui/command";
import { useActionMutation } from "@/hooks/use-action-mutation";
import { fmtConfidence } from "@/lib/decider";
import { apiErrorText, nameFixHref, takenHolder } from "@/lib/litellm";
import type { Artifact, CommandInterpretation, CommandResult, Deployment } from "@/types";
import { InterpretationDialog, interpretationHref, interpretCommand } from "./command-ask";
import { buildCommands, type CommandDef } from "./commands";
import { ConfirmDialog } from "./confirm-dialog";
import { useTheme } from "./theme";

// Values of the Ask items: invisible characters only, so the palette's fuzzy filter never matches (or counts) them.
const ASK_VALUE = "\u2063";
const interpValue = (i: number) => "\u2063".repeat(i + 2);
type Ask = { text: string; status: "loading" } | { text: string; status: "done"; result: CommandResult } | { text: string; status: "error"; error: string };

/** "Interpret: <text>" when no palette entry matches; then the interpretations with their confidence. */
function AskSection({ search, ask, onInterpret, onPick, select }: {
  search: string; ask: Ask | null; onInterpret: () => void; onPick: (i: CommandInterpretation) => void; select: (v: string) => void;
}) {
  const matches = useCommandState((s) => s.filtered.count);
  const text = search.trim();
  const show = !!text && matches === 0;
  const current = ask && ask.text === text ? ask : null;
  const first = current?.status === "done" && current.result.interpretations.length ? interpValue(0) : ASK_VALUE;
  useEffect(() => { if (show) select(first); }, [show, first]); // eslint-disable-line react-hooks/exhaustive-deps
  if (!show) return null;
  if (current?.status === "done" && current.result.interpretations.length) {
    return (
      <CommandGroup heading="Interpretations — nothing runs until you confirm" forceMount>
        {current.result.interpretations.map((i, n) => (
          <CommandItem key={n} value={interpValue(n)} forceMount onSelect={() => onPick(i)}
            aria-label={`${i.label}, confidence ${fmtConfidence(i.confidence)}`}>
            <Sparkles aria-hidden className="mr-2 h-4 w-4" /><span className="min-w-0 flex-1 break-words">{i.label}</span>
            <span className="ml-2 shrink-0 text-xs tabular-nums text-muted-foreground">{fmtConfidence(i.confidence)}</span>
          </CommandItem>
        ))}
      </CommandGroup>
    );
  }
  const busy = current?.status === "loading";
  const note = current?.status === "done" ? current.result.message ?? "Couldn't interpret that." : current?.status === "error" ? current.error : null;
  return (
    <CommandGroup heading="Ask" forceMount>
      <CommandItem value={ASK_VALUE} forceMount disabled={busy} onSelect={onInterpret}>
        {busy ? <Loader2 aria-hidden className="mr-2 h-4 w-4 animate-spin" /> : <Sparkles aria-hidden className="mr-2 h-4 w-4" />}
        <span className="min-w-0 break-words">{busy ? "Interpreting…" : <>Interpret: <span className="font-medium">{text}</span></>}</span>
      </CommandItem>
      {note && <p className="px-2 py-2 text-xs text-muted-foreground">{note}</p>}
    </CommandGroup>
  );
}

const GROUPS: CommandDef["group"][] = ["Pages", "Deployments", "Models", "Preferences"];

export function CommandPalette({ open, onOpenChange }: { open: boolean; onOpenChange: (o: boolean) => void }) {
  const isAdmin = useIsAdmin();
  const { logout } = useAuth();
  const { setTheme } = useTheme();
  const nav = useNavigate();
  const deps = useQuery({ queryKey: ["deployments"], queryFn: () => api<Deployment[]>("/deployments"), enabled: open });
  const inv = useQuery({ queryKey: ["inventory"], queryFn: () => api<Artifact[]>("/inventory"), enabled: open && isAdmin });
  const [confirmStop, setConfirmStop] = useState<{ id: number; name: string; state: string } | null>(null);
  const [search, setSearch] = useState("");
  const [selected, setSelected] = useState("");
  const [ask, setAsk] = useState<Ask | null>(null);
  const [picked, setPicked] = useState<CommandInterpretation | null>(null);
  useEffect(() => { if (!open) { setSearch(""); setAsk(null); } }, [open]);
  const interpret = async () => {
    const text = search.trim();
    if (!text || text.length > 300) {
      setAsk({ text, status: "error", error: "Commands are at most 300 characters." });
      return;
    }
    setAsk({ text, status: "loading" });
    try {
      setAsk({ text, status: "done", result: await interpretCommand(text) });
    } catch (e) {
      setAsk({ text, status: "error", error: apiErrorText(e) });
    }
  };
  const pick = (i: CommandInterpretation) => {
    onOpenChange(false);
    const href = interpretationHref(i);
    if (href) nav(href);
    else setPicked(i);
  };
  const announce = ask?.status === "loading" ? "Interpreting…" : ask?.status === "error" ? ask.error
    : ask?.status === "done" ? (ask.result.interpretations.length ? `${ask.result.interpretations.length} interpretation${ask.result.interpretations.length > 1 ? "s" : ""}: ${ask.result.interpretations.map((i) => `${i.label}, ${fmtConfidence(i.confidence)}`).join("; ")}` : ask.result.message ?? "Couldn't interpret that.") : "";
  const op = useActionMutation(({ id, op }: { id: number; op: string }) => api(`/deployments/${id}/${op}`, { method: "POST" }), {
    success: (_d, v) => `${v.op[0].toUpperCase()}${v.op.slice(1)} requested`,
    invalidate: [["deployments"]],
    errorAction: (err, v) => {
      const d = deps.data?.find((x) => x.id === v.id);
      return !!d && !!takenHolder(err) && { label: "Set model name", onClick: () => nav(nameFixHref(d)) };
    },
  });
  const commands = useMemo(
    () => buildCommands({ isAdmin, deployments: deps.data ?? [], artifacts: inv.data ?? [] }),
    [isAdmin, deps.data, inv.data],
  );

  const run = (c: CommandDef) => {
    onOpenChange(false);
    const a = c.action;
    if (a.type === "navigate") nav(a.to);
    else if (a.type === "theme") setTheme(a.theme);
    else if (a.type === "logout") logout();
    else if (a.op === "stop") setConfirmStop({ id: a.id, name: a.name, state: a.state });
    else op.mutate({ id: a.id, op: a.op });
  };

  const icon = (c: CommandDef) => {
    if (c.action.type === "theme") return <Palette className="mr-2 h-4 w-4" />;
    if (c.action.type === "logout") return <LogOut className="mr-2 h-4 w-4" />;
    if (c.action.type === "deployment") return c.action.op === "stop" ? <Power className="mr-2 h-4 w-4" /> : <Moon className="mr-2 h-4 w-4" />;
    if (c.group === "Deployments") return <Server className="mr-2 h-4 w-4" />;
    if (c.group === "Models") return <Box className="mr-2 h-4 w-4" />;
    return <ArrowRight className="mr-2 h-4 w-4" />;
  };

  return (
    <>
      <CommandDialog open={open} onOpenChange={onOpenChange} commandProps={{ value: selected, onValueChange: setSelected }}>
        <CommandInput placeholder="Search pages, deployments, models, actions… or type a command" value={search} onValueChange={setSearch} maxLength={300} />
        <p aria-live="polite" className="sr-only">{announce}</p>
        <CommandList>
          <AskSection search={search} ask={ask} onInterpret={interpret} onPick={pick} select={setSelected} />
          {GROUPS.map((g) => {
            const items = commands.filter((c) => c.group === g);
            if (!items.length) return null;
            return (
              <CommandGroup key={g} heading={g}>
                {items.map((c) => (
                  <CommandItem key={c.id} value={`${c.label} ${c.keywords}`} onSelect={() => run(c)}
                    className={c.destructive ? "text-status-critical aria-selected:text-status-critical" : undefined}>
                    {icon(c)}{c.label}
                  </CommandItem>
                ))}
              </CommandGroup>
            );
          })}
        </CommandList>
      </CommandDialog>
      <ConfirmDialog open={!!confirmStop} onOpenChange={(o) => !o && setConfirmStop(null)}
        title={`Stop ${confirmStop?.name}?`} description="The container is stopped and its memory released. You can start it again later."
        confirmLabel="Stop" destructive pending={op.isPending}
        requireText={confirmStop?.state === "ready" ? confirmStop.name : undefined}
        onConfirm={() => { if (confirmStop) op.mutate({ id: confirmStop.id, op: "stop" }, { onSettled: () => setConfirmStop(null) }); }} />
      {picked && <InterpretationDialog interp={picked} onClose={() => setPicked(null)} />}
    </>
  );
}
