import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { useNavigate } from "react-router-dom";
import { ArrowRight, Box, LogOut, Moon, Palette, Power, Server } from "lucide-react";
import { api } from "@/api";
import { useAuth, useIsAdmin } from "@/auth";
import { CommandDialog, CommandEmpty, CommandGroup, CommandInput, CommandItem, CommandList } from "@/components/ui/command";
import { useActionMutation } from "@/hooks/use-action-mutation";
import type { Artifact, Deployment } from "@/types";
import { buildCommands, type CommandDef } from "./commands";
import { ConfirmDialog } from "./confirm-dialog";
import { useTheme } from "./theme";

const GROUPS: CommandDef["group"][] = ["Pages", "Deployments", "Models", "Preferences"];

export function CommandPalette({ open, onOpenChange }: { open: boolean; onOpenChange: (o: boolean) => void }) {
  const isAdmin = useIsAdmin();
  const { logout } = useAuth();
  const { setTheme } = useTheme();
  const nav = useNavigate();
  const deps = useQuery({ queryKey: ["deployments"], queryFn: () => api<Deployment[]>("/deployments"), enabled: open });
  const inv = useQuery({ queryKey: ["inventory"], queryFn: () => api<Artifact[]>("/inventory"), enabled: open && isAdmin });
  const [confirmStop, setConfirmStop] = useState<{ id: number; name: string; state: string } | null>(null);
  const op = useActionMutation(({ id, op }: { id: number; op: string }) => api(`/deployments/${id}/${op}`, { method: "POST" }), {
    success: (_d, v) => `${v.op[0].toUpperCase()}${v.op.slice(1)} requested`,
    invalidate: [["deployments"]],
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
      <CommandDialog open={open} onOpenChange={onOpenChange}>
        <CommandInput placeholder="Search pages, deployments, models, actions…" />
        <CommandList>
          <CommandEmpty>No matches.</CommandEmpty>
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
    </>
  );
}
