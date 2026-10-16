import type { Theme } from "./theme";
import { deploymentActions } from "@/logic";
import { visibleGroups } from "./nav";

export type CommandAction =
  | { type: "navigate"; to: string }
  | { type: "deployment"; op: "sleep" | "wake" | "stop"; id: number; name: string; state: string }
  | { type: "theme"; theme: Theme }
  | { type: "logout" };

export interface CommandDef {
  id: string;
  group: "Pages" | "Deployments" | "Models" | "Preferences";
  label: string;
  keywords: string;
  action: CommandAction;
  destructive?: boolean;
}

interface Dep { id: number; name: string; state: string; sleep_capable: boolean; model_ref?: string }
interface Art { ref: string; complete: boolean; kind: string; engines: string[] }

export function buildCommands({ isAdmin, deployments, artifacts }: { isAdmin: boolean; deployments: Dep[]; artifacts: Art[] }): CommandDef[] {
  const out: CommandDef[] = [];
  for (const g of visibleGroups(isAdmin)) for (const i of g.items) {
    out.push({ id: `page:${i.to}`, group: "Pages", label: i.label, keywords: `${g.label} page go${i.keywords ? ` ${i.keywords}` : ""}`, action: { type: "navigate", to: i.to } });
  }
  if (isAdmin) out.push({ id: "page:/deploy", group: "Pages", label: "Deploy a model", keywords: "new create launch", action: { type: "navigate", to: "/deploy" } });
  out.push({ id: "page:/change-password", group: "Pages", label: "Change password", keywords: "account", action: { type: "navigate", to: "/change-password" } });

  for (const d of deployments) {
    out.push({ id: `dep:${d.id}:open`, group: "Deployments", label: `Open ${d.name}`, keywords: `${d.model_ref ?? ""} ${d.state}`, action: { type: "navigate", to: `/deployments/${d.id}` } });
    const acts = deploymentActions(d, isAdmin);
    const target = { id: d.id, name: d.name, state: d.state };
    if (acts.includes("sleep")) out.push({ id: `dep:${d.id}:sleep`, group: "Deployments", label: `Sleep ${d.name}`, keywords: "free memory", action: { type: "deployment", op: "sleep", ...target } });
    if (acts.includes("wake")) out.push({ id: `dep:${d.id}:wake`, group: "Deployments", label: `Wake ${d.name}`, keywords: "resume", action: { type: "deployment", op: "wake", ...target } });
    if (acts.includes("stop")) out.push({ id: `dep:${d.id}:stop`, group: "Deployments", label: `Stop ${d.name}`, keywords: "shutdown kill", action: { type: "deployment", op: "stop", ...target }, destructive: true });
  }

  if (isAdmin) for (const a of artifacts) {
    if (!a.complete || a.kind !== "llm" || !a.engines.length) continue;
    out.push({ id: `model:${a.ref}`, group: "Models", label: `Deploy ${a.ref}`, keywords: a.engines.join(" "), action: { type: "navigate", to: `/deploy?model=${encodeURIComponent(a.ref)}` } });
  }

  for (const t of ["light", "dark", "system"] as const) {
    out.push({ id: `theme:${t}`, group: "Preferences", label: `Theme: ${t[0].toUpperCase()}${t.slice(1)}`, keywords: "appearance mode", action: { type: "theme", theme: t } });
  }
  out.push({ id: "logout", group: "Preferences", label: "Log out", keywords: "sign out", action: { type: "logout" } });
  return out;
}
