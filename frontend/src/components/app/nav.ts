import type { LucideIcon } from "lucide-react";
import { Download, HardDrive, LayoutDashboard, Network, Puzzle, ScrollText, Search, Server, Users } from "lucide-react";

export interface NavItem { to: string; label: string; icon: LucideIcon; end?: boolean; /** Extra ⌘K search words. */ keywords?: string }
export interface NavGroup { label: string; admin?: boolean; items: NavItem[] }

export const NAV_GROUPS: NavGroup[] = [
  { label: "Operate", items: [
    { to: "/", label: "Dashboard", icon: LayoutDashboard, end: true },
    { to: "/deployments", label: "Deployments", icon: Server },
    { to: "/planner", label: "Planner", icon: Puzzle },
  ] },
  { label: "Models", items: [
    { to: "/models", label: "On disk", icon: HardDrive },
    { to: "/catalog", label: "Catalog", icon: Search },
    { to: "/downloads", label: "Downloads", icon: Download },
  ] },
  { label: "Integrations", admin: true, items: [
    { to: "/litellm", label: "LiteLLM", icon: Network, keywords: "virtual keys api endpoint proxy openai master key" },
  ] },
  { label: "Admin", admin: true, items: [
    { to: "/users", label: "Users", icon: Users },
    { to: "/audit", label: "Audit log", icon: ScrollText },
  ] },
];

export const visibleGroups = (isAdmin: boolean) => NAV_GROUPS.filter((g) => !g.admin || isAdmin);
export const MOBILE_TABS = ["/", "/deployments", "/models"];

export function titleFor(pathname: string): { section: string; item?: string } {
  if (pathname === "/") return { section: "Dashboard" };
  if (pathname.startsWith("/deployments/")) return { section: "Deployments", item: "Deployment" };
  if (pathname === "/deploy") return { section: "Deployments", item: "Deploy a model" };
  if (pathname === "/change-password") return { section: "Account", item: "Change password" };
  for (const g of NAV_GROUPS) for (const i of g.items) if (i.to !== "/" && pathname.startsWith(i.to)) return { section: i.label };
  return { section: "DGXMM" };
}
