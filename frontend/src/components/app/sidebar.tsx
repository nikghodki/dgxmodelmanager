import { NavLink } from "react-router-dom";
import { Cpu, PanelLeftClose, PanelLeftOpen } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";
import { visibleGroups } from "./nav";

export function Sidebar({ isAdmin, collapsed, onToggle }: { isAdmin: boolean; collapsed: boolean; onToggle: () => void }) {
  return (
    <aside className={cn("sticky top-0 hidden h-dvh shrink-0 flex-col border-r border-sidebar-border bg-sidebar md:flex", collapsed ? "w-16" : "w-60")}>
      <div className={cn("flex h-14 items-center gap-2 border-b border-sidebar-border px-4", collapsed && "justify-center px-0")}>
        <div className="flex h-7 w-7 items-center justify-center rounded-md bg-primary text-primary-foreground"><Cpu className="h-4 w-4" /></div>
        {!collapsed && <div className="leading-tight"><div className="text-sm font-semibold">DGX Model Manager</div><div className="text-[11px] text-muted-foreground">DGX Spark</div></div>}
      </div>
      <nav className="flex-1 space-y-5 overflow-y-auto px-2 py-4" aria-label="Main">
        {visibleGroups(isAdmin).map((g) => (
          <div key={g.label}>
            {!collapsed && <div className="mb-1 px-3 text-[11px] font-medium uppercase tracking-wider text-muted-foreground">{g.label}</div>}
            <ul className="space-y-0.5">
              {g.items.map(({ to, label, icon: Icon, end }) => {
                const link = (
                  <NavLink to={to} end={end} aria-label={collapsed ? label : undefined}
                    className={({ isActive }) => cn(
                      "flex h-9 items-center gap-3 rounded-md px-3 text-sm text-sidebar-foreground transition-colors hover:bg-sidebar-accent hover:text-foreground",
                      isActive && "bg-sidebar-accent font-medium text-foreground",
                      collapsed && "justify-center px-0",
                    )}>
                    <Icon className="h-4 w-4 shrink-0" />
                    {!collapsed && <span className="truncate">{label}</span>}
                  </NavLink>
                );
                return (
                  <li key={to}>
                    {collapsed ? (
                      <Tooltip><TooltipTrigger asChild>{link}</TooltipTrigger><TooltipContent side="right">{label}</TooltipContent></Tooltip>
                    ) : link}
                  </li>
                );
              })}
            </ul>
          </div>
        ))}
      </nav>
      <div className={cn("flex items-center border-t border-sidebar-border p-2", collapsed ? "justify-center" : "justify-between")}>
        {!collapsed && <span className="px-2 text-[11px] text-muted-foreground">v{__APP_VERSION__}</span>}
        <Button variant="ghost" size="icon" className="h-8 w-8" onClick={onToggle} aria-label={collapsed ? "Expand sidebar" : "Collapse sidebar"}>
          {collapsed ? <PanelLeftOpen className="h-4 w-4" /> : <PanelLeftClose className="h-4 w-4" />}
        </Button>
      </div>
    </aside>
  );
}
