import { useState } from "react";
import { NavLink, useNavigate } from "react-router-dom";
import { HardDrive, KeyRound, LayoutDashboard, LogOut, MoreHorizontal, Server } from "lucide-react";
import { useAuth } from "@/auth";
import { Sheet, SheetContent, SheetHeader, SheetTitle } from "@/components/ui/sheet";
import { cn } from "@/lib/utils";
import { MOBILE_TABS, visibleGroups } from "./nav";

const TABS = [
  { to: "/", label: "Home", icon: LayoutDashboard, end: true },
  { to: "/deployments", label: "Deploys", icon: Server },
  { to: "/models", label: "Models", icon: HardDrive },
];

export function MobileTabBar({ isAdmin }: { isAdmin: boolean }) {
  const [more, setMore] = useState(false);
  const { logout } = useAuth();
  const nav = useNavigate();
  const rest = visibleGroups(isAdmin).flatMap((g) => g.items).filter((i) => !MOBILE_TABS.includes(i.to));
  const tabCls = ({ isActive }: { isActive: boolean }) =>
    cn("flex min-h-[44px] flex-1 flex-col items-center justify-center gap-0.5 text-[11px]", isActive ? "text-primary" : "text-muted-foreground");
  return (
    <>
      <nav aria-label="Primary" className="pb-safe fixed inset-x-0 bottom-0 z-40 flex border-t bg-background/95 backdrop-blur md:hidden">
        {TABS.map(({ to, label, icon: Icon, end }) => (
          <NavLink key={to} to={to} end={end} className={tabCls}><Icon className="h-5 w-5" />{label}</NavLink>
        ))}
        <button type="button" onClick={() => setMore(true)} className="flex min-h-[44px] flex-1 flex-col items-center justify-center gap-0.5 text-[11px] text-muted-foreground">
          <MoreHorizontal className="h-5 w-5" />More
        </button>
      </nav>
      <Sheet open={more} onOpenChange={setMore}>
        <SheetContent side="bottom" className="pb-safe rounded-t-xl">
          <SheetHeader><SheetTitle>More</SheetTitle></SheetHeader>
          <ul className="mt-2 grid grid-cols-1 gap-1">
            {rest.map(({ to, label, icon: Icon }) => (
              <li key={to}><button type="button" onClick={() => { setMore(false); nav(to); }} className="flex min-h-[44px] w-full items-center gap-3 rounded-md px-3 text-sm hover:bg-accent"><Icon className="h-4 w-4" />{label}</button></li>
            ))}
            <li><button type="button" onClick={() => { setMore(false); nav("/change-password"); }} className="flex min-h-[44px] w-full items-center gap-3 rounded-md px-3 text-sm hover:bg-accent"><KeyRound className="h-4 w-4" />Change password</button></li>
            <li><button type="button" onClick={() => { setMore(false); logout(); }} className="flex min-h-[44px] w-full items-center gap-3 rounded-md px-3 text-sm text-status-critical hover:bg-accent"><LogOut className="h-4 w-4" />Log out</button></li>
          </ul>
        </SheetContent>
      </Sheet>
    </>
  );
}
