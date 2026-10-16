import { useCallback, useEffect, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Outlet } from "react-router-dom";
import { api } from "@/api";
import { useAuth, useIsAdmin } from "@/auth";
import { useLiveUpdates } from "@/stream";
import type { HostInfo } from "@/types";
import { AlertBanner } from "./alert-banner";
import { MobileTabBar } from "./mobile-tab-bar";
import { Sidebar } from "./sidebar";
import { TopBar } from "./top-bar";

export const SIDEBAR_KEY = "dgxmm-sidebar";

export function AppShell() {
  const { user } = useAuth();
  const isAdmin = useIsAdmin();
  useLiveUpdates(!!user);
  const host = useQuery({ queryKey: ["host"], queryFn: () => api<HostInfo>("/host") });
  const [collapsed, setCollapsed] = useState(() => localStorage.getItem(SIDEBAR_KEY) === "1");
  const [paletteOpen, setPaletteOpen] = useState(false);
  const toggle = useCallback(() => setCollapsed((c) => { localStorage.setItem(SIDEBAR_KEY, c ? "0" : "1"); return !c; }), []);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") { e.preventDefault(); setPaletteOpen((o) => !o); }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  return (
    <div className="flex min-h-dvh bg-background">
      <Sidebar isAdmin={isAdmin} collapsed={collapsed} onToggle={toggle} />
      <div className="flex min-w-0 flex-1 flex-col">
        <TopBar onOpenPalette={() => setPaletteOpen(true)} />
        <AlertBanner alert={host.data?.alert} />
        <main className="mx-auto w-full max-w-content flex-1 px-4 pb-24 pt-6 md:px-6 md:pb-10">
          <Outlet />
        </main>
      </div>
      <MobileTabBar isAdmin={isAdmin} />
      {/* CommandPalette is mounted here in Task 8: <CommandPalette open={paletteOpen} onOpenChange={setPaletteOpen} /> */}
      <span hidden data-palette-open={paletteOpen} />
    </div>
  );
}
