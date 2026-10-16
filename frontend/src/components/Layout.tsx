import { useQuery } from "@tanstack/react-query";
import { NavLink, Outlet } from "react-router-dom";
import { api } from "../api";
import { useAuth } from "../auth";
import { useLiveUpdates } from "../stream";
import type { HostInfo } from "../types";

const NAV = [
  { to: "/", label: "Dashboard", admin: false }, { to: "/models", label: "Models on disk", admin: false },
  { to: "/catalog", label: "Catalog", admin: false }, { to: "/downloads", label: "Downloads", admin: false },
  { to: "/planner", label: "Planner", admin: false }, { to: "/audit", label: "Audit log", admin: true },
  { to: "/users", label: "Users", admin: true },
];

export function Layout() {
  const { user, logout } = useAuth();
  useLiveUpdates(!!user);
  const host = useQuery({ queryKey: ["host"], queryFn: () => api<HostInfo>("/host") });
  const alert = host.data?.alert;
  return (
    <div className="flex min-h-screen">
      <aside className="w-52 shrink-0 border-r border-slate-200 bg-white">
        <div className="px-4 py-4 text-sm font-bold">DGX Model Manager</div>
        <nav className="flex flex-col gap-0.5 px-2">
          {NAV.filter((n) => !n.admin || user?.role === "admin").map((n) => (
            <NavLink key={n.to} to={n.to} end={n.to === "/"}
              className={({ isActive }) => `rounded px-3 py-1.5 text-sm ${isActive ? "bg-slate-900 text-white" : "text-slate-700 hover:bg-slate-100"}`}>
              {n.label}
            </NavLink>
          ))}
        </nav>
        <div className="mt-6 border-t px-4 py-3 text-xs text-slate-500">
          {user?.username} ({user?.role})
          <button className="ml-2 underline" onClick={logout}>log out</button>
        </div>
      </aside>
      <main className="flex-1 space-y-4 p-6">
        {alert && alert.level !== "ok" && (
          <div className={`rounded-md px-4 py-2 text-sm ${alert.level === "critical" ? "bg-rose-600 text-white" : "bg-amber-100 text-amber-900"}`}>
            <b>{alert.level === "critical" ? "Memory critical" : "Memory warning"}:</b> {alert.message}
            {alert.action && <span className="ml-2">({alert.action})</span>}
          </div>
        )}
        <Outlet />
      </main>
    </div>
  );
}
