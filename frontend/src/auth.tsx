import { createContext, useCallback, useContext, useEffect, useState, type ReactNode } from "react";
import { Navigate, useLocation } from "react-router-dom";
import { api, setCsrf } from "./api";
import type { User } from "./types";

interface AuthState {
  user: User | null;
  loading: boolean;
  /** set for single sign-on sessions */
  sso?: { email: string | null } | null;
  /** true after "Log out" in this tab: the login page then does not send the user straight back to the IdP */
  loggedOut?: boolean;
  login: (u: string, p: string) => Promise<void>;
  logout: () => Promise<void>;
  reload: () => Promise<void>;
}
const Ctx = createContext<AuthState>(null as unknown as AuthState);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);
  const [sso, setSso] = useState<{ email: string | null } | null>(null);
  const [loggedOut, setLoggedOut] = useState(false);
  const reload = useCallback(async () => {
    try {
      const me = await api<{ user: User; csrf_token: string; sso?: { email: string | null } | null }>("/auth/me");
      setCsrf(me.csrf_token);
      setUser(me.user);
      setSso(me.sso ?? null);
    } catch {
      setUser(null);
      setSso(null);
    } finally {
      setLoading(false);
    }
  }, []);
  useEffect(() => {
    reload();
  }, [reload]);
  const login = async (username: string, password: string) => {
    const r = await api<{ user: User; csrf_token: string }>("/auth/login", { method: "POST", body: { username, password } });
    setCsrf(r.csrf_token);
    setUser(r.user);
    setSso(null);
    setLoggedOut(false);
  };
  const logout = async () => {
    await api("/auth/logout", { method: "POST" }).catch(() => undefined);
    setLoggedOut(true);
    setSso(null);
    setUser(null);
  };
  return <Ctx.Provider value={{ user, loading, sso, loggedOut, login, logout, reload }}>{children}</Ctx.Provider>;
}

export const useAuth = () => useContext(Ctx);
export const useIsAdmin = () => useAuth().user?.role === "admin";

export function RequireAuth({ children }: { children: ReactNode }) {
  const { user, loading } = useAuth();
  const loc = useLocation();
  if (loading) return <div className="flex min-h-dvh items-center justify-center text-sm text-muted-foreground">Loading…</div>;
  if (!user) return <Navigate to="/login" replace state={{ from: loc.pathname }} />;
  if (user.must_change_pw && loc.pathname !== "/change-password") return <Navigate to="/change-password" replace />;
  return <>{children}</>;
}
