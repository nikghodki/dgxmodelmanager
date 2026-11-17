import { useEffect, useState } from "react";
import { useLocation, useNavigate, useSearchParams } from "react-router-dom";
import { Loader2, LogIn } from "lucide-react";
import { api, basePath } from "@/api";
import { useAuth } from "@/auth";
import { AuthLayout } from "@/components/app/auth-layout";
import { PasswordInput } from "@/components/app/password-input";
import { ErrorState } from "@/components/app/states";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { ssoErrorMessage, ssoLoginHref } from "@/lib/sso";
import type { AuthConfig } from "@/types";

/** Where the browser goes to sign in with SSO (a full page load; replaced in tests). */
export const goTo = { assign: (url: string) => window.location.assign(url) };

export function Login() {
  const { login, loggedOut } = useAuth();
  const nav = useNavigate();
  const loc = useLocation() as { state?: { from?: string } };
  const [params] = useSearchParams();
  const [u, setU] = useState("");
  const [p, setP] = useState("");
  const [err, setErr] = useState<unknown>(null);
  const [busy, setBusy] = useState(false);
  const [config, setConfig] = useState<AuthConfig | null>(null);
  const [configFailed, setConfigFailed] = useState(false);
  const ssoError = ssoErrorMessage(params.get("sso_error"));
  const wantLocal = params.has("local");
  const oidc = config?.oidc ?? null;
  const ssoHref = oidc ? ssoLoginHref(basePath, oidc.login_path, loc.state?.from) : null;
  const redirecting = !!(oidc?.auto_redirect && ssoHref && !ssoError && !wantLocal && !loggedOut);

  useEffect(() => {
    let live = true;
    api<AuthConfig>("/auth/config").then((c) => { if (live) setConfig(c); }).catch(() => { if (live) setConfigFailed(true); });
    return () => { live = false; };
  }, []);
  useEffect(() => {
    if (redirecting && ssoHref) goTo.assign(ssoHref);
  }, [redirecting, ssoHref]);

  // the password form: when the server allows it (older servers have no /auth/config: always)
  const showLocal = configFailed || (config ? config.local_login : false);
  if (redirecting) {
    return (
      <AuthLayout title="Sign in" description="Manage models on your DGX Spark.">
        <p className="flex items-center gap-2 text-sm text-muted-foreground" role="status"><Loader2 className="h-4 w-4 animate-spin" />Redirecting to single sign-on…</p>
      </AuthLayout>
    );
  }
  return (
    <AuthLayout title="Sign in" description="Manage models on your DGX Spark.">
      <div className="space-y-4">
        {ssoError && <ErrorState error={new Error(ssoError)} />}
        {loggedOut && oidc && <p className="text-sm text-muted-foreground" role="status">You are signed out of DGXMM.</p>}
        {oidc && ssoHref && (
          <Button asChild className="w-full">
            <a href={ssoHref}><LogIn className="mr-2 h-4 w-4" />{oidc.label}</a>
          </Button>
        )}
        {oidc && showLocal && (
          <div className="flex items-center gap-3 text-xs text-muted-foreground" aria-hidden><span className="h-px flex-1 bg-border" />or with a DGXMM password<span className="h-px flex-1 bg-border" /></div>
        )}
        {showLocal && (
          <form className="space-y-4" onSubmit={async (e) => {
            e.preventDefault(); setBusy(true); setErr(null);
            try { await login(u, p); nav(loc.state?.from ?? "/", { replace: true }); } catch (x) { setErr(x); } finally { setBusy(false); }
          }}>
            <div className="space-y-1.5"><Label htmlFor="username">Username</Label>
              <Input id="username" autoFocus={!oidc} autoComplete="username" value={u} onChange={(e) => setU(e.target.value)} /></div>
            <div className="space-y-1.5"><Label htmlFor="password">Password</Label>
              <PasswordInput id="password" autoComplete="current-password" value={p} onChange={(e) => setP(e.target.value)} /></div>
            <ErrorState error={err} />
            <Button type="submit" className="w-full" variant={oidc ? "outline" : "default"} disabled={busy || !u || !p}>{busy && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}Sign in</Button>
          </form>
        )}
        {!config && !configFailed && <p className="flex items-center gap-2 text-sm text-muted-foreground"><Loader2 className="h-4 w-4 animate-spin" />Loading…</p>}
      </div>
    </AuthLayout>
  );
}
