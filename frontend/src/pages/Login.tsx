import { useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { Loader2 } from "lucide-react";
import { useAuth } from "@/auth";
import { AuthLayout } from "@/components/app/auth-layout";
import { PasswordInput } from "@/components/app/password-input";
import { ErrorState } from "@/components/app/states";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";

export function Login() {
  const { login } = useAuth();
  const nav = useNavigate();
  const loc = useLocation() as { state?: { from?: string } };
  const [u, setU] = useState("");
  const [p, setP] = useState("");
  const [err, setErr] = useState<unknown>(null);
  const [busy, setBusy] = useState(false);
  return (
    <AuthLayout title="Sign in" description="Manage models on your DGX Spark.">
      <form className="space-y-4" onSubmit={async (e) => {
        e.preventDefault(); setBusy(true); setErr(null);
        try { await login(u, p); nav(loc.state?.from ?? "/", { replace: true }); } catch (x) { setErr(x); } finally { setBusy(false); }
      }}>
        <div className="space-y-1.5"><Label htmlFor="username">Username</Label>
          <Input id="username" autoFocus autoComplete="username" value={u} onChange={(e) => setU(e.target.value)} /></div>
        <div className="space-y-1.5"><Label htmlFor="password">Password</Label>
          <PasswordInput id="password" autoComplete="current-password" value={p} onChange={(e) => setP(e.target.value)} /></div>
        <ErrorState error={err} />
        <Button type="submit" className="w-full" disabled={busy || !u || !p}>{busy && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}Sign in</Button>
      </form>
    </AuthLayout>
  );
}
