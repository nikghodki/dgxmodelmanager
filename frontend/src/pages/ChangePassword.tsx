import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { Loader2 } from "lucide-react";
import { toast } from "sonner";
import { api } from "@/api";
import { useAuth } from "@/auth";
import { AuthLayout } from "@/components/app/auth-layout";
import { PasswordInput } from "@/components/app/password-input";
import { ErrorState } from "@/components/app/states";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { passwordProblems } from "@/lib/validation";

export function ChangePassword() {
  const { reload, user } = useAuth();
  const nav = useNavigate();
  const [cur, setCur] = useState("");
  const [next, setNext] = useState("");
  const [again, setAgain] = useState("");
  const [err, setErr] = useState<unknown>(null);
  const [busy, setBusy] = useState(false);
  const problems = passwordProblems(next, again, cur);
  return (
    <AuthLayout title={user?.must_change_pw ? "Set a new password" : "Change password"}
      description={user?.must_change_pw ? "Your account requires a new password before you continue." : undefined}>
      <form className="space-y-4" onSubmit={async (e) => {
        e.preventDefault(); setBusy(true); setErr(null);
        try {
          await api("/auth/change-password", { method: "POST", body: { current_password: cur, new_password: next } });
          await reload(); toast.success("Password updated"); nav("/", { replace: true });
        } catch (x) { setErr(x); } finally { setBusy(false); }
      }}>
        <div className="space-y-1.5"><Label htmlFor="cur">Current password</Label><PasswordInput id="cur" autoComplete="current-password" value={cur} onChange={(e) => setCur(e.target.value)} /></div>
        <div className="space-y-1.5"><Label htmlFor="new">New password</Label><PasswordInput id="new" autoComplete="new-password" value={next} onChange={(e) => setNext(e.target.value)} />
          <p className="text-xs text-muted-foreground">At least 12 characters.</p></div>
        <div className="space-y-1.5"><Label htmlFor="again">Repeat new password</Label><PasswordInput id="again" autoComplete="new-password" value={again} onChange={(e) => setAgain(e.target.value)} /></div>
        {next && problems.length > 0 && <ul className="space-y-0.5 text-xs text-status-warning">{problems.map((p) => <li key={p}>• {p}</li>)}</ul>}
        <ErrorState error={err} />
        <Button type="submit" className="w-full" disabled={busy || !cur || !again || problems.length > 0}>
          {busy && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}Save password
        </Button>
      </form>
    </AuthLayout>
  );
}
