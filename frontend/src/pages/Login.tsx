import { useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { useAuth } from "../auth";
import { Button, ErrorBox, Field, Input } from "../components/ui";

export function Login() {
  const { login } = useAuth();
  const nav = useNavigate();
  const loc = useLocation() as { state?: { from?: string } };
  const [u, setU] = useState("");
  const [p, setP] = useState("");
  const [err, setErr] = useState<unknown>(null);
  const [busy, setBusy] = useState(false);
  return (
    <div className="flex min-h-screen items-center justify-center">
      <form className="w-80 space-y-3 rounded-lg border bg-white p-6 shadow-sm"
        onSubmit={async (e) => {
          e.preventDefault();
          setBusy(true);
          setErr(null);
          try {
            await login(u, p);
            nav(loc.state?.from ?? "/", { replace: true });
          } catch (x) {
            setErr(x);
          } finally {
            setBusy(false);
          }
        }}>
        <h1 className="text-lg font-semibold">DGX Model Manager</h1>
        <Field label="Username"><Input autoFocus value={u} onChange={(e) => setU(e.target.value)} autoComplete="username" /></Field>
        <Field label="Password"><Input type="password" value={p} onChange={(e) => setP(e.target.value)} autoComplete="current-password" /></Field>
        <ErrorBox error={err} />
        <Button variant="primary" className="w-full" disabled={busy || !u || !p}>Log in</Button>
      </form>
    </div>
  );
}
