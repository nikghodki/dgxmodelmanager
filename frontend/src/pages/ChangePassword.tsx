import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { api } from "../api";
import { useAuth } from "../auth";
import { Button, Card, ErrorBox, Field, Input } from "../components/ui";

export function ChangePassword() {
  const { reload, user } = useAuth();
  const nav = useNavigate();
  const [cur, setCur] = useState("");
  const [next, setNext] = useState("");
  const [again, setAgain] = useState("");
  const [err, setErr] = useState<unknown>(null);
  return (
    <div className="mx-auto mt-16 max-w-md">
      <Card title={user?.must_change_pw ? "Set a new password to continue" : "Change password"}>
        <form className="space-y-3" onSubmit={async (e) => {
          e.preventDefault();
          if (next !== again) return setErr(new Error("passwords do not match"));
          try {
            await api("/auth/change-password", { method: "POST", body: { current_password: cur, new_password: next } });
            await reload();
            nav("/", { replace: true });
          } catch (x) {
            setErr(x);
          }
        }}>
          <Field label="Current password"><Input type="password" value={cur} onChange={(e) => setCur(e.target.value)} /></Field>
          <Field label="New password" help="At least 12 characters."><Input type="password" value={next} onChange={(e) => setNext(e.target.value)} /></Field>
          <Field label="Repeat new password"><Input type="password" value={again} onChange={(e) => setAgain(e.target.value)} /></Field>
          <ErrorBox error={err} />
          <Button variant="primary" disabled={!cur || next.length < 12}>Save</Button>
        </form>
      </Card>
    </div>
  );
}
