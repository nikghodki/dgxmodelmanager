import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { api } from "../api";
import { useAuth } from "../auth";
import { Badge, Button, Card, ErrorBox, Field, Input, Select, Table } from "../components/ui";
import type { User } from "../types";

export function Users() {
  const qc = useQueryClient();
  const me = useAuth().user;
  const users = useQuery({ queryKey: ["users"], queryFn: () => api<User[]>("/users") });
  const [form, setForm] = useState({ username: "", password: "", role: "viewer" });
  const done = () => qc.invalidateQueries({ queryKey: ["users"] });
  const create = useMutation({ mutationFn: () => api("/users", { method: "POST", body: form }), onSuccess: () => { setForm({ username: "", password: "", role: "viewer" }); done(); } });
  const patch = useMutation({ mutationFn: ({ id, body }: { id: number; body: object }) => api(`/users/${id}`, { method: "PATCH", body }), onSuccess: done });
  const del = useMutation({ mutationFn: (id: number) => api(`/users/${id}`, { method: "DELETE" }), onSuccess: done });
  return (
    <div className="space-y-4">
      <Card title="Users">
        <ErrorBox error={users.error || patch.error || del.error} />
        <Table head={["Username", "Role", "", ""]}>
          {(users.data ?? []).map((u) => (
            <tr key={u.id}>
              <td className="td">{u.username} {u.must_change_pw && <Badge tone="amber">must change password</Badge>}</td>
              <td className="td">
                <Select value={u.role} disabled={u.id === me?.id} onChange={(e) => patch.mutate({ id: u.id, body: { role: e.target.value } })}>
                  <option value="viewer">viewer</option><option value="admin">admin</option>
                </Select>
              </td>
              <td className="td"><Button onClick={() => { const pw = prompt(`New temporary password for ${u.username} (12+ chars)`); if (pw) patch.mutate({ id: u.id, body: { password: pw } }); }}>Reset password</Button></td>
              <td className="td">{u.id !== me?.id && <Button variant="danger" onClick={() => confirm(`Delete ${u.username}?`) && del.mutate(u.id)}>Delete</Button>}</td>
            </tr>
          ))}
        </Table>
      </Card>
      <Card title="Add user">
        <form className="grid grid-cols-1 gap-3 md:grid-cols-4" onSubmit={(e) => { e.preventDefault(); create.mutate(); }}>
          <Field label="Username"><Input value={form.username} onChange={(e) => setForm({ ...form, username: e.target.value.toLowerCase() })} /></Field>
          <Field label="Temporary password"><Input type="password" value={form.password} onChange={(e) => setForm({ ...form, password: e.target.value })} /></Field>
          <Field label="Role"><Select value={form.role} onChange={(e) => setForm({ ...form, role: e.target.value })}><option>viewer</option><option>admin</option></Select></Field>
          <div className="flex items-end"><Button variant="primary">Add</Button></div>
        </form>
        <ErrorBox error={create.error} />
      </Card>
    </div>
  );
}
