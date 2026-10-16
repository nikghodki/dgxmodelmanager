import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Navigate } from "react-router-dom";
import type { ColumnDef } from "@tanstack/react-table";
import { KeyRound, MoreHorizontal, Trash2, UserPlus } from "lucide-react";
import { api } from "@/api";
import { useAuth, useIsAdmin } from "@/auth";
import { ConfirmDialog } from "@/components/app/confirm-dialog";
import { DataTable } from "@/components/app/data-table";
import { PageHeader } from "@/components/app/page-header";
import { PasswordInput } from "@/components/app/password-input";
import { ErrorState } from "@/components/app/states";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from "@/components/ui/dropdown-menu";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { useActionMutation } from "@/hooks/use-action-mutation";
import { passwordProblems } from "@/lib/validation";
import type { User } from "@/types";

function UsersPage() {
  const me = useAuth().user;
  const users = useQuery({ queryKey: ["users"], queryFn: () => api<User[]>("/users") });
  const [adding, setAdding] = useState(false);
  const [form, setForm] = useState({ username: "", password: "", role: "viewer" });
  const [reset, setReset] = useState<User | null>(null);
  const [resetPw, setResetPw] = useState("");
  const [del, setDel] = useState<User | null>(null);
  const inv = [["users"]];
  const create = useActionMutation(() => api("/users", { method: "POST", body: form }), {
    success: () => `Added ${form.username}`, invalidate: inv, onSuccess: () => { setAdding(false); setForm({ username: "", password: "", role: "viewer" }); },
  });
  const patch = useActionMutation(({ id, body }: { id: number; body: object }) => api(`/users/${id}`, { method: "PATCH", body }), { success: "User updated", invalidate: inv });
  const remove = useActionMutation((id: number) => api(`/users/${id}`, { method: "DELETE" }), { success: "User deleted", invalidate: inv, onSuccess: () => setDel(null) });

  const columns = useMemo<ColumnDef<User, any>[]>(() => [
    { accessorKey: "username", header: "Username", cell: ({ row }) => <span className="flex items-center gap-2 font-medium">{row.original.username}
      {row.original.id === me?.id && <Badge variant="secondary">you</Badge>}{row.original.must_change_pw && <Badge variant="outline" className="text-status-warning">must change password</Badge>}</span> },
    { accessorKey: "role", header: "Role", cell: ({ row }) => (
      <Select value={row.original.role} disabled={row.original.id === me?.id || patch.isPending} onValueChange={(r) => patch.mutate({ id: row.original.id, body: { role: r } })}>
        <SelectTrigger className="h-8 w-28" aria-label={`Role for ${row.original.username}`}><SelectValue /></SelectTrigger>
        <SelectContent><SelectItem value="viewer">Viewer</SelectItem><SelectItem value="admin">Admin</SelectItem></SelectContent>
      </Select>) },
  ], [me?.id, patch]);

  const addProblems = passwordProblems(form.password, form.password);
  return (
    <div className="space-y-4">
      <PageHeader title="Users" description="Admins can deploy and change settings; viewers can only look."
        actions={<Button size="sm" onClick={() => setAdding(true)}><UserPlus className="mr-1 h-4 w-4" />Add user</Button>} />
      <ErrorState error={users.error} onRetry={() => users.refetch()} />
      <DataTable columns={columns} data={users.data ?? []} loading={users.isLoading} getRowId={(u) => String(u.id)}
        rowActions={(u) => (
          <DropdownMenu>
            <DropdownMenuTrigger asChild><Button variant="ghost" size="icon" className="h-8 w-8" aria-label={`Actions for ${u.username}`}><MoreHorizontal className="h-4 w-4" /></Button></DropdownMenuTrigger>
            <DropdownMenuContent align="end">
              <DropdownMenuItem onSelect={() => { setReset(u); setResetPw(""); }}><KeyRound className="mr-2 h-4 w-4" />Reset password</DropdownMenuItem>
              {u.id !== me?.id && <DropdownMenuItem className="text-status-critical focus:text-status-critical" onSelect={() => setDel(u)}><Trash2 className="mr-2 h-4 w-4" />Delete</DropdownMenuItem>}
            </DropdownMenuContent>
          </DropdownMenu>
        )} />

      <Dialog open={adding} onOpenChange={setAdding}>
        <DialogContent>
          <DialogHeader><DialogTitle>Add user</DialogTitle><DialogDescription>They must change the temporary password at first sign-in.</DialogDescription></DialogHeader>
          <form id="add-user" className="space-y-4" onSubmit={(e) => { e.preventDefault(); create.mutate(undefined); }}>
            <div className="space-y-1.5"><Label htmlFor="nu">Username</Label><Input id="nu" autoComplete="off" value={form.username} onChange={(e) => setForm({ ...form, username: e.target.value.toLowerCase() })} />
              <p className="text-xs text-muted-foreground">3–32 characters: a–z, 0–9, dot, dash, underscore.</p></div>
            <div className="space-y-1.5"><Label htmlFor="np">Temporary password</Label><PasswordInput id="np" autoComplete="new-password" value={form.password} onChange={(e) => setForm({ ...form, password: e.target.value })} />
              {form.password && addProblems.length > 0 && <p className="text-xs text-status-warning">{addProblems[0]}</p>}</div>
            <div className="space-y-1.5"><Label>Role</Label>
              <Select value={form.role} onValueChange={(r) => setForm({ ...form, role: r })}><SelectTrigger aria-label="Role"><SelectValue /></SelectTrigger>
                <SelectContent><SelectItem value="viewer">Viewer</SelectItem><SelectItem value="admin">Admin</SelectItem></SelectContent></Select></div>
          </form>
          <DialogFooter><Button variant="outline" onClick={() => setAdding(false)}>Cancel</Button>
            <Button type="submit" form="add-user" disabled={create.isPending || !/^[a-z0-9_.-]{3,32}$/.test(form.username) || addProblems.length > 0}>Add user</Button></DialogFooter>
        </DialogContent>
      </Dialog>

      <Dialog open={!!reset} onOpenChange={(o) => !o && setReset(null)}>
        <DialogContent>
          <DialogHeader><DialogTitle>Reset password for {reset?.username}</DialogTitle><DialogDescription>They will be asked to change it at next sign-in; their sessions are ended.</DialogDescription></DialogHeader>
          <div className="space-y-1.5"><Label htmlFor="rp">New temporary password</Label><PasswordInput id="rp" autoComplete="new-password" value={resetPw} onChange={(e) => setResetPw(e.target.value)} />
            {resetPw && passwordProblems(resetPw, resetPw)[0] && <p className="text-xs text-status-warning">{passwordProblems(resetPw, resetPw)[0]}</p>}</div>
          <DialogFooter><Button variant="outline" onClick={() => setReset(null)}>Cancel</Button>
            <Button disabled={patch.isPending || passwordProblems(resetPw, resetPw).length > 0}
              onClick={() => reset && patch.mutate({ id: reset.id, body: { password: resetPw } }, { onSuccess: () => setReset(null) })}>Reset password</Button></DialogFooter>
        </DialogContent>
      </Dialog>

      <ConfirmDialog open={!!del} onOpenChange={(o) => !o && setDel(null)} title={`Delete ${del?.username}?`} description="Their account and sessions are removed."
        confirmLabel="Delete" destructive pending={remove.isPending} onConfirm={() => del && remove.mutate(del.id)} />
    </div>
  );
}

export function Users() {
  return useIsAdmin() ? <UsersPage /> : <Navigate to="/" replace />;
}
