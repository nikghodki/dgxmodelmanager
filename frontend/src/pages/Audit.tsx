import { useQuery } from "@tanstack/react-query";
import { api } from "../api";
import { Card, ErrorBox, Table } from "../components/ui";

export function Audit() {
  const q = useQuery({ queryKey: ["audit"], queryFn: () => api<any[]>("/audit?limit=500"), refetchInterval: 15000 });
  return (
    <Card title="Audit log">
      <ErrorBox error={q.error} />
      <Table head={["Time", "User", "Action", "Target", "Detail"]}>
        {(q.data ?? []).map((a, i) => (
          <tr key={i}>
            <td className="td whitespace-nowrap font-mono text-xs">{new Date(a.ts).toLocaleString()}</td>
            <td className="td">{a.user}</td><td className="td">{a.action}</td><td className="td">{a.target}</td>
            <td className="td font-mono text-xs">{Object.keys(a.detail ?? {}).length ? JSON.stringify(a.detail) : ""}</td>
          </tr>
        ))}
      </Table>
    </Card>
  );
}
