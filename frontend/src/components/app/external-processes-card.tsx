import { Link } from "react-router-dom";
import { Box, Monitor } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { fmtBytes } from "@/format";
import { containerLabel, labelTitle, userLabel } from "@/lib/processes";
import type { ProcMem } from "@/types";

/** Dashboard card: the largest processes DGXMM does not manage, with where they run and what they are. */
export function ExternalProcessesCard({ items }: { items: ProcMem[] }) {
  return (
    <Card className="min-w-0 shadow-card">
      <CardHeader className="flex flex-row items-start justify-between gap-2 space-y-0 pb-2">
        <div className="min-w-0">
          <CardTitle className="text-sm">Largest external memory users</CardTitle>
          <CardDescription className="text-xs">Processes DGXMM does not manage. They are never touched.</CardDescription>
        </div>
        <Button asChild variant="ghost" size="sm" className="shrink-0"><Link to="/processes">View all processes</Link></Button>
      </CardHeader>
      <CardContent>
        <ul className="divide-y text-sm">
          {items.map((p) => {
            const Icon = p.container ? Box : Monitor;
            return (
              <li key={p.pid} className="flex items-start justify-between gap-3 py-2">
                <div className="min-w-0 flex-1">
                  <div className="flex min-w-0 items-center gap-1.5">
                    <Icon className="h-3.5 w-3.5 shrink-0 text-muted-foreground" aria-hidden />
                    <span className="truncate font-medium">{p.container === undefined ? p.name : containerLabel(p.container)}</span>
                    {p.dgxmm_label && (
                      <Badge variant="outline" className="shrink-0 text-[11px]"
                        title={labelTitle(p.dgxmm_label)}>DGXMM-labelled</Badge>
                    )}
                    <span className="shrink-0 text-xs text-muted-foreground">pid {p.pid}</span>
                    {p.user && <span className="truncate text-xs text-muted-foreground" title={userLabel(p.user, p.user_scope).title}>· {userLabel(p.user, p.user_scope).text}</span>}
                  </div>
                  {p.command && (
                    <div className="mt-0.5 truncate font-mono text-xs text-muted-foreground" title={p.command}>{p.command}</div>
                  )}
                </div>
                <div className="shrink-0 text-right tabular-nums">
                  <div>{fmtBytes(p.rss + p.gpu)}</div>
                  <div className="text-xs text-muted-foreground">RAM {fmtBytes(p.rss)}{p.gpu > 0 ? ` · GPU ${fmtBytes(p.gpu)}` : ""}</div>
                </div>
              </li>
            );
          })}
        </ul>
      </CardContent>
    </Card>
  );
}
