import { useQuery } from "@tanstack/react-query";
import { Route } from "lucide-react";
import { api } from "@/api";
import { CopyButton } from "@/components/app/copy-button";
import { ErrorState } from "@/components/app/states";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { aliasSnippet, sourceText } from "@/lib/routing";
import type { RoutingAdvice } from "@/types";

/** "Routing advice" on the LiteLLM page (admin): traffic class -> suggested deployment, with copyable LiteLLM
 *  alias snippets. Advisory only — DGXMM never changes LiteLLM's routing. */
export function RoutingAdviceCard() {
  const q = useQuery({ queryKey: ["routing-advice"], queryFn: () => api<RoutingAdvice>("/decider/routing-advice"), refetchInterval: 60000 });
  const a = q.data;
  if (a && !a.enabled) return null;
  return (
    <Card className="min-w-0 shadow-card" aria-label="Routing advice">
      <CardHeader className="pb-3">
        <CardTitle className="flex items-center gap-2 text-sm"><Route aria-hidden className="h-4 w-4" />Routing advice</CardTitle>
        <CardDescription className="text-xs">
          Which deployment suits each kind of traffic, as LiteLLM aliases you can add to its config. Advisory: DGXMM does not change LiteLLM.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-3">
        {q.isError ? <ErrorState error={q.error} onRetry={() => q.refetch()} /> : !a ? (
          <p role="status" className="text-xs text-muted-foreground">Loading routing advice…</p>
        ) : (
          <>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead><tr className="border-b text-left text-xs uppercase tracking-wider text-muted-foreground">
                  {["Traffic", "Alias", "Model name", "Deployment", "Source", ""].map((h, i) => <th key={i} className="px-2 py-2 font-medium">{h}</th>)}
                </tr></thead>
                <tbody>
                  {a.classes.map((c) => {
                    const snip = aliasSnippet(c);
                    return (
                      <tr key={c.class} className="border-b last:border-0" data-class={c.class}>
                        <td className="px-2 py-2">{c.label}</td>
                        <td className="px-2 py-2 font-mono text-xs">{c.alias}</td>
                        <td className="px-2 py-2 font-mono text-xs">{c.model_name ?? <span className="font-sans text-muted-foreground">No suitable deployment</span>}</td>
                        <td className="px-2 py-2 text-xs" title={c.reason}>{c.deployment ?? "—"}{c.candidates.length > 1 && <span className="text-muted-foreground"> · {c.candidates.length} candidates</span>}
                          {c.note && <span className="block text-status-warning">{c.note}</span>}</td>
                        <td className="whitespace-nowrap px-2 py-2 text-xs text-muted-foreground">{sourceText(c)}</td>
                        <td className="px-2 py-2 text-right">{snip && <CopyButton value={snip} label={`Copy ${c.alias} snippet`} />}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
            {a.snippet && (
              <div className="rounded-md border bg-muted/40">
                <div className="flex items-center justify-between gap-2 border-b px-3 py-1.5 text-xs text-muted-foreground">
                  <span>LiteLLM config (all aliases)</span><CopyButton value={a.snippet} label="Copy all aliases" />
                </div>
                <pre aria-label="LiteLLM alias snippet" className="overflow-x-auto p-3 font-mono text-xs leading-relaxed">{a.snippet}</pre>
              </div>
            )}
          </>
        )}
      </CardContent>
    </Card>
  );
}
