import { useQuery } from "@tanstack/react-query";
import { api } from "@/api";
import { useActionMutation } from "@/hooks/use-action-mutation";
import { insightsKey, insightsPath, type InsightFilters } from "@/lib/insights";
import type { Insight } from "@/types";

/** Insights matching the filters. Kept fresh by the SSE `insights` topic (stream.ts invalidates ["insights"]); the
 *  slow poll only catches expiries while the stream is down. */
export function useInsights(filters: InsightFilters = {}, opts: { enabled?: boolean } = {}) {
  return useQuery({ queryKey: insightsKey(filters), queryFn: () => api<Insight[]>(insightsPath(filters)), refetchInterval: 120000, enabled: opts.enabled ?? true });
}

/** Admin actions on one insight: dismiss, snooze {hours}, applied. Each refreshes every insights query. */
export function useInsightActions() {
  const opts = { invalidate: [["insights"]] };
  const dismiss = useActionMutation((id: number) => api<Insight>(`/insights/${id}/dismiss`, { method: "POST" }), { ...opts, success: "Insight dismissed" });
  const snooze = useActionMutation(({ id, hours }: { id: number; hours: number }) => api<Insight>(`/insights/${id}/snooze`, { method: "POST", body: { hours } }),
    { ...opts, success: (_r, v) => `Snoozed for ${v.hours === 1 ? "1 hour" : v.hours < 24 ? `${v.hours} hours` : v.hours === 24 ? "1 day" : "1 week"}` });
  const applied = useActionMutation((id: number) => api<Insight>(`/insights/${id}/applied`, { method: "POST" }), opts);
  return { dismiss, snooze, applied };
}
