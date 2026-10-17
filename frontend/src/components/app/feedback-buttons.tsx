import { useEffect, useState } from "react";
import { ThumbsDown, ThumbsUp } from "lucide-react";
import { api } from "@/api";
import { Button } from "@/components/ui/button";
import { useActionMutation } from "@/hooks/use-action-mutation";
import { cn } from "@/lib/utils";

/** 👍/👎 on a logged decision (admin): POST /decider/decisions/{id}/feedback. */
export function FeedbackButtons({ decisionId, feedback, invalidate = [], className }: {
  decisionId: number; feedback?: "accepted" | "rejected" | null; invalidate?: unknown[][]; className?: string;
}) {
  const [local, setLocal] = useState<"accepted" | "rejected" | null>(feedback ?? null);
  useEffect(() => { setLocal(feedback ?? null); }, [feedback]);
  const send = useActionMutation((accepted: boolean) => api(`/decider/decisions/${decisionId}/feedback`, { method: "POST", body: { accepted } }), {
    success: "Thanks — feedback recorded", invalidate: [["decider-decisions"], ["decider-status"], ...invalidate],
    onSuccess: (_r, accepted) => setLocal(accepted ? "accepted" : "rejected"),
  });
  const btn = (accepted: boolean) => {
    const on = local === (accepted ? "accepted" : "rejected");
    const Icon = accepted ? ThumbsUp : ThumbsDown;
    return (
      <Button type="button" variant="ghost" size="icon" className={cn("h-7 w-7", on && "bg-accent text-foreground")}
        aria-label={accepted ? "Helpful" : "Not helpful"} title={accepted ? "Helpful" : "Not helpful"} aria-pressed={on}
        disabled={send.isPending} onClick={() => send.mutate(accepted)}>
        <Icon aria-hidden className={cn("h-3.5 w-3.5", on && "fill-current")} />
      </Button>
    );
  };
  return <span className={cn("inline-flex items-center gap-0.5", className)} role="group" aria-label="Feedback on this decision">{btn(true)}{btn(false)}</span>;
}
