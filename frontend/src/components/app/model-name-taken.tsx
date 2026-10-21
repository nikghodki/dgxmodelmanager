import { Link } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { apiErrorText, isModelNameTaken, modelNameRule, takenHolder } from "@/lib/litellm";

/**
 * A 409 `model_name_taken` (deploy / alias / start / wake): why it is refused, a link to the deployment holding the
 * name (none when a LiteLLM entry DGXMM did not create holds it), and a fix: either a one-click suggested alias (`onUse`) or a link to where the alias can be set (`fixHref`).
 * Renders nothing for other errors.
 */
export function ModelNameTaken({ error, suggestion, onUse, fixHref, fixLabel }: {
  error: unknown; suggestion?: string; onUse?: (alias: string) => void; fixHref?: string; fixLabel?: string;
}) {
  if (!isModelNameTaken(error)) return null;
  const h = takenHolder(error);
  return (
    <div role="alert" className="space-y-1.5 text-xs">
      <p className="text-status-critical">
        {apiErrorText(error)}{h && <> <Link to={`/deployments/${h.id}`} className="font-medium underline">Open {h.name}</Link></>}
      </p>
      <p className="text-muted-foreground">{modelNameRule(error)}</p>
      {onUse && suggestion && <Button type="button" variant="outline" size="sm" className="h-7" onClick={() => onUse(suggestion)}>Use {suggestion}</Button>}
      {fixHref && <Link to={fixHref} className="inline-block font-medium text-primary underline">{fixLabel ?? "Set a model name"}</Link>}
    </div>
  );
}
