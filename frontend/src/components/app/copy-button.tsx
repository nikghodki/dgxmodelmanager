import { Check, Copy } from "lucide-react";
import { useState } from "react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";

/** navigator.clipboard only exists in secure contexts; DGXMM is usually opened over plain http://<lan-ip>. */
async function copyText(value: string) {
  if (navigator.clipboard?.writeText) return navigator.clipboard.writeText(value);
  const ta = document.createElement("textarea");
  ta.value = value;
  ta.setAttribute("readonly", "");
  ta.style.position = "fixed";
  ta.style.opacity = "0";
  document.body.appendChild(ta);
  try {
    ta.select();
    if (!document.execCommand("copy")) throw new Error("copy command was rejected");
  } finally {
    ta.remove();
  }
}

export function CopyButton({ value, label = "Copy", onCopied }: { value: string; label?: string; onCopied?: () => void }) {
  const [done, setDone] = useState(false);
  return (
    <Button variant="ghost" size="icon" className="h-7 w-7 shrink-0" aria-label={label}
      onClick={async () => {
        try {
          await copyText(value);
          setDone(true);
          onCopied?.();
          setTimeout(() => setDone(false), 1500);
          toast.success("Copied to clipboard");
        } catch {
          toast.error("Copy failed — select the text and copy it manually");
        }
      }}>
      {done ? <Check className="h-3.5 w-3.5" /> : <Copy className="h-3.5 w-3.5" />}
    </Button>
  );
}
