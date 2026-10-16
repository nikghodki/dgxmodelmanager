import { useState } from "react";
import { Check, ChevronsUpDown } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Command, CommandEmpty, CommandGroup, CommandInput, CommandItem, CommandList } from "@/components/ui/command";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { engineLabel, fmtBytes, shortRef } from "@/format";
import { cn } from "@/lib/utils";
import type { Artifact } from "@/types";

/** `labelledBy` names the trigger after a visible label; the chosen model (or placeholder) stays part of the name. */
export function ModelSelect({ id, labelledBy, artifacts, value, onChange, placeholder = "Choose a downloaded model…" }: {
  id?: string; labelledBy?: string; artifacts: Artifact[]; value: string; onChange: (ref: string) => void; placeholder?: string;
}) {
  const [open, setOpen] = useState(false);
  const current = artifacts.find((a) => a.ref === value);
  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <Button variant="outline" role="combobox" aria-expanded={open} id={id} aria-labelledby={labelledBy && id ? `${labelledBy} ${id}-value` : undefined} className="h-auto min-h-10 w-full justify-between py-2 text-left font-normal">
          <span id={id && `${id}-value`} className="min-w-0 truncate" title={value || undefined}>{current ? shortRef(current.ref) : <span className="text-muted-foreground">{placeholder}</span>}</span>
          <ChevronsUpDown className="ml-2 h-4 w-4 shrink-0 opacity-50" />
        </Button>
      </PopoverTrigger>
      <PopoverContent className="w-[min(92vw,36rem)] p-0" align="start">
        <Command>
          <CommandInput placeholder="Search models…" />
          <CommandList>
            <CommandEmpty>No downloaded model matches.</CommandEmpty>
            <CommandGroup>
              {artifacts.map((a) => (
                <CommandItem key={a.ref} value={a.ref} onSelect={() => { onChange(a.ref); setOpen(false); }} className="items-start gap-2 py-2">
                  <Check className={cn("mt-0.5 h-4 w-4 shrink-0", value === a.ref ? "opacity-100" : "opacity-0")} />
                  <div className="min-w-0 flex-1">
                    <div className="truncate text-sm" title={a.ref}>{shortRef(a.ref)}</div>
                    <div className="mt-1 flex flex-wrap gap-1">
                      {a.quant && <Badge variant="secondary" className="text-[10px]">{a.quant}</Badge>}
                      <Badge variant="outline" className="text-[10px] tabular-nums">{fmtBytes(a.weights_bytes)}</Badge>
                      {a.engines.map((e) => <Badge key={e} variant="outline" className="text-[10px]">{engineLabel(e)}</Badge>)}
                      {a.is_moe && <Badge variant="outline" className="text-[10px]">MoE</Badge>}
                    </div>
                  </div>
                </CommandItem>
              ))}
            </CommandGroup>
          </CommandList>
        </Command>
      </PopoverContent>
    </Popover>
  );
}
