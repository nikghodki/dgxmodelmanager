import type { ReactNode } from "react";
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Sheet, SheetContent, SheetDescription, SheetHeader, SheetTitle } from "@/components/ui/sheet";
import { useIsMobile } from "@/hooks/use-media-query";
import { cn } from "@/lib/utils";

export function ResponsiveDialog({ open, onOpenChange, title, description, children, className, onDismissAttempt }: {
  open: boolean; onOpenChange: (o: boolean) => void; title: string; description?: string; children: ReactNode; className?: string;
  /** When set, outside clicks are ignored and Escape calls this instead of closing (for dialogs that must not vanish by accident). */
  onDismissAttempt?: () => void;
}) {
  const mobile = useIsMobile();
  const guard = onDismissAttempt ? {
    onInteractOutside: (e: Event) => e.preventDefault(),
    onEscapeKeyDown: (e: KeyboardEvent) => { e.preventDefault(); onDismissAttempt(); },
  } : {};
  if (mobile) {
    return (
      <Sheet open={open} onOpenChange={onOpenChange}>
        <SheetContent side="bottom" className="pb-safe h-[94dvh] overflow-y-auto rounded-t-xl" {...guard}>
          <SheetHeader className="text-left"><SheetTitle>{title}</SheetTitle>{description && <SheetDescription>{description}</SheetDescription>}</SheetHeader>
          <div className="mt-4">{children}</div>
        </SheetContent>
      </Sheet>
    );
  }
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className={cn("max-h-[90dvh] max-w-5xl overflow-y-auto", className)} {...guard}>
        <DialogHeader><DialogTitle>{title}</DialogTitle>{description && <DialogDescription>{description}</DialogDescription>}</DialogHeader>
        {children}
      </DialogContent>
    </Dialog>
  );
}
