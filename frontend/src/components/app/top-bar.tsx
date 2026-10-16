import { Link, useLocation, useNavigate } from "react-router-dom";
import { ChevronRight, KeyRound, LogOut, Search } from "lucide-react";
import { useAuth } from "@/auth";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuLabel, DropdownMenuSeparator, DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { MemoryChip } from "./memory-chip";
import { titleFor } from "./nav";
import { ThemeToggle } from "./theme";

export function TopBar({ onOpenPalette }: { onOpenPalette: () => void }) {
  const { user, logout } = useAuth();
  const nav = useNavigate();
  const { pathname } = useLocation();
  const t = titleFor(pathname);
  const isMac = typeof navigator !== "undefined" && /Mac|iPhone|iPad/.test(navigator.platform);
  return (
    <header className="sticky top-0 z-30 flex h-14 items-center gap-2 border-b bg-background/85 px-4 backdrop-blur supports-[backdrop-filter]:bg-background/70 md:px-6">
      <nav aria-label="Breadcrumb" className="min-w-0 flex-1">
        <ol className="flex items-center gap-1.5 text-sm">
          <li className="hidden text-muted-foreground md:block"><Link to="/" className="hover:text-foreground">DGX Spark</Link></li>
          <li className="hidden md:block"><ChevronRight className="h-3.5 w-3.5 text-muted-foreground" /></li>
          <li className={t.item ? "hidden truncate text-muted-foreground md:block" : "truncate font-medium"}>{t.section}</li>
          {t.item && <><li className="hidden md:block"><ChevronRight className="h-3.5 w-3.5 text-muted-foreground" /></li><li className="truncate font-medium">{t.item}</li></>}
        </ol>
      </nav>
      <Button variant="outline" size="sm" onClick={onOpenPalette} aria-label="Search or run a command"
        className="h-8 gap-2 text-muted-foreground md:w-64 md:justify-between">
        <span className="flex items-center gap-2"><Search className="h-3.5 w-3.5" /><span className="hidden md:inline">Search or run…</span></span>
        <kbd className="hidden rounded border bg-muted px-1.5 font-mono text-[10px] md:inline">{isMac ? "⌘" : "Ctrl"} K</kbd>
      </Button>
      <MemoryChip />
      <ThemeToggle />
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <Button variant="ghost" size="icon" className="rounded-full" aria-label="Account menu">
            <Avatar className="h-7 w-7"><AvatarFallback className="bg-primary/15 text-xs font-semibold text-primary">{user?.username.slice(0, 2).toUpperCase()}</AvatarFallback></Avatar>
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="end" className="w-52">
          <DropdownMenuLabel className="font-normal"><div className="text-sm font-medium">{user?.username}</div><div className="text-xs capitalize text-muted-foreground">{user?.role}</div></DropdownMenuLabel>
          <DropdownMenuSeparator />
          <DropdownMenuItem onSelect={() => nav("/change-password")}><KeyRound className="mr-2 h-4 w-4" />Change password</DropdownMenuItem>
          <DropdownMenuItem onSelect={() => logout()}><LogOut className="mr-2 h-4 w-4" />Log out</DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>
    </header>
  );
}
