# DGXMM UI Revamp: Design Spec

- **Date:** 2026-10-16
- **Status:** Approved in brainstorming; awaiting written-spec review
- **Scope:** `frontend/` only. Backend API, routes and behaviour are unchanged.

## 1. Goal

Make the DGXMM web UI look and behave like a modern, professional infrastructure console. The reference points are Vercel, Linear and the Stripe/Datadog dashboards. It should work well:

- on a desktop browser (main operator use);
- on a phone over Tailscale (quick checks and actions);
- when shown to others (demos, where polish and first impressions matter).

### Success criteria

- Every page uses one coherent design system, with light and dark themes.
- The Dashboard answers "can I deploy?" and "is anything wrong?" at a glance.
- Every page works on a phone (≥ 360 px wide) with no horizontal scrolling of the page itself. Wide tables become stacked rows.
- Loading, empty, error and in-progress states are designed for every data view. Browser `confirm()`/`prompt()` are not used anywhere.
- WCAG AA contrast in both themes. Every control is reachable by keyboard and has a visible focus ring. Status is never shown by colour alone.
- Existing features keep working. The frontend unit tests and build pass, and the backend pytest suite stays green.

## 2. Decisions (from brainstorming)

| Topic | Decision |
|---|---|
| Visual direction | **Dark = "Graphite"** (Vercel/Linear: near-black, hairline borders). **Light = "Enterprise"** (Stripe/Datadog: bright, soft shadows). One shared indigo-blue accent. |
| Theme selection | Light / Dark / System toggle in the top bar. Default System. Persisted in localStorage and applied before first paint (no flash). |
| Navigation | Grouped, collapsible sidebar plus top bar on desktop. Bottom tab bar plus "More" sheet on phones. |
| Dashboard | Memory hero → live KPI tiles → deployments **table** → unmanaged containers → external processes. |
| Component approach | **shadcn/ui** (Radix primitives, Tailwind, CSS-variable tokens; source copied into the repo), lucide-react icons, `cmdk` command palette, `sonner` toasts, TanStack Table, Recharts restyled. |

## 3. Design tokens

CSS variables in `src/index.css`, in shadcn HSL format. Dark values apply under `.dark` on `<html>`; light values are the default `:root` values.

| Token | Light (Enterprise) | Dark (Graphite) |
|---|---|---|
| `--background` | `#f6f8fb` | `#0a0a0b` |
| `--card` / `--popover` | `#ffffff` | `#111114` |
| `--muted` | `#f1f4f8` | `#18181b` |
| `--border` / `--input` | `#e5e9f0` | `#1f1f23` |
| `--foreground` | `#0f172a` | `#ededef` |
| `--muted-foreground` | `#475569` | `#a1a1aa` |
| `--primary` (accent) | `#2563eb` (blue-600) | `#6366f1` (indigo-500) |
| `--primary-foreground` | `#ffffff` | `#ffffff` |
| `--ring` | `#2563eb` | `#818cf8` |
| `--sidebar` | `#ffffff` | `#0f0f11` |
| shadow (cards) | `0 1px 2px rgb(16 24 40 / .04)` | none (borders only) |

**Status colours** are reserved and never used for chart series. Each always comes with an icon and a text label.

| State | Meaning | Light fg/bg | Dark fg/bg | Icon |
|---|---|---|---|---|
| good | ready, synced, done | `#047857` / `#e7f8ef` | `#4ade80` / `#10291c` | CheckCircle2 |
| warning | starting, loading, pending, warning alert | `#b45309` / `#fef3c7` | `#fbbf24` / `#2d2208` | Loader2 (spin) / AlertTriangle |
| sleeping | sleeping | `#6d28d9` / `#f1ecff` | `#c4b5fd` / `#231b3a` | Moon |
| critical | failed, critical alert | `#b91c1c` / `#fee2e2` | `#f87171` / `#3b1111` | XCircle |
| neutral | stopped, unknown, none | `#475569` / `#f1f5f9` | `#a1a1aa` / `#1c1c21` | Circle |

**Chart colours:** the data-viz reference 8-slot categorical palette in fixed order.
- Light steps: `#2a78d6 #eb6834 #1baf7a #eda100 #e87ba4 #008300 #4a3aa7 #e34948`.
- Dark steps: `#3987e5 #d95926 #199e70 #c98500 #d55181 #008300 #9085e9 #e66767`.
- Both are validated with `scripts/validate_palette.js` against `--card` in each mode.
- Memory-bar fixed roles: managed deployments use slots 1…n in deployment-id order (colour follows the entity). External uses neutral gray (`#94a3b8` light / `#6b6b70` dark). Margin and OS reserve use `--muted`. Free uses a 30% tint of good.

**Typography:**
- Inter (self-hosted via `@fontsource-variable/inter`, so no CDN is needed on the LAN). Monospace is JetBrains Mono (self-hosted) for model refs, endpoints, logs and hashes.
- `font-variant-numeric: tabular-nums` on all numeric cells and stats.
- Scale: page title 20/28 semibold; section title 14/20 semibold; body 14/20; small 12/16; label 11/16 uppercase, tracking 0.05em.

**Shape and spacing:** radius 8 px for cards and inputs, 6 px for buttons, full for pills. 4 px spacing grid. Content max width 1440 px.

## 4. Component layer

### 4.1 shadcn/ui primitives (`src/components/ui/`)

Button, Card, Badge, Table, Dialog, AlertDialog, Sheet, DropdownMenu, Tabs, Tooltip, Popover, Select, Input, Label, Switch, Checkbox, Command, Skeleton, Progress, Separator, Avatar, ScrollArea, and Sonner (`Toaster`).

Source is generated or copied per the shadcn conventions (`components.json`, `cn()` in `src/lib/utils.ts`, class-variance-authority).

### 4.2 App components (`src/components/app/`)

| Component | Responsibility |
|---|---|
| `ThemeProvider` / `ThemeToggle` | light/dark/system; localStorage key `dgxmm-theme`. An inline script in `index.html` sets the class before React mounts. |
| `AppShell` | Sidebar + top bar + content + mobile tab bar. Sidebar collapse state persisted (`dgxmm-sidebar`). |
| `CommandPalette` | ⌘K / Ctrl-K / the top-bar search button. Groups: Pages; Deployments (open, sleep, wake, stop); Models (deploy); Actions (deploy a model, toggle theme, log out). Destructive actions go through `ConfirmDialog`. Admin-only actions are hidden for viewers. |
| `MemoryChip` | Top-bar pill with free-for-new memory and a status dot. Opens a popover containing `MemoryBar`. |
| `AlertBanner` | Watchdog warning/critical banner under the top bar. |
| `PageHeader` | Title, description, actions slot, breadcrumbs. |
| `StatusBadge` | Maps a deployment, download or LiteLLM state to the status table in §3 (icon + label). |
| `StatTile` | Label, value, optional sub-text, optional sparkline (2 px line, no axes, hover tooltip with value). |
| `MemoryBar` | Stacked horizontal bar, 2 px gaps, 4 px rounded ends, legend with values, per-segment tooltip, "view as table" toggle. |
| `DataTable` | TanStack Table wrapper: column sorting, a text filter, optional facet filter chips, sticky header, row actions (`⋯` menu). Below 768 px it renders each row as a stacked card (primary column as the title, others as label/value pairs). |
| `EmptyState` / `ErrorState` | Icon, title, text, primary action / retry. |
| `ConfirmDialog` | AlertDialog wrapper used for stop/remove/delete/reset-password; optional typed confirmation for destructive actions on ready deployments. |
| `CopyButton` | Copies an endpoint or ref, with toast feedback. |
| `ChartCard` | Card with a title, window selector and "view as table" toggle around a Recharts chart (theme-aware colours read from CSS variables). |

## 5. App shell

- **Sidebar** (≥ 768 px), with lucide icons and labels:
  - **Operate:** Dashboard (`LayoutDashboard`), Deployments (`Server`), Planner (`Puzzle`).
  - **Models:** On disk (`HardDrive`), Catalog (`Search`), Downloads (`Download`).
  - **Admin**, only for admins: Users (`Users`), Audit log (`ScrollText`).
  - Collapsible to an icon rail (tooltips on hover). The footer shows the host name (`spark-83dc`) and app version.
- **Top bar:**
  - breadcrumbs (`spark-83dc / Page / Item`);
  - the command palette trigger ("Search or run… ⌘K");
  - `MemoryChip`, `ThemeToggle`;
  - a user menu (avatar initials): Change password, Log out.
- **Mobile** (< 768 px):
  - The top bar shows the page title and the palette and theme buttons.
  - A fixed bottom tab bar (Home, Deployments, Models, More) with 44 px targets. "More" opens a bottom Sheet with Planner, Catalog, Downloads, Users, Audit, Change password and Log out.
  - Dialogs become full-height Sheets. Content has bottom padding for the tab bar and the safe-area inset.
- **Feedback:**
  - Action results show as sonner toasts (success, or error carrying the API `detail.message`).
  - Mutating buttons show a spinner and are disabled while pending.
  - The current inline `ErrorBox` stays only for form validation and query load errors.

## 6. Pages

All routes stay as they are, with one addition: **`/deployments`** (list). Every API call stays the same.

- **Login / Change password:** a centred card on a subtle grid/radial background, with the product mark and name. Password show/hide. Inline validation (12+ characters, match). Enter submits.
- **Dashboard** (`/`):
  1. **Memory hero card:**
     - Left: "Free for new deployments" (large number, green/amber/red by headroom), the total and margin sub-text, and buttons "Deploy model" (primary, admin) and "Plan combinations".
     - Right: `MemoryBar` with legend.
  2. **KPI row:** four `StatTile`s, aggregated over ready deployments from the live SSE samples.
     - Generation tok/s (sum, sparkline).
     - TTFT p50 (worst ready deployment, sparkline).
     - KV cache (max, progress bar).
     - Queue (waiting/running).
  3. **Deployments `DataTable`:**
     - Columns: Name (link), Model, Engine, State (`StatusBadge` + error text on hover), Memory (reserved / measured), Tok/s, KV, LiteLLM.
     - Row menu: Open, Sleep/Wake, Switch…, Reconfigure…, Stop, Remove.
     - `EmptyState` when there are none.
  4. **"Containers not managed by DGXMM"** (collapsible card): adoptable or not-adoptable reason, and an Adopt action.
  5. **"Largest external memory users"**: a compact list (pid, name, memory).
  6. **"Memory over the last hour"**: a `ChartCard` (used / external / free for new).
- **Deployments** (`/deployments`): the same table full-width, with state facet chips (All / Ready / Sleeping / Starting / Failed / Stopped) and search.
- **Deployment detail** (`/deployments/:id`):
  - Header: name, `StatusBadge`, model, engine and action buttons (Sleep/Wake, Reconfigure, Stop, Logs, more menu with Remove). `last_error` shows in a critical callout.
  - Tabs:
    - **Overview:** endpoint cards with `CopyButton`; memory reserved / measured with estimate breakdown; configuration as key/value.
    - **Performance:** `ChartCard`s for throughput, TTFT p50/p95, KV and queue, and ITL / spec-accept, with a live / 24h / 7d selector.
    - **Advice:** cards with severity badge, rationale, expected effect, confidence badge and Apply… (only when applicable).
    - **Benchmarks:** a run button, a results table, and a tok/s-by-concurrency chart per config.
    - **Events:** a timeline with expandable log tails.
  - Logs open in a right-side Sheet (mono, auto-scroll, refresh).
- **Deploy wizard** (`/deploy`):
  - Left column: four stacked sections (Model with a searchable select showing quant/size badges; Engine & name; Sizing; Advanced, collapsed by default).
  - Right column: a sticky **Memory check** card with a fit verdict badge, "needs X of Y free", a breakdown bar (weights / KV / state / activations / runtime / margin), KV pool tokens, the vLLM utilisation, llmfit line and confidence.
  - When it doesn't fit, plans appear as selectable cards and the Deploy button label reflects the chosen plan.
  - Mobile: one column, with a sticky bottom summary bar (verdict + Deploy).
- **Switch / Reconfigure:** the same two-column layout inside a large Dialog (desktop) or a full Sheet (mobile).
- **Models on disk:**
  - A `DataTable` (Model with revision, Kind, Format/quant badges, Weights, Context, Architecture, Status), search and Kind/Status facets.
  - Problems show in an expandable row detail.
  - Row action: Deploy. Header action: Rescan (admin).
- **Catalog:**
  - A search bar (query, GGUF toggle, context input).
  - A results `DataTable` (Model with badges, Downloads, Approx size, Fit badge, Engines as icons with tooltips, llmfit).
  - Row click opens a right-side **details Sheet** (estimate, support, GGUF files with per-file download, variants, download safetensors). The llmfit-unavailable banner is a warning callout.
- **Downloads:** a `DataTable` with a progress bar (percent, bytes, speed and ETA computed client-side from successive SSE updates), state badge, error text and Cancel.
- **Planner:**
  - A capacity summary card.
  - Best combinations as ranked cards, each with a stacked capacity bar (models + headroom) and model badges.
  - A candidates `DataTable` with a favourite star toggle and a must-include checkbox.
- **Users:** a `DataTable` (username, role select, must-change badge, actions). Add user, Reset password and Delete go through Dialogs / `ConfirmDialog`.
- **Audit log:** a `DataTable` with an action facet filter, relative time with an absolute tooltip, and expandable JSON detail.

## 7. Data and behaviour

- **Unchanged:** react-query keys and SSE handling (`useLiveUpdates`, `useLive`), and `api.ts` with its CSRF handling.
- Toast-on-mutation is added through a shared `useActionMutation(fn, {success, invalidate})` helper.
- Theme-aware chart colours are read once from CSS variables via `useChartColors()`, which re-reads on theme change.
- No new runtime dependencies beyond:
  - `@radix-ui/*` (via shadcn components), `class-variance-authority`, `clsx`, `tailwind-merge`, `tailwindcss-animate`;
  - `lucide-react`, `cmdk`, `sonner`, `@tanstack/react-table`;
  - `@fontsource-variable/inter`, `@fontsource-variable/jetbrains-mono`.
- Keep the bundle reasonable: route-level code splitting with `React.lazy` for the Catalog, Planner, Deployment detail and Audit pages. Target an initial JS of no more than about 350 kB gzip.

## 8. Accessibility

- Radix provides focus handling and ARIA for dialogs, menus, tabs and popovers.
- Every icon-only button has an `aria-label`.
- Visible 2 px focus ring (`--ring`).
- Contrast AA for text in both themes.
- `prefers-reduced-motion` disables non-essential animation.
- Every chart has a "view as table" toggle. Status is always shown as icon + label.

## 9. Testing

- **vitest + @testing-library/react + jsdom** (switch the vitest environment to jsdom for component tests):
  - `ThemeToggle` persists and applies the class.
  - `CommandPalette` filters, runs navigation, and hides admin actions for viewers.
  - `DataTable` sorts, filters and renders stacked cards under a mocked narrow viewport.
  - `MemoryBar` segment widths/values and the table view.
  - `StatusBadge` mapping for every state.
  - The existing `format`/`logic` tests are kept.
- Run the palette validator for both modes and record the output in the PR/commit notes.
- `npm run build` must succeed with no TypeScript errors. Backend `pytest -q` stays green (the backend is untouched).
- **Visual check** with the DevTools browser tool, using a temporary smoke account (created and deleted): every page in light and dark at 1440×900 and 390×844, with screenshots reviewed for overflow, label collisions and contrast.

## 10. Out of scope

- Backend/API changes.
- New product features beyond the `/deployments` list page and replacing `confirm()`/`prompt()` with dialogs.
- Internationalisation.
- PWA/offline.
