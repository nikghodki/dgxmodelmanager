---
description: "The React single-page app: pages, design system, data fetching and live updates."
---

# Frontend

`frontend/` is a React 18 + TypeScript app built with Vite (`base: /dgxmm/`, router basename
`/dgxmm`). The Docker image builds it and serves it from `/app/static`.

- **Pages** (`src/pages/`): Login, Change password, Dashboard, Deployments, Deployment detail, Deploy
  wizard, Models (on disk), Catalog, Downloads, Planner, Processes, LiteLLM, Decision model, Users, Audit
  log. Deploy wizard, Catalog, Planner, Processes, LiteLLM, Decision model, Deployment detail and Audit
  are lazy-loaded.
- **Design system** (`src/components/ui/`): shadcn/ui components on Radix primitives, styled with
  Tailwind CSS tokens. There is a light theme and a dark theme. Status is always shown as icon plus
  text, and Inter and JetBrains Mono fonts are self-hosted. App building blocks (shell, sidebar, data
  tables, memory bar, plan chooser, command palette) live in `src/components/app/`.
- **Data**: TanStack Query for every API call (`src/api.ts` adds the CSRF header). `src/stream.ts`
  opens one `EventSource` on `/dgxmm/api/stream`: `budget` events update the host data in place,
  `metrics` events are appended to the live chart series, and `deployment`, `download` and `inventory`
  events refresh the matching queries; `diagnosis` refreshes a deployment's diagnosis and the
  admin-only `decider` topic carries image build progress.
- **Charts**: Recharts with a fixed palette that has light and dark variants.

## Next steps

- [Request and data flow](request-flow.md)
- [Contributing](../contributing/index.md)
