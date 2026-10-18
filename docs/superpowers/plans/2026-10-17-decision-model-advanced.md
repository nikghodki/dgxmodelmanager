# Advanced decision-model features — implementation plan

> For agentic workers: executed with superpowers:subagent-driven-development (lane = task; implementer per lane, task review, fix rounds, final whole-branch review).

**Goal:** implement the eleven features of the spec, deploy and verify them live on the DGX, and leave the public repo ready for other DGX owners to deploy.

**Architecture:**
- A foundation lane adds the feature registry, the insights feed and the background engine.
- Three feature lanes then build on it in parallel worktrees.
- An integration lane merges them and evaluates on real data live.
- A public-readiness lane prepares the repo for other users.
- Docs, the security review and the final review close it out.

**Spec:** /tmp/dgxmm-adv/spec.md (= `docs/superpowers/specs/2026-10-17-decision-model-advanced-design.md` on `feature/decider-advanced`). **Shared rules:** /tmp/dgxmm-adv/context.md.

## Global constraints
- Identity is nikghodki noreply only, with no trailers.
- The repo is public, so these never appear in it: forbidden names, box identifiers, real IPs.
- Principles from the spec are binding.
- The live app stays up and healthy, and the user's containers are never touched.
- MemAvailable ≥ 20 GiB.

## Review focus
Inputs the lane tests are unlikely to exercise; reviewers check these deliberately:
1. **Upgrading an existing install** from main 7c055b7 with the decider disabled: nothing new starts and nothing breaks; migrations are additive and run once.
2. **Decider slow or down while the background jobs run:** jobs back off, interactive features are not starved, and at most one "degraded" insight is raised.
3. **Hostile inputs:** ⌘K text, a model card or config fetched for the compatibility check, or a `.py` file in a snapshot crafted to steer the answer, crash the parser, or run code.
4. **A schedule firing at a bad moment:** while traffic arrives, during a deploy or startup queue, after the deployment was removed, across a DST change, or during a DGXMM restart. It must never sleep a busy deployment, never evict anything, and never run twice.
5. **Viewer-only users:** they see insights and navigation commands but cannot act; no admin data leaks through SSE.

## Lanes (tasks)

| # | Lane | Branch / worktree | Base | Depends on |
|---|---|---|---|---|
| 1 | **A0 Foundation:** FeatureSpec registry (the existing 5 migrated with unchanged behaviour), insights table/API/SSE/UI (page, top-bar badge, deployment card), insight engine (jobs, budget, background lane in the decider client limiter), migrations, generalised evaluation harness (`tools/eval_feature.py`), commit the plan to `docs/superpowers/plans/` | `adv-f0` / `dgxmm-wt/f0` | `feature/decider-advanced` | — |
| 2 | **A1 Deploy-time:** features 2 config advice, 3 compat, 4 draft pairing, 5 engine choice, 11 remote-code risk (backend, wizard/catalog UI, corpora, real-model evaluation) | `adv-a1` / `dgxmm-wt/a1` | `adv-f0` head | 1 |
| 3 | **A2 Operations:** features 1 health triage, 8 reading labels, 9 threshold tuning, 10 routing advice (backend jobs, insights, UI on deployment detail / benchmarks / LiteLLM page / Decision model page, corpora, evaluation) | `adv-a2` / `dgxmm-wt/a2` | `adv-f0` head | 1 |
| 4 | **A3 Schedules + ⌘K:** feature 7 traffic profiles, proposals, schedules table/API/runner/UI; feature 6 command interpretation API and ⌘K "Ask" mode (corpus ≥ 150 utterances, evaluation) | `adv-a3` / `dgxmm-wt/a3` | `adv-f0` head | 1 |
| 5 | **P Public readiness** (parallel with lanes 2–4, files disjoint from feature code): SECURITY.md (reporting, scope), issue/PR templates, pin the llama.cpp image by digest in config defaults, `config.example.yaml`/`.env.example` sanity for a fresh DGX, fresh-clone install verification in an isolated environment (DinD or equivalent; dry-run plus the installer suite where a real install is impossible), README quick start checked command by command, CI workflows valid (actionlint), repo-wide hygiene sweep (incl. `docs/superpowers/` history docs read as public), `.gitignore`/`.dockerignore` sanity, no dependabot version-update config (only nikghodki contributes) | `adv-pub` / `dgxmm-wt/pub` | `feature/decider-advanced` | — |
| 6 | **A4 Integration + live:** merge lanes 2–5; reconcile; full suites; deploy to the live box via `./install.sh` (umask 022, before/after state, rollback path); live end-to-end of all 16 features; real-data evaluation numbers into `docs/decider-evaluation.md`; browser check of every new surface (light/dark/mobile) | `adv-int` / `dgxmm-wt/int` | `adv-f0` | 2–5 |
| 7 | **Security review** of `main...adv-int`, then one fix wave and a scoped re-review | `adv-int` | — | 6 |
| 8 | **D Docs:** README (feature list, screenshots re-captured with privacy check), usage (Insights, each new feature, ⌘K Ask, schedules), configuration (every new setting from the code), architecture (registry, insights, engine, scheduler), operations (upgrade, troubleshooting new jobs), security.md (new paths), CHANGELOG release entry, CONTRIBUTING; link check | `adv-int` | — | 7 |
| 9 | **Final whole-branch review** (most capable model), then one fix wave | `adv-int` | — | 8 |
| 10 | **Release:** squash `adv-int` onto main as one nikghodki commit (no trailers); verify the tree hygiene and that the tree equals `adv-int`; push main (fast-forward); check the GitHub Actions runs pass via the API; fast-forward the live checkout and `./install.sh`; verify; clean up worktrees and branches; update memory | — | — | 9 |

Each lane brief: /tmp/dgxmm-adv/brief-<lane>.md, report /tmp/dgxmm-adv/report-<lane>.md, ledger /tmp/dgxmm-adv/progress.md.
