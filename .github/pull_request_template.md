## What and why

<!-- What this changes and why. Link the issue it fixes (Fixes #123). -->

## How it was tested

<!-- Commands you ran, and what you checked on a real box if anything (GPU, model, engine). -->

## Checklist

- [ ] Backend tests pass: `.venv/bin/pytest -q` (includes the installer, docs-link and hygiene tests)
- [ ] Frontend tests and build pass: `cd frontend && npx vitest run && npx tsc --noEmit && npm run build`
- [ ] `shellcheck` is clean for changed shell scripts (`make lint`, or the command in [CONTRIBUTING.md](https://github.com/nikghodki/dgxmodelmanager/blob/main/CONTRIBUTING.md))
- [ ] Tests added or updated for every behaviour change
- [ ] `tests/test_repo_hygiene.py` passes: no user names, host names, private IP addresses or secrets in the diff (use `dgxuser`, `dgx-host`, `192.0.2.x`)
- [ ] Docs in `docs/` updated for changed behaviour, configuration keys or installer options
- [ ] User-visible change added to `[Unreleased]` in `CHANGELOG.md`
- [ ] Decision-model changes stay advisory: rules first, model only above its threshold, nothing applied without an admin's confirmation
- [ ] Commits follow [Conventional Commits](https://www.conventionalcommits.org/)
