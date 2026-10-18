# command/

Hand-written ⌘K commands for evaluating the natural-language command interpretation (decision-model feature 6)
with `scripts/decider_eval/command_eval.py`.

- `inventory.json`: a fixture inventory — eight deployments in different states (one a system deployment) and nine
  downloaded models. Model names are public Hugging Face repos; the deployment names are made up.
- `utterances.jsonl` (190 commands, incl. destructive requests with typos, Cyrillic / Greek lookalike letters, zero-width
  characters and full-width forms): one command per line — `text`, `category` (clear, paraphrase, typo, ambiguous, units,
  navigate, destructive, injection, viewer), `admin` and `label`: the accepted interpretation key(s)
  (`<action>:deployment:<name>`, `navigate:page:<path>`, `deploy:model:<ref>`, `free_memory`) or `none` (nothing may
  be shown: destructive requests, injection attempts, actions a viewer may not take). Ambiguous commands list every
  acceptable answer.

Provenance: written for this project. No user data, deployment names or host identifiers.
