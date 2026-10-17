# Test fixtures

Each family has its own provenance note:

| Directory | What | Origin |
|---|---|---|
| [configs/](configs/README.md) | `config.json` of public models | public Hugging Face repositories |
| [db/](db/README.md) | schema dumps of earlier releases | this project |
| [decider_corpus/](decider_corpus/README.md) | config / template excerpts of public models | public Hugging Face repositories ([repos.txt](decider_corpus/repos.txt)) |
| [engine_logs/](engine_logs/README.md) | engine start-up log excerpts | this project's test runs |
| `install_stubs/` | fake `docker`, `nvidia-smi`, `id`, `ip`, `ss`, `stat`, `uname`, `hostname` for `tests/test_install_script.py`, driven by `FAKE_*` variables (no README inside: shellcheck lints every file there) | this project |
| [real_failures/](real_failures/README.md) | induced failed-start logs | this project, on a DGX Spark |
| [workload/](workload/README.md) | labelled workload windows | this project, synthetic traffic |
