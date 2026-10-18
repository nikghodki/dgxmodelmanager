# remote_code corpus

Provenance / licence: excerpts (config keys, file names, model-card starts, code-scan summaries) of the PUBLIC Hugging Face repositories named in the cases, fetched anonymously from huggingface.co for interoperability testing only; each remains under its repository's own licence. No weights. Built with scripts/decider_eval/build_deploy_corpora.py; remove a repo here on request.

Real repos: scans (findings with one-line excerpts, per-file import lists) of the repos' published `*.py` files, labelled `routine`, except the two d3 decision models (`unusual`: they ship an HTTP server and an urllib client). `samples/`: synthetic files written for this test — malicious patterns (network calls to documentation addresses, subprocess, exec of base64, writes to ~/.ssh, credential reads) labelled `unusual`, each starting with `raise ImportError("inert test sample")`, and ordinary model code (`routine_*`). They are only ever parsed, never run.
