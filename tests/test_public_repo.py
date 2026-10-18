"""The public repository, as a stranger who clones it sees it: reproducible image references, local state kept out
of git, and the community files (security policy, issue and pull request templates, licence, README badge)."""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest
import yaml

from dgxmm.config import Images

ROOT = Path(__file__).resolve().parents[1]
DIGEST = re.compile(r"@sha256:[0-9a-f]{64}$")
# a fixed version: a tag with a digit in it (v1.2, 16.15-trixie, 22-bookworm-slim), never latest/stable/main-*
FIXED_TAG = re.compile(r":(?=[^:@/]*\d)[^:@/]+@sha256:[0-9a-f]{64}$")


def tracked(*patterns: str) -> list[Path]:
    out = subprocess.run(["git", "ls-files", "-z", "--", *patterns], cwd=ROOT, capture_output=True, check=True).stdout
    return [ROOT / n for n in out.decode().split("\0") if n]


# --- images --------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("key", ["vllm", "llamacpp"])
def test_engine_image_defaults_are_pinned_by_digest_and_match_the_example(key):
    default = getattr(Images(), key)
    assert DIGEST.search(default), default
    example = yaml.safe_load((ROOT / "config.example.yaml").read_text())["images"][key]
    assert example == default


def test_example_config_says_how_to_update_the_llamacpp_pin():
    text = (ROOT / "config.example.yaml").read_text()
    assert "docker buildx imagetools inspect ghcr.io/ggml-org/llama.cpp:server-cuda" in text


def test_dockerfile_base_images_are_pinned_by_digest():
    files = tracked("*Dockerfile")
    assert len(files) >= 2
    for f in files:
        stages: set[str] = set()
        for m in re.finditer(r"^FROM\s+(\S+)(?:\s+AS\s+(\S+))?", f.read_text(), re.M | re.I):
            base, stage = m.group(1), m.group(2)
            if base not in stages:
                assert DIGEST.search(base), f"{f.relative_to(ROOT)}: FROM {base}"
            if stage:
                stages.add(stage)


def test_compose_images_are_pinned_to_a_fixed_version_and_digest():
    services = yaml.safe_load((ROOT / "docker-compose.yaml").read_text())["services"]
    for name, svc in services.items():
        if "build" in svc:  # built locally from this checkout
            continue
        assert FIXED_TAG.search(svc["image"]), f"{name}: {svc['image']}"


def test_ci_container_images_are_pinned():
    for wf in tracked(".github/workflows/*.yml"):
        for ref in re.findall(r"docker run [^\n]*?\s((?:[\w.-]+/)*[\w.-]+:[\w.-]+(?:@sha256:[0-9a-f]{64})?)\s", wf.read_text()):
            assert DIGEST.search(ref), f"{wf.name}: {ref}"


# --- local state ---------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("path", [".env", "config.yaml", "secrets/admin_password", "data/dgxmm.db", ".venv/bin/python",
                                  "frontend/node_modules/x/index.js", "frontend/dist/index.html", "dgxmm.bundle"])
def test_local_state_is_gitignored(path):
    p = subprocess.run(["git", "check-ignore", "-q", "--no-index", path], cwd=ROOT)
    assert p.returncode == 0, f"{path} is not ignored"


@pytest.mark.parametrize("path", ["config.example.yaml", "deploy/litellm/config.yaml", "frontend/src/lib/x.ts"])
def test_tracked_config_is_not_ignored(path):
    # frontend/src/lib/ is matched by the Python template's lib/ (CONTRIBUTING: git add -f); everything else must not be
    p = subprocess.run(["git", "check-ignore", "-q", "--no-index", path], cwd=ROOT)
    assert (p.returncode == 0) == path.startswith("frontend/src/lib/"), path


def test_no_large_files_are_tracked():
    big = {str(p.relative_to(ROOT)): p.stat().st_size for p in tracked() if p.is_file() and p.stat().st_size > 1_000_000}
    assert big == {}


# --- community files -----------------------------------------------------------------------------------------------

def test_security_policy():
    text = (ROOT / "SECURITY.md").read_text()
    assert "https://github.com/nikghodki/dgxmodelmanager/security/advisories/new" in text
    assert "](docs/security.md)" in text
    for heading in ("## Reporting a vulnerability", "## Supported versions", "## Scope"):
        assert heading in text


def _form(name: str) -> dict:
    form = yaml.safe_load((ROOT / ".github" / "ISSUE_TEMPLATE" / name).read_text())
    assert form["name"] and form["description"] and isinstance(form["body"], list)
    return form


def test_bug_report_form_asks_for_what_triage_needs():
    form = _form("bug_report.yml")
    ids = {b.get("id") for b in form["body"]}
    assert {"version", "host", "status", "logs", "steps"} <= ids
    required = {b["id"] for b in form["body"] if b.get("validations", {}).get("required")}
    assert {"version", "host", "steps"} <= required
    status = next(b for b in form["body"] if b.get("id") == "status")
    assert "./install.sh status" in status["attributes"]["label"] + status["attributes"].get("description", "")
    assert "secret" in yaml.safe_dump(form).lower()


def test_feature_request_form():
    form = _form("feature_request.yml")
    assert any(b.get("validations", {}).get("required") for b in form["body"])


def test_issue_config_points_security_reports_away_from_public_issues():
    cfg = yaml.safe_load((ROOT / ".github" / "ISSUE_TEMPLATE" / "config.yml").read_text())
    assert cfg["blank_issues_enabled"] is False
    assert any("security/advisories/new" in link["url"] for link in cfg["contact_links"])


def test_pull_request_template_checklist():
    text = (ROOT / ".github" / "pull_request_template.md").read_text()
    for item in ("pytest -q", "vitest run", "npm run build", "shellcheck", "tests/test_repo_hygiene.py", "docs/",
                 "CHANGELOG.md"):
        assert item in text, item
    assert text.count("- [ ]") >= 5


def test_no_dependabot_version_updates():
    # only the owner contributes: security alerts can be enabled in the repository settings without a bot opening PRs
    assert not tracked(".github/dependabot.yml", ".github/dependabot.yaml")


def test_readme_licence_badge_and_policy_links():
    readme = (ROOT / "README.md").read_text()
    assert (ROOT / "LICENSE").read_text().startswith("MIT License")
    assert re.search(r"^## License\n\nMIT\b", readme, re.M)
    badge = "https://github.com/nikghodki/dgxmodelmanager/actions/workflows/tests.yml/badge.svg"
    assert badge in readme.split("\n## ")[0]
    assert "](SECURITY.md)" in readme and "](CONTRIBUTING.md)" in readme
