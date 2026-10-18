"""L5: GitHub Actions workflows pin every action by full commit SHA and run with read-only repository access (the
Pages deploy job alone may write to Pages)."""
import pathlib
import re

import pytest

WF = sorted((pathlib.Path(__file__).resolve().parents[1] / ".github" / "workflows").glob("*.y*ml"))


def test_there_are_workflows():
    assert WF


@pytest.mark.parametrize("path", WF, ids=[p.name for p in WF])
def test_actions_are_pinned_by_commit_sha(path):
    uses = re.findall(r"^\s*-?\s*uses:\s*(\S+)", path.read_text(), re.M)
    assert uses
    for u in uses:
        if u.startswith(("./", "docker://")):
            continue
        assert re.fullmatch(r"[\w.-]+/[\w./-]+@[0-9a-f]{40}", u), u


@pytest.mark.parametrize("path", WF, ids=[p.name for p in WF])
def test_default_token_permissions_are_read_only(path):
    text = path.read_text()
    assert re.search(r"^permissions:\s*\n\s+contents:\s*read\s*$", text, re.M), "top-level permissions: contents: read"
    writes = re.findall(r"^\s*([\w-]+):\s*write\b", text, re.M)
    # the only write access: the Pages deploy job (pages.yml), which needs pages + id-token (OIDC) to publish
    allowed = ["pages", "id-token"] if path.name == "pages.yml" else []
    assert writes == allowed, writes
