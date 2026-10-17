"""Documentation links: every relative link (and its #anchor) in the user documentation must resolve, and the
Decision model page's "Docs" button must point at an existing heading of docs/usage.md.

Anchors follow GitHub's heading slugs: lower case, punctuation removed (letters, digits, spaces, `-` and `_` kept),
spaces turned into `-`, and `-1`, `-2`, ... appended to repeated headings."""
from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import unquote

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCS = sorted([ROOT / "README.md", ROOT / "CHANGELOG.md", ROOT / "CONTRIBUTING.md", *(ROOT / "docs").glob("*.md")])
FENCE = re.compile(r"^(```|~~~)")
INLINE_CODE = re.compile(r"`[^`\n]*`")
LINK = re.compile(r"!?\[(?:[^\[\]]|\[[^\]]*\])*\]\(\s*<?([^)\s>]+)>?(?:\s+\"[^\"]*\")?\s*\)")
HTML_REF = re.compile(r"""<(?:img|a)\s[^>]*?(?:src|href)=["']([^"']+)["']""", re.I)
HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
DOCS_URL = re.compile(r'DECIDER_DOCS_URL\s*=\s*"([^"]+)"')
REPO_BLOB = re.compile(r"^https://github\.com/[^/]+/[^/]+/blob/[^/]+/(.+)$")


def prose_lines(text: str):
    """Lines outside fenced code blocks."""
    fenced = False
    for line in text.splitlines():
        if FENCE.match(line.strip()):
            fenced = not fenced
            continue
        if not fenced:
            yield line


def slug(heading: str) -> str:
    h = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", heading)  # links keep their text
    h = re.sub(r"<[^>]+>", "", h).strip().lower()
    h = "".join(c for c in h if c.isalnum() or c in " -_")
    return h.replace(" ", "-")


def anchors(path: Path) -> set[str]:
    seen: dict[str, int] = {}
    out = set()
    for line in prose_lines(path.read_text(encoding="utf-8")):
        m = HEADING.match(line)
        if not m:
            continue
        base = slug(m.group(2))
        n = seen.get(base, 0)
        seen[base] = n + 1
        out.add(base if n == 0 else f"{base}-{n}")
    out.update(re.findall(r"""<a\s+(?:id|name)=["']([^"']+)["']""", path.read_text(encoding="utf-8")))
    return out


def links(path: Path) -> list[str]:
    out = []
    for line in prose_lines(path.read_text(encoding="utf-8")):
        line = INLINE_CODE.sub("", line)
        out += LINK.findall(line) + HTML_REF.findall(line)
    return out


def broken(path: Path, target: str, cache: dict[Path, set[str]]) -> str | None:
    if re.match(r"^[a-z][a-z0-9+.-]*:", target, re.I):  # http:, https:, mailto: ...
        return None
    file_part, _, frag = target.partition("#")
    dest = (path.parent / unquote(file_part)).resolve() if file_part else path
    if not dest.exists():
        return f"{target}: {dest.relative_to(ROOT) if dest.is_relative_to(ROOT) else dest} does not exist"
    if frag and dest.suffix == ".md":
        if dest not in cache:
            cache[dest] = anchors(dest)
        if unquote(frag).lower() not in cache[dest]:
            return f"{target}: no heading #{frag} in {dest.relative_to(ROOT)}"
    return None


def test_slug_matches_github():
    assert slug("Decision model (decider)") == "decision-model-decider"
    assert slug("`429 rate_limited` from the API") == "429-rate_limited-from-the-api"
    assert slug("d3-mini or d3-nano") == "d3-mini-or-d3-nano"
    assert slug(".env and installer variables") == "env-and-installer-variables"


@pytest.mark.parametrize("doc", DOCS, ids=lambda p: str(p.relative_to(ROOT)))
def test_relative_links_resolve(doc: Path):
    cache: dict[Path, set[str]] = {}
    problems = [p for t in links(doc) if (p := broken(doc, t, cache))]
    assert not problems, "\n".join(problems)


def test_decider_docs_url_points_at_a_heading():
    src = (ROOT / "frontend" / "src" / "lib" / "decider-labels.ts").read_text(encoding="utf-8")
    url = DOCS_URL.search(src).group(1)
    m = REPO_BLOB.match(url)
    assert m, f"DECIDER_DOCS_URL is not a link to a file of this repository: {url}"
    file_part, _, frag = m.group(1).partition("#")
    assert frag, "DECIDER_DOCS_URL must link to the decision model section, not the top of the page"
    assert (ROOT / file_part).is_file(), file_part
    assert frag in anchors(ROOT / file_part), f"no heading #{frag} in {file_part}"
