"""Documentation links: every relative link (and its #anchor) in the user documentation must resolve, every link to
the documentation site must point at an existing page (and heading), and the Decision model page's "Docs" button
must point at the decision model's page of the site.

Anchors follow GitHub's heading slugs: lower case, punctuation removed (letters, digits, spaces, `-` and `_` kept),
spaces turned into `-`, and `-1`, `-2`, ... appended to repeated headings. The site uses the same ids
(mkdocs.yml: toc slugify = pymdownx.slugs.slugify, lower case), so one #anchor works on github.com and on the site;
`mkdocs build --strict` checks the site's side."""
from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import unquote

import pytest

ROOT = Path(__file__).resolve().parents[1]
SITE = "https://nikghodki.github.io/dgxmodelmanager/"
NOT_PAGES = ("superpowers", "overrides")  # design history and theme overrides: not part of the site (mkdocs.yml)
PAGES = sorted(p for p in (ROOT / "docs").rglob("*.md") if p.relative_to(ROOT / "docs").parts[0] not in NOT_PAGES)
DOCS = sorted([ROOT / "README.md", ROOT / "CHANGELOG.md", ROOT / "CONTRIBUTING.md", ROOT / "SECURITY.md", *PAGES])
FENCE = re.compile(r"^(```|~~~)")
INLINE_CODE = re.compile(r"`[^`\n]*`")
LINK = re.compile(r"!?\[(?:[^\[\]]|\[[^\]]*\])*\]\(\s*<?([^)\s>]+)>?(?:\s+\"[^\"]*\")?\s*\)")
HTML_REF = re.compile(r"""<(?:img|a)\s[^>]*?(?:src|href)=["']([^"']+)["']""", re.I)
HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
DOCS_URL = re.compile(r'DECIDER_DOCS_URL\s*=\s*"([^"]+)"')
SITE_LINK = re.compile(re.escape(SITE) + r"[^\s)\"'<>]*")


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


def site_page(url: str) -> tuple[Path | None, str]:
    """The docs/ source of a documentation-site URL (directory URLs: a/b/ is docs/a/b.md or docs/a/b/index.md)."""
    path, _, frag = url[len(SITE):].partition("#")
    path = path.strip("/")
    candidates = [ROOT / "docs" / "index.md"] if not path else [ROOT / "docs" / f"{path}.md",
                                                                ROOT / "docs" / path / "index.md"]
    return next((c for c in candidates if c.is_file()), None), frag


def broken_site_link(url: str, cache: dict[Path, set[str]]) -> str | None:
    page, frag = site_page(url)
    if page is None or page.relative_to(ROOT / "docs").parts[0] in NOT_PAGES or page == ROOT / "docs" / "README.md":
        return f"{url}: no page of the site"
    if frag:
        if page not in cache:
            cache[page] = anchors(page)
        if frag not in cache[page]:
            return f"{url}: no heading #{frag} in {page.relative_to(ROOT)}"
    return None


def broken(path: Path, target: str, cache: dict[Path, set[str]]) -> str | None:
    if target.startswith(SITE):
        return broken_site_link(target, cache)
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


def test_site_page_maps_directory_urls_to_sources():
    assert site_page(SITE) == (ROOT / "docs" / "index.md", "")
    assert site_page(SITE + "architecture/") == (ROOT / "docs" / "architecture" / "index.md", "")
    assert site_page(SITE + "user-guide/deploying/#planner") == (ROOT / "docs" / "user-guide" / "deploying.md", "planner")
    assert site_page(SITE + "no-such-page/")[0] is None


def test_site_urls_in_source_and_docs_resolve():
    # the README links to the site; so may code comments and the installer's messages
    files = [ROOT / "README.md", ROOT / "install.sh", *(ROOT / "frontend" / "src").rglob("*.ts*"),
             *(ROOT / "dgxmm").rglob("*.py")]
    cache: dict[Path, set[str]] = {}
    problems = [f"{f.relative_to(ROOT)}: {p}" for f in files
                for url in SITE_LINK.findall(f.read_text(encoding="utf-8")) if (p := broken_site_link(url, cache))]
    assert not problems, "\n".join(problems)


def test_decider_docs_url_points_at_the_decision_model_page():
    src = (ROOT / "frontend" / "src" / "lib" / "decider-labels.ts").read_text(encoding="utf-8")
    url = DOCS_URL.search(src).group(1)
    assert url.startswith(SITE), f"DECIDER_DOCS_URL is not a link to the documentation site: {url}"
    page, frag = site_page(url)
    assert page == ROOT / "docs" / "user-guide" / "decision-model.md", url
    assert not frag or frag in anchors(page), f"no heading #{frag} in {page.relative_to(ROOT)}"
