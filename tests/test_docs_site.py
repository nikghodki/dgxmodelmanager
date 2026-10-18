"""The documentation site (mkdocs.yml, built by .github/workflows/pages.yml): every nav entry exists, every page is in
the nav, pages describe themselves for search, the build's dependencies are pinned with hashes, and the site loads
nothing from third parties. `mkdocs build --strict` (the tests workflow's docs job) checks links and anchors."""
from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
NOT_PAGES = ("superpowers", "overrides")


class _Loader(yaml.SafeLoader):
    """mkdocs.yml uses !!python/name and !!python/object/apply tags (slugify, emoji); they are not needed here."""


_Loader.add_multi_constructor("tag:yaml.org,2002:python/", lambda loader, suffix, node: None)


def config() -> dict:
    return yaml.load((ROOT / "mkdocs.yml").read_text(encoding="utf-8"), Loader=_Loader)  # noqa: S506 (SafeLoader)


def nav_files(nav) -> list[str]:
    out: list[str] = []
    for item in nav:
        value = next(iter(item.values())) if isinstance(item, dict) else item
        if isinstance(value, list):
            out += nav_files(value)
        else:
            out.append(value)
    return out


def pages() -> list[str]:
    return sorted(str(p.relative_to(DOCS)) for p in DOCS.rglob("*.md")
                  if p.relative_to(DOCS).parts[0] not in NOT_PAGES and p != DOCS / "README.md")


def test_every_nav_entry_exists():
    missing = [f for f in nav_files(config()["nav"]) if not (DOCS / f).is_file()]
    assert missing == []


def test_every_page_is_in_the_nav_once():
    nav = nav_files(config()["nav"])
    assert sorted(nav) == pages()
    assert len(nav) == len(set(nav))


def test_excluded_paths_match_what_is_not_a_page():
    excluded = config()["exclude_docs"].split()
    assert excluded == ["README.md", "superpowers/", "overrides/"]


@pytest.mark.parametrize("page", pages())
def test_page_has_a_title_and_a_description(page):
    text = (DOCS / page).read_text(encoding="utf-8")
    m = re.match(r"---\n(.*?)\n---\n", text, re.S)
    assert m, f"{page}: no front matter"
    meta = yaml.safe_load(m.group(1))
    assert 40 <= len(meta.get("description", "")) <= 200, f"{page}: description"
    assert re.search(r"^# \S", text[m.end():], re.M), f"{page}: no H1 title"


def test_site_settings():
    cfg = config()
    assert cfg["site_url"] == "https://nikghodki.github.io/dgxmodelmanager/"
    assert cfg["repo_url"] == "https://github.com/nikghodki/dgxmodelmanager"
    assert cfg["edit_uri"] == "edit/main/docs/"
    assert cfg["strict"] is True
    assert "MIT" in cfg["copyright"]


def test_site_loads_nothing_from_third_parties():
    cfg = config()
    assert cfg["theme"]["font"] is False  # no Google Fonts
    assert "extra" not in cfg or "analytics" not in cfg["extra"]
    assert all(not re.match(r"https?://", js if isinstance(js, str) else js["path"]) for js in cfg["extra_javascript"])
    # Mermaid comes from the site itself (the theme's CDN fallback stays unused) ...
    assert "assets/javascripts/mermaid.min.js" in cfg["extra_javascript"]
    hook = (DOCS / "overrides" / "hooks" / "mermaid.py").read_text(encoding="utf-8")
    assert re.search(r'^INTEGRITY = "sha512-[A-Za-z0-9+/]{86}=="$', hook, re.M)
    # ... and the repository button does not call the GitHub API for stars and forks
    source = (DOCS / "overrides" / "partials" / "source.html").read_text(encoding="utf-8")
    link = re.search(r"<a [^>]*>", source).group(0)
    assert 'class="md-source"' in link and "data-md-component" not in link


def test_docs_requirements_are_pinned_with_hashes():
    reqs = (ROOT / "requirements-docs.txt").read_text(encoding="utf-8")
    entries = re.split(r"\n(?=[A-Za-z0-9])", re.sub(r"(?m)^#.*\n", "", reqs).strip())
    assert len(entries) >= 10
    for e in entries:
        assert re.match(r"^[A-Za-z0-9_.\[\],-]+==[\w.+-]+ \\$", e.splitlines()[0]), e.splitlines()[0]
        assert "--hash=sha256:" in e, e.splitlines()[0]
    names = {e.split("==")[0].lower() for e in entries}
    assert {"mkdocs", "mkdocs-material", "pymdown-extensions"} <= names


def test_pages_workflow_least_privilege():
    wf = yaml.safe_load((ROOT / ".github" / "workflows" / "pages.yml").read_text(encoding="utf-8"))
    assert wf["permissions"] == {"contents": "read"}
    assert wf["concurrency"]["group"] == "pages"
    jobs = wf["jobs"]
    assert jobs["build"].get("permissions", {"contents": "read"}) == {"contents": "read"}
    assert jobs["deploy"]["permissions"] == {"pages": "write", "id-token": "write"}
    assert all(j["runs-on"] == "ubuntu-24.04" for j in jobs.values())
    on = wf[True]  # YAML 1.1 reads the key `on` as True
    assert "workflow_dispatch" in on and on["push"]["branches"] == ["main"]


def _html_images():
    import importlib.util
    spec = importlib.util.spec_from_file_location("html_images", DOCS / "overrides" / "hooks" / "html_images.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_html_image_with_width_becomes_markdown_for_the_site():
    hook = _html_images()
    src = ('Text\n<img src="../images/a.png" width="300" alt="A phone">\n\n'
           '```html\n<img src="../images/b.png" width="10" alt="code">\n```\n')
    out = hook.on_page_markdown(src, None, None, None)
    assert '![A phone](../images/a.png){ width="300" }' in out
    assert '<img src="../images/b.png" width="10" alt="code">' in out  # fenced code is left alone


@pytest.mark.parametrize("page", pages())
def test_no_attribute_lists_on_images(page):
    """On github.com `{ width=... }` after an image is shown as text: pages use <img ... width=...> instead
    (html_images.py turns it into an attribute list for the site)."""
    text = (DOCS / page).read_text(encoding="utf-8")
    assert not re.search(r"!\[[^\]]*\]\([^)]*\)\{", text), page
