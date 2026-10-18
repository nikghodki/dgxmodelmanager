"""MkDocs hook: include a repository root file (CHANGELOG.md, CONTRIBUTING.md, SECURITY.md) in a page.

A page includes one with a line in pymdownx.snippets syntax:

    --8<-- "CHANGELOG.md"

The file's H1 is dropped (the page has its own) and its links, written for github.com relative to the repository
root, are rewritten for the page: ``docs/...`` to the site page, README/CHANGELOG/CONTRIBUTING/SECURITY to their
site pages, anything else in the repository to its GitHub URL. That keeps one source for each file, valid on GitHub
and on the site, and lets ``mkdocs build --strict`` check every link and anchor."""
from __future__ import annotations

import posixpath
import re
from pathlib import Path

from mkdocs.exceptions import PluginError

GITHUB = "https://github.com/nikghodki/dgxmodelmanager"
INCLUDABLE = ("CHANGELOG.md", "CONTRIBUTING.md", "SECURITY.md")
ROOT_PAGES = {"README.md": "index.md", "CHANGELOG.md": "changelog.md", "CONTRIBUTING.md": "contributing/index.md",
              "SECURITY.md": "security/reporting.md"}
MARKER = re.compile(r'^--8<-- "([^"]+)"[ \t]*$', re.M)
FENCE = re.compile(r"^\s*(```|~~~)")
LINK = re.compile(r"(!?\[(?:[^\[\]]|\[[^\]]*\])*\]\()(\s*<?)([^)\s>]+)")
CODE = re.compile(r"`[^`\n]*`")
SCHEME = re.compile(r"^[a-z][a-z0-9+.-]*:", re.I)


def _target(target: str, page_uri: str, root: Path) -> str:
    if SCHEME.match(target) or target.startswith("#"):
        return target
    path, _, frag = target.partition("#")
    path = posixpath.normpath(path)
    if path.startswith("docs/"):
        dest = path[len("docs/"):]
    elif path in ROOT_PAGES:
        dest = ROOT_PAGES[path]
    else:
        kind = "tree" if (root / path).is_dir() else "blob"
        return f"{GITHUB}/{kind}/main/{path}" + (f"#{frag}" if frag else "")
    rel = posixpath.relpath(dest, posixpath.dirname(page_uri) or ".")
    return rel + (f"#{frag}" if frag else "")


def rewrite(text: str, page_uri: str, root: Path) -> str:
    out, fenced = [], False
    for line in text.splitlines():
        if FENCE.match(line):
            fenced = not fenced
        elif not fenced:
            spans: list[str] = []

            def hide(m: re.Match) -> str:
                spans.append(m.group(0))
                return f"\0{len(spans) - 1}\0"

            line = CODE.sub(hide, line)
            line = LINK.sub(lambda m: m.group(1) + m.group(2) + _target(m.group(3), page_uri, root), line)
            line = re.sub(r"\0(\d+)\0", lambda m: spans[int(m.group(1))], line)
        out.append(line)
    return "\n".join(out)


def include(name: str, page_uri: str, root: Path) -> str:
    if name not in INCLUDABLE:
        raise PluginError(f"{page_uri}: only {', '.join(INCLUDABLE)} can be included, not {name!r}")
    text = (root / name).read_text(encoding="utf-8")
    text = re.sub(r"\A# [^\n]*\n+", "", text)  # the page has its own title
    return rewrite(text, page_uri, root)


def on_page_markdown(markdown, page, config, files):  # noqa: ARG001 (MkDocs hook signature)
    root = Path(config.config_file_path).parent
    return MARKER.sub(lambda m: include(m.group(1), page.file.src_uri, root), markdown)
