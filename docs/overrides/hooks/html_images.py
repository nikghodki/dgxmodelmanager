"""MkDocs hook: an HTML image with a width becomes a Markdown image with an attribute list for the site.

github.com shows ``{ width="300" }`` after a Markdown image as text, so pages size a screenshot with HTML that both
render, in exactly this form (one per line, attributes in this order):

    <img src="../images/dashboard-mobile.png" width="300" alt="Dashboard on a phone">

For the site the line becomes ``![alt](src){ width="300" }`` before Markdown runs, so MkDocs still rewrites the
relative path for the page's URL and ``mkdocs build --strict`` still checks that the image exists."""
from __future__ import annotations

import re

IMG = re.compile(r'^([ \t]*)<img src="([^"\s]+)" width="(\d+)" alt="([^"\[\]]*)">[ \t]*$')
FENCE = re.compile(r"^\s*(```|~~~)")


def convert(markdown: str) -> str:
    out, fenced = [], False
    for line in markdown.split("\n"):
        if FENCE.match(line):
            fenced = not fenced
        elif not fenced:
            m = IMG.match(line)
            if m:
                indent, src, width, alt = m.groups()
                line = f'{indent}![{alt}]({src}){{ width="{width}" }}'
        out.append(line)
    return "\n".join(out)


def on_page_markdown(markdown, page, config, files):  # noqa: ARG001 (MkDocs hook signature)
    return convert(markdown)
