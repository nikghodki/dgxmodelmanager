"""MkDocs hook: serve Mermaid from the site instead of a CDN.

Material for MkDocs loads Mermaid from unpkg.com unless a global ``mermaid`` already exists. This hook puts a pinned
``mermaid.min.js`` into the built site (``extra_javascript`` in mkdocs.yml loads it), so readers' browsers never
contact a third party. The file is taken from the npm registry tarball of the pinned version, checked against the
registry's published SHA-512 integrity, and cached (``DGXMM_DOCS_CACHE``, default ``~/.cache/dgxmm-docs``). It is
not committed: it is larger than the repository's 1 MB file limit. The build fails if it cannot be fetched or does
not match."""
from __future__ import annotations

import base64
import hashlib
import io
import os
import shutil
import tarfile
import urllib.request
from pathlib import Path

from mkdocs.exceptions import PluginError

VERSION = "11.17.2"
TARBALL = f"https://registry.npmjs.org/mermaid/-/mermaid-{VERSION}.tgz"
INTEGRITY = "sha512-V6K3C8EBdEsPFZXSKMJe6ppQOENxuHARr9GvHX4hh47lAbhMRD9qf4oEK7LoaRQxULMa80/qt5gHO73aCleBBg=="
MEMBER = "package/dist/mermaid.min.js"
TARGET = "assets/javascripts/mermaid.min.js"  # as listed in extra_javascript


def _cache() -> Path:
    return Path(os.environ.get("DGXMM_DOCS_CACHE") or Path.home() / ".cache" / "dgxmm-docs") / f"mermaid-{VERSION}.min.js"


def _fetch(dest: Path) -> None:
    with urllib.request.urlopen(TARBALL, timeout=60) as r:  # noqa: S310 (fixed https URL)
        data = r.read()
    digest = "sha512-" + base64.b64encode(hashlib.sha512(data).digest()).decode()
    if digest != INTEGRITY:
        raise PluginError(f"mermaid {VERSION}: tarball integrity mismatch ({digest})")
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        js = tar.extractfile(MEMBER)
        if js is None:
            raise PluginError(f"mermaid {VERSION}: {MEMBER} missing from the tarball")
        body = js.read()
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp")
    tmp.write_bytes(body)
    tmp.replace(dest)
    dest.with_suffix(".sha256").write_text(hashlib.sha256(body).hexdigest())


def _verified(path: Path) -> bool:
    sums = path.with_suffix(".sha256")
    return path.is_file() and sums.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == sums.read_text().strip()


def on_post_build(config):
    cached = _cache()
    if not _verified(cached):
        try:
            _fetch(cached)
        except OSError as e:
            raise PluginError(f"mermaid {VERSION}: could not fetch {TARBALL}: {e}") from e
    out = Path(config.site_dir) / TARGET
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(cached, out)
