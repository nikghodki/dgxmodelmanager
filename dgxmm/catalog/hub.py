"""Thin, testable wrapper over huggingface_hub (spec §7)."""
from __future__ import annotations

import os
import re
import time
import zlib
from dataclasses import asdict, dataclass, field
from typing import Any

import httpx

from dgxmm.config import GIB, Settings

DTYPE_BYTES = {"F64": 8, "F32": 4, "BF16": 2, "F16": 2, "F8_E4M3": 1, "F8_E5M2": 1, "F8_E8M0": 1, "I8": 1, "U8": 1,
               "I16": 2, "U16": 2, "I32": 4, "U32": 4, "I64": 8, "F4": 0.5, "BOOL": 1}
_QUANT = re.compile(r"(IQ\d_[A-Z]+|Q\d_K_[SML]|Q\d_K|Q\d_\d|Q\d|NVFP4|MXFP4|FP8|AWQ|GPTQ|BF16|F16|INT4|INT8|W4A16)", re.I)


@dataclass
class HubModel:
    repo_id: str
    downloads: int
    likes: int
    pipeline_tag: str | None
    library: str | None
    tags: list[str]
    gated: Any
    est_bytes: int | None
    params: int | None
    has_gguf: bool
    last_modified: str | None

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class HubFile:
    name: str
    size: int


@dataclass
class HubDetails:
    repo_id: str
    files: list[HubFile]
    config: dict | None
    gated: Any
    sha: str | None = None
    extra: dict = field(default_factory=dict)


# Pre-download compatibility check (dgxmm/decider/compat.py): the only files it may read from the Hub, each held in
# memory (never written to the HF cache), at most FETCH_CAP bytes (README.md: CARD_CAP) and FETCH_TIMEOUT_S.
FETCH_ALLOWLIST = frozenset({"config.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json",
                             "README.md"})
FETCH_CAP = 64 * 1024
CARD_CAP = 8 * 1024
FETCH_TIMEOUT_S = 5.0
HUB_TIMEOUT_S = 20.0   # every Hub API call (model_info here; huggingface_hub's own client: install_hub_timeout)
MAX_REDIRECTS = 3
# Redirects are followed only to the configured endpoint's host and the Hub's own CDN / storage hosts, over https
# (http only to the configured endpoint itself, e.g. a local mirror): never to engine ports or other hosts.
REDIRECT_DOMAINS = ("huggingface.co", "hf.co")


class HubGated(Exception):
    """The Hub refused a file (401 / 403): a gated repo without a token that has access."""


@dataclass
class RepoMeta:
    repo_id: str
    sha: str | None
    files: list[str]
    gated: Any
    private: bool = False


@dataclass
class Fetched:
    data: bytes | None  # None: not found or larger than the cap (where truncation is not allowed)
    truncated: bool = False
    too_large: bool = False


def hub_endpoint() -> str:
    """The configured Hub endpoint (HF_ENDPOINT, as huggingface_hub reads it)."""
    return (os.environ.get("HF_ENDPOINT") or "https://huggingface.co").rstrip("/")


def redirect_allowed(url: httpx.URL) -> bool:
    """A redirect target the Hub client may follow (see REDIRECT_DOMAINS)."""
    ep = httpx.URL(hub_endpoint())
    host = (url.host or "").lower()
    if url.scheme == ep.scheme and host == (ep.host or "").lower() and url.port == ep.port:
        return True
    return url.scheme == "https" and url.port in (None, 443) and any(
        host == d or host.endswith("." + d) for d in REDIRECT_DOMAINS)


def _check_redirect(response: httpx.Response) -> None:
    """httpx response hook: refuse a redirect off the Hub before it is followed."""
    if response.has_redirect_location:
        target = response.request.url.join(response.headers["location"])
        if not redirect_allowed(target):
            raise ValueError(f"refused redirect from the Hub to {target.scheme}://{target.host}")


def install_hub_timeout(timeout_s: float = HUB_TIMEOUT_S) -> None:
    """huggingface_hub's shared HTTP client has no timeout (a black-holed Hub holds a worker thread forever):
    give it one. Downloads pass their own per-request timeouts and are unaffected."""
    try:
        from huggingface_hub.utils import _http
    except ImportError:  # pragma: no cover - huggingface_hub is a dependency
        return

    def factory():
        c = _http.default_client_factory()
        c.timeout = type(c.timeout)(timeout_s)
        return c
    _http.set_client_factory(factory)


def guess_quant(name: str) -> str | None:
    m = _QUANT.findall(name)
    return m[-1].upper() if m else None


def quick_required(est_bytes: int, settings: Settings) -> int:
    """Rough requirement for list views: weights + 5% + vLLM runtime + 2 GiB default KV allowance."""
    runtime = int((settings.vllm_runtime_fixed_gib + settings.vllm_cuda_graph_gib) * GIB)
    return int(est_bytes * 1.05) + runtime + 2 * GIB


def fit_badge(required: int, budget, evictable: int) -> dict:
    if required <= budget.free_for_new:
        badge = "fits_now"
    elif required <= budget.free_for_new + evictable:
        badge = "fits_after_evict"
    else:
        badge = "too_big"
    return {"badge": badge, "required": required, "free": budget.free_for_new}


class HfHubClient:
    def __init__(self, token: str | None = None, api=None, http: httpx.Client | None = None):
        if api is None:
            from huggingface_hub import HfApi
            install_hub_timeout()
            api = HfApi(token=token)
        self.api, self.token = api, token
        self.http = http or httpx.Client(timeout=HUB_TIMEOUT_S, follow_redirects=True)
        self.http.max_redirects = MAX_REDIRECTS
        hooks = self.http.event_hooks
        if _check_redirect not in hooks.get("response", []):
            self.http.event_hooks = {**hooks, "response": [*hooks.get("response", []), _check_redirect]}

    @staticmethod
    def _bytes(st) -> tuple[int | None, int | None]:
        if not st or not getattr(st, "parameters", None):
            return None, None
        params = sum(st.parameters.values())
        return int(sum(n * DTYPE_BYTES.get(d, 2) for d, n in st.parameters.items())), params

    def search(self, query: str | None, limit: int = 30, gguf: bool = False, anonymous: bool = False) -> list[HubModel]:
        """`anonymous`: without the HF token (a viewer's search: private repos the token can read stay invisible)."""
        kw: dict[str, Any] = dict(search=query or None, sort="downloads", limit=limit,
                                  expand=["downloads", "likes", "pipeline_tag", "library_name", "tags", "gated",
                                          "safetensors", "lastModified", "gguf"])
        if anonymous:
            kw["token"] = False
        if gguf:
            kw["library"] = "gguf"
        else:
            kw["pipeline_tag"] = "text-generation"
        out = []
        for m in self.api.list_models(**kw):
            est, params = self._bytes(getattr(m, "safetensors", None))
            tags = list(getattr(m, "tags", None) or [])
            lm = getattr(m, "last_modified", None)
            out.append(HubModel(m.id, int(getattr(m, "downloads", 0) or 0), int(getattr(m, "likes", 0) or 0),
                                getattr(m, "pipeline_tag", None), getattr(m, "library_name", None), tags,
                                getattr(m, "gated", False), est, params,
                                bool(getattr(m, "gguf", None)) or "gguf" in tags, lm.isoformat() if hasattr(lm, "isoformat") else lm))
        return out

    def details(self, repo_id: str, *, anonymous: bool = False) -> HubDetails:
        """Files, config, gating and sha of a repo. `anonymous`: without the HF token (a viewer's lookup); a private
        repo is then never described (LookupError), whatever the Hub answers."""
        kw: dict[str, Any] = {"files_metadata": True, "timeout": HUB_TIMEOUT_S}
        if anonymous:
            kw["token"] = False
        info = self.api.model_info(repo_id, **kw)
        if anonymous and getattr(info, "private", False):
            raise LookupError(f"{repo_id} not found")
        files = [HubFile(s.rfilename, int(s.size or 0)) for s in (info.siblings or [])]
        config = None
        if any(f.name == "config.json" for f in files):
            headers = {"Authorization": f"Bearer {self.token}"} if self.token and not anonymous else {}
            r = self.http.get(f"https://huggingface.co/{repo_id}/resolve/{info.sha or 'main'}/config.json", headers=headers,
                              follow_redirects=True)
            if r.status_code == 200:
                try:
                    config = r.json()
                except ValueError:
                    config = None
        return HubDetails(repo_id, files, config, getattr(info, "gated", False), getattr(info, "sha", None))

    def repo_meta(self, repo_id: str, *, anonymous: bool = False) -> RepoMeta:
        """Sha, file names, gating and visibility of a repo (metadata only, no file contents). `anonymous`: without
        the HF token (a viewer's check: private repos the token can read stay invisible)."""
        kw: dict[str, Any] = {"timeout": HUB_TIMEOUT_S}
        if anonymous:
            kw["token"] = False
        info = self.api.model_info(repo_id, **kw)
        return RepoMeta(repo_id, getattr(info, "sha", None), [s.rfilename for s in (info.siblings or [])],
                        getattr(info, "gated", False), bool(getattr(info, "private", False)))

    def fetch_small(self, repo_id: str, filename: str, *, revision: str | None = None, cap: int = FETCH_CAP,
                    truncate: bool = False, timeout_s: float = FETCH_TIMEOUT_S, anonymous: bool = False) -> Fetched:
        """One allow-listed small file, streamed into memory: at most `cap` bytes within `timeout_s` overall.
        `truncate`: keep the first `cap` bytes of a larger file (model cards); else a larger file is `too_large`
        (a cut JSON would mislead). 404 -> data None; 401 / 403 -> HubGated. `anonymous`: without the HF token.
        Redirects: at most MAX_REDIRECTS, only to Hub hosts (``redirect_allowed``; else ValueError); httpx drops
        the token on a redirect to another host.

        Raw network reads are checked against the deadline and the cap one by one (a server that dribbles bytes
        cannot hold the thread past the deadline plus one read timeout); identity encoding is asked for, a gzip /
        deflate body is inflated incrementally with the cap applied to the inflated bytes, other encodings are
        refused (ValueError)."""
        if filename not in FETCH_ALLOWLIST:
            raise ValueError(f"{filename} is not an allowed compatibility-check file")
        if cap > FETCH_CAP:
            raise ValueError("cap above FETCH_CAP")
        headers = {"Accept-Encoding": "identity"}
        if self.token and not anonymous:
            headers["Authorization"] = f"Bearer {self.token}"
        url = f"{hub_endpoint()}/{repo_id}/resolve/{revision or 'main'}/{filename}"
        deadline = time.monotonic() + timeout_s
        read_timeout = max(min(timeout_s / 2, 2.5), 0.1)
        buf = bytearray()
        with self.http.stream("GET", url, headers=headers, follow_redirects=True,
                              timeout=httpx.Timeout(timeout_s, read=read_timeout)) as r:
            if r.status_code in (401, 403):
                raise HubGated(repo_id)
            if r.status_code == 404:
                return Fetched(None)
            r.raise_for_status()
            enc = (r.headers.get("content-encoding") or "identity").strip().lower()
            if enc in ("gzip", "x-gzip", "deflate"):
                inflate = zlib.decompressobj(16 + zlib.MAX_WBITS if "gzip" in enc else zlib.MAX_WBITS)
            elif enc in ("identity", ""):
                inflate = None
            else:
                raise ValueError(f"unsupported content encoding {enc!r}")
            for raw in r.iter_raw():
                if time.monotonic() > deadline:
                    raise TimeoutError(f"{filename} of {repo_id} took longer than {timeout_s} s")
                if inflate is not None:
                    buf += inflate.decompress(raw, cap + 1 - len(buf))
                    while inflate.unconsumed_tail and len(buf) <= cap:
                        buf += inflate.decompress(inflate.unconsumed_tail, cap + 1 - len(buf))
                else:
                    buf += raw
                if len(buf) > cap:
                    if truncate:
                        return Fetched(bytes(buf[:cap]), truncated=True)
                    return Fetched(None, too_large=True)
            if time.monotonic() > deadline:
                raise TimeoutError(f"{filename} of {repo_id} took longer than {timeout_s} s")
        return Fetched(bytes(buf))
