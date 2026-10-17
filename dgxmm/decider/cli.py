"""``python -m dgxmm.admin decider <command>``: manage the decision model from the host (used by ./install.sh).

    docker exec dgxmm python -m dgxmm.admin decider status [--json]
    docker exec dgxmm python -m dgxmm.admin decider build [--force]
    docker exec dgxmm python -m dgxmm.admin decider download [--model vllm-sr/d3-nano] [--verify-only]
    docker exec dgxmm python -m dgxmm.admin decider enable [--wait] [--timeout S]
    docker exec dgxmm python -m dgxmm.admin decider disable [--remove]

``build`` and ``download`` work on their own (Docker socket / HF cache). ``status``, ``enable`` and ``disable`` go
through the running app's API on its admin unix socket (only the app owns the deployer) with a short-lived scoped
admin token (dgxmm/auth/cli_tokens.py) that is created for the command, never printed, and deleted afterwards.
``enable`` uses the normal fit check: if the decider does not fit it lists the plans and changes nothing (those plans
stop or sleep your deployments, so choose one in the UI: Integrations > Decision model).
"""
from __future__ import annotations

import argparse
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

import httpx
from sqlalchemy import select

from dgxmm.auth import cli_tokens
from dgxmm.db import Database, User
from dgxmm.decider import snapshot
from dgxmm.decider.image import BuildFailed, DeciderImage
from dgxmm.engines.decision import DECIDER_MODELS, DECIDER_REPO, hash_list_name

EXIT_OK, EXIT_ERROR, EXIT_UNREACHABLE, EXIT_NO_MEMORY, EXIT_IMAGE_MISSING, EXIT_TIMEOUT, EXIT_MODEL_MISSING = (
    0, 1, 2, 3, 4, 5, 6)
GIB = 1024 ** 3
TOKEN_MARGIN_S = 600
MAX_WAIT_S = cli_tokens.MAX_TTL_S - TOKEN_MARGIN_S  # the token must outlive the longest --wait


class CliError(Exception):
    def __init__(self, message: str, code: int = EXIT_ERROR):
        super().__init__(message)
        self.code = code


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m dgxmm.admin decider", description="Manage the decision model")
    p.add_argument("--user", help="admin user the API calls are made as (default: the oldest admin)")
    sub = p.add_subparsers(dest="cmd", required=True)
    st = sub.add_parser("status", help="show the decider's state")
    st.add_argument("--json", action="store_true", help="print GET /api/decider/status as JSON")
    st.add_argument("--check", action="store_true", help="exit 1 unless the decider is enabled and healthy")
    b = sub.add_parser("build", help="build the decider image (skipped when it exists)")
    b.add_argument("--force", action="store_true", help="rebuild even if the image exists")
    d = sub.add_parser("download", help="download and verify the pinned model snapshot into the HF cache")
    d.add_argument("--verify-only", action="store_true", help="only check the snapshot on disk")
    d.add_argument("--model", type=_decider_model, default=DECIDER_REPO,
                   help=f"which supported decision model (default {DECIDER_REPO}; also: "
                        f"{', '.join(r for r in DECIDER_MODELS if r != DECIDER_REPO)})")
    e = sub.add_parser("enable", help="enable the decider (deploys it through the normal fit check)")
    e.add_argument("--wait", action="store_true", help="wait until it is healthy")
    e.add_argument("--timeout", type=_wait_seconds, default=900.0, help="seconds to wait with --wait (default 900)")
    e.add_argument("--poll", type=_seconds, default=5.0, help=argparse.SUPPRESS)
    sub.add_parser("admin-disabled", help="exit 0 if an admin explicitly disabled the decider (installer use)")
    x = sub.add_parser("disable", help="disable the decider and stop its deployment")
    x.add_argument("--remove", action="store_true", help="also remove the decider deployment")
    return p


def _seconds(v: str) -> float:
    f = float(v)
    if f <= 0:
        raise argparse.ArgumentTypeError("must be a positive number of seconds")
    return f


def _wait_seconds(v: str) -> float:
    f = _seconds(v)
    if f > MAX_WAIT_S:
        raise argparse.ArgumentTypeError(f"at most {MAX_WAIT_S} seconds")
    return f


# ---------- API session ----------

def _admin_id(db: Database, username: str | None) -> int:
    with db.session() as s:
        q = select(User).where(User.role == "admin")
        if username:
            q = q.where(User.username == username.strip().lower())
        user = s.scalars(q.order_by(User.created_at, User.id)).first()
        if user is None:
            raise CliError(f"no admin user{' named ' + repr(username) if username else ''} in the DGXMM database")
        return user.id


def default_client(settings) -> httpx.Client:
    """The app's admin unix socket (CLI tokens are refused on the TCP listener). DGXMM_ADMIN_SOCKET may point at
    another socket path; there is deliberately no URL override."""
    path = os.environ.get("DGXMM_ADMIN_SOCKET") or str(settings.admin_socket_path)
    if not Path(path).is_socket():
        raise CliError(f"DGXMM's admin socket {path} does not exist; is the dgxmm container running and healthy? "
                       "(./install.sh status)", EXIT_UNREACHABLE)
    return httpx.Client(transport=httpx.HTTPTransport(uds=path), base_url="http://dgxmm-admin",
                        timeout=httpx.Timeout(120, connect=5), trust_env=False)


@contextmanager
def api_session(db: Database, client: httpx.Client, base_path: str, user: str | None,
                ttl_s: int) -> Iterator["Api"]:
    token = cli_tokens.issue(db, _admin_id(db, user), ttl_s=ttl_s)
    try:
        yield Api(client, token, base_path)
    finally:
        cli_tokens.revoke(db, token)


class Api:
    def __init__(self, client: httpx.Client, token: str, base_path: str):
        self.client, self.base_path, self._headers = client, base_path, {cli_tokens.HEADER: token}

    def call(self, method: str, path: str, body: dict | None = None) -> httpx.Response:
        try:
            return self.client.request(method, self.base_path + path, headers=self._headers,
                                       json=body if method != "GET" else None)
        except httpx.TransportError as e:
            raise CliError(f"DGXMM is not reachable at {self.client.base_url} ({e}); is the dgxmm container "
                           "running and healthy? (./install.sh status)", EXIT_UNREACHABLE) from e

    def json(self, method: str, path: str, body: dict | None = None, ok=(200,)) -> Any:
        r = self.call(method, path, body)
        if r.status_code not in ok:
            raise CliError(f"{method} {path} failed: HTTP {r.status_code}: {_detail(r)}")
        return r.json()


def _detail(r: httpx.Response) -> str:
    try:
        d = r.json().get("detail")
    except ValueError:
        return r.text[:300]
    if isinstance(d, dict):
        return f"{d.get('code', '')}: {d.get('message', '')}".strip(": ")
    return str(d)[:300]


# ---------- commands ----------

def _status_lines(st: dict, image: dict) -> list[str]:
    if st["enabled"]:
        state = st.get("state") or "no deployment"
        head = f"enabled, {'healthy' if st['healthy'] else 'not healthy'} ({state}"
        head += f", deployment {st['deployment_id']})" if st.get("deployment_id") is not None else ")"
    else:
        head = "disabled" + (f" (deployment {st['deployment_id']}: {st.get('state')})"
                             if st.get("deployment_id") is not None else "")
    img = image.get("image") or "?"
    present = {True: "present", False: "not built", None: "unknown (docker unavailable)"}[image.get("present")]
    lines = [f"Decision model: {head}",
             f"    model   {st['model']} @ {(st.get('revision') or '?')[:12]}",
             f"    image   {img}: {present}" + (" (building)" if image.get("building") else "")
             + (f" (last build error: {image['error']})" if image.get("error") else "")]
    p50 = f", p50 {st['p50_ms']:.0f} ms" if st.get("p50_ms") is not None else ""
    lines.append(f"    decisions (24h): {st.get('decisions_24h', 0)}{p50}")
    return lines


def cmd_status(api: Api, out, as_json: bool, check: bool = False) -> int:
    st = api.json("GET", "/api/decider/status")
    image = api.json("GET", "/api/decider/image")
    if as_json:
        out(json.dumps({**st, "image": image}, indent=2, sort_keys=True))
    else:
        for line in _status_lines(st, image):
            out(line)
    if check and not (st["enabled"] and st["healthy"]):
        return EXIT_ERROR
    return EXIT_OK


def _explain_no_fit(out, detail: dict) -> None:
    fit = detail.get("fit") or {}
    req, free = fit.get("required"), fit.get("free_for_new")
    if isinstance(req, (int, float)) and isinstance(free, (int, float)):
        out(f"The decider does not fit: it needs {req / GIB:.1f} GiB, {free / GIB:.1f} GiB is free for new deployments.")
    else:
        out("The decider does not fit in the free memory.")
    plans = [p for p in detail.get("plans") or [] if p.get("fits") and p.get("actions")]
    if plans:
        out("Plans that would make room (they stop or sleep deployments, so nothing was changed):")
        for p in plans:
            out(f"    - {p.get('title')}: {p.get('detail')}")
    out("Free memory, or open the UI (Integrations > Decision model > Enable) and pick a plan, then re-run.")


def cmd_enable(api: Api, out, *, wait: bool, timeout: float, poll: float, sleep: Callable[[float], None],
               now: Callable[[], float]) -> int:
    # A snapshot downloaded a moment ago must be in the inventory before the deploy checks for the model.
    api.json("POST", "/api/inventory/refresh")
    r = api.call("POST", "/api/decider/enable", {})
    if r.status_code != 202:
        detail = (r.json().get("detail") if r.headers.get("content-type", "").startswith("application/json")
                  else None) or {}
        code = detail.get("code") if isinstance(detail, dict) else None
        if code == "image_missing":
            out(f"The decider image {detail.get('image', '')} is not built: run "
                "'python -m dgxmm.admin decider build' (./install.sh --with-decider does this).")
            return EXIT_IMAGE_MISSING
        if code == "model_not_found":
            out("The decider model snapshot is not in the HF cache: run 'python -m dgxmm.admin decider download' "
                "(./install.sh --with-decider does this).")
            return EXIT_MODEL_MISSING
        if code == "insufficient_memory":
            _explain_no_fit(out, detail)
            return EXIT_NO_MEMORY
        raise CliError(f"enabling the decider failed: HTTP {r.status_code}: {_detail(r)}")
    body = r.json()
    out(f"Decision model enabled (deployment {body['deployment_id']}, {body['state']}).")
    if not wait:
        return EXIT_OK
    start, last = now(), None
    out(f"Waiting for the decider to become healthy (up to {timeout:.0f}s; the first start compiles kernels and "
        "takes a few minutes)")
    while True:
        st = api.json("GET", "/api/decider/status")
        if st["healthy"]:
            out(f"    healthy after {now() - start:.0f}s")
            return EXIT_OK
        state = st.get("state")
        if state != last:
            out(f"    state: {state}")
            last = state
        if state in ("failed", "stopped", None):
            out(f"The decider deployment is {state or 'gone'}: see its Events and Diagnosis in the UI "
                f"(Deployments > decider), then re-run.")
            return EXIT_ERROR
        if now() - start >= timeout:
            out(f"The decider did not become healthy within {timeout:.0f}s (state: {state}). It keeps starting in "
                "the background: check 'python -m dgxmm.admin decider status' or the UI.")
            return EXIT_TIMEOUT
        sleep(poll)


def cmd_disable(api: Api, out, remove: bool) -> int:
    body = api.json("POST", "/api/decider/disable", {"remove": remove})
    if remove:
        out("Decision model disabled; its deployment was removed (features fall back to rules).")
    else:
        out(f"Decision model disabled (deployment state: {body.get('state') or 'none'}; features fall back to rules).")
    return EXIT_OK


def cmd_build(settings, docker, out, force: bool) -> int:
    if docker is None:
        from dgxmm.docker_ctl import DockerController
        docker = DockerController.from_env()
    image = DeciderImage(settings, lambda: docker, events=None)
    if not force and docker.image_present(image.tag):
        out(f"Decider image {image.tag} is already present (--force rebuilds it).")
        return EXIT_OK
    out(f"==> Building the decider image {image.tag} from {image.build_dir} (first build: the pinned vLLM base is "
        "several GB)")
    try:
        image.build_blocking(lambda line: out(f"    {line}"))
    except (BuildFailed, OSError) as e:
        out(f"Building {image.tag} failed: {e}")
        return EXIT_ERROR
    out(f"    built {image.tag}")
    return EXIT_OK


def _decider_model(value: str) -> str:
    """A supported decision model by repo id or short name (``d3-nano``)."""
    for repo in DECIDER_MODELS:
        if value in (repo, repo.rsplit("/", 1)[-1]):
            return repo
    raise argparse.ArgumentTypeError(f"{value!r} is not a supported decision model "
                                     f"(supported: {', '.join(DECIDER_MODELS)})")


def cmd_download(settings, out, verify_only: bool, repo: str = DECIDER_REPO) -> int:
    m = DECIDER_MODELS[repo]
    pins = DeciderImage(settings, lambda: None, events=None).build_dir / hash_list_name(repo, m.revision)
    try:
        ok = snapshot.ensure_snapshot(Path(settings.hf_cache.local), pins, echo=out, verify_only=verify_only,
                                      repo=repo)
    except snapshot.DownloadFailed as e:
        out(f"{e}. Check network access to huggingface.co, then re-run: the download resumes where it stopped.")
        return EXIT_ERROR
    except PermissionError as e:
        out(f"Cannot write to the HF cache ({e}). Files there must belong to the DGXMM user (DGXMM_UID in .env).")
        return EXIT_ERROR
    if not ok:
        out("The model snapshot is not complete and verified." + ("" if verify_only else
            " Nothing was trusted; remove the files listed above from the HF cache and re-run."))
        return EXIT_ERROR
    return EXIT_OK


def cmd_admin_disabled(db: Database, settings, out) -> int:
    """``decider.enabled`` is stored as false (an admin switched it off), as opposed to never enabled."""
    from dgxmm.decider.service import DECIDER_ENABLED
    from dgxmm.settings_store import SettingsStore
    raw = SettingsStore(db, settings).get(DECIDER_ENABLED)
    if raw is not None and raw.strip().lower() in ("false", "0", "no", "off"):
        out("The decision model was disabled by an admin.")
        return EXIT_OK
    return EXIT_ERROR


def run(argv: list[str], *, db: Database | Callable[[], Database], settings, client: httpx.Client | None = None, docker=None,
        out: Callable[[str], None] = print, sleep: Callable[[float], None] = time.sleep,
        now: Callable[[], float] = time.monotonic) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.cmd == "build":
            return cmd_build(settings, docker, out, args.force)
        if args.cmd == "download":
            return cmd_download(settings, out, args.verify_only, args.model)
        ttl = int((args.timeout if args.cmd == "enable" else 0) + TOKEN_MARGIN_S)  # <= MAX_TTL_S by _wait_seconds
        if not isinstance(db, Database):
            db = db()
        if args.cmd == "admin-disabled":
            return cmd_admin_disabled(db, settings, out)
        client = client or default_client(settings)
        with api_session(db, client, settings.base_path, args.user, ttl) as api:
            if args.cmd == "status":
                return cmd_status(api, out, args.json, args.check)
            if args.cmd == "enable":
                return cmd_enable(api, out, wait=args.wait, timeout=args.timeout, poll=args.poll, sleep=sleep, now=now)
            return cmd_disable(api, out, args.remove)
    except CliError as e:
        out(f"error: {e}")
        return e.code
