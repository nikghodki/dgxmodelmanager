"""``python -m dgxmm.admin decider ...``: the local CLI the installer uses to build, download, enable, disable and
show the decision model through the running app's API (dgxmm/decider/cli.py)."""
from __future__ import annotations

import json

import httpx
import pytest
from sqlalchemy import select

from dgxmm import admin
from dgxmm.db import AuditLog, CliToken
from dgxmm.decider import cli
from tests.test_api_decider import B, api  # noqa: F401  (fixture)
from tests.test_api_deployments import SMALL


def local(api):  # noqa: F811
    """What the CLI talks to: the admin listener. Shares the fixture's transport (and so its event loop, where the
    deployer's start-up tasks run)."""
    from dgxmm.admin_listener import AdminListener
    if not isinstance(api._transport.app, AdminListener):
        api._transport.app = AdminListener(api._transport.app)
    return api


class Out:
    def __init__(self):
        self.lines: list[str] = []

    def __call__(self, line: str = "") -> None:
        self.lines.append(line)

    @property
    def text(self) -> str:
        return "\n".join(self.lines)


def run(api, *argv, sleep=lambda s: None, **kw):  # noqa: F811
    out = Out()
    rc = cli.run(list(argv), db=api.ctx.db, settings=api.ctx.settings, client=local(api), out=out, sleep=sleep, **kw)
    return rc, out


def no_tokens_left(db) -> bool:
    with db.session() as s:
        return s.scalars(select(CliToken)).all() == []


def test_status_shows_disabled_decider_and_cleans_up_its_token(api):  # noqa: F811
    rc, out = run(api, "status")
    assert rc == 0, out.text
    assert "disabled" in out.text and "vllm-sr/d3-mini" in out.text and "dgxmm-decider:3" in out.text
    assert no_tokens_left(api.ctx.db)
    rc, out = run(api, "status", "--json")
    st = json.loads(out.text)
    assert rc == 0 and st["enabled"] is False and st["image"]["image"] == "dgxmm-decider:3"


def test_enable_waits_until_healthy_then_disable_removes(api):  # noqa: F811
    api.stack.http.all_healthy = True
    api.ctx.http = api.stack.http.client()  # the decider's own client (healthy needs one)
    rc, out = run(api, "enable", "--wait", "--timeout", "30", "--poll", "0.05", sleep=lambda s: __import__("time").sleep(s))
    assert rc == 0, out.text
    assert "healthy" in out.text
    st = api.get(f"{B}/status").json()
    assert st["enabled"] and st["healthy"] and st["state"] == "ready"
    rc, _ = run(api, "enable")  # idempotent
    assert rc == 0 and len([v for v in api.get("/dgxmm/api/deployments").json() if v["engine"] == "decision"]) == 1
    rc, out = run(api, "disable", "--remove")
    assert rc == 0 and api.get(f"{B}/status").json()["deployment_id"] is None
    with api.ctx.db.session() as s:
        actions = [a.action for a in s.scalars(select(AuditLog))]
    assert "decider_enable" in actions and "decider_disable" in actions and "inventory_refreshed" in actions
    assert no_tokens_left(api.ctx.db)
    assert "dgxmmcli_" not in out.text


def test_enable_timeout_reports_state(api):  # noqa: F811
    clock = iter(range(0, 1000, 10))
    rc, out = run(api, "enable", "--wait", "--timeout", "30", now=lambda: next(clock))
    assert rc == cli.EXIT_TIMEOUT
    assert "did not become healthy" in out.text and any(s in out.text for s in ("starting", "loading"))


def test_enable_without_image_says_how_to_build(api):  # noqa: F811
    api.stack.sdk.missing_images.add("dgxmm-decider:3")
    rc, out = run(api, "enable")
    assert rc == cli.EXIT_IMAGE_MISSING and "decider build" in out.text


def test_enable_without_model_says_how_to_download(api_stack):
    from tests.test_api_deployments import admin as make_admin
    make_admin(api_stack)
    rc, out = run(api_stack, "enable")
    assert rc == cli.EXIT_MODEL_MISSING and "decider download" in out.text


def test_enable_that_does_not_fit_lists_plans_and_changes_nothing(api):  # noqa: F811
    from dgxmm.config import GIB
    api.stack.http.all_healthy = True
    tiny = api.post("/dgxmm/api/deployments", headers=api.h,
                    json={"name": "tiny", "engine": "vllm", "model_ref": "Qwen/Qwen3-Tiny", "config": SMALL}).json()
    reserve = api.ctx.settings.margin_bytes + api.ctx.settings.os_reserve_bytes
    api.stack.probe.available = 13 * GIB + reserve - 1 * GIB
    rc, out = run(api, "enable")
    assert rc == cli.EXIT_NO_MEMORY
    assert "does not fit" in out.text and "Decision model" in out.text and "UI" in out.text
    assert api.get(f"/dgxmm/api/deployments/{tiny['id']}").json()["state"] != "stopped"
    assert api.get(f"{B}/status").json()["enabled"] is False


def test_unreachable_app(settings, db, make_user):
    make_user("root", "root-password-123", "admin")

    def refuse(request):
        raise httpx.ConnectError("connection refused")
    client = httpx.Client(transport=httpx.MockTransport(refuse), base_url="http://127.0.0.1:8090")
    out = Out()
    rc = cli.run(["status"], db=db, settings=settings, client=client, out=out)
    assert rc == cli.EXIT_UNREACHABLE and "not reachable" in out.text
    assert no_tokens_left(db)


def test_no_admin_user(settings, db):
    out = Out()
    rc = cli.run(["status"], db=db, settings=settings, client=httpx.Client(base_url="http://127.0.0.1:1"), out=out)
    assert rc == 1 and "no admin user" in out.text


class FakeDocker:
    def __init__(self, present=False, chunks=None):
        self.present, self.chunks, self.builds = present, chunks or [], []

    def image_present(self, tag):
        return self.present

    def build_image(self, fileobj, tag, labels=None):
        self.builds.append(tag)
        for c in self.chunks:
            yield c
        if not any("error" in c for c in self.chunks):
            self.present = True


def test_build_streams_progress_and_skips_when_present(settings, db):
    d = FakeDocker(chunks=[{"stream": "Step 1/5 : FROM vllm/vllm-openai@sha256:22d3"}, {"stream": "\n"},
                           {"status": "Downloading"}])
    out = Out()
    assert cli.run(["build"], db=db, settings=settings, docker=d, out=out) == 0
    assert d.builds == ["dgxmm-decider:3"] and "Step 1/5" in out.text and "built dgxmm-decider:3" in out.text
    out = Out()
    assert cli.run(["build"], db=db, settings=settings, docker=d, out=out) == 0
    assert d.builds == ["dgxmm-decider:3"] and "already present" in out.text
    assert cli.run(["build", "--force"], db=db, settings=settings, docker=d, out=Out()) == 0
    assert len(d.builds) == 2


def test_build_failure_is_reported(settings, db):
    d = FakeDocker(chunks=[{"error": "pull access denied"}])
    out = Out()
    assert cli.run(["build"], db=db, settings=settings, docker=d, out=out) == 1
    assert "pull access denied" in out.text


def test_download_uses_the_hf_cache_and_pinned_list(settings, db, monkeypatch):
    seen = {}

    def fake_ensure(hf_home, pins, *, echo, verify_only=False, **kw):
        seen.update(hf_home=str(hf_home), pins=str(pins), verify_only=verify_only)
        echo("verified")
        return True
    monkeypatch.setattr(cli.snapshot, "ensure_snapshot", fake_ensure)
    out = Out()
    assert cli.run(["download"], db=db, settings=settings, out=out) == 0
    assert seen["hf_home"] == settings.hf_cache.local and seen["pins"].endswith(".sha256")
    assert cli.run(["download", "--verify-only"], db=db, settings=settings, out=Out()) == 0 and seen["verify_only"]
    monkeypatch.setattr(cli.snapshot, "ensure_snapshot", lambda *a, **k: False)
    assert cli.run(["download"], db=db, settings=settings, out=Out()) == 1

    def broken(*a, **k):
        raise cli.snapshot.DownloadFailed("connection reset")
    monkeypatch.setattr(cli.snapshot, "ensure_snapshot", broken)
    out = Out()
    assert cli.run(["download"], db=db, settings=settings, out=out) == 1
    assert "connection reset" in out.text and "re-run" in out.text


def test_download_of_another_supported_model_uses_its_own_pinned_list(settings, db, monkeypatch, capsys):
    from dgxmm.engines.decision import DECIDER_MODELS
    seen = {}

    def fake_ensure(hf_home, pins, *, echo, verify_only=False, repo=None, **kw):
        seen.update(pins=str(pins), repo=repo)
        return True
    monkeypatch.setattr(cli.snapshot, "ensure_snapshot", fake_ensure)
    for arg in ("vllm-sr/d3-nano", "d3-nano"):
        assert cli.run(["download", "--model", arg], db=db, settings=settings, out=Out()) == 0
        assert seen["repo"] == "vllm-sr/d3-nano"
        assert seen["pins"].endswith(f"d3-nano-{DECIDER_MODELS['vllm-sr/d3-nano'].revision}.sha256")
    assert cli.run(["download"], db=db, settings=settings, out=Out()) == 0 and seen["repo"] == "vllm-sr/d3-mini"
    with pytest.raises(SystemExit) as e:
        cli.run(["download", "--model", "acme/other"], db=db, settings=settings, out=Out())
    assert e.value.code == 2 and "vllm-sr/d3-nano" in capsys.readouterr().err  # lists the supported models


def test_admin_main_routes_decider_commands(settings, db, monkeypatch, tmp_path):
    import yaml
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(yaml.safe_dump({"data_dir": settings.data_dir}))
    got = {}

    def fake_run(argv, **kw):
        got.update(argv=argv, kw=kw)
        return 7
    monkeypatch.setattr(cli, "run", fake_run)
    assert admin.main(["--config", str(cfg), "decider", "enable", "--wait"]) == 7
    assert got["argv"] == ["enable", "--wait"] and got["kw"]["settings"].data_dir == settings.data_dir


@pytest.mark.parametrize("argv", [["bogus"], ["enable", "--timeout", "-1"], ["enable", "--timeout", "10201"]])
def test_bad_arguments(settings, db, argv):
    with pytest.raises(SystemExit):
        cli.run(argv, db=db, settings=settings, out=Out())


def test_status_check_exit_code_reflects_health(api):  # noqa: F811
    rc, out = run(api, "status", "--check")
    assert rc == cli.EXIT_ERROR and "disabled" in out.text  # not enabled -> not ok for --check
    api.stack.http.all_healthy = True
    api.ctx.http = api.stack.http.client()
    assert run(api, "enable", "--wait", "--timeout", "30", "--poll", "0.05",
               sleep=lambda s: __import__("time").sleep(s))[0] == 0
    assert run(api, "status", "--check")[0] == 0


def test_build_and_download_need_no_database(tmp_path, monkeypatch):
    """The installer may download before DGXMM ever started (no database yet)."""
    import yaml
    data = tmp_path / "fresh-data"
    data.mkdir()
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(yaml.safe_dump({"data_dir": str(data)}))
    monkeypatch.setattr(cli.snapshot, "ensure_snapshot", lambda *a, **k: True)
    assert admin.main(["--config", str(cfg), "decider", "download"]) == 0
    assert not (data / "dgxmm.sqlite3").exists()
    with pytest.raises(SystemExit, match="no DGXMM database"):
        admin.main(["--config", str(cfg), "decider", "status"])


def test_token_outlives_the_longest_wait():
    from dgxmm.auth import cli_tokens
    assert cli.MAX_WAIT_S + 600 <= cli_tokens.MAX_TTL_S


def test_missing_admin_socket_is_a_clear_unreachable_error(settings, db, make_user):
    make_user("root", "root-password-123", "admin")
    out = Out()
    assert cli.run(["status"], db=db, settings=settings, out=out) == cli.EXIT_UNREACHABLE
    assert "admin socket" in out.text and no_tokens_left(db)


def test_admin_disabled_reports_an_explicit_disable_only(settings, db):
    from dgxmm.settings_store import SettingsStore
    store = SettingsStore(db, settings)
    assert cli.run(["admin-disabled"], db=db, settings=settings, out=Out()) == 1  # never set: not disabled
    store.set_bool("decider.enabled", True)
    assert cli.run(["admin-disabled"], db=lambda: db, settings=settings, out=Out()) == 1
    store.set_bool("decider.enabled", False)
    out = Out()
    assert cli.run(["admin-disabled"], db=db, settings=settings, out=out) == 0 and "disabled" in out.text
