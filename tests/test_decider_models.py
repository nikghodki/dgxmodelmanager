"""Configurable decision model: vllm-sr/d3-mini (default) or vllm-sr/d3-nano, each pinned to a revision with
checked-in hashes, selected with the `decider.model` setting."""
import pytest

from dgxmm.config import GIB
from dgxmm.engines.base import LaunchContext
from dgxmm.engines.decision import DECIDER_MODELS, DEFAULT_MODEL, hash_list_name
from tests.test_api_deployments import admin
from tests.test_decider_engine import D3_CONFIG

MINI, NANO = "vllm-sr/d3-mini", "vllm-sr/d3-nano"
B = "/dgxmm/api/decider"


def add_models(fake_hf_repo, inv, which=(MINI, NANO)):
    for repo in which:
        m = DECIDER_MODELS[repo]
        files = ({"model-00001-of-00002.safetensors": 5 * GIB, "model-00002-of-00002.safetensors": 4 * GIB}
                 if repo == MINI else {"model.safetensors": 4 * GIB})
        fake_hf_repo(repo, rev=m.revision, config=D3_CONFIG, files={**files, "d3_server.py": 100})
    inv.refresh()


def test_registry_pins_both_models_and_mini_stays_default():
    assert DEFAULT_MODEL == MINI
    assert DECIDER_MODELS[MINI].revision == "97dd054df4b3e0b3beb6af23ab8c82b7c0052e0e"
    assert DECIDER_MODELS[NANO].revision == "aeb70d4dd53ed25b855c3b0a66394b8fe265db7f"
    assert DECIDER_MODELS[NANO].served_name == "d3-nano"
    assert DECIDER_MODELS[NANO].reservation_bytes < DECIDER_MODELS[MINI].reservation_bytes


def test_checked_in_hash_lists_cover_code_manifest_and_config():
    from dgxmm.decider.image import CONTEXT_FILES, default_build_dir
    d = default_build_dir()
    df = (d / "Dockerfile").read_text()
    for repo, m in DECIDER_MODELS.items():
        name = hash_list_name(repo, m.revision)
        assert name in CONTEXT_FILES and m.revision in df
        lines = (d / name).read_text().splitlines()
        names = [ln.split()[1] for ln in lines]
        for must in ("d3_server.py", "d3_runtime.py", "modeling_d3.py", "MODEL_MANIFEST.json", "config.json",
                     "decision_config.json", "readout.safetensors", "tokenizer.json"):
            assert must in names, (repo, must)
        assert all(len(ln.split()[0]) == 64 for ln in lines)
    ep = (d / "entrypoint.sh").read_text()
    assert "d3-mini-$D3_REVISION" not in ep  # the hash list follows the selected repo


def test_engine_supports_estimates_and_launches_the_selected_model(stack, fake_hf_repo):
    add_models(fake_hf_repo, stack.inv)
    eng = stack.deployer.engines["decision"]
    for repo, m in DECIDER_MODELS.items():
        art = stack.inv.get(repo)
        assert eng.supports(art)
        prep = stack.deployer.prepare("decision", repo, {})
        assert prep.estimate.total == m.reservation_bytes
        cfg = eng.validate_config({})
        assert eng.served_name(cfg, repo) == m.served_name
        spec = eng.container_spec(art, cfg, None, LaunchContext(1, "decider", 8150, m.reservation_bytes, 121 * GIB),
                                  stack.settings)
        assert spec.environment["D3_REPO"] == repo and spec.environment["D3_REVISION"] == m.revision
        ro = [x.target for x in spec.mounts if x.mode == "ro"]
        assert ro == [f"/models/hf/hub/models--{repo.replace('/', '--')}"]


def test_wrong_revision_is_not_supported(stack, fake_hf_repo):
    fake_hf_repo(NANO, rev="c" * 40, config=D3_CONFIG, files={"model.safetensors": GIB})
    stack.inv.refresh()
    assert not stack.deployer.engines["decision"].supports(stack.inv.get(NANO))


@pytest.fixture
def api(api_stack, fake_hf_repo):
    add_models(fake_hf_repo, api_stack.stack.inv)
    api_stack.h = admin(api_stack)
    api_stack.stack.http.all_healthy = True
    return api_stack


def test_status_lists_models_and_the_configured_one(api):
    st = api.get(f"{B}/status").json()
    assert st["configured_model"] == MINI and st["model"] == MINI
    by = {m["repo"]: m for m in st["models"]}
    assert set(by) == {MINI, NANO}
    assert by[NANO]["revision"] == DECIDER_MODELS[NANO].revision and by[NANO]["downloaded"] is True
    assert by[NANO]["reservation_bytes"] == DECIDER_MODELS[NANO].reservation_bytes
    assert by[MINI]["default"] is True and by[NANO]["default"] is False


def test_select_model_in_settings(api):
    r = api.put(f"{B}/settings", headers=api.h, json={"model": NANO})
    assert r.status_code == 200, r.text
    assert r.json()["configured_model"] == NANO and r.json()["revision"] == DECIDER_MODELS[NANO].revision
    r = api.put(f"{B}/settings", headers=api.h, json={"model": "evil/model"})
    assert r.status_code == 422 and r.json()["detail"]["code"] == "unknown_model"


def test_enable_deploys_the_configured_model(api):
    api.put(f"{B}/settings", headers=api.h, json={"model": NANO})
    r = api.post(f"{B}/enable", headers=api.h, json={})
    assert r.status_code == 202, r.text
    view = api.get(f"/dgxmm/api/deployments/{r.json()['deployment_id']}").json()
    assert view["model_ref"] == NANO and view["reserved_bytes"] == DECIDER_MODELS[NANO].reservation_bytes


def test_enable_switches_a_running_decider_to_the_configured_model(api):
    import time
    r = api.post(f"{B}/enable", headers=api.h, json={})
    dep_id = r.json()["deployment_id"]
    for _ in range(200):
        if api.get(f"/dgxmm/api/deployments/{dep_id}").json()["state"] == "ready":
            break
        time.sleep(0.01)
    api.put(f"{B}/settings", headers=api.h, json={"model": NANO})
    st = api.get(f"{B}/status").json()
    assert st["model"] == MINI and st["configured_model"] == NANO and st["restart_required"] is True
    r = api.post(f"{B}/enable", headers=api.h, json={})
    assert r.status_code == 202, r.text
    assert r.json()["deployment_id"] == dep_id  # same deployment, new model
    view = api.get(f"/dgxmm/api/deployments/{dep_id}").json()
    assert view["model_ref"] == NANO and view["state"] in ("queued", "starting", "loading", "ready")
    assert api.ctx.decider.switch_from == MINI  # the supervisor rolls back to it if d3-nano fails to start


def test_download_endpoint_uses_the_pinned_snapshot_downloader(api):
    """Only the files the decider needs, verified: tests/test_decider_snapshot_download.py covers the run itself."""
    calls = []

    class Downloads:
        def start(self, *a, **kw):
            raise AssertionError("the generic hf_hub_download path is never used for a decision model")

        def start_pinned(self, repo, revision=None, user_id=None):
            calls.append((repo, revision))
            return {"id": 9, "repo_id": repo, "state": "queued"}

        async def launch(self, dl_id):
            calls.append(("launch", dl_id))

    api.ctx.downloads = Downloads()
    r = api.post(f"{B}/models/download", headers=api.h, json={"model": NANO})
    assert r.status_code == 202, r.text
    assert calls[0] == (NANO, DECIDER_MODELS[NANO].revision)
    assert calls[1] == ("launch", 9)
    assert api.post(f"{B}/models/download", headers=api.h, json={"model": "x/y"}).status_code == 422


def test_enable_refuses_an_image_built_before_the_model_was_supported(api):
    # an image from before d3-nano: only the single-revision label of d3-mini
    api.stack.sdk.image_labels["dgxmm-decider:3"] = {"dgxmm.decider.revision": DECIDER_MODELS[MINI].revision}
    api.put(f"{B}/settings", headers=api.h, json={"model": NANO})
    r = api.post(f"{B}/enable", headers=api.h, json={})
    assert r.status_code == 409 and r.json()["detail"]["code"] == "image_outdated", r.text
    assert r.json()["detail"]["build_url"].endswith("/api/decider/image/build")
    st = api.get(f"{B}/status").json()
    assert {m["repo"]: m["image_supports"] for m in st["models"]} == {MINI: True, NANO: False}
    api.put(f"{B}/settings", headers=api.h, json={"model": MINI})
    assert api.post(f"{B}/enable", headers=api.h, json={}).status_code == 202  # d3-mini still runs on it


def test_image_labels_list_every_supported_model():
    from dgxmm.decider.image import default_build_dir
    df = (default_build_dir() / "Dockerfile").read_text()
    for repo, m in DECIDER_MODELS.items():
        assert f"{repo}@{m.revision}" in df


async def test_orphan_decider_container_is_adopted_with_its_model(stack, fake_hf_repo):
    from dgxmm.deployer.reconcile import reconcile
    from dgxmm.engines.base import managed_labels
    from dgxmm.engines.decision import MODEL_LABEL, SYSTEM_LABEL
    add_models(fake_hf_repo, stack.inv)
    eng = stack.deployer.engines["decision"]
    spec = eng.container_spec(stack.inv.get(NANO), eng.validate_config({}), None,
                              LaunchContext(1, "decider", 8150, 8 * GIB, 121 * GIB), stack.settings)
    assert spec.labels[MODEL_LABEL] == NANO
    stack.sdk.add_unmanaged("dgxmm-decider", labels={**managed_labels(77, "decision"), SYSTEM_LABEL: "true",
                                                     MODEL_LABEL: NANO},
                            command=["--port", "8150", "--batch-size", "8"], ports=[(8150, "127.0.0.1")])
    await reconcile(stack.deployer)
    v = next(x for x in stack.deployer.list_views() if x["engine"] == "decision")
    assert v["model_ref"] == NANO and v["system"]


def _xet_layout(stack, repo, rev, files):
    """huggingface_hub >= 2 layout: snapshot -> ../../blobs/<sha> (repo) -> ../../blobs/<xx>/<hash> (shared store)."""
    import os
    from pathlib import Path
    root = Path(stack.settings.hf_cache.local) / "hub"
    d = root / ("models--" + repo.replace("/", "--"))
    snap = d / "snapshots" / rev
    (d / "blobs").mkdir(parents=True, exist_ok=True)
    snap.mkdir(parents=True, exist_ok=True)
    for i, (name, content) in enumerate(files.items()):
        h = f"{i:02x}" + "a" * 62
        shared = root / "blobs" / h[:2] / h
        shared.parent.mkdir(parents=True, exist_ok=True)
        shared.write_bytes(content)
        sha = f"{i:040x}"
        os.symlink(os.path.relpath(shared, d / "blobs"), d / "blobs" / sha)
        os.symlink(os.path.relpath(d / "blobs" / sha, snap), snap / name)
    return root, snap


def test_shared_store_files_of_the_snapshot_are_mounted_one_by_one(stack, fake_hf_repo):
    """huggingface_hub >= 2 keeps contents in a shared store (hub/blobs/<xx>/<hash>); only the files this snapshot
    links to are bind-mounted, never the store itself, other repos' files or its .refs."""
    import json
    import os
    from pathlib import Path
    root, snap = _xet_layout(stack, NANO, DECIDER_MODELS[NANO].revision,
                             {"config.json": json.dumps(D3_CONFIG).encode(), "model.safetensors": b"w" * 64,
                              "d3_server.py": b"print()"})
    (root / "blobs" / ".refs").mkdir()
    (root / "blobs" / ".refs" / "inventory").write_text("other repos")
    (root / "blobs" / "ff").mkdir()
    (root / "blobs" / "ff" / ("f" * 64)).write_bytes(b"someone else's private model")
    os.symlink("/etc/hostname", snap / "escape.txt")  # escapes the HF cache: never mounted
    add_models(fake_hf_repo, stack.inv, which=(MINI,))  # d3-mini: regular-file blobs inside its repo dir
    stack.inv.refresh()
    eng = stack.deployer.engines["decision"]
    launch = LaunchContext(1, "decider", 8150, 9 * GIB, 121 * GIB)
    host = stack.settings.hf_cache.host.rstrip("/")
    mini = eng.container_spec(stack.inv.get(MINI), eng.validate_config({}), None, launch, stack.settings)
    assert [(m.target, m.mode) for m in mini.mounts] == [
        ("/models/hf/hub/models--vllm-sr--d3-mini", "ro")]  # no shared-store mount at all
    nano = eng.container_spec(stack.inv.get(NANO), eng.validate_config({}), None, launch, stack.settings)
    got = sorted((m.host, m.target, m.mode) for m in nano.mounts)
    expect = [(f"{host}/hub/models--vllm-sr--d3-nano", "/models/hf/hub/models--vllm-sr--d3-nano", "ro")]
    for i in range(3):
        h = f"{i:02x}" + "a" * 62
        expect.append((f"{host}/hub/blobs/{h[:2]}/{h}", f"/models/hf/hub/blobs/{h[:2]}/{h}", "ro"))
    assert got == sorted(expect)
    assert not any(".refs" in t or t.endswith("/hub/blobs") or "ffff" in t or "hostname" in h for h, t, _ in got)


def test_shared_store_link_that_is_not_a_regular_file_is_rejected(stack):
    import os
    root, snap = _xet_layout(stack, NANO, DECIDER_MODELS[NANO].revision, {"config.json": b"{}"})
    (root / "blobs" / "dd").mkdir()
    os.symlink(os.path.relpath(root / "blobs" / "dd", snap), snap / "a_directory")
    from dgxmm.engines.decision import shared_blob_mounts
    targets = [m.target for m in shared_blob_mounts(str(snap), NANO, stack.settings)]
    assert len(targets) == 1 and not any(t.endswith("/dd") for t in targets)
