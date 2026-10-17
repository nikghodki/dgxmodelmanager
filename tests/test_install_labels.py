"""Per-install labels (``dgxmm.install=<DGXMM_INSTALL_ID>``) on the containers DGXMM creates and the images it
builds, so ``./install.sh uninstall --purge`` removes only this install's decider image and container."""
from __future__ import annotations

from tests.test_api_decider import B, api  # noqa: F401  (fixture)


def test_containers_and_decider_image_carry_the_install_label(api):  # noqa: F811
    api.ctx.settings.install_id = "inst42abc"
    api.stack.sdk.missing_images.add("dgxmm-decider:3")
    assert api.post(f"{B}/image/build", headers=api.h).status_code == 202
    for _ in range(100):
        if not api.ctx.decider_image.building:
            break
        __import__("time").sleep(0.02)
    call = api.stack.sdk.api.build_calls[-1]
    assert call["labels"] == {"dgxmm.built": "true", "dgxmm.install": "inst42abc"}
    assert api.post(f"{B}/enable", headers=api.h, json={}).status_code == 202
    c = next(c for c in api.stack.sdk._containers.values() if c.labels.get("dgxmm.engine") == "decision")
    assert c.labels["dgxmm.install"] == "inst42abc" and c.labels["dgxmm.managed"] == "true"
    # the decider's internal network is created by this install too: uninstall --purge removes only a labelled one
    net = api.stack.sdk.networks.get(api.ctx.settings.decider_network)
    assert net.created_kwargs["labels"] == {"dgxmm.managed": "true", "dgxmm.purpose": "decider",
                                            "dgxmm.install": "inst42abc"}


def test_no_install_label_without_an_install_id(api):  # noqa: F811
    assert api.ctx.settings.install_id is None
    assert api.post(f"{B}/enable", headers=api.h, json={}).status_code == 202
    c = next(c for c in api.stack.sdk._containers.values() if c.labels.get("dgxmm.engine") == "decision")
    assert "dgxmm.install" not in c.labels
    net = api.stack.sdk.networks.get(api.ctx.settings.decider_network)
    assert "dgxmm.install" not in net.created_kwargs["labels"]


def test_install_id_from_the_environment(monkeypatch, tmp_path):
    from dgxmm.config import load_settings
    monkeypatch.setenv("DGXMM_INSTALL_ID", "fromenv12")
    cfg = tmp_path / "c.yaml"
    cfg.write_text("{}\n")
    assert load_settings(str(cfg)).install_id == "fromenv12"
