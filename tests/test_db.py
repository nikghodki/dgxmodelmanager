import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from dgxmm.db import Deployment, DeploymentEvent, User


def test_roundtrip_user_and_deployment(db):
    with db.session() as s:
        s.add(User(username="admin", pw_hash="x", role="admin"))
        s.add(Deployment(name="q8b", engine="vllm", model_ref="Qwen/Qwen3-8B", config={"max_model_len": 4096},
                         state="planned", port=8100, reserved_bytes=10, estimate={}))
    with db.session() as s:
        dep = s.scalar(select(Deployment).where(Deployment.name == "q8b"))
        assert dep.config == {"max_model_len": 4096}
        assert dep.litellm_status == "none"
        s.add(DeploymentEvent(deployment_id=dep.id, from_state="planned", to_state="starting", message="go"))
    with db.session() as s:
        assert len(s.scalars(select(DeploymentEvent)).all()) == 1


def test_unique_username_and_rollback(db):
    with db.session() as s:
        s.add(User(username="a", pw_hash="x", role="viewer"))
    with pytest.raises(IntegrityError):
        with db.session() as s:
            s.add(User(username="a", pw_hash="y", role="viewer"))
    with db.session() as s:
        assert len(s.scalars(select(User)).all()) == 1
