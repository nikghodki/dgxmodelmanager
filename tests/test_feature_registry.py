"""The decision-model feature registry (dgxmm/decider/registry.py): the five shipped features keep their names,
groups and thresholds; the eleven advanced ones are pre-registered; settings, status, rate limits and the Decider
are driven from it."""
from __future__ import annotations

import pytest

from dgxmm.decider import registry
from dgxmm.decider.registry import FeatureSpec, all_features, get, groups, register

SHIPPED = {"diagnosis": 0.8, "parser": 0.8, "plan": 0.6, "kind": 0.8, "workload": 0.75}
CALIBRATED = {"command": 0.7, "health": 0.85, "reading": 0.95}  # recalibrated on the real corpora (docs/decider-evaluation.md)
NEW = ["health", "config", "compat", "draft", "engine", "command", "schedule", "reading", "tuning", "routing",
       "remote_code"]


def test_shipped_features_keep_names_groups_and_thresholds():
    for g, th in SHIPPED.items():
        assert get(g).group == g and get(g).default_threshold == th
    assert get("parser_tool").group == get("parser_reasoning").group == "parser"
    assert get("parser_tool").default_threshold == get("parser_reasoning").default_threshold == 0.8
    assert [s.name for s in groups()][:5] == list(SHIPPED)


def test_background_flags_of_the_shipped_features_match_the_old_set():
    assert {s.name for s in all_features() if s.background and s.name in SHIPPED} == {"diagnosis", "kind", "workload"}
    assert not get("parser_tool").background and not get("plan").background


@pytest.mark.parametrize("name", NEW)
def test_new_features_are_preregistered(name):
    s = get(name)
    assert s.group == name and s.label and s.description
    assert s.default_threshold == CALIBRATED.get(name, 0.8)
    assert s.needs_model is (name != "tuning")


def test_settings_groups_are_the_sixteen_features_in_display_order():
    assert [s.name for s in groups()] == list(SHIPPED) + NEW
    assert registry.DEFAULT_THRESHOLDS == {s.name: s.default_threshold for s in groups()}


def test_register_validates_and_rejects_duplicates():
    with pytest.raises(ValueError):
        register(FeatureSpec("diagnosis", "diagnosis", "x", "y", 0.8))
    with pytest.raises(ValueError):
        FeatureSpec("Bad-Name", "x", "x", "y", 0.8)
    with pytest.raises(ValueError):
        FeatureSpec("ok", "ok", "x", "y", 1.5)
    with pytest.raises(ValueError):  # a member of a group that does not exist
        register(FeatureSpec("ghost_member", "ghost", "x", "y", 0.8))


def test_register_a_new_feature_updates_every_derived_table():
    from dgxmm.api.ratelimit import RATE_LIMITS, bucket_limits
    from dgxmm.decider.service import FEATURES, feature_group
    spec = FeatureSpec("zz_probe", "zz_probe", "Probe", "Test-only feature.", 0.65, background=True)
    try:
        register(spec)
        assert FEATURES["zz_probe"] == 0.65 and feature_group("zz_probe") == "zz_probe"
        assert bucket_limits(registry.feature_bucket("zz_probe")) == spec.rate_limit
        assert "decider_admin" in RATE_LIMITS  # the static buckets are untouched
    finally:
        registry._unregister("zz_probe")
    assert "zz_probe" not in FEATURES


def test_unknown_feature_lookups_fall_back():
    assert registry.find("nope") is None
    assert registry.group_of("nope") == "nope"
    with pytest.raises(KeyError):
        get("nope")


def test_rate_limit_buckets_come_from_the_registry():
    from dgxmm.api.ratelimit import bucket_limits
    assert bucket_limits("decider_suggest") == (30, 60.0, 120)  # parser: unchanged shipped bucket
    assert bucket_limits("decider_plan") == (30, 60.0, 120)
    assert registry.feature_bucket("parser") == "decider_suggest" and registry.feature_bucket("plan") == "decider_plan"
    for name in NEW:
        assert bucket_limits(registry.feature_bucket(name)) == get(name).rate_limit
    with pytest.raises(KeyError):
        bucket_limits("nope")


def test_feature_info_for_the_ui():
    info = registry.feature_info()
    assert [i["name"] for i in info] == list(SHIPPED) + NEW
    assert set(info[0]) == {"name", "label", "description", "default_threshold", "background", "needs_model",
                            "viewer_visible", "members"}
    assert next(i for i in info if i["name"] == "parser")["members"] == ["parser_tool", "parser_reasoning"]


def test_a_feature_cannot_take_a_static_bucket():
    from dgxmm.api.ratelimit import RATE_LIMITS
    for name in ("admin", "read", "feedback"):
        with pytest.raises(ValueError):
            register(FeatureSpec(name, name, "x", "y", 0.8))
    with pytest.raises(ValueError):
        register(FeatureSpec("zz_probe2", "zz_probe2", "x", "y", 0.8, bucket="insights_read"))
    assert registry.find("admin") is None
    static = set(RATE_LIMITS) - set(registry.buckets())
    assert static == registry.STATIC_BUCKETS


async def test_status_defaults_to_the_viewer_view(api_stack):
    from dgxmm.api.decider import status_out
    out = await status_out(api_stack.ctx)
    assert "jobs" not in out and "feature_specs" in out
