"""Integration (spec 2026-10-17): the evaluation overview in docs/decider-evaluation.md has one row per registered
feature group with the registry's threshold, and protocol.py's calibration notes name every advanced feature."""
import pathlib
import re

from dgxmm.decider import registry

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _overview_rows() -> dict[str, str]:
    doc = (ROOT / "docs" / "decider-evaluation.md").read_text()
    section = doc.split("## Overview: all 16 features", 1)[1].split("\n## ", 1)[0]
    rows = {}
    for line in section.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) >= 2 and re.fullmatch(r"[a-z_]+", cells[0]) and cells[0] != "feature":
            rows[cells[0]] = cells[1]
    return rows


def test_overview_has_one_row_per_feature_group_with_its_threshold():
    groups = [g.name for g in registry.groups()]
    rows = _overview_rows()
    assert sorted(rows) == sorted(groups) and len(rows) == 16
    for g in groups:
        spec = registry.get(g)
        want = "rules only" if not spec.needs_model else f"{spec.default_threshold:g}"
        assert rows[g] == want, (g, rows[g], want)


def test_protocol_notes_name_every_advanced_feature_and_its_threshold():
    text = (ROOT / "dgxmm" / "decider" / "protocol.py").read_text()
    for g in ("health", "config", "compat", "draft", "engine", "command", "schedule", "reading", "routing",
              "remote_code"):
        assert f"# - {g} {registry.get(g).default_threshold:g}" in text, g
    assert "# - tuning: rules only" in text


def test_docs_say_a_switch_that_is_off_turns_the_feature_off_except_the_code_scan():
    """Ruling: off means the feature is off; the remote-code static scan is the one exception (only the model's
    question is skipped). "Rules keep working" is true only of the decision model being disabled."""
    for rel in ("docs/user-guide/decision-model.md", "docs/configuration/decision-model.md"):
        text = " ".join((ROOT / rel).read_text().split())
        assert "its rules keep working" not in text, rel
        assert "rules of the interactive features keep working" not in text, rel
        assert "turns the feature off" in text, rel
        assert "code scan always runs" in text, rel
