"""The operations features' real-data corpora (tests/fixtures/decider_corpus/{health,reading,routing}): every case
loads through its evaluation adapter, and wherever a rule answers, it is one of the accepted labels (rules must not
be wrong on real data; docs/decider-evaluation.md). Cases the rules leave open go to the model."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from dgxmm.decider import health, reading, routing

CORPORA = Path(__file__).parent / "fixtures" / "decider_corpus"
ADAPTERS = {"health": health.eval_case, "reading": reading.eval_case, "routing": routing.eval_case}
# Known rules misses, documented in docs/decider-evaluation.md (the rules cannot see the cause).
KNOWN_MISSES = {
    "reading": {"history-benchmark-15"},  # a cold first run whose deployment's start events were deleted with it
}


def cases(feature):
    out = []
    for f in sorted((CORPORA / feature).glob("*.jsonl")):
        out += [json.loads(line) for line in f.read_text().splitlines() if line.strip()]
    return out


@pytest.mark.parametrize("feature", sorted(ADAPTERS))
def test_corpus_rules_are_never_wrong(feature):
    items = cases(feature)
    assert len(items) >= 20, f"{feature}: corpus missing or too small"
    ruled = wrong = 0
    for c in items:
        ec = ADAPTERS[feature](c)
        assert ec.options and ec.question and ec.state
        if ec.rules is None:
            continue
        ruled += 1
        accepted = c["label"] if isinstance(c["label"], list) else [c["label"]]
        if ec.rules not in accepted and c["name"] not in KNOWN_MISSES.get(feature, set()):
            wrong += 1
            print(feature, c["name"], ec.rules, accepted)
    assert wrong == 0
    assert ruled > 0


def test_corpus_names_are_unique_and_labels_are_options():
    for feature, fn in ADAPTERS.items():
        items = cases(feature)
        assert len({c["name"] for c in items}) == len(items)
        for c in items:
            keys = {o.key for o in fn(c).options}
            accepted = c["label"] if isinstance(c["label"], list) else [c["label"]]
            assert set(accepted) <= keys, (feature, c["name"])
