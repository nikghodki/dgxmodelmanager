"""Insight body templates read as clean text: singular / plural time and count phrases ("the last minute", "1
request was preempted"), and the tuning evidence table is aligned column by column."""
from __future__ import annotations

from types import SimpleNamespace

from dgxmm.decider import health as hl
from dgxmm.decider import schedule as sc
from dgxmm.decider import tuning
from dgxmm.decider.common import last_span, plural


def test_plural_and_last_span():
    assert plural(1, "minute") == "1 minute"
    assert plural(0, "minute") == "0 minutes"
    assert plural(3, "class", "classes") == "3 classes"
    assert plural(1.0, "hour") == "1 hour" and plural(2.5, "hour") == "2.5 hours"
    assert last_span(1, "minute") == "the last minute"
    assert last_span(0.98, "minute") == "the last minute"  # shown rounded: never "the last 1 minutes"
    assert last_span(5, "minute") == "the last 5 minutes"
    assert last_span(None, "minute") == "the last few minutes"


def _s(**over):
    return {"minutes": 1.0, "kv_high_fraction": 0.5, "kv_p95": 0.97, "preemptions": 1, "preemptions_per_min": 1.0,
            "per_seq_tps": 10.0, "throughput_ratio": 0.4, "baseline_per_seq_tps": 25.0, "baseline_minutes": 1, **over}


def test_health_explanations_pluralise():
    one = hl.explain("kv_pressure", _s(), {})
    assert "over the last minute" in one and "1 minutes" not in one and "1 request was preempted" in one
    many = hl.explain("kv_pressure", _s(minutes=5.0, preemptions=3), {})
    assert "over the last 5 minutes" in many and "3 requests were preempted" in many
    storm = hl.explain("preemption_storm", _s(), {})
    assert "over the last minute:" in storm and "1 minutes" not in storm
    drop = hl.explain("throughput_drop", _s(), {})
    assert "1 minute)" in drop and "1 minutes" not in drop
    err = hl.explain("error_creep", _s(), {"status_5xx": 1, "error_lines": 1})
    assert "1 HTTP 5xx answer and 1 server error line in" in err
    err2 = hl.explain("error_creep", _s(), {"status_5xx": 2, "error_lines": 0})
    assert "2 HTTP 5xx answers and 0 server error lines" in err2
    mis = hl.explain("client_misuse", _s(), {"requests": 1, "status_4xx": 1})
    assert "1 of 1 request (100 %) was rejected" in mis
    mis2 = hl.explain("client_misuse", _s(), {"requests": 5, "status_4xx": 2})
    assert "2 of 5 requests (40 %) were rejected" in mis2


def test_schedule_proposal_pluralises():
    cand = SimpleNamespace(windows=[(0, 8)], min_weeks=3, sleep_at="22:00", wake_at="06:00", sleep_days=127,
                           wake_days=127, idle_hours=1.0)
    body = sc.proposal_body({"name": "d1", "reserved_bytes": 0}, cand, "UTC")
    assert "the same window every week" in body and "(s)" not in body and "about 1 hour a week" in body
    cand.windows, cand.idle_hours = [(0, 8), (24, 8)], 56.0
    body = sc.proposal_body({"name": "d1", "reserved_bytes": 0}, cand, "UTC")
    assert "the same 2 windows every week" in body and "about 56 hours a week" in body


def test_tuning_body_table_is_aligned_and_counts_pluralise():
    table = [{"threshold": 0.8, "n": 120, "accepted": 100, "precision": 0.8333, "wilson_lo": 0.75},
             {"threshold": 0.9, "n": 9, "accepted": 9, "precision": 1.0, "wilson_lo": 0.7},
             {"threshold": 0.95, "n": 0, "accepted": 0, "precision": None, "wilson_lo": None}]
    body = tuning.render_body("Health triage", 0.8, 0.9, 0.7, 1, table)
    assert "supported by 1 judged model answer of" in body
    lines = body.split("\n")
    head = next(i for i, ln in enumerate(lines) if ln.startswith("threshold"))
    rows = lines[head:head + 2 + len(table)]
    assert all(set(c) <= {"-", " ", "|"} for c in [rows[1]])
    # every row has its column separators at the same offsets
    cols = [[i for i, ch in enumerate(r) if ch == "|"] for r in rows]
    assert len(cols[0]) == 4 and all(c == cols[0] for c in cols)
    assert "0.90 (proposed)" in body and "0.80 (current)" in body
    assert tuning.render_body("x", 0.8, 0.9, 0.7, 2, table).count("2 judged model answers") == 1


def test_job_results_pluralise_their_counts(monkeypatch):
    """The Background jobs table shows these: never "1 proposals" / "1 profiles"."""
    monkeypatch.setattr(tuning, "tune", lambda ctx: {"checked": 15, "proposed": [object()]})
    assert tuning.job(SimpleNamespace(ctx=None)) == "15 features checked, 1 proposal"
    monkeypatch.setattr(tuning, "tune", lambda ctx: {"checked": 1, "proposed": []})
    assert tuning.job(SimpleNamespace(ctx=None)) == "1 feature checked, 0 proposals"
    from dgxmm.decider import features
    assert features.profiles_text({"profiles": 1}) == "1 profile"
    assert features.profiles_text({"profiles": 3}) == "3 profiles"
