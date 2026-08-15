"""Tests of the second iteration's features: reconcile, probes, findings, the
library, flags and the report."""
from __future__ import annotations

import pytest

from app.config import AnalysisConfig, LLMConfig
from app.coverage.engine import CoverageEngine
from app.domain import Segment, Speaker
from app.guide.library import GuideLibrary
from app.guide.schemas import Guide, Section, Topic
from app.storage.report import build_report_markdown
from app.storage.session_store import SessionStore
from app.transcript.store import TranscriptStore


def make_guide() -> Guide:
    return Guide(
        guide_id="g-test", language="en", title="Test",
        sections=[
            Section(id="s1", title="Section", topics=[
                Topic(id="s1.t1", question="Question one?"),
                Topic(id="s1.t2", question="Question two?"),
            ]),
        ],
    )


def make_engine(**cfg_overrides) -> CoverageEngine:
    async def notify(type_, payload):
        pass

    return CoverageEngine(
        guide=make_guide(), transcript=TranscriptStore(), llm=None,
        analysis_cfg=AnalysisConfig(**cfg_overrides), llm_cfg=LLMConfig(),
        notify=notify, session_store=None, session_id="test",
    )


# -------------------------------------------------------------------- modes

def test_pick_mode_reconcile_every_n():
    eng = make_engine(reconcile_every=3)
    assert eng._pick_mode() == "delta"          # iteration 1
    eng.state.analysis_iteration = 1
    assert eng._pick_mode() == "delta"          # iteration 2
    eng.state.analysis_iteration = 2
    assert eng._pick_mode() == "reconcile"      # iteration 3
    eng.state.analysis_iteration = 5
    assert eng._pick_mode() == "reconcile"      # iteration 6


def test_pick_mode_disabled():
    eng = make_engine(reconcile_every=0)
    eng.state.analysis_iteration = 3
    assert eng._pick_mode() == "delta"


def test_split_windows_preserves_order_and_bounds():
    segs = [
        Segment(id=i, speaker=Speaker.RESPONDENT, t0=i, t1=i + 1, text="x" * 100)
        for i in range(10)
    ]
    windows = CoverageEngine._split_windows(segs, max_chars=300)
    assert all(w for w in windows)
    flat = [s.id for w in windows for s in w]
    assert flat == list(range(10))
    assert len(windows) > 1
    for w in windows:
        assert sum(len(s.text) + 24 for s in w) <= 300 or len(w) == 1


# ------------------------------------------------------------------- probes

def test_probes_live_one_cycle():
    eng = make_engine()
    r1 = eng._validate({"topic_updates": [], "recommendations": [
        {"type": "probe", "note": "a hook", "suggested_question": "Tell me more about that?",
         "quote": "I gave up on Todoist"},
    ]})
    eng._apply_response(r1)
    probes = eng.current_probes()
    assert len(probes) == 1 and probes[0]["quote"] == "I gave up on Todoist"

    eng._apply_response(eng._validate({"topic_updates": [], "recommendations": []}))
    assert eng.current_probes() == []  # probes go stale after a cycle


def test_probes_capped_and_unknown_topic_detached():
    eng = make_engine(max_probes=2)
    recs = [
        {"type": "probe", "note": f"n{i}", "suggested_question": f"q{i}?"} for i in range(3)
    ]
    recs[0]["topic_id"] = "s9.t9"  # an unknown topic
    eng._apply_response(eng._validate({"topic_updates": [], "recommendations": recs}))
    probes = eng.current_probes()
    assert len(probes) == 2
    assert probes[0]["topic_id"] is None


def test_probes_do_not_displace_gap_recommendations():
    eng = make_engine()
    eng._apply_response(eng._validate({"topic_updates": [], "recommendations": [
        {"type": "coverage_gap", "topic_id": "s1.t1", "note": "never came up",
         "suggested_question": "Question one?", "urgency": "high"},
        {"type": "probe", "note": "a hook", "suggested_question": "Dig into it?"},
    ]}))
    assert len(eng.current_recommendations()) == 1
    assert len(eng.current_probes()) == 1


# ----------------------------------------------------------------- findings

def test_findings_collected_only_in_final_mode():
    eng = make_engine()
    resp = eng._validate({
        "topic_updates": [],
        "findings": [
            {"topic_id": "s1.t1", "finding": "uses Jira"},
            {"topic_id": "s1.t1", "finding": "uses Jira"},         # a duplicate
            {"topic_id": "s9.t9", "finding": "outside the guide"},  # an unknown topic
        ],
    })
    eng._apply_response(resp, mode="delta")
    assert eng.findings() == {}
    eng._apply_response(resp, mode="final")
    assert eng.findings() == {"s1.t1": ["uses Jira"]}


def test_final_mode_ignores_recommendations():
    eng = make_engine()
    eng._apply_response(
        eng._validate({"topic_updates": [], "recommendations": [
            {"type": "coverage_gap", "topic_id": "s1.t1", "note": "x", "suggested_question": "y"},
        ]}),
        mode="final",
    )
    assert eng.current_recommendations() == []


# ------------------------------------------------------------ manual control

def test_manual_status_blocks_llm_updates():
    eng = make_engine()
    eng.set_manual_status("s1.t1", "partial")
    eng._apply_response(eng._validate({
        "topic_updates": [{"topic_id": "s1.t1", "status": "covered"}],
    }))
    assert eng.state.topics["s1.t1"].status == "partial"  # the researcher has the last word

    eng.set_manual_status("s1.t1", None)  # the mark is cleared — the LLM may update again
    eng._apply_response(eng._validate({
        "topic_updates": [{"topic_id": "s1.t1", "status": "covered"}],
    }))
    assert eng.state.topics["s1.t1"].status == "covered"


def test_manual_downgrade_allowed_for_user():
    eng = make_engine()
    eng._apply_response(eng._validate({
        "topic_updates": [{"topic_id": "s1.t1", "status": "covered"}],
    }))
    eng.set_manual_status("s1.t1", "not_covered")  # the user corrected the LLM downwards
    st = eng.state.topics["s1.t1"]
    assert st.status == "not_covered" and st.manual


def test_manual_covered_drops_recommendation():
    eng = make_engine()
    eng._apply_response(eng._validate({"topic_updates": [], "recommendations": [
        {"type": "coverage_gap", "topic_id": "s1.t1", "note": "x", "suggested_question": "y"},
    ]}))
    assert len(eng.current_recommendations()) == 1
    eng.set_manual_status("s1.t1", "covered")
    assert eng.current_recommendations() == []


def test_dismiss_mutes_until_manual_change():
    eng = make_engine()
    rec = {"type": "coverage_gap", "topic_id": "s1.t2", "note": "x", "suggested_question": "y"}
    eng._apply_response(eng._validate({"topic_updates": [], "recommendations": [rec]}))
    eng.dismiss_recommendation("s1.t2")
    assert eng.current_recommendations() == []

    # The next cycle recommends the same topic again — it stays hidden.
    eng._apply_response(eng._validate({"topic_updates": [], "recommendations": [rec]}))
    assert eng.current_recommendations() == []

    # A manual status edit clears the muting.
    eng.set_manual_status("s1.t2", "partial")
    eng.set_manual_status("s1.t2", None)
    eng._apply_response(eng._validate({"topic_updates": [], "recommendations": [rec]}))
    assert [r["topic_id"] for r in eng.current_recommendations()] == ["s1.t2"]


def test_set_manual_status_unknown_topic_raises():
    eng = make_engine()
    with pytest.raises(ValueError):
        eng.set_manual_status("s9.t9", "covered")


# --------------------------------------------------------------------- library

def test_guide_library_roundtrip(tmp_path):
    lib = GuideLibrary(tmp_path)
    assert lib.list() == []
    file_id = lib.save(make_guide(), source_text="raw text")
    items = lib.list()
    assert len(items) == 1 and items[0]["topics_count"] == 2
    loaded = lib.load(file_id)
    assert loaded["guide"]["title"] == "Test"
    assert loaded["source_text"] == "raw text"
    lib.delete(file_id)
    assert lib.list() == []


def test_guide_library_rejects_path_traversal(tmp_path):
    lib = GuideLibrary(tmp_path)
    with pytest.raises(ValueError):
        lib.load("../evil")


# ------------------------------------------------------- flags and the report

def test_flags_interleaved_in_markdown(tmp_path):
    store = SessionStore(tmp_path, "s-test")
    segs = [
        Segment(id=1, speaker=Speaker.INTERVIEWER, t0=1.0, t1=2.0, text="First utterance"),
        Segment(id=2, speaker=Speaker.RESPONDENT, t0=10.0, t1=12.0, text="Second utterance"),
    ]
    store.add_flag(5.0, "important")
    store.render_markdown(segs, make_guide())
    md = (tmp_path / "s-test" / "transcript.md").read_text(encoding="utf-8")
    assert md.index("First utterance") < md.index("🚩") < md.index("Second utterance")
    assert "important" in md
    assert (tmp_path / "s-test" / "flags.jsonl").exists()
    store.close()


def test_comment_keeps_the_question_it_was_written_under(tmp_path):
    """A comment without the guide question cannot be deciphered a week later."""
    store = SessionStore(tmp_path, "s-anchor")
    store.add_flag(
        7.0, "confused about the pricing tiers",
        anchor="Tell me about your channel, please. What is it about?",
        anchor_section="The channel and the role of Telegram",
        topic_id="s1.t1",
    )
    saved = store.flags[0]
    assert saved["anchor_section"] == "The channel and the role of Telegram"
    assert saved["topic_id"] == "s1.t1"
    store.render_markdown([], make_guide())
    md = (tmp_path / "s-anchor" / "transcript.md").read_text(encoding="utf-8")
    assert "Researcher comment" in md and "confused about the pricing tiers" in md
    assert "The channel and the role of Telegram → Tell me about your channel" in md
    store.close()


def test_bare_flag_still_reads_as_a_bookmark(tmp_path):
    store = SessionStore(tmp_path, "s-bare")
    store.add_flag(3.0)
    store.render_markdown([], make_guide())
    md = (tmp_path / "s-bare" / "transcript.md").read_text(encoding="utf-8")
    assert "Researcher flag" in md and "comment" not in md.lower()
    store.close()


def test_questions_survive_toggle_and_reach_transcript(tmp_path):
    store = SessionStore(tmp_path, "s-q")
    first = store.add_question(12.0, "ask about the second channel")
    store.add_question(30.0, "find out who answers at night")
    assert store.set_question_done(first["id"], True)["done"] is True
    assert store.set_question_done(999, True) is None

    rows = (tmp_path / "s-q" / "questions.jsonl").read_text(encoding="utf-8").strip().split("\n")
    assert len(rows) == 2 and '"done": true' in rows[0]

    store.render_markdown([], make_guide())
    md = (tmp_path / "s-q" / "transcript.md").read_text(encoding="utf-8")
    assert "## Extra questions" in md
    assert "- [x] [00:12] ask about the second channel" in md
    assert "- [ ] [00:30] find out who answers at night" in md
    store.close()


def test_original_guide_text_saved_next_to_the_parsed_one(tmp_path):
    store = SessionStore(tmp_path, "s-src")
    store.save_guide_text("Introductions\nGood afternoon! My name is Paul.")
    saved = (tmp_path / "s-src" / "guide_source.txt").read_text(encoding="utf-8")
    assert saved.startswith("Introductions")
    store.save_guide_text("   ")  # empty text does not wipe the saved original
    assert (tmp_path / "s-src" / "guide_source.txt").read_text(encoding="utf-8") == saved
    store.close()


def test_report_markdown_structure():
    eng = make_engine()
    eng._apply_response(eng._validate({
        "topic_updates": [
            {"topic_id": "s1.t1", "status": "covered", "evidence": "talked about Jira"},
        ],
    }))
    md = build_report_markdown(
        session_id="s-test",
        guide=make_guide(),
        state=eng.state,
        findings={"s1.t1": ["uses Jira", "loses tasks among the notes"]},
        flags=[{"t": 65.0, "note": "a striking quote"}],
        meta={"started_at": "2026-07-11", "segments": 42},
        summary="A short summary of the session.",
    )
    assert "## Summary" in md and "A short summary of the session." in md
    assert "✅ Question one?" in md and "❌ Question two?" in md
    assert "- uses Jira" in md
    assert "## Not covered" in md and "Question two?" in md
    assert "🚩" in md and "1:05" in md and "a striking quote" in md
