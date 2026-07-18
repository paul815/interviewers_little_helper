"""Тесты фич второй итерации: сверка, пробы, findings, библиотека, флаги, отчёт."""
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
        guide_id="g-test", language="ru", title="Тест",
        sections=[
            Section(id="s1", title="Секция", topics=[
                Topic(id="s1.t1", question="Вопрос один?"),
                Topic(id="s1.t2", question="Вопрос два?"),
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


# ------------------------------------------------------------------- режимы

def test_pick_mode_reconcile_every_n():
    eng = make_engine(reconcile_every=3)
    assert eng._pick_mode() == "delta"          # итерация 1
    eng.state.analysis_iteration = 1
    assert eng._pick_mode() == "delta"          # итерация 2
    eng.state.analysis_iteration = 2
    assert eng._pick_mode() == "reconcile"      # итерация 3
    eng.state.analysis_iteration = 5
    assert eng._pick_mode() == "reconcile"      # итерация 6


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


# -------------------------------------------------------------------- пробы

def test_probes_live_one_cycle():
    eng = make_engine()
    r1 = eng._validate({"topic_updates": [], "recommendations": [
        {"type": "probe", "note": "зацепка", "suggested_question": "Расскажите подробнее?",
         "quote": "я бросил Todoist"},
    ]})
    eng._apply_response(r1)
    probes = eng.current_probes()
    assert len(probes) == 1 and probes[0]["quote"] == "я бросил Todoist"

    eng._apply_response(eng._validate({"topic_updates": [], "recommendations": []}))
    assert eng.current_probes() == []  # пробы устаревают за цикл


def test_probes_capped_and_unknown_topic_detached():
    eng = make_engine(max_probes=2)
    recs = [
        {"type": "probe", "note": f"n{i}", "suggested_question": f"q{i}?"} for i in range(3)
    ]
    recs[0]["topic_id"] = "s9.t9"  # неизвестная тема
    eng._apply_response(eng._validate({"topic_updates": [], "recommendations": recs}))
    probes = eng.current_probes()
    assert len(probes) == 2
    assert probes[0]["topic_id"] is None


def test_probes_do_not_displace_gap_recommendations():
    eng = make_engine()
    eng._apply_response(eng._validate({"topic_updates": [], "recommendations": [
        {"type": "coverage_gap", "topic_id": "s1.t1", "note": "не поднималось",
         "suggested_question": "Вопрос один?", "urgency": "high"},
        {"type": "probe", "note": "зацепка", "suggested_question": "Копнуть?"},
    ]}))
    assert len(eng.current_recommendations()) == 1
    assert len(eng.current_probes()) == 1


# ----------------------------------------------------------------- findings

def test_findings_collected_only_in_final_mode():
    eng = make_engine()
    resp = eng._validate({
        "topic_updates": [],
        "findings": [
            {"topic_id": "s1.t1", "finding": "пользуется Jira"},
            {"topic_id": "s1.t1", "finding": "пользуется Jira"},   # дубль
            {"topic_id": "s9.t9", "finding": "мимо гайда"},        # неизвестная тема
        ],
    })
    eng._apply_response(resp, mode="delta")
    assert eng.findings() == {}
    eng._apply_response(resp, mode="final")
    assert eng.findings() == {"s1.t1": ["пользуется Jira"]}


def test_final_mode_ignores_recommendations():
    eng = make_engine()
    eng._apply_response(
        eng._validate({"topic_updates": [], "recommendations": [
            {"type": "coverage_gap", "topic_id": "s1.t1", "note": "x", "suggested_question": "y"},
        ]}),
        mode="final",
    )
    assert eng.current_recommendations() == []


# --------------------------------------------------------- ручное управление

def test_manual_status_blocks_llm_updates():
    eng = make_engine()
    eng.set_manual_status("s1.t1", "partial")
    eng._apply_response(eng._validate({
        "topic_updates": [{"topic_id": "s1.t1", "status": "covered"}],
    }))
    assert eng.state.topics["s1.t1"].status == "partial"  # слово исследователя — последнее

    eng.set_manual_status("s1.t1", None)  # метка снята — LLM снова может обновлять
    eng._apply_response(eng._validate({
        "topic_updates": [{"topic_id": "s1.t1", "status": "covered"}],
    }))
    assert eng.state.topics["s1.t1"].status == "covered"


def test_manual_downgrade_allowed_for_user():
    eng = make_engine()
    eng._apply_response(eng._validate({
        "topic_updates": [{"topic_id": "s1.t1", "status": "covered"}],
    }))
    eng.set_manual_status("s1.t1", "not_covered")  # пользователь поправил LLM вниз
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

    # Следующий цикл снова рекомендует ту же тему — остаётся скрытой.
    eng._apply_response(eng._validate({"topic_updates": [], "recommendations": [rec]}))
    assert eng.current_recommendations() == []

    # Ручная правка статуса снимает скрытие.
    eng.set_manual_status("s1.t2", "partial")
    eng.set_manual_status("s1.t2", None)
    eng._apply_response(eng._validate({"topic_updates": [], "recommendations": [rec]}))
    assert [r["topic_id"] for r in eng.current_recommendations()] == ["s1.t2"]


def test_set_manual_status_unknown_topic_raises():
    eng = make_engine()
    with pytest.raises(ValueError):
        eng.set_manual_status("s9.t9", "covered")


# ------------------------------------------------------------------ библиотека

def test_guide_library_roundtrip(tmp_path):
    lib = GuideLibrary(tmp_path)
    assert lib.list() == []
    file_id = lib.save(make_guide(), source_text="сырой текст")
    items = lib.list()
    assert len(items) == 1 and items[0]["topics_count"] == 2
    loaded = lib.load(file_id)
    assert loaded["guide"]["title"] == "Тест"
    assert loaded["source_text"] == "сырой текст"
    lib.delete(file_id)
    assert lib.list() == []


def test_guide_library_rejects_path_traversal(tmp_path):
    lib = GuideLibrary(tmp_path)
    with pytest.raises(ValueError):
        lib.load("../evil")


# ------------------------------------------------------------- флаги и отчёт

def test_flags_interleaved_in_markdown(tmp_path):
    store = SessionStore(tmp_path, "s-test")
    segs = [
        Segment(id=1, speaker=Speaker.INTERVIEWER, t0=1.0, t1=2.0, text="Первая реплика"),
        Segment(id=2, speaker=Speaker.RESPONDENT, t0=10.0, t1=12.0, text="Вторая реплика"),
    ]
    store.add_flag(5.0, "важно")
    store.render_markdown(segs, make_guide())
    md = (tmp_path / "s-test" / "transcript.md").read_text(encoding="utf-8")
    assert md.index("Первая реплика") < md.index("🚩") < md.index("Вторая реплика")
    assert "важно" in md
    assert (tmp_path / "s-test" / "flags.jsonl").exists()
    store.close()


def test_report_markdown_structure():
    eng = make_engine()
    eng._apply_response(eng._validate({
        "topic_updates": [
            {"topic_id": "s1.t1", "status": "covered", "evidence": "рассказал про Jira"},
        ],
    }))
    md = build_report_markdown(
        session_id="s-test",
        guide=make_guide(),
        state=eng.state,
        findings={"s1.t1": ["пользуется Jira", "теряет задачи в заметках"]},
        flags=[{"t": 65.0, "note": "яркая цитата"}],
        meta={"started_at": "2026-07-11", "segments": 42},
        summary="Краткое резюме сессии.",
    )
    assert "## Резюме" in md and "Краткое резюме сессии." in md
    assert "✅ Вопрос один?" in md and "❌ Вопрос два?" in md
    assert "- пользуется Jira" in md
    assert "## Не раскрыто" in md and "Вопрос два?" in md
    assert "🚩" in md and "1:05" in md and "яркая цитата" in md
