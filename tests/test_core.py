"""Тесты ядра без тяжёлых зависимостей (ASR/аудио-железо/Ollama не нужны)."""
from __future__ import annotations

import numpy as np
import pytest

from app.audio.chunker import ChunkAssembler
from app.audio.vad import EnergyDetector
from app.config import AudioConfig, AnalysisConfig, LLMConfig
from app.coverage.engine import CoverageEngine
from app.domain import Speaker
from app.guide.parser import build_guide
from app.guide.schemas import Guide, Section, Topic
from app.llm.ollama_client import robust_json_parse
from app.transcript.store import TranscriptStore

SR = 16000


# ------------------------------------------------------------- JSON parsing

def test_robust_json_parse_plain():
    assert robust_json_parse('{"a": 1}') == {"a": 1}


def test_robust_json_parse_fenced_and_prefixed():
    text = 'Вот ответ:\n```json\n{"a": [1, 2], "b": "x"}\n```\nНадеюсь, помог!'
    assert robust_json_parse(text) == {"a": [1, 2], "b": "x"}


def test_robust_json_parse_think_tags():
    text = '<think>Так, тема s1.t1 — {"вложенный": "мусор"}</think>{"ok": true}'
    assert robust_json_parse(text) == {"ok": True}


def test_robust_json_parse_garbage():
    with pytest.raises(ValueError):
        robust_json_parse("никакого джсона тут нет")


# ----------------------------------------------------------------- чанкер

def tone(seconds: float, amp: float = 0.1) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    return (amp * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def silence(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * SR), dtype=np.float32)


def make_assembler(**overrides) -> ChunkAssembler:
    cfg = AudioConfig(**overrides)
    return ChunkAssembler(EnergyDetector(), cfg, Speaker.RESPONDENT)


def feed_blocks(asm: ChunkAssembler, audio: np.ndarray, block_s: float = 0.5):
    chunks = []
    step = int(block_s * SR)
    for i in range(0, len(audio), step):
        chunks.extend(asm.feed(audio[i : i + step]))
    return chunks


def test_chunk_cut_on_pause():
    asm = make_assembler()
    audio = np.concatenate([silence(1.0), tone(2.0), silence(1.5)])
    chunks = feed_blocks(asm, audio)
    assert len(chunks) == 1
    ch = chunks[0]
    assert ch.speaker == Speaker.RESPONDENT
    assert 0.5 <= ch.t0 <= 1.05
    assert 2.8 <= ch.t1 <= 3.6
    dur = len(ch.audio) / SR
    assert 1.8 <= dur <= 3.0


def test_chunk_forced_cut_on_max_length():
    asm = make_assembler(max_chunk_s=5.0)
    chunks = feed_blocks(asm, tone(8.0))
    assert chunks, "непрерывная речь длиннее max_chunk_s должна порезаться"
    assert len(chunks[0].audio) / SR >= 4.0


def test_chunk_flush_returns_tail():
    asm = make_assembler()
    got = feed_blocks(asm, np.concatenate([silence(0.5), tone(1.5)]))
    assert got == []  # пауза после речи ещё не наступила
    final = asm.flush()
    assert final is not None
    assert 1.0 <= len(final.audio) / SR <= 2.2


def test_silence_only_never_chunks():
    asm = make_assembler()
    assert feed_blocks(asm, silence(12.0)) == []
    assert asm.flush() is None


# ------------------------------------------------------------------- гайд

def test_build_guide_assigns_ids():
    parsed = {
        "title": "Тест",
        "language": "ru",
        "sections": [
            {"title": "Секция А", "topics": [
                {"question": "Вопрос 1?"},
                {"question": "Вопрос 2?", "notes": "если молчит — уточнить"},
            ]},
            {"title": "Пустая", "topics": [{"question": "   "}]},
            {"title": "Секция Б", "topics": [{"question": "Вопрос 3?"}]},
        ],
    }
    guide = build_guide(parsed)
    assert [s.id for s in guide.sections] == ["s1", "s3"]
    assert guide.sections[0].topics[0].id == "s1.t1"
    assert guide.sections[0].topics[1].notes == "если молчит — уточнить"
    assert guide.topic_ids() == ["s1.t1", "s1.t2", "s3.t1"]


def test_build_guide_empty_raises():
    with pytest.raises(ValueError):
        build_guide({"title": "x", "language": "ru", "sections": []})


# -------------------------------------------------------------- транскрипт

def test_transcript_delta_cursor_and_order():
    store = TranscriptStore()
    store.add(Speaker.INTERVIEWER, 10.0, 12.0, "поздний", "ru")
    store.add(Speaker.RESPONDENT, 1.0, 3.0, "ранний", "ru")
    delta, cursor = store.delta_since(0)
    assert [s.text for s in delta] == ["ранний", "поздний"]
    assert cursor == 2
    delta2, cursor2 = store.delta_since(cursor)
    assert delta2 == [] and cursor2 == 2
    store.add(Speaker.INTERVIEWER, 20.0, 22.0, "новый", "ru")
    delta3, _ = store.delta_since(cursor)
    assert [s.text for s in delta3] == ["новый"]


# ------------------------------------------------------------------ движок

def make_engine() -> CoverageEngine:
    guide = Guide(
        guide_id="g-test", language="ru", title="Тест",
        sections=[
            Section(id="s1", title="Секция", topics=[
                Topic(id="s1.t1", question="Вопрос один?"),
                Topic(id="s1.t2", question="Вопрос два?"),
            ]),
        ],
    )

    async def notify(type_, payload):
        pass

    return CoverageEngine(
        guide=guide, transcript=TranscriptStore(), llm=None,
        analysis_cfg=AnalysisConfig(), llm_cfg=LLMConfig(),
        notify=notify, session_store=None, session_id="test",
    )


def test_engine_monotonic_merge():
    eng = make_engine()
    resp = eng._validate({
        "topic_updates": [
            {"topic_id": "s1.t1", "status": "covered", "evidence": "рассказал", "confidence": 0.9},
            {"topic_id": "s1.t9", "status": "covered"},   # неизвестный id
            {"topic_id": "s1.t2", "status": "partial"},
            "мусор",                                        # невалидный элемент
        ],
        "recommendations": [],
    })
    applied = eng._apply_response(resp)
    assert applied == 2
    assert eng.state.topics["s1.t1"].status == "covered"
    assert eng.state.topics["s1.t2"].status == "partial"

    # Понижение статуса игнорируется.
    resp2 = eng._validate({
        "topic_updates": [{"topic_id": "s1.t1", "status": "not_covered"}],
        "recommendations": [],
    })
    assert eng._apply_response(resp2) == 0
    assert eng.state.topics["s1.t1"].status == "covered"


def test_engine_recommendations_lifecycle():
    eng = make_engine()
    resp = eng._validate({
        "topic_updates": [],
        "recommendations": [
            {"type": "coverage_gap", "topic_id": "s1.t2", "urgency": "normal",
             "note": "затронуто мельком", "suggested_question": "Расскажите про два?"},
            {"type": "coverage_gap", "topic_id": "s1.t1", "urgency": "high",
             "note": "не поднималось", "suggested_question": "Расскажите про один?"},
        ],
    })
    eng._apply_response(resp)
    recs = eng.current_recommendations()
    assert len(recs) == 2
    assert recs[0]["topic_id"] == "s1.t1"          # high — первым
    assert recs[0]["topic_question"] == "Вопрос один?"
    assert recs[0]["section_title"] == "Секция"

    # Тема закрылась — её рекомендация исчезает.
    resp2 = eng._validate({
        "topic_updates": [{"topic_id": "s1.t1", "status": "covered"}],
        "recommendations": [],
    })
    eng._apply_response(resp2)
    recs2 = eng.current_recommendations()
    assert [r["topic_id"] for r in recs2] == ["s1.t2"]


def test_engine_rejects_rec_for_covered_topic():
    eng = make_engine()
    eng._apply_response(eng._validate({
        "topic_updates": [{"topic_id": "s1.t1", "status": "covered"}],
        "recommendations": [
            {"type": "coverage_gap", "topic_id": "s1.t1", "urgency": "high",
             "note": "x", "suggested_question": "y"},
        ],
    }))
    assert eng.current_recommendations() == []
