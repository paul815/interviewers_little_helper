"""Тесты ядра без тяжёлых зависимостей (ASR/аудио-железо/Ollama не нужны)."""
from __future__ import annotations

import numpy as np
import pytest

from app.audio.chunker import ChunkAssembler, StreamingChunkAssembler
from app.audio.speech_events import (
    FRAME_SAMPLES,
    EnergyStreamProcessor,
    SpeechEnd,
    SpeechStart,
    SpeechStateMachine,
    create_stream_processor,
)
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


# ------------------------------------------------------- потоковый VAD

def make_machine(**overrides) -> SpeechStateMachine:
    params = dict(min_speech_s=0.25, redemption_s=0.6)
    params.update(overrides)
    return SpeechStateMachine(**params)


def run_probs(machine: SpeechStateMachine, probs: list[float]) -> list:
    events = []
    for p in probs:
        events.extend(machine.advance(p))
    return events


def frames(seconds: float) -> int:
    """Сколько кадров VAD укладывается в отрезок времени."""
    return int(seconds * SR / FRAME_SAMPLES)


def test_state_machine_emits_start_and_end():
    machine = make_machine()
    events = run_probs(machine, [0.9] * frames(1.0) + [0.0] * frames(1.0))
    assert [type(e) for e in events] == [SpeechStart, SpeechEnd]
    start, end = events
    assert start.timestamp_samples == 0
    # Конец реплики — там, где началась тишина, а не там, где сработал redemption.
    assert abs(end.end_timestamp_samples - 1.0 * SR) < 0.05 * SR


def test_state_machine_redemption_survives_pause_inside_phrase():
    """Пауза короче redemption не рвёт реплику на две."""
    machine = make_machine()
    events = run_probs(
        machine,
        [0.9] * frames(0.5) + [0.0] * frames(0.4) + [0.9] * frames(0.5) + [0.0] * frames(1.0),
    )
    assert [type(e) for e in events] == [SpeechStart, SpeechEnd]
    assert events[1].end_timestamp_samples > 1.3 * SR  # обе половины внутри одной реплики


def test_state_machine_hysteresis_sustains_on_marginal_frames():
    """Вероятность между порогами реплику продолжает, но не начинает."""
    machine = make_machine()
    assert run_probs(machine, [0.4] * frames(1.0)) == []
    events = run_probs(machine, [0.9] * frames(0.5) + [0.4] * frames(1.0) + [0.0] * frames(1.0))
    assert [type(e) for e in events] == [SpeechStart, SpeechEnd]
    assert events[1].end_timestamp_samples > 1.4 * SR


def test_state_machine_drops_too_short_speech():
    """Щелчок короче min_speech даёт SpeechStart, но не SpeechEnd."""
    machine = make_machine()
    events = run_probs(machine, [0.9] * frames(0.1) + [0.0] * frames(1.0))
    assert [type(e) for e in events] == [SpeechStart]
    assert not machine.in_speech


def test_state_machine_flush_closes_open_speech():
    machine = make_machine()
    run_probs(machine, [0.9] * frames(1.0))
    assert machine.in_speech
    events = machine.flush(extra_samples=100)
    assert [type(e) for e in events] == [SpeechEnd]
    assert events[0].end_timestamp_samples == machine.cursor_samples + 100
    assert not machine.in_speech


def test_energy_processor_ignores_quiet_noise():
    cfg = AudioConfig(vad="energy-stream")
    proc = create_stream_processor(cfg)
    assert isinstance(proc, EnergyStreamProcessor)
    assert proc.process(tone(2.0, amp=0.0005)) == []
    assert not proc.in_speech


# ------------------------------------------- потоковая нарезка на чанки

def make_stream_assembler(**overrides) -> StreamingChunkAssembler:
    cfg = AudioConfig(vad="energy-stream", **overrides)
    return StreamingChunkAssembler(create_stream_processor(cfg), cfg, Speaker.RESPONDENT)


def test_stream_cuts_on_end_of_speech_not_on_max_chunk():
    """Главный выигрыш: реплика уходит в ASR сразу после паузы."""
    asm = make_stream_assembler(max_chunk_s=25.0)
    chunks = feed_blocks(asm, np.concatenate([silence(0.5), tone(2.0), silence(1.5)]))
    assert len(chunks) == 1
    ch = chunks[0]
    assert ch.speaker == Speaker.RESPONDENT
    # Речь идёт с 0.5 до 2.5 c; допуски — на pre/post-pad и кадр VAD.
    assert 0.1 <= ch.t0 <= 0.55
    assert 2.4 <= ch.t1 <= 2.9
    assert abs(len(ch.audio) / SR - (ch.t1 - ch.t0)) < 0.01


def test_stream_two_replies_split_by_pause():
    asm = make_stream_assembler()
    audio = np.concatenate([tone(1.0), silence(1.2), tone(1.0), silence(1.2)])
    chunks = feed_blocks(asm, audio)
    assert len(chunks) == 2
    assert chunks[0].t1 < chunks[1].t0  # не перекрываются


def test_stream_forced_cut_on_monologue_without_overlap():
    """Монолог режется по max_chunk_s, но без дублей и дыр на стыке."""
    asm = make_stream_assembler(max_chunk_s=3.0)
    chunks = feed_blocks(asm, np.concatenate([tone(8.0), silence(1.5)]))
    assert len(chunks) >= 2
    for prev, nxt in zip(chunks, chunks[1:]):
        assert abs(nxt.t0 - prev.t1) < 0.35, "на стыке не должно быть ни дыры, ни перекрытия"


def test_stream_timestamps_do_not_drift_over_many_feeds():
    """t0/t1 — абсолютные секунды от старта захвата: дрейф сломал бы порядок
    реплик между каналами (они сортируются по t0)."""
    asm = make_stream_assembler()
    audio = np.concatenate([tone(0.8), silence(1.2)] * 20)
    chunks = feed_blocks(asm, audio, block_s=0.1)
    assert len(chunks) == 20
    for i, ch in enumerate(chunks):
        expected_start = i * 2.0
        assert abs(ch.t0 - expected_start) < 0.4, f"чанк {i}: t0={ch.t0}, ждали ~{expected_start}"
    assert chunks == sorted(chunks, key=lambda c: c.t0)


def test_stream_is_sample_exact():
    """Строгая версия проверки стыков: ни один сэмпл не отдан дважды и ни один
    сэмпл речи не потерян.

    Проверяется на шуме с огибающей — каждый сэмпл уникален, поэтому содержимое
    чанка можно сверить с источником точным равенством, а не «примерно по
    длительности». Блоки подаются рваными (вплоть до одного сэмпла), чтобы
    задеть придержанный хвост неполного кадра VAD. `max_chunk_s` мал, а один
    участок речи длинный — значит, обязательно случится принудительный разрез,
    ради которого и заведён `_emitted_through`.
    """
    rng = np.random.default_rng(1234)
    spans = [("-", 1.0), ("v", 2.0), ("-", 1.2), ("v", 9.0), ("-", 1.2), ("v", 1.5), ("-", 1.5)]
    parts, speech_spans, pos = [], [], 0
    for kind, dur in spans:
        n = int(dur * SR)
        parts.append((rng.standard_normal(n) * (0.3 if kind == "v" else 0.0)).astype(np.float32))
        if kind == "v":
            speech_spans.append((pos, pos + n))
        pos += n
    src = np.concatenate(parts)

    asm = make_stream_assembler(max_chunk_s=3.0)
    chunks, i = [], 0
    for step in [3000, 512, 7777, 1, 4096, 999] * 200:
        if i >= len(src):
            break
        chunks.extend(asm.feed(src[i : i + step]))
        i += step
    tail = asm.flush()
    if tail is not None:
        chunks.append(tail)
    assert len(chunks) > 3, "длинный монолог обязан был порезаться на несколько чанков"

    coverage = np.zeros(len(src), dtype=np.int32)
    for ch in chunks:
        a, b = round(ch.t0 * SR), round(ch.t1 * SR)
        # t0/t1 честно описывают то, что лежит внутри чанка.
        assert np.array_equal(ch.audio, src[a:b]), f"чанк {ch.t0:.2f}–{ch.t1:.2f} не совпал с источником"
        coverage[a:b] += 1

    assert coverage.max() <= 1, "какой-то сэмпл отдан в ASR дважды — слова задвоятся"
    for start, end in speech_spans:
        assert coverage[start:end].all(), f"потеряна речь на {start / SR:.2f}–{end / SR:.2f} c"


def test_stream_flush_returns_tail():
    asm = make_stream_assembler()
    assert feed_blocks(asm, np.concatenate([silence(0.5), tone(1.5)])) == []
    final = asm.flush()
    assert final is not None
    assert 1.2 <= len(final.audio) / SR <= 2.2


def test_stream_silence_only_never_chunks():
    asm = make_stream_assembler()
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
