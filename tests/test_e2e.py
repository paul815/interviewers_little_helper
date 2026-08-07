"""E2E без железа: конвейер аудио->чанкер->ASR-воркер->транскрипт на фейковом
ASR-бэкенде и полный цикл движка (анализ, сверка, финал, отчёт) на фейковой LLM."""
from __future__ import annotations

import asyncio
import json
import threading
import time
import queue as queue_mod

import numpy as np
import pytest

from app.asr.base import ASRBackend, ASRResult
from app.asr.worker import ASRWorker
from app.audio.capture import RingBuffer
from app.audio.chunker import ChunkerThread, create_assembler
from app.config import ASRConfig, AnalysisConfig, AudioConfig, LLMConfig
from app.coverage.engine import CoverageEngine
from app.domain import Speaker
from app.guide.schemas import Guide, Section, Topic
from app.llm.ollama_client import ChatMeta
from app.storage.report import build_report_markdown
from app.storage.session_store import SessionStore
from app.transcript.store import TranscriptStore

SR = 16000


# ------------------------------------------------------- фейковые бэкенды

class FakeASRBackend(ASRBackend):
    """Отдаёт заготовленные реплики по очереди, независимо от аудио."""

    def __init__(self, lines: list[str]):
        self.lines = list(lines)
        self.name = "fake"
        self._i = 0

    def load(self) -> None:
        pass

    def transcribe(self, audio: np.ndarray) -> ASRResult:
        text = self.lines[self._i % len(self.lines)]
        self._i += 1
        return ASRResult(text=text, language="ru", no_speech_prob=0.0)


class FakeLLM:
    """Детерминированные ответы: live-схема -> обновления+рекомендации,
    final-схема (по наличию findings) -> обновления+тезисы."""

    def __init__(self):
        self.calls: list[str] = []

    async def chat_json(self, system: str, user: str, schema: dict):
        final = "findings" in schema.get("properties", {})
        self.calls.append("final" if final else "live")
        if final:
            parsed = {
                "topic_updates": [
                    {"topic_id": "s1.t2", "status": "partial", "evidence": "мельком упомянул"},
                ],
                "findings": [
                    {"topic_id": "s1.t1", "finding": "пользуется Jira и Notion"},
                ],
            }
        else:
            parsed = {
                "topic_updates": [
                    {"topic_id": "s1.t1", "status": "covered",
                     "evidence": "рассказал про инструменты", "confidence": 0.9},
                ],
                "recommendations": [
                    {"type": "coverage_gap", "topic_id": "s1.t2", "urgency": "high",
                     "note": "не поднималось", "suggested_question": "А что с потерями задач?"},
                    {"type": "probe", "note": "зацепка про стикеры",
                     "suggested_question": "Почему стикеры, а не приложение?",
                     "quote": "клею стикеры на монитор"},
                ],
            }
        meta = ChatMeta(duration_s=0.01, prompt_chars=len(system) + len(user),
                        raw_response=json.dumps(parsed, ensure_ascii=False))
        return parsed, meta


def make_guide() -> Guide:
    return Guide(
        guide_id="g-e2e", language="ru", title="E2E",
        sections=[Section(id="s1", title="Секция", topics=[
            Topic(id="s1.t1", question="Какие инструменты?"),
            Topic(id="s1.t2", question="Теряются ли задачи?"),
        ])],
    )


# ------------------------------------------------------------- конвейер

def tone(seconds: float) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    return (0.1 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def silence(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * SR), dtype=np.float32)


# Оба пути нарезки: потоковый (по умолчанию) и батчевый по паузам (запасной).
# Ни один не требует onnxruntime — энергетический источник событий встроен.
@pytest.mark.parametrize("vad", ["energy-stream", "energy"])
def test_audio_to_transcript_pipeline(vad):
    audio_cfg = AudioConfig(
        poll_interval_s=0.05, min_pause_s=0.4, min_speech_s=0.2, pad_s=0.05, vad=vad,
    )
    ring = RingBuffer(audio_cfg.ring_seconds, SR)
    out_q: "queue_mod.Queue" = queue_mod.Queue()
    stop = threading.Event()
    transcript = TranscriptStore()

    assembler = create_assembler(audio_cfg, Speaker.RESPONDENT)
    chunker = ChunkerThread(Speaker.RESPONDENT, ring, assembler, out_q, audio_cfg, stop)
    backend = FakeASRBackend(["первая реплика", "вторая реплика"])
    worker = ASRWorker(
        backend, out_q,
        on_segment=lambda ch, txt, lang: transcript.add(ch.speaker, ch.t0, ch.t1, txt, lang),
        on_status=lambda phase, msg: None,
        cfg=ASRConfig(), stop_event=stop,
    )
    chunker.start()
    worker.start()

    # Две «реплики», разделённые паузой; хвост без паузы дожмётся при остановке.
    ring.append(np.concatenate([tone(1.0), silence(0.8), tone(1.0)]))

    deadline = time.monotonic() + 8.0
    while len(transcript) < 1 and time.monotonic() < deadline:
        time.sleep(0.05)
    assert len(transcript) >= 1, "первый чанк не дошёл до транскрипта"

    stop.set()
    chunker.join(5.0)
    worker.join(5.0)
    assert not chunker.is_alive() and not worker.is_alive()

    segs = transcript.all_segments()
    assert len(segs) == 2, [s.text for s in segs]
    assert segs[0].text == "первая реплика" and segs[1].text == "вторая реплика"
    assert segs[0].speaker == Speaker.RESPONDENT
    assert 0 <= segs[0].t0 < segs[1].t0  # хронология сохранена


def test_pipeline_latency_in_real_time():
    """Задержка «человек договорил → строка в транскрипте» на живом темпе.

    Аудио подаётся в RingBuffer блоками по 20 мс по стенным часам, как это
    делал бы драйвер, и работают настоящие ChunkerThread + ASRWorker. ASR
    отвечает мгновенно, поэтому замер показывает вклад **самого конвейера**:
    `vad_redemption_s` (0.6) + до `poll_interval_s` (0.2) + работа VAD.
    Ради этого числа всё и переделывалось, так что регресс тут должен ронять
    тест, а не всплывать на живом интервью.

    Границы широкие: нижняя ловит разрез посреди речи, верхняя — грубое
    замедление, не мигая на медленном раннере.
    """
    cfg = AudioConfig(vad="energy-stream")
    ring = RingBuffer(cfg.ring_seconds, SR)
    out_q: "queue_mod.Queue" = queue_mod.Queue()
    stop = threading.Event()
    transcript = TranscriptStore()
    arrivals: list[float] = []
    transcript.add_listener(lambda seg: arrivals.append(time.monotonic()))

    chunker = ChunkerThread(
        Speaker.RESPONDENT, ring, create_assembler(cfg, Speaker.RESPONDENT),
        out_q, cfg, stop,
    )
    worker = ASRWorker(
        FakeASRBackend(["реплика"]), out_q,
        on_segment=lambda ch, txt, lang: transcript.add(ch.speaker, ch.t0, ch.t1, txt, lang),
        on_status=lambda phase, msg: None,
        cfg=ASRConfig(), stop_event=stop,
    )
    chunker.start()
    worker.start()

    # Шум как «речь»: энергетический источник событий смотрит на RMS. Тишины
    # в хвосте заведомо больше, чем vad_redemption_s, — иначе реплику закрыл бы
    # только flush на остановке, и замер потерял бы смысл.
    rng = np.random.default_rng(7)
    block_s, blocks_fed = 0.02, 0
    started = time.monotonic()
    speech_ended_at = None
    for amplitude, duration in ((0.0, 0.3), (0.3, 1.0), (0.0, 1.4)):
        for _ in range(int(duration / block_s)):
            samples = rng.standard_normal(int(block_s * SR)) * amplitude
            ring.append(samples.astype(np.float32))
            blocks_fed += 1
            time.sleep(max(0.0, started + blocks_fed * block_s - time.monotonic()))
        if amplitude:
            speech_ended_at = time.monotonic()

    deadline = time.monotonic() + 5.0
    while not arrivals and time.monotonic() < deadline:
        time.sleep(0.01)
    stop.set()
    chunker.join(5.0)
    worker.join(5.0)

    assert arrivals, "реплика так и не доехала до транскрипта за отведённое время"
    latency = arrivals[0] - speech_ended_at
    assert latency >= 0.4, f"сегмент выдан через {latency:.2f} c — раньше, чем VAD мог закрыть реплику"
    assert latency <= 3.0, f"задержка конвейера {latency:.2f} c — что-то заметно тормозит"


# ------------------------------------------------- движок + отчёт целиком

def test_engine_cycles_and_report(tmp_path):
    store = SessionStore(tmp_path, "e2e")
    transcript = TranscriptStore()
    transcript.add_listener(store.append_segment)
    guide = make_guide()
    llm = FakeLLM()
    events: list[str] = []

    async def notify(type_, payload):
        events.append(type_)

    engine = CoverageEngine(
        guide=guide, transcript=transcript, llm=llm,
        analysis_cfg=AnalysisConfig(max_probes=2), llm_cfg=LLMConfig(),
        notify=notify, session_store=store, session_id="e2e",
    )

    async def scenario():
        transcript.add(Speaker.INTERVIEWER, 1.0, 3.0, "Какими инструментами пользуетесь?", "ru")
        transcript.add(Speaker.RESPONDENT, 4.0, 9.0, "Jira, Notion, клею стикеры на монитор", "ru")
        await engine.analyze(manual=True)

        assert engine.state.topics["s1.t1"].status == "covered"
        assert [r["topic_id"] for r in engine.current_recommendations()] == ["s1.t2"]
        assert len(engine.current_probes()) == 1

        # Пустая дельта -> цикл пропускается, счётчик не растёт.
        before = engine.state.analysis_iteration
        await engine.analyze(manual=True)
        assert engine.state.analysis_iteration == before

        ok = await engine.final_pass()
        assert ok
        assert engine.state.topics["s1.t2"].status == "partial"
        assert engine.findings() == {"s1.t1": ["пользуется Jira и Notion"]}
        assert llm.calls[-1] == "final"

    asyncio.run(scenario())

    store.add_flag(5.0, "стикеры!")
    md = build_report_markdown(
        session_id="e2e", guide=guide, state=engine.state,
        findings=engine.findings(), flags=store.flags,
        meta={"started_at": "t", "segments": len(transcript)}, summary=None,
    )
    store.save_report(md)
    store.finalize({"stopped_at": "t2", "segments": len(transcript)})

    d = tmp_path / "e2e"
    for name in ("transcript.jsonl", "coverage_state.json", "recommendations.jsonl",
                 "analysis_log.jsonl", "flags.jsonl", "report.md", "meta.json"):
        assert (d / name).exists(), name

    saved = json.loads((d / "coverage_state.json").read_text(encoding="utf-8"))
    assert saved["topics"]["s1.t1"]["status"] == "covered"
    report_text = (d / "report.md").read_text(encoding="utf-8")
    assert "пользуется Jira и Notion" in report_text and "стикеры!" in report_text
    assert "coverage" in events and "recommendations" in events
