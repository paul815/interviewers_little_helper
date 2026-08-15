"""E2E without hardware: the audio -> chunker -> ASR worker -> transcript pipeline
on a fake ASR backend, and the engine's full cycle (analysis, reconcile, final,
report) on a fake LLM."""
from __future__ import annotations

import asyncio
import json
import queue as queue_mod
import threading
import time

import numpy as np
import pytest

from app.asr.base import ASRBackend, ASRResult
from app.asr.worker import ASRWorker
from app.audio.capture import RingBuffer
from app.audio.chunker import ChunkerThread, create_assembler
from app.config import AnalysisConfig, ASRConfig, AudioConfig, LLMConfig
from app.coverage.engine import CoverageEngine
from app.domain import Speaker
from app.guide.schemas import Guide, Section, Topic
from app.llm.ollama_client import ChatMeta
from app.storage.report import build_report_markdown
from app.storage.session_store import SessionStore
from app.transcript.store import TranscriptStore

SR = 16000


# ------------------------------------------------------------ fake backends

class FakeASRBackend(ASRBackend):
    """Hands out prepared utterances in turn, regardless of the audio."""

    def __init__(self, lines: list[str]):
        self.lines = list(lines)
        self.name = "fake"
        self._i = 0

    def load(self) -> None:
        pass

    def transcribe(self, audio: np.ndarray) -> ASRResult:
        text = self.lines[self._i % len(self.lines)]
        self._i += 1
        return ASRResult(text=text, language="en", no_speech_prob=0.0)


class FakeLLM:
    """Deterministic answers: the live schema -> updates plus recommendations,
    the final schema (detected by the presence of findings) -> updates plus points."""

    def __init__(self):
        self.calls: list[str] = []

    async def chat_json(self, system: str, user: str, schema: dict):
        final = "findings" in schema.get("properties", {})
        self.calls.append("final" if final else "live")
        if final:
            parsed = {
                "topic_updates": [
                    {"topic_id": "s1.t2", "status": "partial", "evidence": "mentioned in passing"},
                ],
                "findings": [
                    {"topic_id": "s1.t1", "finding": "uses Jira and Notion"},
                ],
            }
        else:
            parsed = {
                "topic_updates": [
                    {"topic_id": "s1.t1", "status": "covered",
                     "evidence": "talked about the tools", "confidence": 0.9},
                ],
                "recommendations": [
                    {"type": "coverage_gap", "topic_id": "s1.t2", "urgency": "high",
                     "note": "never came up",
                     "suggested_question": "And what about tasks getting lost?"},
                    {"type": "probe", "note": "a hook about the sticky notes",
                     "suggested_question": "Why sticky notes rather than an app?",
                     "quote": "I stick notes on the monitor"},
                ],
            }
        meta = ChatMeta(duration_s=0.01, prompt_chars=len(system) + len(user),
                        raw_response=json.dumps(parsed, ensure_ascii=False))
        return parsed, meta


def make_guide() -> Guide:
    return Guide(
        guide_id="g-e2e", language="en", title="E2E",
        sections=[Section(id="s1", title="Section", topics=[
            Topic(id="s1.t1", question="Which tools?"),
            Topic(id="s1.t2", question="Do tasks get lost?"),
        ])],
    )


# ------------------------------------------------------------- pipeline

def tone(seconds: float) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    return (0.1 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def silence(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * SR), dtype=np.float32)


# Both cutting routes: streaming (the default) and batch by pauses (the fallback).
# Neither needs onnxruntime — the energy event source is built in.
@pytest.mark.parametrize("vad", ["energy-stream", "energy"])
def test_audio_to_transcript_pipeline(vad):
    audio_cfg = AudioConfig(
        poll_interval_s=0.05, min_pause_s=0.4, min_speech_s=0.2, pad_s=0.05, vad=vad,
    )
    ring = RingBuffer(audio_cfg.ring_seconds, SR)
    out_q: queue_mod.Queue = queue_mod.Queue()
    stop = threading.Event()
    transcript = TranscriptStore()

    assembler = create_assembler(audio_cfg, Speaker.RESPONDENT)
    chunker = ChunkerThread(Speaker.RESPONDENT, ring, assembler, out_q, audio_cfg, stop)
    backend = FakeASRBackend(["first utterance", "second utterance"])
    worker = ASRWorker(
        backend, out_q,
        on_segment=lambda ch, txt, lang: transcript.add(ch.speaker, ch.t0, ch.t1, txt, lang),
        on_status=lambda phase, msg: None,
        cfg=ASRConfig(), stop_event=stop,
    )
    chunker.start()
    worker.start()

    # Two "utterances" separated by a pause; the tail without a pause is flushed on stop.
    ring.append(np.concatenate([tone(1.0), silence(0.8), tone(1.0)]))

    deadline = time.monotonic() + 8.0
    while len(transcript) < 1 and time.monotonic() < deadline:
        time.sleep(0.05)
    assert len(transcript) >= 1, "the first chunk never reached the transcript"

    stop.set()
    chunker.join(5.0)
    worker.join(5.0)
    assert not chunker.is_alive() and not worker.is_alive()

    segs = transcript.all_segments()
    assert len(segs) == 2, [s.text for s in segs]
    assert segs[0].text == "first utterance" and segs[1].text == "second utterance"
    assert segs[0].speaker == Speaker.RESPONDENT
    assert 0 <= segs[0].t0 < segs[1].t0  # the chronology is preserved


def test_pipeline_latency_in_real_time():
    """The "the person finished speaking -> a line in the transcript" latency at a live pace.

    Audio is fed into the RingBuffer in 20 ms blocks by wall clock, as a driver
    would, and real ChunkerThread and ASRWorker instances are running. ASR answers
    instantly, so the measurement shows the contribution of **the pipeline
    itself**: `vad_redemption_s` (0.6) plus up to `poll_interval_s` (0.2) plus the
    VAD's own work. This number is what the whole rework was for, so a regression
    here should fail the test rather than surface in a live interview.

    The bounds are wide: the lower one catches a cut in mid-speech, the upper one
    a gross slowdown, without flickering on a slow runner.
    """
    cfg = AudioConfig(vad="energy-stream")
    ring = RingBuffer(cfg.ring_seconds, SR)
    out_q: queue_mod.Queue = queue_mod.Queue()
    stop = threading.Event()
    transcript = TranscriptStore()
    arrivals: list[float] = []
    transcript.add_listener(lambda seg: arrivals.append(time.monotonic()))

    chunker = ChunkerThread(
        Speaker.RESPONDENT, ring, create_assembler(cfg, Speaker.RESPONDENT),
        out_q, cfg, stop,
    )
    worker = ASRWorker(
        FakeASRBackend(["utterance"]), out_q,
        on_segment=lambda ch, txt, lang: transcript.add(ch.speaker, ch.t0, ch.t1, txt, lang),
        on_status=lambda phase, msg: None,
        cfg=ASRConfig(), stop_event=stop,
    )
    chunker.start()
    worker.start()

    # Noise as "speech": the energy event source looks at RMS. There is
    # deliberately more silence in the tail than vad_redemption_s — otherwise the
    # utterance would only be closed by the flush on stop and the measurement
    # would be meaningless.
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

    assert arrivals, "the utterance never reached the transcript within the time allowed"
    latency = arrivals[0] - speech_ended_at
    assert latency >= 0.4, (
        f"the segment came out after {latency:.2f} s — sooner than the VAD could "
        f"close the utterance"
    )
    assert latency <= 3.0, f"pipeline latency {latency:.2f} s — something is markedly slow"


# ------------------------------------------------ the engine and the report

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
        transcript.add(Speaker.INTERVIEWER, 1.0, 3.0, "Which tools do you use?", "en")
        transcript.add(Speaker.RESPONDENT, 4.0, 9.0,
                       "Jira, Notion, I stick notes on the monitor", "en")
        await engine.analyze(manual=True)

        assert engine.state.topics["s1.t1"].status == "covered"
        assert [r["topic_id"] for r in engine.current_recommendations()] == ["s1.t2"]
        assert len(engine.current_probes()) == 1

        # An empty delta -> the cycle is skipped and the counter does not grow.
        before = engine.state.analysis_iteration
        await engine.analyze(manual=True)
        assert engine.state.analysis_iteration == before

        ok = await engine.final_pass()
        assert ok
        assert engine.state.topics["s1.t2"].status == "partial"
        assert engine.findings() == {"s1.t1": ["uses Jira and Notion"]}
        assert llm.calls[-1] == "final"

    asyncio.run(scenario())

    store.add_flag(5.0, "sticky notes!")
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
    assert "uses Jira and Notion" in report_text and "sticky notes!" in report_text
    assert "coverage" in events and "recommendations" in events
