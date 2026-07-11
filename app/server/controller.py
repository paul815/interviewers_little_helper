"""Оркестратор сессии: связывает захват аудио, чанкеры, ASR-воркер,
транскрипт, движок покрытия и персистентность."""
from __future__ import annotations

import asyncio
import logging
import queue
import threading
import time
from datetime import datetime

from .. import __version__
from ..config import AppConfig
from ..coverage.engine import CoverageEngine
from ..domain import AudioChunk, Speaker
from ..guide.parser import parse_guide_text
from ..guide.schemas import Guide
from ..llm.ollama_client import OllamaClient, OllamaError
from ..transcript.store import TranscriptStore
from ..storage.session_store import SessionStore
from .hub import WsHub

log = logging.getLogger("ilh.controller")


class ControllerError(Exception):
    """Ошибка уровня API с человекочитаемым сообщением."""


class SessionRuntime:
    """Все живые части одной сессии."""

    def __init__(self):
        self.store: SessionStore | None = None
        self.transcript: TranscriptStore | None = None
        self.captures: list = []
        self.chunkers: list = []
        self.asr_worker = None
        self.asr_queue: "queue.Queue[AudioChunk]" = queue.Queue()
        self.engine: CoverageEngine | None = None
        self.scheduler_task: asyncio.Task | None = None
        self.thread_stop = threading.Event()
        self.engine_stop: asyncio.Event | None = None
        self.manual_event: asyncio.Event | None = None
        self.guide: Guide | None = None
        self.started_wall: str = ""
        self.asr_status: str = "loading"


class AppController:
    def __init__(self, cfg: AppConfig, hub: WsHub):
        self.cfg = cfg
        self.hub = hub
        self.llm = OllamaClient(cfg.llm)
        self.state = "idle"  # idle | starting | running | stopping
        self.rt: SessionRuntime | None = None
        self._transition_lock = asyncio.Lock()

    # ------------------------------------------------------------------ гайд

    async def parse_guide(self, text: str) -> Guide:
        check = await self.llm.check()
        if not check["ok"]:
            raise ControllerError(check["error"] or "LLM недоступна")
        try:
            return await parse_guide_text(self.llm, text)
        except OllamaError as e:
            raise ControllerError(str(e)) from e
        except ValueError as e:
            raise ControllerError(str(e)) from e

    # ---------------------------------------------------------------- сессия

    async def start_session(self, mic_index: int, system_index: int, guide_data: dict) -> dict:
        async with self._transition_lock:
            if self.state != "idle":
                raise ControllerError("Сессия уже запущена")
            if mic_index == system_index:
                raise ControllerError("Микрофон и системный звук — одно и то же устройство")
            try:
                guide = Guide.model_validate(guide_data)
            except Exception as e:
                raise ControllerError(f"Невалидная структура гайда: {e}") from e
            if not guide.topic_ids():
                raise ControllerError("В гайде нет ни одной темы")

            check = await self.llm.check()
            if not check["ok"]:
                raise ControllerError(check["error"] or "LLM недоступна")

            self.state = "starting"
            await self._status("Запуск сессии…")
            rt = SessionRuntime()
            try:
                await self._build_runtime(rt, mic_index, system_index, guide)
            except Exception as e:
                log.exception("Не удалось запустить сессию")
                await self._teardown_runtime(rt)
                self.state = "idle"
                await self._status("Ошибка запуска")
                raise ControllerError(str(e)) from e

            self.rt = rt
            self.state = "running"
            await self._status("Сессия идёт. ASR догружается…"
                               if rt.asr_status == "loading" else "Сессия идёт")
            await self.hub.broadcast("session", {"state": "running", "session_id": rt.store.session_id})
            return {"session_id": rt.store.session_id}

    async def _build_runtime(self, rt: SessionRuntime, mic_index: int, system_index: int, guide: Guide) -> None:
        from ..asr.factory import create_asr_backend
        from ..asr.worker import ASRWorker
        from ..audio.capture import ChannelCapture
        from ..audio.chunker import ChunkerThread
        from ..audio.vad import create_detector

        rt.guide = guide
        rt.started_wall = datetime.now().astimezone().isoformat()

        rt.store = SessionStore.create(self.cfg, {"guide_title": guide.title})
        rt.store.save_guide(guide)

        rt.transcript = TranscriptStore()
        rt.transcript.add_listener(rt.store.append_segment)
        rt.transcript.add_listener(
            lambda seg: self.hub.broadcast_threadsafe("segment", seg.to_dict())
        )

        def asr_status(phase: str, message: str) -> None:
            rt.asr_status = {"asr_loading": "loading", "asr_ready": "ready"}.get(phase, "error")
            self.hub.broadcast_threadsafe("status", {"message": message, "asr": rt.asr_status})

        def on_segment(chunk: AudioChunk, text: str, language: str | None) -> None:
            rt.transcript.add(chunk.speaker, chunk.t0, chunk.t1, text, language)

        backend = create_asr_backend(self.cfg.asr)
        rt.asr_worker = ASRWorker(
            backend, rt.asr_queue, on_segment, asr_status, self.cfg.asr, rt.thread_stop
        )
        rt.asr_worker.start()  # модель грузится в фоне; аудио тем временем копится

        for device_index, speaker in ((mic_index, Speaker.INTERVIEWER), (system_index, Speaker.RESPONDENT)):
            capture = ChannelCapture(device_index, speaker, self.cfg.audio)
            capture.start()
            rt.captures.append(capture)
            chunker = ChunkerThread(
                speaker, capture.ring, create_detector(self.cfg.audio.vad),
                rt.asr_queue, self.cfg.audio, rt.thread_stop,
            )
            chunker.start()
            rt.chunkers.append(chunker)

        rt.engine_stop = asyncio.Event()
        rt.manual_event = asyncio.Event()
        rt.engine = CoverageEngine(
            guide=guide,
            transcript=rt.transcript,
            llm=self.llm,
            analysis_cfg=self.cfg.analysis,
            llm_cfg=self.cfg.llm,
            notify=self.hub.broadcast,
            session_store=rt.store,
            session_id=rt.store.session_id,
        )
        rt.scheduler_task = asyncio.create_task(rt.engine.run(rt.engine_stop, rt.manual_event))

    async def stop_session(self) -> dict:
        async with self._transition_lock:
            if self.state not in ("running", "starting"):
                raise ControllerError("Сессия не запущена")
            self.state = "stopping"
            rt = self.rt
            await self._status("Остановка: дораспознаём хвост аудио…")
            await self._teardown_runtime(rt)
            session_id = rt.store.session_id if rt.store else None
            self.rt = None
            self.state = "idle"
            await self._status(f"Сессия сохранена: sessions/{session_id}" if session_id else "Сессия остановлена")
            await self.hub.broadcast("session", {"state": "idle", "session_id": session_id})
            await self.hub.broadcast("timer", {"next_analysis_at": None, "interval_s": self.cfg.analysis.interval_s})
            return {"session_id": session_id}

    async def _teardown_runtime(self, rt: SessionRuntime) -> None:
        # 1. Останавливаем планировщик анализа.
        if rt.engine_stop is not None:
            rt.engine_stop.set()
        if rt.scheduler_task is not None:
            rt.scheduler_task.cancel()
            try:
                await rt.scheduler_task
            except (asyncio.CancelledError, Exception):
                pass
        # 2. Захват: новые сэмплы больше не поступают.
        for capture in rt.captures:
            await asyncio.to_thread(capture.stop)
        # 3. Чанкеры дожимают буферы и кладут финальные чанки в очередь.
        rt.thread_stop.set()
        for chunker in rt.chunkers:
            await asyncio.to_thread(chunker.join, 10.0)
        # 4. ASR-воркер дорабатывает очередь (см. условие выхода в run()).
        if rt.asr_worker is not None:
            await asyncio.to_thread(rt.asr_worker.join, 60.0)
            if rt.asr_worker.is_alive():
                log.warning("ASR-воркер не завершился за 60 c — хвост может быть потерян")
        # 5. Финальная запись на диск.
        if rt.store is not None:
            try:
                if rt.engine is not None:
                    rt.store.save_coverage(rt.engine.state)
                    rt.store.append_recommendations(
                        rt.engine.state.analysis_iteration, rt.engine.current_recommendations()
                    )
                if rt.transcript is not None and rt.guide is not None:
                    rt.store.render_markdown(rt.transcript.all_segments(), rt.guide)
                stats = {
                    f"xruns_{c.speaker.value.lower()}": c.stats.xruns for c in rt.captures
                }
                rt.store.finalize(
                    {"stopped_at": datetime.now().astimezone().isoformat(),
                     "segments": len(rt.transcript) if rt.transcript else 0,
                     **stats}
                )
            except Exception:
                log.exception("Ошибка финального сохранения сессии")

    # ------------------------------------------------------------- прочее API

    def analyze_now(self) -> None:
        if self.state != "running" or self.rt is None or self.rt.manual_event is None:
            raise ControllerError("Сессия не запущена")
        if self.rt.engine is not None and self.rt.engine.analyzing:
            raise ControllerError("Анализ уже выполняется")
        self.rt.manual_event.set()

    def snapshot(self) -> dict:
        rt = self.rt
        snap: dict = {
            "state": self.state,
            "app_version": __version__,
            "interval_s": self.cfg.analysis.interval_s,
            "llm_model": self.cfg.llm.model,
            "session_id": None,
            "guide": None,
            "coverage": None,
            "recommendations": None,
            "segments": [],
            "next_analysis_at": None,
            "analyzing": False,
            "asr_status": None,
        }
        if rt is not None:
            snap.update(
                session_id=rt.store.session_id if rt.store else None,
                guide=rt.guide.model_dump() if rt.guide else None,
                segments=[s.to_dict() for s in rt.transcript.tail(300)] if rt.transcript else [],
                asr_status=rt.asr_status,
            )
            if rt.engine is not None:
                snap.update(
                    coverage=rt.engine.coverage_payload(),
                    recommendations=rt.engine.recommendations_payload(),
                    next_analysis_at=rt.engine.next_analysis_at,
                    analyzing=rt.engine.analyzing,
                )
        return snap

    async def _status(self, message: str) -> None:
        log.info("Статус: %s", message)
        await self.hub.broadcast("status", {"state": self.state, "message": message})
