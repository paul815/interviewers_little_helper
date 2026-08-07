"""Оркестратор сессии: связывает захват аудио, чанкеры, ASR-воркер,
транскрипт, движок покрытия и персистентность."""
from __future__ import annotations

import asyncio
import dataclasses
import logging
import queue
import threading
import time
from datetime import datetime

from .. import __version__
from ..config import AppConfig
from ..coverage.engine import CoverageEngine
from ..domain import SPEAKER_FULL_RU, AudioChunk, Speaker
from ..guide.library import GuideLibrary
from ..guide.parser import parse_guide_text
from ..guide.schemas import Guide
from ..llm.ollama_client import OllamaClient, OllamaError
from ..storage import report
from ..storage.session_store import SessionStore
from ..transcript.store import TranscriptStore
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
        self.watchdog_task: asyncio.Task | None = None
        self.thread_stop = threading.Event()
        self.engine_stop: asyncio.Event | None = None
        self.manual_event: asyncio.Event | None = None
        self.guide: Guide | None = None
        self.started_wall: str = ""
        self.started_at_epoch: float = 0.0
        self.t0_mono: float = 0.0
        self.duration_min: int = 60
        self.asr_vocabulary: str = ""
        self.asr_status: str = "loading"
        self.channel_alive: dict[str, bool] = {}


class AppController:
    def __init__(self, cfg: AppConfig, hub: WsHub):
        self.cfg = cfg
        self.hub = hub
        self.llm = OllamaClient(cfg.llm)
        self.library = GuideLibrary(cfg.guides_path)
        self.state = "idle"  # idle | starting | running | stopping
        self.rt: SessionRuntime | None = None
        self._transition_lock = asyncio.Lock()
        self._monitor_caps: list = []
        self._monitor_task: asyncio.Task | None = None

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

    # ------------------------------------------------- монитор уровней (idle)

    async def start_monitor(self, mic_index: int | None, system_index: int | None) -> dict:
        """Открывает выбранные устройства до старта сессии, чтобы UI показывал
        уровни и пользователь убедился в маршрутизации звука."""
        if self.state != "idle":
            raise ControllerError("Сессия уже идёт — уровни транслируются из неё")
        from ..audio.capture import ChannelCapture

        await self.stop_monitor()
        caps, errors = [], {}
        pairs = ((mic_index, Speaker.INTERVIEWER), (system_index, Speaker.RESPONDENT))
        for idx, speaker in pairs:
            if idx is None or idx < 0:
                continue
            try:
                cap = ChannelCapture(idx, speaker, self.cfg.audio)
                await asyncio.to_thread(cap.start)
                caps.append(cap)
            except Exception as e:
                errors[speaker.value.lower()] = str(e)
        self._monitor_caps = caps
        if caps:
            self._monitor_task = asyncio.create_task(self._levels_loop(caps, monitor=True))
        return {"ok": bool(caps), "errors": errors}

    async def stop_monitor(self) -> None:
        task, self._monitor_task = self._monitor_task, None
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        caps, self._monitor_caps = self._monitor_caps, []
        for cap in caps:
            try:
                await asyncio.to_thread(cap.stop)
            except Exception:
                log.exception("Ошибка остановки монитора")

    async def _levels_loop(self, captures: list, monitor: bool) -> None:
        try:
            while True:
                payload = {"monitor": monitor}
                for cap in captures:
                    payload[cap.speaker.value.lower()] = round(cap.level, 3)
                await self.hub.broadcast("levels", payload)
                await asyncio.sleep(0.12)
        except asyncio.CancelledError:
            raise

    # ---------------------------------------------------------------- сессия

    async def start_session(
        self,
        mic_index: int,
        system_index: int,
        guide_data: dict,
        duration_min: int | None = None,
        asr_vocabulary: str | None = None,
    ) -> dict:
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

            await self.stop_monitor()  # освобождаем устройства перед сессией
            self.state = "starting"
            await self._status("Запуск сессии…")
            rt = SessionRuntime()
            rt.duration_min = duration_min or self.cfg.analysis.default_duration_min
            rt.asr_vocabulary = (asr_vocabulary or self.cfg.asr.vocabulary).strip()
            try:
                await self._build_runtime(rt, mic_index, system_index, guide)
            except Exception as e:
                log.exception("Не удалось запустить сессию")
                await self._stop_audio_asr(rt)
                await self._final_persist(rt)
                self.state = "idle"
                await self._status("Ошибка запуска")
                raise ControllerError(str(e)) from e

            self.rt = rt
            self.state = "running"
            await self._status("Сессия идёт. ASR догружается…"
                               if rt.asr_status == "loading" else "Сессия идёт")
            await self.hub.broadcast(
                "session",
                {"state": "running", "session_id": rt.store.session_id,
                 "started_at": rt.started_at_epoch, "duration_min": rt.duration_min},
            )
            return {"session_id": rt.store.session_id}

    async def _build_runtime(self, rt: SessionRuntime, mic_index: int, system_index: int, guide: Guide) -> None:
        from ..asr.factory import create_asr_backend
        from ..asr.worker import ASRWorker
        from ..audio.capture import ChannelCapture
        from ..audio.chunker import ChunkerThread, create_assembler

        rt.guide = guide
        rt.started_wall = datetime.now().astimezone().isoformat()
        rt.started_at_epoch = time.time()

        rt.store = SessionStore.create(
            self.cfg,
            {"guide_title": guide.title, "duration_min": rt.duration_min,
             "asr_vocabulary": rt.asr_vocabulary},
        )
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

        asr_cfg = dataclasses.replace(self.cfg.asr, vocabulary=rt.asr_vocabulary)
        backend = create_asr_backend(asr_cfg)
        rt.asr_worker = ASRWorker(
            backend, rt.asr_queue, on_segment, asr_status, asr_cfg, rt.thread_stop
        )
        rt.asr_worker.start()  # модель грузится в фоне; аудио тем временем копится

        rt.t0_mono = time.monotonic()
        for device_index, speaker in ((mic_index, Speaker.INTERVIEWER), (system_index, Speaker.RESPONDENT)):
            capture = ChannelCapture(device_index, speaker, self.cfg.audio)
            capture.start()
            rt.captures.append(capture)
            rt.channel_alive[speaker.value.lower()] = True
            chunker = ChunkerThread(
                speaker, capture.ring, create_assembler(self.cfg.audio, speaker),
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
        rt.watchdog_task = asyncio.create_task(self._watchdog_loop(rt))

    # ------------------------------------------------------- watchdog каналов

    async def _watchdog_loop(self, rt: SessionRuntime) -> None:
        """VU-уровни в UI + контроль, что устройства живы; мёртвые переоткрываются."""
        silence_s = self.cfg.audio.watchdog_silence_s
        last_reopen: dict[str, float] = {}
        tick = 0
        try:
            while True:
                await asyncio.sleep(0.12)
                payload = {"monitor": False}
                for cap in rt.captures:
                    payload[cap.speaker.value.lower()] = round(cap.level, 3)
                await self.hub.broadcast("levels", payload)

                tick += 1
                if tick % 25 != 0:  # проверка живости ~раз в 3 c
                    continue
                now = time.monotonic()
                for cap in rt.captures:
                    key = cap.speaker.value.lower()
                    alive = (now - cap.last_sample_time) <= silence_s
                    if alive != rt.channel_alive.get(key, True):
                        rt.channel_alive[key] = alive
                        name = SPEAKER_FULL_RU[cap.speaker]
                        await self.hub.broadcast(
                            "channel", {"speaker": key, "alive": alive, "device": cap.device_name}
                        )
                        if alive:
                            await self.hub.broadcast(
                                "status", {"message": f"Канал «{name}» восстановлен"}
                            )
                    if not alive and now - last_reopen.get(key, 0.0) > 10.0:
                        last_reopen[key] = now
                        await self.hub.broadcast("error", {
                            "message": f"Канал «{SPEAKER_FULL_RU[cap.speaker]}» не получает звук "
                                       f"(устройство «{cap.device_name}») — переоткрываю",
                        })
                        try:
                            await asyncio.to_thread(cap.stop)
                            await asyncio.to_thread(cap.start)
                        except Exception as e:
                            log.warning("Не удалось переоткрыть устройство %s: %s", cap.device_name, e)
        except asyncio.CancelledError:
            raise

    # ------------------------------------------------------------- остановка

    async def stop_session(self) -> dict:
        async with self._transition_lock:
            if self.state not in ("running", "starting"):
                raise ControllerError("Сессия не запущена")
            self.state = "stopping"
            rt = self.rt
            await self._status("Остановка: дораспознаём хвост аудио…")
            await self._stop_audio_asr(rt)

            has_transcript = rt.transcript is not None and len(rt.transcript) > 0
            if self.cfg.analysis.final_sweep and rt.engine is not None and has_transcript:
                try:
                    await rt.engine.final_pass()
                except Exception:
                    log.exception("Финальная сверка не удалась")
                    await self.hub.broadcast("error", {"message": "Финальная сверка не удалась — статусы сохранены как есть"})
            if self.cfg.analysis.report and rt.engine is not None and has_transcript:
                await self._status("Готовлю отчёт сессии…")
                try:
                    await self._write_report(rt)
                except Exception:
                    log.exception("Не удалось собрать отчёт")
                    await self.hub.broadcast("error", {"message": "Отчёт не собран — подробности в logs/app.log"})

            await self._final_persist(rt)
            session_id = rt.store.session_id if rt.store else None
            self.rt = None
            self.state = "idle"
            await self._status(f"Сессия сохранена: sessions/{session_id}" if session_id else "Сессия остановлена")
            await self.hub.broadcast("session", {"state": "idle", "session_id": session_id})
            await self.hub.broadcast("timer", {"next_analysis_at": None, "interval_s": self.cfg.analysis.interval_s})
            return {"session_id": session_id}

    async def _stop_audio_asr(self, rt: SessionRuntime) -> None:
        # 1. Останавливаем планировщик анализа и watchdog.
        if rt.engine_stop is not None:
            rt.engine_stop.set()
        for task in (rt.scheduler_task, rt.watchdog_task):
            if task is not None:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
        rt.scheduler_task = rt.watchdog_task = None
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

    async def _final_persist(self, rt: SessionRuntime) -> None:
        if rt.store is None:
            return
        try:
            if rt.engine is not None:
                rt.store.save_coverage(rt.engine.state)
                rt.store.append_recommendations(
                    rt.engine.state.analysis_iteration, rt.engine.current_recommendations()
                )
            if rt.transcript is not None and rt.guide is not None:
                rt.store.render_markdown(rt.transcript.all_segments(), rt.guide)
            stats = {f"xruns_{c.speaker.value.lower()}": c.stats.xruns for c in rt.captures}
            rt.store.finalize(
                {"stopped_at": datetime.now().astimezone().isoformat(),
                 "segments": len(rt.transcript) if rt.transcript else 0,
                 **stats}
            )
        except Exception:
            log.exception("Ошибка финального сохранения сессии")

    async def _write_report(self, rt: SessionRuntime) -> None:
        findings = rt.engine.findings()
        summary = None
        try:
            user = report.summary_user_prompt(rt.guide, rt.engine.state, findings)
            parsed, _ = await self.llm.chat_json(report.SUMMARY_SYSTEM, user, report.SUMMARY_SCHEMA)
            summary = (parsed.get("summary") or "").strip() or None
        except OllamaError as e:
            log.warning("Резюме отчёта не удалось: %s — отчёт будет без него", e)
        md = report.build_report_markdown(
            session_id=rt.store.session_id,
            guide=rt.guide,
            state=rt.engine.state,
            findings=findings,
            flags=rt.store.flags,
            meta={"started_at": rt.started_wall,
                  "segments": len(rt.transcript) if rt.transcript else 0},
            summary=summary,
        )
        rt.store.save_report(md)
        await self._status("Отчёт сохранён: report.md")

    # ------------------------------------------------------------- прочее API

    def analyze_now(self) -> None:
        if self.state != "running" or self.rt is None or self.rt.manual_event is None:
            raise ControllerError("Сессия не запущена")
        if self.rt.engine is not None and self.rt.engine.analyzing:
            raise ControllerError("Анализ уже выполняется")
        self.rt.manual_event.set()

    async def set_topic_status(self, topic_id: str, status: str | None) -> None:
        if self.state != "running" or self.rt is None or self.rt.engine is None:
            raise ControllerError("Сессия не запущена")
        try:
            self.rt.engine.set_manual_status(topic_id, status)
        except ValueError as e:
            raise ControllerError(str(e)) from e
        await self.hub.broadcast("coverage", self.rt.engine.coverage_payload())
        await self.hub.broadcast("recommendations", self.rt.engine.recommendations_payload())

    async def dismiss_recommendation(self, topic_id: str) -> None:
        if self.state != "running" or self.rt is None or self.rt.engine is None:
            raise ControllerError("Сессия не запущена")
        try:
            self.rt.engine.dismiss_recommendation(topic_id)
        except ValueError as e:
            raise ControllerError(str(e)) from e
        await self.hub.broadcast("recommendations", self.rt.engine.recommendations_payload())

    async def add_flag(self, note: str = "") -> dict:
        if self.state != "running" or self.rt is None or self.rt.store is None:
            raise ControllerError("Сессия не запущена")
        t = time.monotonic() - self.rt.t0_mono
        flag = self.rt.store.add_flag(t, note.strip())
        await self.hub.broadcast("flag", flag)
        return flag

    def snapshot(self) -> dict:
        rt = self.rt
        snap: dict = {
            "state": self.state,
            "app_version": __version__,
            "interval_s": self.cfg.analysis.interval_s,
            "llm_model": self.cfg.llm.model,
            "default_duration_min": self.cfg.analysis.default_duration_min,
            "session_id": None,
            "guide": None,
            "coverage": None,
            "recommendations": None,
            "segments": [],
            "flags": [],
            "next_analysis_at": None,
            "analyzing": False,
            "asr_status": None,
            "started_at": None,
            "duration_min": None,
            "channels": {},
        }
        if rt is not None:
            snap.update(
                session_id=rt.store.session_id if rt.store else None,
                guide=rt.guide.model_dump() if rt.guide else None,
                segments=[s.to_dict() for s in rt.transcript.tail(300)] if rt.transcript else [],
                flags=list(rt.store.flags) if rt.store else [],
                asr_status=rt.asr_status,
                started_at=rt.started_at_epoch,
                duration_min=rt.duration_min,
                channels=dict(rt.channel_alive),
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
