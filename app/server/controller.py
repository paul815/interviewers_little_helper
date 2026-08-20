"""Session orchestrator: wires together audio capture, the chunkers, the ASR
worker, the transcript, the coverage engine and persistence."""
from __future__ import annotations

import asyncio
import dataclasses
import logging
import queue
import threading
import time
from datetime import datetime
from functools import partial

from .. import __version__
from ..asr import router  # a plain table, no onnx-asr behind it
from ..audio.echo import EchoDetector
from ..config import AppConfig
from ..coverage.engine import CoverageEngine
from ..domain import SPEAKER_FULL, AudioChunk, Speaker
from ..guide.library import GuideLibrary
from ..guide.parser import parse_guide_text
from ..guide.schemas import Guide
from ..llm.ollama_client import OllamaClient, OllamaError
from ..storage import report
from ..storage.project_store import ProjectError, ProjectStore
from ..storage.prompt_store import PromptStore
from ..storage.session_store import SessionStore
from ..transcript.store import TranscriptStore
from .hub import WsHub

log = logging.getLogger("ilh.controller")


class ControllerError(Exception):
    """An API-level error with a human-readable message."""


class SessionRuntime:
    """All the live parts of one session."""

    def __init__(self):
        self.store: SessionStore | None = None
        self.transcript: TranscriptStore | None = None
        self.captures: list = []
        self.chunkers: list = []
        self.recorder = None
        self.audio_meta: dict = {}  # recording summary, ends up in meta.json
        self.audio_warnings: list[str] = []  # recording problems: on screen and in meta
        self.asr_worker = None
        self.asr_queue: queue.Queue[AudioChunk] = queue.Queue()
        self.engine: CoverageEngine | None = None
        self.scheduler_task: asyncio.Task | None = None
        self.watchdog_task: asyncio.Task | None = None
        self.thread_stop = threading.Event()
        self.engine_stop: asyncio.Event | None = None
        self.manual_event: asyncio.Event | None = None
        self.guide: Guide | None = None
        self.guide_text: str = ""  # the original, as the researcher pasted it
        self.project_id: str | None = None  # None — an interview outside a project
        self.project_title: str = ""
        self.started_wall: str = ""
        self.started_at_epoch: float = 0.0
        self.t0_mono: float = 0.0
        self.duration_min: int = 60
        self.asr_vocabulary: str = ""
        self.asr_language: str = ""  # "" — whatever config.json says, no routing
        self.llm_instructions: str = ""  # the project's extra analysis instructions
        self.asr_status: str = "loading"
        self.channel_alive: dict[str, bool] = {}
        self.echo: EchoDetector | None = None  # None — switched off in the config


class AppController:
    def __init__(self, cfg: AppConfig, hub: WsHub):
        self.cfg = cfg
        self.hub = hub
        self.llm = OllamaClient(cfg.llm)
        self.library = GuideLibrary(cfg.guides_path)
        self.projects = ProjectStore(cfg.projects_path)
        self.prompts = PromptStore(cfg.prompts_path)
        self.state = "idle"  # idle | starting | running | stopping
        self.rt: SessionRuntime | None = None
        self.recovered: list[dict] = []  # sessions repaired after a crash
        self._transition_lock = asyncio.Lock()
        self._monitor_caps: list = []
        self._monitor_task: asyncio.Task | None = None

    # ----------------------------------------------------------------- guide

    async def parse_guide(self, text: str) -> Guide:
        check = await self.llm.check()
        if not check["ok"]:
            raise ControllerError(check["error"] or "The LLM is unavailable")
        try:
            return await parse_guide_text(
                self.llm, text, self.prompts.templates()["guide_parse"]
            )
        except OllamaError as e:
            raise ControllerError(str(e)) from e
        except ValueError as e:
            raise ControllerError(str(e)) from e

    # ------------------------------------------------- level monitor (idle)

    async def start_monitor(self, mic_index: int | None, system_index: int | None) -> dict:
        """Opens the selected devices before the session starts, so the UI can
        show levels and the user can confirm the audio routing."""
        if self.state != "idle":
            raise ControllerError("A session is already running — levels come from it")
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
                log.exception("Error stopping the monitor")

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

    # --------------------------------------------------------------- session

    async def start_session(
        self,
        mic_index: int,
        system_index: int,
        guide_data: dict,
        duration_min: int | None = None,
        asr_vocabulary: str | None = None,
        asr_language: str | None = None,
        project_id: str | None = None,
        guide_text: str = "",
    ) -> dict:
        async with self._transition_lock:
            if self.state != "idle":
                raise ControllerError("A session is already running")
            if mic_index == system_index:
                raise ControllerError("The microphone and the system audio are the same device")
            try:
                guide = Guide.model_validate(guide_data)
            except Exception as e:
                raise ControllerError(f"Invalid guide structure: {e}") from e
            if not guide.topic_ids():
                raise ControllerError("The guide has no topics at all")

            # Check the project before starting the hardware: learning about a
            # typo in the id only after the devices are captured means losing the
            # first utterances.
            project = None
            if project_id:
                try:
                    project = self.projects.load(project_id)
                except ProjectError as e:
                    raise ControllerError(str(e)) from e

            check = await self.llm.check()
            if not check["ok"]:
                raise ControllerError(check["error"] or "The LLM is unavailable")

            await self.stop_monitor()  # release the devices before the session
            self.state = "starting"
            await self._status("Starting the session…")
            rt = SessionRuntime()
            rt.duration_min = duration_min or self.cfg.analysis.default_duration_min
            rt.asr_vocabulary = (asr_vocabulary or self.cfg.asr.vocabulary).strip()
            rt.asr_language = router.normalise(asr_language or self.cfg.asr.language) or ""
            rt.guide_text = guide_text or ""
            if project is not None:
                rt.project_id = project["project_id"]
                rt.project_title = project.get("title", "")
                # The guide may have been pasted last time: the interview screen
                # must not end up empty just because the window was reopened.
                if not rt.guide_text.strip():
                    rt.guide_text = project.get("guide_text") or ""
                rt.llm_instructions = (project.get("llm_instructions") or "").strip()
            try:
                await self._build_runtime(rt, mic_index, system_index, guide)
            except Exception as e:
                log.exception("Could not start the session")
                await self._stop_audio_asr(rt)
                await self._final_persist(rt)
                self.state = "idle"
                await self._status("Startup failed")
                raise ControllerError(str(e)) from e

            self.rt = rt
            self.state = "running"
            await self._status("Session running. ASR is still loading…"
                               if rt.asr_status == "loading" else "Session running")
            await self.hub.broadcast(
                "session",
                {"state": "running", "session_id": rt.store.session_id,
                 "started_at": rt.started_at_epoch, "duration_min": rt.duration_min,
                 "project_id": rt.project_id, "project_title": rt.project_title},
            )
            return {"session_id": rt.store.session_id}

    async def _build_runtime(
        self, rt: SessionRuntime, mic_index: int, system_index: int, guide: Guide
    ) -> None:
        from ..asr.factory import create_asr_backend
        from ..asr.worker import ASRWorker
        from ..audio.capture import ChannelCapture
        from ..audio.chunker import ChunkerThread, create_assembler
        from ..audio.recorder import SessionRecorder

        rt.guide = guide
        rt.started_wall = datetime.now().astimezone().isoformat()
        rt.started_at_epoch = time.time()

        rt.store = SessionStore.create(
            self.cfg,
            {"guide_title": guide.title, "duration_min": rt.duration_min,
             "asr_vocabulary": rt.asr_vocabulary, "asr_language": rt.asr_language,
             "project_id": rt.project_id, "project_title": rt.project_title},
            root=self.projects.sessions_root(rt.project_id) if rt.project_id else None,
        )
        rt.store.save_guide(guide)
        rt.store.save_guide_text(rt.guide_text)
        if rt.project_id:
            # The project preset catches up with what actually went into the
            # interview: an edit to the guide or a new term in the vocabulary
            # needs no separate save button — the next interview in the series
            # picks them up on its own.
            try:
                self.projects.update(
                    rt.project_id,
                    guide=guide.model_dump(),
                    guide_text=rt.guide_text or None,
                    asr_vocabulary=rt.asr_vocabulary,
                    asr_language=rt.asr_language,
                    duration_min=rt.duration_min,
                )
            except ProjectError as e:  # recording the session matters more than the preset
                log.warning("Could not update the preset of project %s: %s", rt.project_id, e)

        rt.transcript = TranscriptStore()
        rt.transcript.add_listener(rt.store.append_segment)
        rt.transcript.add_listener(
            lambda seg: self.hub.broadcast_threadsafe("segment", seg.to_dict())
        )

        def asr_status(phase: str, message: str) -> None:
            rt.asr_status = {"asr_loading": "loading", "asr_ready": "ready"}.get(phase, "error")
            self.hub.broadcast_threadsafe("status", {"message": message, "asr": rt.asr_status})

        if self.cfg.audio.echo_detect:
            rt.echo = EchoDetector(
                window_s=self.cfg.audio.echo_window_s,
                threshold=self.cfg.audio.echo_jaccard,
                min_hits=self.cfg.audio.echo_min_hits,
            )

        def on_segment(chunk: AudioChunk, text: str, language: str | None) -> None:
            rt.transcript.add(chunk.speaker, chunk.t0, chunk.t1, text, language)
            if rt.echo is None:
                return
            quiet = rt.echo.observe(chunk.speaker, chunk.t0, chunk.t1, text, chunk.level)
            if quiet is not None:
                self._audio_warning(
                    rt,
                    f"The «{SPEAKER_FULL[quiet]}» channel is also picking up the other "
                    f"side of the call — the same phrases are landing in the transcript "
                    f"twice, and the analysis reads them as said by both. Headphones "
                    f"fix it; nothing is being deleted automatically.",
                )

        asr_cfg = router.route(
            dataclasses.replace(self.cfg.asr, vocabulary=rt.asr_vocabulary), rt.asr_language
        )
        backend = create_asr_backend(asr_cfg)
        rt.asr_worker = ASRWorker(
            backend, rt.asr_queue, on_segment, asr_status, asr_cfg, rt.thread_stop
        )
        rt.asr_worker.start()  # the model loads in the background; audio piles up meanwhile

        if self.cfg.storage.save_audio:
            recorder = SessionRecorder(rt.store.dir / "audio.wav", self.cfg.audio.sample_rate)
            recorder.on_warning = partial(self._audio_warning, rt)
            try:
                recorder.start()
                rt.recorder = recorder
            except Exception as e:  # no space, no permissions — the interview beats the recording
                log.warning("Audio recording did not start: %s", e)
                await self.hub.broadcast("error", {
                    "message": f"Audio is not being recorded ({e}) — "
                               f"the transcript continues as usual",
                })

        rt.t0_mono = time.monotonic()
        for device_index, speaker in ((mic_index, Speaker.INTERVIEWER),
                                      (system_index, Speaker.RESPONDENT)):
            capture = ChannelCapture(device_index, speaker, self.cfg.audio)
            if rt.recorder is not None:
                capture.on_audio = partial(rt.recorder.submit, speaker)
                capture.on_audio_failed = lambda err, sp=speaker: self._audio_warning(
                    rt, f"Recording of the «{SPEAKER_FULL[sp]}» channel stopped ({err}) — "
                        f"the transcript continues as usual"
                )
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
            # Prompts are frozen at start: an edit in "Settings" mid-interview
            # must not change the rules of the game between two analysis cycles.
            templates=self.prompts.templates(),
            instructions=rt.llm_instructions,
        )
        rt.scheduler_task = asyncio.create_task(rt.engine.run(rt.engine_stop, rt.manual_event))
        rt.watchdog_task = asyncio.create_task(self._watchdog_loop(rt))

    def _audio_warning(self, rt: SessionRuntime, message: str) -> None:
        """A recording problem from the audio thread: on screen immediately, in
        meta.json at the end.

        Called from outside the event loop, so the broadcast uses the
        threadsafe variant.
        """
        if message not in rt.audio_warnings:
            rt.audio_warnings.append(message)
        self.hub.broadcast_threadsafe("error", {"message": message})

    # ------------------------------------------------------ channel watchdog

    async def _watchdog_loop(self, rt: SessionRuntime) -> None:
        """VU levels in the UI plus a check that the devices are alive; dead
        ones are reopened."""
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
                if tick % 25 != 0:  # liveness check roughly every 3 s
                    continue
                now = time.monotonic()
                for cap in rt.captures:
                    key = cap.speaker.value.lower()
                    alive = (now - cap.last_sample_time) <= silence_s
                    if alive != rt.channel_alive.get(key, True):
                        rt.channel_alive[key] = alive
                        name = SPEAKER_FULL[cap.speaker]
                        await self.hub.broadcast(
                            "channel", {"speaker": key, "alive": alive, "device": cap.device_name}
                        )
                        if alive:
                            await self.hub.broadcast(
                                "status", {"message": f"The «{name}» channel is back"}
                            )
                    if not alive and now - last_reopen.get(key, 0.0) > 10.0:
                        last_reopen[key] = now
                        await self.hub.broadcast("error", {
                            "message": f"The «{SPEAKER_FULL[cap.speaker]}» channel is getting no "
                                       f"audio (device «{cap.device_name}») — reopening it",
                        })
                        try:
                            await asyncio.to_thread(cap.stop)
                            await asyncio.to_thread(cap.start)
                        except Exception as e:
                            log.warning("Could not reopen device %s: %s",
                                        cap.device_name, e)
        except asyncio.CancelledError:
            raise

    # ------------------------------------------------------------- shutdown

    async def stop_session(self) -> dict:
        async with self._transition_lock:
            if self.state not in ("running", "starting"):
                raise ControllerError("No session is running")
            self.state = "stopping"
            rt = self.rt
            await self._status("Stopping: transcribing the tail of the audio…")
            await self._stop_audio_asr(rt)

            has_transcript = rt.transcript is not None and len(rt.transcript) > 0
            if self.cfg.analysis.final_sweep and rt.engine is not None and has_transcript:
                try:
                    await rt.engine.final_pass()
                except Exception:
                    log.exception("The final reconciliation failed")
                    await self.hub.broadcast("error", {
                        "message": "The final reconciliation failed — statuses saved as they were"
                    })
            if self.cfg.analysis.report and rt.engine is not None and has_transcript:
                await self._status("Preparing the session report…")
                try:
                    await self._write_report(rt)
                except Exception:
                    log.exception("Could not build the report")
                    await self.hub.broadcast("error", {
                        "message": "The report was not built — details in logs/app.log"
                    })

            await self._final_persist(rt)
            session_id = rt.store.session_id if rt.store else None
            self.rt = None
            self.state = "idle"
            await self._status(f"Session saved: sessions/{session_id}"
                               if session_id else "Session stopped")
            await self.hub.broadcast("session", {"state": "idle", "session_id": session_id})
            await self.hub.broadcast("timer", {"next_analysis_at": None,
                                               "interval_s": self.cfg.analysis.interval_s})
            return {"session_id": session_id}

    async def _stop_audio_asr(self, rt: SessionRuntime) -> None:
        # 1. Stop the analysis scheduler and the watchdog.
        if rt.engine_stop is not None:
            rt.engine_stop.set()
        for task in (rt.scheduler_task, rt.watchdog_task):
            if task is not None:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
        rt.scheduler_task = rt.watchdog_task = None
        # 2. Capture: no new samples arrive from here on.
        for capture in rt.captures:
            await asyncio.to_thread(capture.stop)
        # 2a. The recorder writes the tail and closes the WAV (after the devices
        #     stop, otherwise not everything makes it into the file).
        if rt.recorder is not None:
            recorder, rt.recorder = rt.recorder, None
            try:
                rt.audio_meta = await asyncio.to_thread(recorder.stop)
            except Exception:
                log.exception("Could not close the audio recording cleanly")
        # 3. The chunkers flush their buffers and put the final chunks on the queue.
        rt.thread_stop.set()
        for chunker in rt.chunkers:
            await asyncio.to_thread(chunker.join, 10.0)
        # 4. The ASR worker drains the queue (see the exit condition in run()).
        if rt.asr_worker is not None:
            await asyncio.to_thread(rt.asr_worker.join, 60.0)
            if rt.asr_worker.is_alive():
                log.warning("The ASR worker did not finish in 60 s — the tail may be lost")

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
            extra = {"stopped_at": datetime.now().astimezone().isoformat(),
                     "segments": len(rt.transcript) if rt.transcript else 0,
                     **stats, **rt.audio_meta}
            if rt.audio_warnings:
                extra["audio_warnings"] = list(rt.audio_warnings)
            rt.store.finalize(extra)
        except Exception:
            log.exception("Error during the final save of the session")

    async def _write_report(self, rt: SessionRuntime) -> None:
        findings = rt.engine.findings()
        summary = None
        try:
            system = report.summary_system_prompt(rt.guide, self.cfg.analysis.output_language)
            user = report.summary_user_prompt(rt.guide, rt.engine.state, findings)
            parsed, _ = await self.llm.chat_json(system, user, report.SUMMARY_SCHEMA)
            summary = (parsed.get("summary") or "").strip() or None
        except OllamaError as e:
            log.warning("The report summary failed: %s — the report will go without it", e)
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
        await self._status("Report saved: report.md")

    # ------------------------------------------------------------- other API

    def recover_crashed_sessions(self) -> list[dict]:
        """Repair session folders cut short by a crash or a power cut.

        A blocking disk walk — called from to_thread at server start, and the
        result reaches the UI in the very first state snapshot.
        """
        from ..storage.recovery import recover_sessions

        self.recovered = recover_sessions(self.cfg)
        return self.recovered

    def analyze_now(self) -> None:
        if self.state != "running" or self.rt is None or self.rt.manual_event is None:
            raise ControllerError("No session is running")
        if self.rt.engine is not None and self.rt.engine.analyzing:
            raise ControllerError("Analysis is already running")
        self.rt.manual_event.set()

    async def set_topic_status(self, topic_id: str, status: str | None) -> None:
        if self.state != "running" or self.rt is None or self.rt.engine is None:
            raise ControllerError("No session is running")
        try:
            self.rt.engine.set_manual_status(topic_id, status)
        except ValueError as e:
            raise ControllerError(str(e)) from e
        await self.hub.broadcast("coverage", self.rt.engine.coverage_payload())
        await self.hub.broadcast("recommendations", self.rt.engine.recommendations_payload())

    async def dismiss_recommendation(self, topic_id: str) -> None:
        if self.state != "running" or self.rt is None or self.rt.engine is None:
            raise ControllerError("No session is running")
        try:
            self.rt.engine.dismiss_recommendation(topic_id)
        except ValueError as e:
            raise ControllerError(str(e)) from e
        await self.hub.broadcast("recommendations", self.rt.engine.recommendations_payload())

    async def add_flag(
        self,
        note: str = "",
        anchor: str = "",
        anchor_section: str = "",
        topic_id: str | None = None,
    ) -> dict:
        if self.state != "running" or self.rt is None or self.rt.store is None:
            raise ControllerError("No session is running")
        t = time.monotonic() - self.rt.t0_mono
        flag = self.rt.store.add_flag(
            t, note.strip(), anchor.strip(), anchor_section.strip(), topic_id
        )
        await self.hub.broadcast("flag", flag)
        return flag

    async def add_question(self, text: str) -> dict:
        if self.state != "running" or self.rt is None or self.rt.store is None:
            raise ControllerError("No session is running")
        text = text.strip()
        if not text:
            raise ControllerError("The question is empty")
        t = time.monotonic() - self.rt.t0_mono
        q = self.rt.store.add_question(t, text)
        await self.hub.broadcast("question", q)
        return q

    async def set_question_done(self, question_id: int, done: bool) -> dict:
        if self.state != "running" or self.rt is None or self.rt.store is None:
            raise ControllerError("No session is running")
        q = self.rt.store.set_question_done(question_id, done)
        if q is None:
            raise ControllerError("Question not found")
        await self.hub.broadcast("question", q)
        return q

    def snapshot(self) -> dict:
        rt = self.rt
        snap: dict = {
            "state": self.state,
            "app_version": __version__,
            "interval_s": self.cfg.analysis.interval_s,
            "llm_model": self.cfg.llm.model,
            "default_duration_min": self.cfg.analysis.default_duration_min,
            "session_id": None,
            "project_id": None,
            "project_title": "",
            "guide": None,
            "guide_text": "",
            "coverage": None,
            "recommendations": None,
            "segments": [],
            "flags": [],
            "questions": [],
            "next_analysis_at": None,
            "analyzing": False,
            "asr_status": None,
            "started_at": None,
            "duration_min": None,
            "channels": {},
            "recovered": self.recovered,
        }
        if rt is not None:
            snap.update(
                session_id=rt.store.session_id if rt.store else None,
                project_id=rt.project_id,
                project_title=rt.project_title,
                guide=rt.guide.model_dump() if rt.guide else None,
                guide_text=rt.guide_text,
                segments=[s.to_dict() for s in rt.transcript.tail(300)] if rt.transcript else [],
                flags=list(rt.store.flags) if rt.store else [],
                questions=list(rt.store.questions) if rt.store else [],
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
        log.info("Status: %s", message)
        await self.hub.broadcast("status", {"state": self.state, "message": message})
