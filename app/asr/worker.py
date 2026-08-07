"""Единственный ASR-воркер: сериализует доступ обоих каналов к одной модели."""
from __future__ import annotations

import logging
import queue
import re
import threading
import time
from typing import Callable

from ..config import ASRConfig
from ..domain import AudioChunk
from .base import ASRBackend

log = logging.getLogger("ilh.asr")

_WS_RE = re.compile(r"\s+")

# on_segment(chunk, text, language) -> None
SegmentCallback = Callable[[AudioChunk, str, str | None], None]
StatusCallback = Callable[[str, str], None]  # (phase, message)


class ASRWorker(threading.Thread):
    def __init__(
        self,
        backend: ASRBackend,
        in_queue: "queue.Queue[AudioChunk]",
        on_segment: SegmentCallback,
        on_status: StatusCallback,
        cfg: ASRConfig,
        stop_event: threading.Event,
    ):
        super().__init__(name="asr-worker", daemon=True)
        self.backend = backend
        self.in_queue = in_queue
        self.on_segment = on_segment
        self.on_status = on_status
        self.cfg = cfg
        self.stop_event = stop_event
        self.ready = False
        self.load_error: str | None = None

    def run(self) -> None:
        try:
            self.on_status("asr_loading", f"Загрузка модели распознавания ({self.backend.name})…")
            t = time.monotonic()
            self.backend.load()
            self.ready = True
            message = f"ASR готов: {self.backend.describe()} ({time.monotonic() - t:.0f} c)"
            warnings = self.backend.warnings()
            if warnings:
                message += " — внимание: " + "; ".join(warnings)
            self.on_status("asr_ready", message)
        except Exception as e:
            self.load_error = str(e)
            log.exception("Не удалось загрузить ASR-модель")
            self.on_status("asr_error", f"Ошибка загрузки ASR: {e}")
            return

        # После сигнала стоп дорабатываем очередь до конца (финальные чанки).
        while not (self.stop_event.is_set() and self.in_queue.empty()):
            try:
                chunk = self.in_queue.get(timeout=0.25)
            except queue.Empty:
                continue
            self._process(chunk)

    def _process(self, chunk: AudioChunk) -> None:
        backlog = self.in_queue.qsize()
        if backlog > 20:
            log.warning("Очередь ASR растёт: %d чанков в ожидании", backlog)
        try:
            t = time.monotonic()
            result = self.backend.transcribe(chunk.audio)
            elapsed = time.monotonic() - t
            text = _WS_RE.sub(" ", result.text).strip()
            if not text:
                return
            if (
                result.no_speech_prob is not None
                and result.no_speech_prob > self.cfg.drop_no_speech_prob
            ):
                log.debug("Отброшен вероятный не-речевой сегмент (%.2f): %r",
                          result.no_speech_prob, text[:60])
                return
            log.debug(
                "%s %.1f–%.1f c распознан за %.1f c: %r",
                chunk.speaker.value, chunk.t0, chunk.t1, elapsed, text[:80],
            )
            self.on_segment(chunk, text, result.language)
        except Exception:
            log.exception("Ошибка транскрипции чанка %s %.1f–%.1f c — пропускаю",
                          chunk.speaker.value, chunk.t0, chunk.t1)
