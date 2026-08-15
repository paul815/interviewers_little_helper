"""The single ASR worker: serialises both channels' access to one model."""
from __future__ import annotations

import logging
import queue
import re
import threading
import time
from collections.abc import Callable

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
        in_queue: queue.Queue[AudioChunk],
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
            self.on_status("asr_loading", f"Loading the recognition model ({self.backend.name})…")
            t = time.monotonic()
            self.backend.load()
            self.ready = True
            message = f"ASR ready: {self.backend.describe()} ({time.monotonic() - t:.0f} s)"
            warnings = self.backend.warnings()
            if warnings:
                message += " — note: " + "; ".join(warnings)
            self.on_status("asr_ready", message)
        except Exception as e:
            self.load_error = str(e)
            log.exception("Could not load the ASR model")
            self.on_status("asr_error", f"ASR failed to load: {e}")
            return

        # After the stop signal we drain the queue to the end (the final chunks).
        while not (self.stop_event.is_set() and self.in_queue.empty()):
            try:
                chunk = self.in_queue.get(timeout=0.25)
            except queue.Empty:
                continue
            self._process(chunk)

    def _process(self, chunk: AudioChunk) -> None:
        backlog = self.in_queue.qsize()
        if backlog > 20:
            log.warning("The ASR queue is growing: %d chunks waiting", backlog)
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
                log.debug("Dropped a likely non-speech segment (%.2f): %r",
                          result.no_speech_prob, text[:60])
                return
            log.debug(
                "%s %.1f–%.1f s recognised in %.1f s: %r",
                chunk.speaker.value, chunk.t0, chunk.t1, elapsed, text[:80],
            )
            self.on_segment(chunk, text, result.language)
        except Exception:
            log.exception("Error transcribing chunk %s %.1f–%.1f s — skipping it",
                          chunk.speaker.value, chunk.t0, chunk.t1)
