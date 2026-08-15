"""Transcript storage: a thread-safe list of segments plus a delta cursor for
the coverage engine plus subscribers (disk, WebSocket)."""
from __future__ import annotations

import logging
import threading
from collections.abc import Callable

from ..domain import Segment, Speaker

log = logging.getLogger("ilh.transcript")


class TranscriptStore:
    def __init__(self):
        self._segments: list[Segment] = []
        self._lock = threading.Lock()
        self._listeners: list[Callable[[Segment], None]] = []

    def add_listener(self, fn: Callable[[Segment], None]) -> None:
        self._listeners.append(fn)

    def add(
        self, speaker: Speaker, t0: float, t1: float, text: str, language: str | None
    ) -> Segment:
        with self._lock:
            seg = Segment(
                id=len(self._segments) + 1,
                speaker=speaker, t0=t0, t1=t1, text=text, language=language,
            )
            self._segments.append(seg)
        for fn in self._listeners:
            try:
                fn(seg)
            except Exception:
                log.exception("A transcript subscriber failed on segment %d", seg.id)
        return seg

    def all_segments(self) -> list[Segment]:
        with self._lock:
            return list(self._segments)

    def tail(self, n: int) -> list[Segment]:
        with self._lock:
            return sorted(self._segments[-n:], key=lambda s: s.t0)

    def delta_since(self, cursor: int) -> tuple[list[Segment], int]:
        """Segments added after cursor (in insertion order), sorted
        chronologically; returns the new cursor."""
        with self._lock:
            delta = self._segments[cursor:]
            new_cursor = len(self._segments)
        return sorted(delta, key=lambda s: s.t0), new_cursor

    def __len__(self) -> int:
        with self._lock:
            return len(self._segments)
