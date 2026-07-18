"""Нарезка непрерывного аудиопотока на чанки речи по паузам.

Чистая логика вынесена в ChunkAssembler (тестируется без потоков и железа),
ChunkerThread — тонкая обёртка: RingBuffer -> ChunkAssembler -> очередь ASR.
"""
from __future__ import annotations

import logging
import queue
import threading

import numpy as np

from ..config import AudioConfig
from ..domain import AudioChunk, Speaker
from .capture import RingBuffer
from .vad import SpeechDetector

log = logging.getLogger("ilh.chunker")


class ChunkAssembler:
    def __init__(self, detector: SpeechDetector, cfg: AudioConfig, speaker: Speaker):
        self.detector = detector
        self.cfg = cfg
        self.speaker = speaker
        self.sr = cfg.sample_rate
        self._buf = np.zeros(0, dtype=np.float32)
        self._offset = 0  # сэмплов потока уже отброшено слева от _buf

    def feed(self, new_audio: np.ndarray) -> list[AudioChunk]:
        if len(new_audio):
            self._buf = np.concatenate([self._buf, new_audio])
        out: list[AudioChunk] = []
        while True:
            chunk = self._try_cut()
            if chunk is None:
                break
            out.append(chunk)
        return out

    def flush(self) -> AudioChunk | None:
        """Финальный чанк при остановке сессии — забираем остаток речи."""
        if len(self._buf) / self.sr < self.cfg.min_speech_s:
            return None
        regions = self.detector.speech_regions(self._buf)
        chunk = self._emit(regions, regions[-1][1]) if regions else None
        self._buf = np.zeros(0, dtype=np.float32)
        return chunk

    def _try_cut(self) -> AudioChunk | None:
        dur = len(self._buf) / self.sr
        if dur < max(1.0, self.cfg.min_speech_s + self.cfg.min_pause_s):
            return None
        regions = self.detector.speech_regions(self._buf)
        if not regions:
            # Долгая тишина: держим только небольшой хвост, чтобы буфер не рос.
            if dur > 5.0:
                keep = self.sr  # 1 секунда
                consumed = len(self._buf) - keep
                self._offset += consumed
                self._buf = self._buf[consumed:]
            return None
        # Пауза внутри буфера: завершённая реплика отрезается сразу, не дожидаясь
        # тишины в хвосте (критично, когда аудио поступает пачкой и следующая
        # реплика уже началась).
        for (_, end_a), (start_b, _) in zip(regions, regions[1:]):
            if start_b - end_a >= self.cfg.min_pause_s:
                return self._emit([r for r in regions if r[1] <= end_a], end_a)
        trailing = dur - regions[-1][1]
        if trailing >= self.cfg.min_pause_s or dur >= self.cfg.max_chunk_s:
            return self._emit(regions, regions[-1][1])
        return None

    def _emit(self, regions: list[tuple[float, float]], end_s: float) -> AudioChunk | None:
        """Вырезает [начало первой реплики − pad; end_s + pad] и сдвигает буфер."""
        dur = len(self._buf) / self.sr
        start_s = max(regions[0][0] - self.cfg.pad_s, 0.0)
        end_pad = min(end_s + self.cfg.pad_s, dur)
        speech_total = sum(e - s for s, e in regions)

        a, b = int(start_s * self.sr), int(end_pad * self.sr)
        chunk_audio = self._buf[a:b].copy()
        t0 = (self._offset + a) / self.sr
        t1 = (self._offset + b) / self.sr

        self._offset += b
        self._buf = self._buf[b:]

        if speech_total < self.cfg.min_speech_s:
            return None
        return AudioChunk(speaker=self.speaker, audio=chunk_audio, t0=t0, t1=t1)


class ChunkerThread(threading.Thread):
    def __init__(
        self,
        speaker: Speaker,
        ring: RingBuffer,
        detector: SpeechDetector,
        out_queue: "queue.Queue[AudioChunk]",
        cfg: AudioConfig,
        stop_event: threading.Event,
    ):
        super().__init__(name=f"chunker-{speaker.value.lower()}", daemon=True)
        self.speaker = speaker
        self.ring = ring
        self.assembler = ChunkAssembler(detector, cfg, speaker)
        self.out_queue = out_queue
        self.cfg = cfg
        self.stop_event = stop_event
        self._last_dropped = 0

    def run(self) -> None:
        log.info("Чанкер %s запущен", self.speaker.value)
        try:
            while not self.stop_event.wait(self.cfg.poll_interval_s):
                self._pump()
            self._pump()
            final = self.assembler.flush()
            if final is not None:
                self.out_queue.put(final)
        except Exception:
            log.exception("Чанкер %s аварийно остановлен", self.speaker.value)
        log.info("Чанкер %s остановлен", self.speaker.value)

    MAX_BACKLOG = 60  # чанков; дальше ASR уже не догонит — не копим память впустую

    def _pump(self) -> None:
        if self.ring.dropped_samples > self._last_dropped:
            log.warning(
                "Канал %s: переполнение буфера, потеряно %d сэмплов",
                self.speaker.value, self.ring.dropped_samples - self._last_dropped,
            )
            self._last_dropped = self.ring.dropped_samples
        for chunk in self.assembler.feed(self.ring.pop_all()):
            if self.out_queue.qsize() >= self.MAX_BACKLOG:
                log.error(
                    "Очередь ASR переполнена (%d) — чанк %s %.1f–%.1f c отброшен. "
                    "ASR мёртв или безнадёжно отстаёт.",
                    self.out_queue.qsize(), self.speaker.value, chunk.t0, chunk.t1,
                )
                continue
            self.out_queue.put(chunk)
            log.debug(
                "Чанк %s: %.1f–%.1f c (%.1f c аудио)",
                self.speaker.value, chunk.t0, chunk.t1, chunk.t1 - chunk.t0,
            )
