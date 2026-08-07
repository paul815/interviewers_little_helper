"""Нарезка непрерывного аудиопотока на чанки речи.

Две реализации с одним контрактом `feed()` / `flush()`:

- `StreamingChunkAssembler` (основной) — режет по событиям потокового VAD:
  реплика уходит в ASR через `vad_redemption_s` после того, как человек
  замолчал. Каждый кадр аудио обрабатывается ровно один раз.
- `ChunkAssembler` (запасной) — исходная нарезка по паузам: копит буфер и
  ищет в нём паузу батчевым VAD'ом. Задержка привязана к `max_chunk_s`,
  а разметка одного и того же аудио пересчитывается на каждом опросе.

Обе тестируются без потоков и железа; ChunkerThread — тонкая обёртка:
RingBuffer -> ассемблер -> очередь ASR.
"""
from __future__ import annotations

import logging
import queue
import threading
from typing import Protocol

import numpy as np

from ..config import AudioConfig
from ..domain import AudioChunk, Speaker
from .capture import RingBuffer
from .speech_events import SpeechEnd, SpeechEventSource, SpeechStart, create_stream_processor
from .vad import SpeechDetector

log = logging.getLogger("ilh.chunker")


class ChunkAssemblerLike(Protocol):
    def feed(self, new_audio: np.ndarray) -> list[AudioChunk]:
        ...

    def flush(self) -> AudioChunk | None:
        ...


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


class StreamingChunkAssembler:
    """Нарезка по событиям потокового VAD.

    Держит скользящее окно недавнего аудио (`_buf`, начинающийся на абсолютной
    позиции `_buf_start`) и вырезает из него реплику, как только пришёл
    `SpeechEnd`. Ждать заполнения буфера не нужно, поэтому задержка сегмента
    определяется только `vad_redemption_s` и скоростью ASR.

    Инварианты:

    - `t0/t1` — абсолютные секунды от старта захвата (на этом держится
      восстановление порядка реплик двух каналов сортировкой по `t0`);
    - соседние чанки не перекрываются и не оставляют дыр: `_emitted_through`
      помнит, до какого сэмпла аудио уже отдано, и pad добавляется только на
      настоящих границах речи, а не на принудительном разрезе монолога.
    """

    # Короче этого чанк не имеет смысла отдавать в ASR (модели выдают мусор).
    MIN_EMIT_S = 0.1

    def __init__(self, source: SpeechEventSource, cfg: AudioConfig, speaker: Speaker):
        self.source = source
        self.cfg = cfg
        self.speaker = speaker
        self.sr = cfg.sample_rate
        self._buf = np.zeros(0, dtype=np.float32)
        self._buf_start = 0        # абсолютная позиция сэмпла _buf[0]
        self._stream_end = 0       # сколько сэмплов всего прошло через feed()
        self._emitted_through = 0  # до какого сэмпла аудио уже отдано в ASR
        self._speech_start: int | None = None
        self._pre_pad = int(cfg.pre_pad_s * self.sr)
        self._post_pad = int(cfg.pad_s * self.sr)
        self._max_samples = int(cfg.max_chunk_s * self.sr)
        # Между репликами гарантированно есть vad_redemption_s тишины — именно
        # столько ждёт VAD, прежде чем закрыть реплику. Если pad'ы в сумме
        # перекрывают этот зазор, конец одного чанка залезет на начало
        # следующего и слова задвоятся в транскрипте.
        if cfg.pre_pad_s + cfg.pad_s >= cfg.vad_redemption_s:
            log.warning(
                "pre_pad_s + pad_s (%.2f) >= vad_redemption_s (%.2f): соседние чанки "
                "будут перекрываться, реплики задвоятся. Уменьшите pad'ы или увеличьте "
                "vad_redemption_s.",
                cfg.pre_pad_s + cfg.pad_s, cfg.vad_redemption_s,
            )

    def feed(self, new_audio: np.ndarray) -> list[AudioChunk]:
        if len(new_audio):
            self._buf = np.concatenate([self._buf, new_audio])
            self._stream_end += len(new_audio)

        out: list[AudioChunk] = []
        for ev in self.source.process(new_audio):
            if isinstance(ev, SpeechStart):
                self._speech_start = ev.timestamp_samples
            elif isinstance(ev, SpeechEnd):
                self._append(out, self._emit(ev.start_timestamp_samples,
                                             ev.end_timestamp_samples, pad_end=True))
                self._speech_start = None
        if not self.source.in_speech:
            # Реплику отсеял фильтр минимальной длительности: SpeechStart был,
            # SpeechEnd не будет.
            self._speech_start = None

        # Монолог длиннее max_chunk_s: режем принудительно, чтобы транскрипт
        # не молчал до конца ответа.
        if (
            self._speech_start is not None
            and self._stream_end - self._speech_start >= self._max_samples
        ):
            self._append(out, self._emit(self._speech_start, self._stream_end, pad_end=False))
            self._speech_start = self._stream_end

        self._trim()
        return out

    def flush(self) -> AudioChunk | None:
        """Хвост при остановке сессии: закрываем незавершённую реплику."""
        chunk = None
        for ev in self.source.flush():
            if isinstance(ev, SpeechEnd):
                chunk = self._emit(ev.start_timestamp_samples,
                                   ev.end_timestamp_samples, pad_end=True) or chunk
        self._buf = np.zeros(0, dtype=np.float32)
        self._buf_start = self._stream_end
        self._speech_start = None
        return chunk

    @staticmethod
    def _append(out: list[AudioChunk], chunk: AudioChunk | None) -> None:
        if chunk is not None:
            out.append(chunk)

    def _emit(self, start: int, end: int, pad_end: bool) -> AudioChunk | None:
        # Всё, что уже отдано, не отдаём повторно: после принудительного разреза
        # SpeechEnd приносит исходное начало реплики, которое давно позади.
        start = max(start, self._emitted_through)
        if end - start < self.MIN_EMIT_S * self.sr:
            return None

        # Pre-pad только на настоящем начале реплики: Silero срабатывает чуть
        # позже первого слога. На стыке после принудительного разреза padding
        # продублировал бы уже отданное аудио.
        a = start - self._pre_pad if start > self._emitted_through else start
        b = end + self._post_pad if pad_end else end
        a = max(a, self._buf_start)
        b = min(b, self._stream_end)
        if b <= a:
            return None

        audio = self._buf[a - self._buf_start : b - self._buf_start].copy()
        self._emitted_through = end
        return AudioChunk(speaker=self.speaker, audio=audio, t0=a / self.sr, t1=b / self.sr)

    def _trim(self) -> None:
        """Отбросить аудио, которое уже никому не понадобится."""
        anchor = self._speech_start if self._speech_start is not None else self._stream_end
        keep_from = max(0, anchor - self._pre_pad)
        if keep_from > self._buf_start:
            self._buf = self._buf[keep_from - self._buf_start :]
            self._buf_start = keep_from


def create_assembler(cfg: AudioConfig, speaker: Speaker) -> ChunkAssemblerLike:
    """Ассемблер по `audio.vad`: потоковый по умолчанию, батчевый — по запросу."""
    if cfg.vad in ("silero", "energy"):
        from .vad import create_detector

        log.info("Нарезка %s: батчевая по паузам (vad=%s)", speaker.value, cfg.vad)
        return ChunkAssembler(create_detector(cfg.vad), cfg, speaker)
    return StreamingChunkAssembler(create_stream_processor(cfg), cfg, speaker)


class ChunkerThread(threading.Thread):
    def __init__(
        self,
        speaker: Speaker,
        ring: RingBuffer,
        assembler: ChunkAssemblerLike,
        out_queue: "queue.Queue[AudioChunk]",
        cfg: AudioConfig,
        stop_event: threading.Event,
    ):
        super().__init__(name=f"chunker-{speaker.value.lower()}", daemon=True)
        self.speaker = speaker
        self.ring = ring
        self.assembler = assembler
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
