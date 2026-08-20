"""Cutting a continuous audio stream into speech chunks.

Two implementations sharing one `feed()` / `flush()` contract:

- `StreamingChunkAssembler` (the main one) cuts on streaming-VAD events: an
  utterance goes to ASR `vad_redemption_s` after the person fell silent. Every
  frame of audio is processed exactly once.
- `ChunkAssembler` (the fallback) is the original pause-based cutting: it
  accumulates a buffer and looks for a pause in it with a batch VAD. The latency
  is tied to `max_chunk_s`, and the same audio is re-annotated on every poll.

Both are testable without threads or hardware; ChunkerThread is a thin wrapper:
RingBuffer -> assembler -> the ASR queue.
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
from .preprocess import create_highpass
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
        self._offset = 0  # stream samples already discarded to the left of _buf

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
        """The final chunk when the session stops — take the remaining speech."""
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
            # Long silence: keep only a short tail so the buffer does not grow.
            if dur > 5.0:
                keep = self.sr  # 1 second
                consumed = len(self._buf) - keep
                self._offset += consumed
                self._buf = self._buf[consumed:]
            return None
        # A pause inside the buffer: a finished utterance is cut off at once,
        # without waiting for silence in the tail (critical when audio arrives in
        # a batch and the next utterance has already started).
        for (_, end_a), (start_b, _) in zip(regions, regions[1:], strict=False):
            if start_b - end_a >= self.cfg.min_pause_s:
                return self._emit([r for r in regions if r[1] <= end_a], end_a)
        trailing = dur - regions[-1][1]
        if trailing >= self.cfg.min_pause_s or dur >= self.cfg.max_chunk_s:
            return self._emit(regions, regions[-1][1])
        return None

    def _emit(self, regions: list[tuple[float, float]], end_s: float) -> AudioChunk | None:
        """Cuts out [start of the first utterance − pad; end_s + pad] and shifts the buffer."""
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
    """Cutting driven by streaming-VAD events.

    Keeps a sliding window of recent audio (`_buf`, starting at absolute position
    `_buf_start`) and cuts an utterance out of it as soon as a `SpeechEnd`
    arrives. There is no need to wait for a buffer to fill, so a segment's latency
    is set only by `vad_redemption_s` and the speed of ASR.

    Invariants:

    - `t0/t1` are absolute seconds from the start of capture (recovering the
      order of the two channels' utterances by sorting on `t0` rests on this);
    - neighbouring chunks neither overlap nor leave gaps: `_emitted_through`
      remembers how far the audio has already been handed out, and padding is
      added only at real speech boundaries, not at a forced cut in a monologue.
    """

    # Shorter than this, a chunk is not worth sending to ASR (the models produce garbage).
    MIN_EMIT_S = 0.1

    def __init__(self, source: SpeechEventSource, cfg: AudioConfig, speaker: Speaker):
        self.source = source
        self.cfg = cfg
        self.speaker = speaker
        self.sr = cfg.sample_rate
        self._buf = np.zeros(0, dtype=np.float32)
        self._buf_start = 0        # absolute position of sample _buf[0]
        self._stream_end = 0       # how many samples have passed through feed() in total
        self._emitted_through = 0  # how far the audio has already gone to ASR
        self._speech_start: int | None = None
        self._pre_pad = int(cfg.pre_pad_s * self.sr)
        self._post_pad = int(cfg.pad_s * self.sr)
        self._max_samples = int(cfg.max_chunk_s * self.sr)
        # Between utterances there are guaranteed to be vad_redemption_s of
        # silence — that is exactly how long the VAD waits before closing an
        # utterance. If the pads together exceed that gap, the end of one chunk
        # runs into the start of the next and words get doubled in the transcript.
        if cfg.pre_pad_s + cfg.pad_s >= cfg.vad_redemption_s:
            log.warning(
                "pre_pad_s + pad_s (%.2f) >= vad_redemption_s (%.2f): neighbouring chunks "
                "will overlap and utterances will be doubled. Reduce the pads or raise "
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
            # The minimum-duration filter dropped the utterance: there was a
            # SpeechStart, there will be no SpeechEnd.
            self._speech_start = None

        # A monologue longer than max_chunk_s: cut it forcibly so the transcript
        # does not stay silent until the answer ends.
        if (
            self._speech_start is not None
            and self._stream_end - self._speech_start >= self._max_samples
        ):
            self._append(out, self._emit(self._speech_start, self._stream_end, pad_end=False))
            self._speech_start = self._stream_end

        self._trim()
        return out

    def flush(self) -> AudioChunk | None:
        """The tail when the session stops: close the unfinished utterance."""
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
        # Nothing already handed out is handed out twice: after a forced cut,
        # SpeechEnd brings the utterance's original start, long since passed.
        start = max(start, self._emitted_through)
        if end - start < self.MIN_EMIT_S * self.sr:
            return None

        # Pre-pad only at a real start of an utterance: Silero fires slightly
        # after the first syllable. At the seam following a forced cut, padding
        # would duplicate audio that has already gone out.
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
        """Discard audio nobody will need any more."""
        anchor = self._speech_start if self._speech_start is not None else self._stream_end
        keep_from = max(0, anchor - self._pre_pad)
        if keep_from > self._buf_start:
            self._buf = self._buf[keep_from - self._buf_start :]
            self._buf_start = keep_from


def create_assembler(cfg: AudioConfig, speaker: Speaker) -> ChunkAssemblerLike:
    """The assembler chosen by `audio.vad`: streaming by default, batch on request."""
    if cfg.vad in ("silero", "energy"):
        from .vad import create_detector

        log.info("Chunking %s: batch, by pauses (vad=%s)", speaker.value, cfg.vad)
        return ChunkAssembler(create_detector(cfg.vad), cfg, speaker)
    return StreamingChunkAssembler(create_stream_processor(cfg), cfg, speaker)


class ChunkerThread(threading.Thread):
    def __init__(
        self,
        speaker: Speaker,
        ring: RingBuffer,
        assembler: ChunkAssemblerLike,
        out_queue: queue.Queue[AudioChunk],
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
        # The filter sits here rather than in ChannelCapture on purpose: the
        # recording tap hangs off the capture callback, and the audio.wav next
        # to the transcript has to stay what the room actually sounded like.
        self._highpass = create_highpass(cfg.highpass_hz, cfg.sample_rate)
        if self._highpass is not None:
            log.info(
                "Channel %s: high-pass at %.0f Hz (%d taps)",
                speaker.value, cfg.highpass_hz, self._highpass.taps,
            )

    def run(self) -> None:
        log.info("Chunker %s started", self.speaker.value)
        try:
            while not self.stop_event.wait(self.cfg.poll_interval_s):
                self._pump()
            self._pump()
            final = self.assembler.flush()
            if final is not None:
                self.out_queue.put(final)
        except Exception:
            log.exception("Chunker %s stopped abnormally", self.speaker.value)
        log.info("Chunker %s stopped", self.speaker.value)

    MAX_BACKLOG = 60  # chunks; beyond this ASR will never catch up — do not waste memory

    def _pump(self) -> None:
        if self.ring.dropped_samples > self._last_dropped:
            log.warning(
                "Channel %s: buffer overflow, %d samples lost",
                self.speaker.value, self.ring.dropped_samples - self._last_dropped,
            )
            self._last_dropped = self.ring.dropped_samples
        audio = self.ring.pop_all()
        if self._highpass is not None:
            audio = self._highpass.process(audio)
        for chunk in self.assembler.feed(audio):
            if self.out_queue.qsize() >= self.MAX_BACKLOG:
                log.error(
                    "The ASR queue is full (%d) — chunk %s %.1f–%.1f s was dropped. "
                    "ASR is dead or hopelessly behind.",
                    self.out_queue.qsize(), self.speaker.value, chunk.t0, chunk.t1,
                )
                continue
            self.out_queue.put(chunk)
            log.debug(
                "Chunk %s: %.1f–%.1f s (%.1f s of audio)",
                self.speaker.value, chunk.t0, chunk.t1, chunk.t1 - chunk.t0,
            )
