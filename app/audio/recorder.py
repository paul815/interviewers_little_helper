"""Recording the session's audio track into a WAV next to the transcript.

Both tracks are written into a single stereo file, `audio.wav`: the left channel
is the interviewer (microphone), the right one the respondent (system audio). One
file rather than two: any player opens it, and the channels stay separated — in
an editor you can hear who was speaking, and `tools/bench_asr.py` can take just
the one it needs.

The format is PCM16 16 kHz: exactly what goes to ASR, with no second resampling.
An hour of interview is roughly 230 MB.

How the alignment works. A sample's position in the file equals its index in the
channel's stream, i.e. the same scale on which the chunker builds the segments'
`t0/t1`. The channels are physically different devices with independent clocks,
so the writer takes as many frames at a time as both have managed to hand over
(`min`). If a device dropped out (or has not warmed up yet), silence takes its
place while the other channel keeps being written: otherwise one microphone
falling out would shift the entire recording. The padded silence goes into
meta.json — it shows that the channel's track drifted from its transcript
timecodes by exactly that many seconds.

The WAV header is repaired on the fly and pushed to disk with fsync, so the
recording survives both a process crash and a power cut: no more than
HEADER_SYNC_S of tail is lost. Whatever remains beyond the last sync is written
by `storage/recovery.py` on the next launch.
"""
from __future__ import annotations

import logging
import os
import struct
import threading
import time
from collections import deque
from collections.abc import Callable
from pathlib import Path

import numpy as np

from ..domain import Speaker

log = logging.getLogger("ilh.recorder")

_BYTES_PER_SAMPLE = 2


class WavWriter:
    """A 16-bit PCM WAV whose header is repaired on the fly.

    The standard `wave` module fills the sizes in only on `close()`: after a
    process crash the file stays unplayable. Here the header is rewritten every
    few seconds, so an interrupted recording opens in a player — only the tail
    after the last sync is lost.
    """

    def __init__(self, path: Path, sample_rate: int, channels: int):
        self.path = Path(path)
        self.sample_rate = sample_rate
        self.channels = channels
        self.frames = 0
        self._synced = -1
        self._fsync_failed = False
        self._f = open(self.path, "wb")
        self._f.write(self._header(0))

    def _header(self, data_bytes: int) -> bytes:
        block_align = self.channels * _BYTES_PER_SAMPLE
        return struct.pack(
            "<4sI4s4sIHHIIHH4sI",
            b"RIFF", 36 + data_bytes, b"WAVE",
            b"fmt ", 16, 1, self.channels, self.sample_rate,
            self.sample_rate * block_align, block_align, 8 * _BYTES_PER_SAMPLE,
            b"data", data_bytes,
        )

    def write(self, pcm: bytes, frames: int) -> None:
        self._f.write(pcm)
        self.frames += frames

    def sync(self) -> None:
        """Write the sizes into the header and push the file all the way to disk.

        `flush()` alone is not enough: it hands the bytes to the OS cache, and on
        a power cut what is lost is not a couple of seconds of tail but everything
        the cache had not flushed (tens of seconds on an external drive). `fsync`
        makes the disk confirm the write: 128 KB every couple of seconds against
        a lost interview.
        """
        if self.frames == self._synced:
            return
        end = self._f.tell()
        self._f.seek(0)
        self._f.write(self._header(self.frames * self.channels * _BYTES_PER_SAMPLE))
        self._f.seek(end)
        self._f.flush()
        try:
            os.fsync(self._f.fileno())
        except OSError as e:  # network drive, a yanked USB stick
            if not self._fsync_failed:
                self._fsync_failed = True
                log.warning("fsync is unavailable for %s: %s — the recording rests on the OS cache",
                            self.path, e)
        self._synced = self.frames

    def close(self) -> None:
        try:
            self.sync()
        finally:
            self._f.close()


class SessionRecorder:
    """Stereo recording of the session: two independent channels -> one WAV."""

    CHANNELS = (Speaker.INTERVIEWER, Speaker.RESPONDENT)

    FLUSH_INTERVAL_S = 0.5   # how often the writer picks up what has accumulated
    HEADER_SYNC_S = 2.0      # how often the header is repaired and pushed to disk
    STALL_S = 3.0            # no blocks for longer — the channel stops holding the recording up
    MAX_PENDING_S = 60.0     # the disk cannot keep up: from here on we sacrifice audio, not memory
    WARN_INTERVAL_S = 30.0   # no more often than this do we complain to the UI about one problem

    def __init__(self, path: Path, sample_rate: int, stall_s: float = STALL_S):
        self.path = Path(path)
        self.sample_rate = sample_rate
        self._stall_s = stall_s
        # A recording problem must reach the screen, not only the log: learning
        # about lost audio an hour after the interview is the same as not learning.
        self.on_warning: Callable[[str], None] | None = None
        self._last_warn = 0.0
        self._max_pending = int(self.MAX_PENDING_S * sample_rate)
        self._lock = threading.Lock()        # the channel queues; taken from callbacks
        self._write_lock = threading.Lock()  # frame order within the file
        self._pending: dict[Speaker, deque[np.ndarray]] = {s: deque() for s in self.CHANNELS}
        self._available = {s: 0 for s in self.CHANNELS}
        self._last_block = {s: 0.0 for s in self.CHANNELS}
        self._started = {s: False for s in self.CHANNELS}
        self._padded = {s: 0 for s in self.CHANNELS}
        self._dropped = {s: 0 for s in self.CHANNELS}
        self._writer: WavWriter | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # ---------------------------------------------------------------- lifecycle

    def start(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = WavWriter(self.path, self.sample_rate, len(self.CHANNELS))
        self._thread = threading.Thread(target=self._run, name="recorder", daemon=True)
        self._thread.start()
        log.info("Recording the session audio: %s", self.path)

    def submit(self, speaker: Speaker, block: np.ndarray) -> None:
        """Accepting a block from the audio callback: a copy into the queue, nothing more."""
        if not len(block):
            return
        dropped = False
        with self._lock:
            q = self._pending[speaker]
            q.append(block)
            self._available[speaker] += len(block)
            self._last_block[speaker] = time.monotonic()
            self._started[speaker] = True
            while self._available[speaker] > self._max_pending and len(q) > 1:
                old = q.popleft()
                self._available[speaker] -= len(old)
                self._dropped[speaker] += len(old)
                dropped = True
        if dropped:  # notify outside the lock: a callback must not wait on the UI
            self._warn("The disk cannot keep up with the recording — some audio is being lost. "
                       "Free up space and close heavy programs")

    def _warn(self, message: str, throttle: bool = True) -> None:
        """Report a recording problem outwards. Called from the audio and writer threads."""
        now = time.monotonic()
        if throttle and now - self._last_warn < self.WARN_INTERVAL_S:
            return
        self._last_warn = now
        log.warning("Audio recording: %s", message)
        fn = self.on_warning
        if fn is None:
            return
        try:
            fn(message)
        except Exception:
            log.exception("Could not deliver the recording warning")

    def stop(self) -> dict:
        """Write the tail, close the file, return the summary for meta.json."""
        if self._writer is None:
            return {}
        self._stop.set()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(15.0)
        if thread is not None and thread.is_alive():
            log.warning("The audio writer did not finish in 15 s — "
                        "the tail of the recording is lost")
            self._warn("The recording did not close cleanly — the last few seconds "
                       "may be missing", throttle=False)
        else:
            try:
                self._drain(final=True)
            except Exception:
                log.exception("Could not write the audio tail")
        writer, self._writer = self._writer, None
        writer.close()
        return self._summary(writer.frames)

    def _summary(self, frames: int) -> dict:
        info = {"audio_file": self.path.name,
                "audio_seconds": round(frames / self.sample_rate, 1)}
        padded = {s.value.lower(): round(n / self.sample_rate, 1)
                  for s, n in self._padded.items() if n}
        if padded:
            info["audio_silence_padded_s"] = padded
            log.warning("Silence was padded into the recording in place of "
                        "silent channels: %s", padded)
        dropped = {s.value.lower(): round(n / self.sample_rate, 1)
                   for s, n in self._dropped.items() if n}
        if dropped:
            info["audio_dropped_s"] = dropped
            log.error("The recording could not keep up with capture, audio lost: %s", dropped)
        log.info("Session audio saved: %s (%.1f s)", self.path, frames / self.sample_rate)
        return info

    # ------------------------------------------------------------------- writer

    def _run(self) -> None:
        last_sync = time.monotonic()
        try:
            while not self._stop.wait(self.FLUSH_INTERVAL_S):
                self._drain(final=False)
                now = time.monotonic()
                if now - last_sync >= self.HEADER_SYNC_S:
                    self._writer.sync()
                    last_sync = now
        except Exception:
            log.exception("Audio recording interrupted — the file will hold up to the last sync")
            self._warn("The audio recording broke off — from here on only the transcript "
                       "is written. What was saved up to this second is still on disk",
                       throttle=False)

    def _drain(self, final: bool) -> None:
        # Taking the frames and writing them happen under one lock: otherwise two
        # drains in a row (the writer's and the final one from stop()) could swap
        # pieces around.
        with self._write_lock:
            now = time.monotonic()
            with self._lock:
                frames = self._target(now, final)
                if frames <= 0:
                    return
                columns = [self._take(s, frames) for s in self.CHANNELS]
            pcm = np.stack(columns, axis=1)  # (frames, channels) — the interleaving order
            np.clip(pcm, -1.0, 1.0, out=pcm)
            self._writer.write((pcm * 32767.0).astype("<i2").tobytes(), frames)

    def _target(self, now: float, final: bool) -> int:
        """How many frames can be written without the channels drifting apart."""
        if final:
            return max(self._available.values())
        live = [s for s in self.CHANNELS
                if self._started[s] and now - self._last_block[s] <= self._stall_s]
        if not live:
            # The channel has not handed over a single block yet, or the device
            # dropped out — it cannot be waited for, or the whole recording stalls.
            return max(self._available.values())
        return min(self._available[s] for s in live)

    def _take(self, speaker: Speaker, frames: int) -> np.ndarray:
        """Take `frames` samples off the channel, padding with silence if there are too few."""
        out = np.zeros(frames, dtype=np.float32)
        q = self._pending[speaker]
        filled = 0
        while filled < frames and q:
            block = q[0]
            take = min(len(block), frames - filled)
            out[filled:filled + take] = block[:take]
            if take == len(block):
                q.popleft()
            else:
                q[0] = block[take:]
            filled += take
            self._available[speaker] -= take
        if filled < frames:
            self._padded[speaker] += frames - filled
        return out
