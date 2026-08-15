"""Audio capture: device enumeration and two independent input channels.

sounddevice/soxr are imported lazily so that the rest of the application runs
(and is tested) without PortAudio installed.
"""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from ..config import AudioConfig
from ..domain import Speaker
from . import loopback

log = logging.getLogger("ilh.audio")


def list_input_devices() -> list[dict]:
    """Every recording source: microphones, virtual cables and — on Windows —
    output devices via WASAPI loopback (system audio without a cable)."""
    import sounddevice as sd

    devices = []
    try:
        default_in = sd.default.device[0]
    except Exception:
        default_in = -1
    hostapis = sd.query_hostapis()
    for idx, dev in enumerate(sd.query_devices()):
        if dev.get("max_input_channels", 0) <= 0:
            continue
        devices.append(
            {
                "index": idx,
                "name": dev["name"],
                "hostapi": (
                    hostapis[dev["hostapi"]]["name"] if dev.get("hostapi") is not None else ""
                ),
                "max_input_channels": dev["max_input_channels"],
                "default_samplerate": dev.get("default_samplerate"),
                "is_default_input": idx == default_in,
                "is_loopback": False,
                "is_default_loopback": False,
            }
        )
    return devices + loopback.list_loopback_devices()


class RingBuffer:
    """A bounded FIFO of audio blocks. On overflow it drops the oldest blocks
    (ring semantics) and counts the lost samples."""

    def __init__(self, capacity_seconds: float, sample_rate: int):
        self._blocks: deque[np.ndarray] = deque()
        self._lock = threading.Lock()
        self._capacity = int(capacity_seconds * sample_rate)
        self._size = 0
        self.dropped_samples = 0

    def append(self, block: np.ndarray) -> None:
        with self._lock:
            self._blocks.append(block)
            self._size += len(block)
            while self._size > self._capacity and self._blocks:
                old = self._blocks.popleft()
                self._size -= len(old)
                self.dropped_samples += len(old)

    def pop_all(self) -> np.ndarray:
        with self._lock:
            if not self._blocks:
                return np.zeros(0, dtype=np.float32)
            blocks = list(self._blocks)
            self._blocks.clear()
            self._size = 0
        return np.concatenate(blocks)


@dataclass
class CaptureStats:
    xruns: int = 0
    samples: int = 0


class ChannelCapture:
    """One input channel: device -> mono float32 16 kHz -> RingBuffer."""

    def __init__(self, device_index: int, speaker: Speaker, cfg: AudioConfig):
        self.device_index = device_index
        self.speaker = speaker
        self.cfg = cfg
        self.ring = RingBuffer(cfg.ring_seconds, cfg.sample_rate)
        self.stats = CaptureStats()
        self.device_name = ""
        self.level = 0.0  # decaying peak for the VU meter, 0..1
        self.last_sample_time = 0.0  # time.monotonic() of the last callback
        # A tap on the same audio that goes into the ring: the session recording
        # on disk (see audio/recorder.py). None — the channel is not recorded.
        self.on_audio: Callable[[np.ndarray], None] | None = None
        # Called once if the tap failed and recording of the channel was disabled.
        self.on_audio_failed: Callable[[str], None] | None = None
        self._stream = None
        self._resampler = None

    def start(self) -> None:
        if loopback.is_loopback_index(self.device_index):
            self._start_loopback()
        else:
            self._start_input_device()

    def _start_input_device(self) -> None:
        import sounddevice as sd

        target = self.cfg.sample_rate
        info = sd.query_devices(self.device_index)
        self.device_name = info["name"]

        stream = None
        actual_rate = target
        # First try opening at 16 kHz directly, otherwise the native rate plus resampling.
        for rate in (target, int(info.get("default_samplerate") or 48000)):
            try:
                stream = sd.InputStream(
                    device=self.device_index,
                    channels=1,
                    samplerate=rate,
                    dtype="float32",
                    callback=self._callback,
                )
                stream.start()
                actual_rate = rate
                break
            except Exception as e:
                log.debug("Channel %s: did not open at %d Hz: %s", self.speaker.value, rate, e)
                stream = None
        if stream is None:
            raise RuntimeError(
                f"Could not open the audio device «{self.device_name}» (#{self.device_index})"
            )

        self._finish_start(stream, actual_rate)

    def _start_loopback(self) -> None:
        """System audio from an output device. In shared mode the client does not
        pick the WASAPI rate, so we open at the native one and always resample."""
        info = loopback.device_info(self.device_index)
        self.device_name = info["name"]
        stream = loopback.LoopbackStream(info, self._ingest)
        try:
            stream.start()
        except Exception as e:
            raise RuntimeError(
                f"Could not open the system audio «{self.device_name}»: {e}"
            ) from e
        self._finish_start(stream, stream.samplerate)

    def _finish_start(self, stream, actual_rate: int) -> None:
        target = self.cfg.sample_rate
        self.last_sample_time = time.monotonic()
        if actual_rate != target:
            import soxr

            self._resampler = soxr.ResampleStream(actual_rate, target, 1, dtype="float32")
            log.info(
                "Channel %s: device «%s» at %d Hz, resampling to %d Hz",
                self.speaker.value, self.device_name, actual_rate, target,
            )
        else:
            log.info("Channel %s: device «%s» at %d Hz",
                     self.speaker.value, self.device_name, target)
        self._stream = stream

    def _callback(self, indata, frames, time_info, status) -> None:
        mono = indata[:, 0] if indata.ndim > 1 else indata
        self._ingest(mono.copy(), bool(status))

    def _ingest(self, mono: np.ndarray, xrun: bool) -> None:
        # Inside the callback: only a copy into the buffer, no heavy work.
        if xrun:
            self.stats.xruns += 1
        self.last_sample_time = time.monotonic()
        if len(mono):
            peak = float(np.max(np.abs(mono)))
            self.level = max(self.level * 0.82, min(peak, 1.0))
        if self._resampler is not None:
            mono = self._resampler.resample_chunk(mono)
        if len(mono):
            block = mono.astype(np.float32, copy=False)
            self.ring.append(block)
            self.stats.samples += len(mono)
            if self.on_audio is not None:
                try:
                    self.on_audio(block)
                except Exception as e:
                    # Writing to disk has no right to take capture down: disable
                    # the tap and complain exactly once.
                    self.on_audio = None
                    log.exception("Channel %s: audio recording disabled", self.speaker.value)
                    fn, self.on_audio_failed = self.on_audio_failed, None
                    if fn is not None:
                        try:
                            fn(str(e))
                        except Exception:
                            log.exception("Could not report the recording failure")

    def stop(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception as e:
                log.warning("Channel %s: error while stopping the stream: %s",
                            self.speaker.value, e)
            self._stream = None
