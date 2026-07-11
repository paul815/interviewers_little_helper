"""Захват аудио: перечисление устройств и два независимых входных канала.

sounddevice/soxr импортируются лениво, чтобы остальная часть приложения
работала (и тестировалась) без установленного PortAudio.
"""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass

import numpy as np

from ..config import AudioConfig
from ..domain import Speaker

log = logging.getLogger("ilh.audio")


def list_input_devices() -> list[dict]:
    """Все устройства с входными каналами (микрофоны и виртуальные кабели)."""
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
                "hostapi": hostapis[dev["hostapi"]]["name"] if dev.get("hostapi") is not None else "",
                "max_input_channels": dev["max_input_channels"],
                "default_samplerate": dev.get("default_samplerate"),
                "is_default_input": idx == default_in,
            }
        )
    return devices


class RingBuffer:
    """Ограниченный FIFO аудио-блоков. При переполнении выбрасывает старейшие
    блоки (кольцевая семантика) и считает потерянные сэмплы."""

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
    """Один входной канал: устройство -> mono float32 16 kHz -> RingBuffer."""

    def __init__(self, device_index: int, speaker: Speaker, cfg: AudioConfig):
        self.device_index = device_index
        self.speaker = speaker
        self.cfg = cfg
        self.ring = RingBuffer(cfg.ring_seconds, cfg.sample_rate)
        self.stats = CaptureStats()
        self.device_name = ""
        self.level = 0.0  # затухающий пик для VU-метра, 0..1
        self.last_sample_time = 0.0  # time.monotonic() последнего callback'а
        self._stream = None
        self._resampler = None

    def start(self) -> None:
        import sounddevice as sd

        target = self.cfg.sample_rate
        info = sd.query_devices(self.device_index)
        self.device_name = info["name"]

        stream = None
        actual_rate = target
        # Сначала пробуем открыть сразу на 16 kHz, иначе — родная частота + ресемплинг.
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
                log.debug("Канал %s: не открылся на %d Гц: %s", self.speaker.value, rate, e)
                stream = None
        if stream is None:
            raise RuntimeError(
                f"Не удалось открыть аудиоустройство «{self.device_name}» (#{self.device_index})"
            )

        self.last_sample_time = time.monotonic()
        if actual_rate != target:
            import soxr

            self._resampler = soxr.ResampleStream(actual_rate, target, 1, dtype="float32")
            log.info(
                "Канал %s: устройство «%s» на %d Гц, ресемплинг в %d Гц",
                self.speaker.value, self.device_name, actual_rate, target,
            )
        else:
            log.info("Канал %s: устройство «%s» на %d Гц", self.speaker.value, self.device_name, target)
        self._stream = stream

    def _callback(self, indata, frames, time_info, status) -> None:
        # Внутри callback — только копирование в буфер, никакой тяжёлой работы.
        if status:
            self.stats.xruns += 1
        self.last_sample_time = time.monotonic()
        mono = indata[:, 0].copy() if indata.ndim > 1 else indata.copy()
        if len(mono):
            peak = float(np.max(np.abs(mono)))
            self.level = max(self.level * 0.82, min(peak, 1.0))
        if self._resampler is not None:
            mono = self._resampler.resample_chunk(mono)
        if len(mono):
            self.ring.append(mono.astype(np.float32, copy=False))
            self.stats.samples += len(mono)

    def stop(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception as e:
                log.warning("Канал %s: ошибка при остановке потока: %s", self.speaker.value, e)
            self._stream = None
