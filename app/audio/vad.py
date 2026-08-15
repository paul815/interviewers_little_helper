"""Speech detectors. The main one is Silero VAD (the ONNX model from the
faster-whisper package, no torch); the fallback is energy-based, for when Silero
is unavailable."""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import numpy as np

log = logging.getLogger("ilh.vad")

SAMPLE_RATE = 16000


class SpeechDetector(ABC):
    @abstractmethod
    def speech_regions(self, audio: np.ndarray) -> list[tuple[float, float]]:
        """Speech intervals (start_s, end_s) within a 16 kHz float32 buffer."""


class SileroDetector(SpeechDetector):
    def __init__(self, threshold: float = 0.4):
        from faster_whisper import vad as fw_vad  # check availability at construction time

        self._fw_vad = fw_vad
        self._options = self._build_options(
            threshold=threshold,
            min_speech_duration_ms=150,
            min_silence_duration_ms=250,
            speech_pad_ms=0,
        )

    def _build_options(self, **desired):
        """VadOptions differs between faster-whisper versions (dataclass /
        NamedTuple, different field sets) — pass only the fields that exist."""
        import dataclasses

        opt_cls = self._fw_vad.VadOptions
        if dataclasses.is_dataclass(opt_cls):
            fields = {f.name for f in dataclasses.fields(opt_cls)}
        elif hasattr(opt_cls, "_fields"):
            fields = set(opt_cls._fields)
        else:
            fields = set(desired)
        return opt_cls(**{k: v for k, v in desired.items() if k in fields})

    def speech_regions(self, audio: np.ndarray) -> list[tuple[float, float]]:
        if len(audio) < 512:
            return []
        timestamps = self._fw_vad.get_speech_timestamps(audio, self._options)
        return [(t["start"] / SAMPLE_RATE, t["end"] / SAMPLE_RATE) for t in timestamps]


class EnergyDetector(SpeechDetector):
    """A simple RMS detector: the fallback, and a tool for tests."""

    def __init__(self, frame_ms: int = 30, abs_floor: float = 0.006, merge_gap_s: float = 0.3):
        self.frame = int(SAMPLE_RATE * frame_ms / 1000)
        self.abs_floor = abs_floor
        self.merge_gap_s = merge_gap_s

    def speech_regions(self, audio: np.ndarray) -> list[tuple[float, float]]:
        n_frames = len(audio) // self.frame
        if n_frames == 0:
            return []
        frames = audio[: n_frames * self.frame].reshape(n_frames, self.frame)
        rms = np.sqrt((frames.astype(np.float64) ** 2).mean(axis=1))
        noise = np.percentile(rms, 20)
        thr = max(self.abs_floor, noise * 3.5)
        active = rms > thr
        if not active.any() and noise > self.abs_floor:
            # The whole buffer is loud (continuous speech): the adaptive
            # threshold built from the "noise floor" ended up above the signal.
            active = rms > self.abs_floor

        regions: list[tuple[float, float]] = []
        frame_s = self.frame / SAMPLE_RATE
        start = None
        for i, a in enumerate(active):
            if a and start is None:
                start = i * frame_s
            elif not a and start is not None:
                regions.append((start, i * frame_s))
                start = None
        if start is not None:
            regions.append((start, n_frames * frame_s))

        merged: list[tuple[float, float]] = []
        for s, e in regions:
            if merged and s - merged[-1][1] <= self.merge_gap_s:
                merged[-1] = (merged[-1][0], e)
            else:
                merged.append((s, e))
        return merged


def create_detector(kind: str = "auto") -> SpeechDetector:
    if kind in ("auto", "silero"):
        try:
            det = SileroDetector()
            log.info("VAD: Silero (onnx, faster-whisper)")
            return det
        except Exception as e:
            if kind == "silero":
                raise
            log.warning("Silero VAD is unavailable (%s) — switching to the energy one", e)
    log.info("VAD: energy (fallback)")
    return EnergyDetector()
