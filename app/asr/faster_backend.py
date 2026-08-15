"""faster-whisper (CTranslate2): CUDA on Windows/NVIDIA, int8 on CPU — the fallback everywhere."""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import numpy as np

from ..config import ASRConfig
from .base import ASRBackend, ASRResult

log = logging.getLogger("ilh.asr")


def _add_nvidia_dll_dirs() -> None:
    """On Windows, cuBLAS/cuDNN come from the nvidia-* pip packages; their DLL
    directories must be added to the search path explicitly before CTranslate2 loads."""
    if sys.platform != "win32":
        return
    try:
        import nvidia  # a namespace package

        for base in nvidia.__path__:
            for sub in Path(base).iterdir():
                for dll_dir in (sub / "bin", sub / "lib"):
                    if dll_dir.is_dir():
                        os.add_dll_directory(str(dll_dir))
    except Exception as e:
        log.debug("The nvidia pip packages were not found: %s", e)


class FasterWhisperBackend(ASRBackend):
    def __init__(self, cfg: ASRConfig, device: str = "auto"):
        self.cfg = cfg
        self.device_pref = device  # auto | cuda | cpu
        self.device = None
        self.compute_type = None
        self.model = None
        self.name = "faster-whisper"

    def load(self) -> None:
        from faster_whisper import WhisperModel

        _add_nvidia_dll_dirs()
        attempts = []
        if self.device_pref in ("auto", "cuda"):
            attempts.append(
                ("cuda", self.cfg.compute_type if self.cfg.compute_type != "auto" else "float16")
            )
        if self.device_pref in ("auto", "cpu"):
            attempts.append(
                ("cpu", self.cfg.compute_type if self.cfg.compute_type != "auto" else "int8")
            )

        last_err: Exception | None = None
        for device, compute in attempts:
            try:
                log.info("Loading faster-whisper %s (%s, %s)…", self.cfg.model, device, compute)
                self.model = WhisperModel(self.cfg.model, device=device, compute_type=compute)
                # Warm-up, and a check that the device actually works.
                list(self.model.transcribe(np.zeros(1600, dtype=np.float32), beam_size=1)[0])
                self.device, self.compute_type = device, compute
                log.info("faster-whisper is ready: %s / %s", device, compute)
                return
            except Exception as e:
                last_err = e
                log.warning("faster-whisper did not come up on %s: %s", device, e)
                self.model = None
        raise RuntimeError(f"Could not load faster-whisper: {last_err}")

    def transcribe(self, audio: np.ndarray) -> ASRResult:
        segments, info = self.model.transcribe(
            audio,
            language=self.cfg.language,
            beam_size=self.cfg.beam_size,
            temperature=0.0,
            condition_on_previous_text=False,
            initial_prompt=self.cfg.vocabulary or None,
            vad_filter=False,  # VAD already did its work in the chunker
            without_timestamps=True,
        )
        parts, weights, probs = [], [], []
        for seg in segments:
            txt = seg.text.strip()
            if txt:
                parts.append(txt)
            w = max(seg.end - seg.start, 0.1)
            weights.append(w)
            probs.append(getattr(seg, "no_speech_prob", 0.0) or 0.0)
        no_speech = float(np.average(probs, weights=weights)) if probs else 1.0
        return ASRResult(
            text=" ".join(parts),
            language=getattr(info, "language", None),
            no_speech_prob=no_speech,
        )

    def describe(self) -> str:
        return f"faster-whisper {self.cfg.model} ({self.device}, {self.compute_type})"
