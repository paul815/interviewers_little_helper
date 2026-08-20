"""Choosing the ASR backend.

`auto` prefers Parakeet (onnx-asr): it is roughly an order of magnitude faster
than Whisper and takes no VRAM. If onnx-asr is not installed, the old behaviour
applies: mlx on Apple Silicon, faster-whisper otherwise.

`whisper` is `auto` with the Parakeet branch removed — the fastest Whisper this
machine can run. The router picks it for interviews in a language no onnx-asr
model of ours was trained on (app/asr/router.py); naming a concrete backend
there would cost Apple Silicon its Metal acceleration.

An extension point: on Apple Silicon, Parakeet is noticeably faster through
parakeet-mlx (Metal) than through onnx-asr (CPU). A separate backend class plus
a branch below is all that would take; the rest of the code does not depend on
the ASR engine.
"""
from __future__ import annotations

import logging
import platform
import sys

from ..config import ASRConfig
from .base import ASRBackend

log = logging.getLogger("ilh.asr")


def create_asr_backend(cfg: ASRConfig) -> ASRBackend:
    backend = cfg.backend
    if backend in ("auto", "whisper"):
        if backend == "auto" and _onnx_asr_available():
            backend = "parakeet"
        elif sys.platform == "darwin" and platform.machine() == "arm64" and _mlx_available():
            backend = "mlx"
        else:
            backend = "faster"

    if backend == "parakeet":
        from .parakeet_backend import ParakeetOnnxBackend

        return ParakeetOnnxBackend(cfg)
    if backend == "mlx":
        from .mlx_backend import MLXWhisperBackend

        return MLXWhisperBackend(cfg)
    if backend == "faster":
        from .faster_backend import FasterWhisperBackend

        return FasterWhisperBackend(cfg, device="auto")
    if backend == "faster-cpu":
        from .faster_backend import FasterWhisperBackend

        return FasterWhisperBackend(cfg, device="cpu")
    raise ValueError(f"Unknown ASR backend in the config: {cfg.backend}")


def _onnx_asr_available() -> bool:
    try:
        import onnx_asr  # noqa: F401

        return True
    except ImportError:
        log.warning("onnx-asr is not installed — Parakeet is unavailable, falling back to Whisper")
        return False


def _mlx_available() -> bool:
    try:
        import mlx_whisper  # noqa: F401

        return True
    except ImportError:
        log.warning("mlx-whisper is not installed — using faster-whisper")
        return False
