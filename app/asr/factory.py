"""Выбор ASR-бэкенда: mlx на Apple Silicon, faster-whisper (CUDA/CPU) иначе."""
from __future__ import annotations

import logging
import platform
import sys

from ..config import ASRConfig
from .base import ASRBackend

log = logging.getLogger("ilh.asr")


def create_asr_backend(cfg: ASRConfig) -> ASRBackend:
    backend = cfg.backend
    if backend == "auto":
        if sys.platform == "darwin" and platform.machine() == "arm64" and _mlx_available():
            backend = "mlx"
        else:
            backend = "faster"

    if backend == "mlx":
        from .mlx_backend import MLXWhisperBackend

        return MLXWhisperBackend(cfg)
    if backend == "faster":
        from .faster_backend import FasterWhisperBackend

        return FasterWhisperBackend(cfg, device="auto")
    if backend == "faster-cpu":
        from .faster_backend import FasterWhisperBackend

        return FasterWhisperBackend(cfg, device="cpu")
    raise ValueError(f"Неизвестный ASR-бэкенд в конфиге: {cfg.backend}")


def _mlx_available() -> bool:
    try:
        import mlx_whisper  # noqa: F401

        return True
    except ImportError:
        log.warning("mlx-whisper не установлен — использую faster-whisper")
        return False
