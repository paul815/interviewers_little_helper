"""Выбор ASR-бэкенда.

`auto` предпочитает Parakeet (onnx-asr): он примерно на порядок быстрее Whisper
и не занимает VRAM. Если onnx-asr не установлен — прежнее поведение: mlx на
Apple Silicon, faster-whisper иначе.

Точка расширения: на Apple Silicon Parakeet заметно быстрее через parakeet-mlx
(Metal), чем через onnx-asr (CPU). Отдельный бэкенд-класс + ветка ниже — всё,
что для этого нужно; остальной код от движка ASR не зависит.
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
    if backend == "auto":
        if _onnx_asr_available():
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
    raise ValueError(f"Неизвестный ASR-бэкенд в конфиге: {cfg.backend}")


def _onnx_asr_available() -> bool:
    try:
        import onnx_asr  # noqa: F401

        return True
    except ImportError:
        log.warning("onnx-asr не установлен — Parakeet недоступен, откатываюсь на Whisper")
        return False


def _mlx_available() -> bool:
    try:
        import mlx_whisper  # noqa: F401

        return True
    except ImportError:
        log.warning("mlx-whisper не установлен — использую faster-whisper")
        return False
