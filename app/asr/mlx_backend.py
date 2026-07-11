"""mlx-whisper: ускорение Metal на Apple Silicon."""
from __future__ import annotations

import logging

import numpy as np

from ..config import ASRConfig
from .base import ASRBackend, ASRResult

log = logging.getLogger("ilh.asr")


class MLXWhisperBackend(ASRBackend):
    def __init__(self, cfg: ASRConfig):
        self.cfg = cfg
        self.name = "mlx-whisper"
        self._mlx_whisper = None

    def load(self) -> None:
        import mlx_whisper

        self._mlx_whisper = mlx_whisper
        log.info("Загружаю mlx-whisper %s…", self.cfg.mlx_model)
        # Прогрев: форсирует скачивание и компиляцию модели.
        mlx_whisper.transcribe(
            np.zeros(1600, dtype=np.float32), path_or_hf_repo=self.cfg.mlx_model
        )
        log.info("mlx-whisper готов")

    def transcribe(self, audio: np.ndarray) -> ASRResult:
        result = self._mlx_whisper.transcribe(
            audio,
            path_or_hf_repo=self.cfg.mlx_model,
            language=self.cfg.language,
            condition_on_previous_text=False,
        )
        return ASRResult(
            text=(result.get("text") or "").strip(),
            language=result.get("language"),
            no_speech_prob=None,
        )

    def describe(self) -> str:
        return f"mlx-whisper {self.cfg.mlx_model}"
