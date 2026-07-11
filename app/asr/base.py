"""Интерфейс ASR-бэкенда: чанк 16 kHz float32 -> текст."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class ASRResult:
    text: str
    language: str | None = None
    no_speech_prob: float | None = None


class ASRBackend(ABC):
    name: str = "base"

    @abstractmethod
    def load(self) -> None:
        """Загрузка модели (медленно; вызывается в рабочем потоке)."""

    @abstractmethod
    def transcribe(self, audio: np.ndarray) -> ASRResult:
        ...

    def describe(self) -> str:
        return self.name
