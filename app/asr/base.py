"""The ASR backend interface: a 16 kHz float32 chunk -> text."""
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
        """Load the model (slow; called on a worker thread)."""

    @abstractmethod
    def transcribe(self, audio: np.ndarray) -> ASRResult:
        ...

    def describe(self) -> str:
        return self.name

    def warnings(self) -> list[str]:
        """What the backend could not honour from the settings — shown to the user."""
        return []
