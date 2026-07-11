"""Общие доменные типы: говорящие, аудио-чанки, сегменты транскрипта."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

import numpy as np


class Speaker(str, Enum):
    INTERVIEWER = "INTERVIEWER"
    RESPONDENT = "RESPONDENT"


SPEAKER_SHORT_RU = {Speaker.INTERVIEWER: "И", Speaker.RESPONDENT: "Р"}
SPEAKER_FULL_RU = {Speaker.INTERVIEWER: "Интервьюер", Speaker.RESPONDENT: "Респондент"}


@dataclass
class AudioChunk:
    """Кусок речи одного говорящего; t0/t1 — секунды от старта сессии."""

    speaker: Speaker
    audio: np.ndarray  # float32 mono, 16 kHz
    t0: float
    t1: float


@dataclass
class Segment:
    """Распознанный сегмент транскрипта."""

    id: int
    speaker: Speaker
    t0: float
    t1: float
    text: str
    language: str | None = None
    created_at: str = field(default_factory=lambda: datetime.now().astimezone().isoformat())

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "speaker": self.speaker.value,
            "t0": round(self.t0, 2),
            "t1": round(self.t1, 2),
            "text": self.text,
            "language": self.language,
            "created_at": self.created_at,
        }


def fmt_ts(seconds: float) -> str:
    s = max(0, int(seconds))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h:02d}:{m:02d}:{sec:02d}"
    return f"{m:02d}:{sec:02d}"
