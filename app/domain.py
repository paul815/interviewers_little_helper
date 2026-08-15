"""Shared domain types: speakers, audio chunks, transcript segments."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

import numpy as np


class Speaker(str, Enum):  # noqa: UP042 — StrEnum would change the result of str()
    INTERVIEWER = "INTERVIEWER"
    RESPONDENT = "RESPONDENT"


SPEAKER_SHORT = {Speaker.INTERVIEWER: "I", Speaker.RESPONDENT: "R"}
SPEAKER_FULL = {Speaker.INTERVIEWER: "Interviewer", Speaker.RESPONDENT: "Respondent"}


@dataclass
class AudioChunk:
    """A piece of speech from one speaker; t0/t1 are seconds from session start."""

    speaker: Speaker
    audio: np.ndarray  # float32 mono, 16 kHz
    t0: float
    t1: float


@dataclass
class Segment:
    """A recognised transcript segment."""

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
