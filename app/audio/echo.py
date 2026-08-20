"""Noticing the respondent's voice arriving on both channels at once.

The respondent's channel is WASAPI loopback — literally what the output device
is playing. Through headphones that is the end of the story. Through speakers
the same voice also reaches the microphone, a fraction of a second later and
quieter, and then both channels recognise the same sentence: the transcript
carries it twice, and the coverage engine reads half of it as something the
interviewer said. A guide gets marked as covered by the interviewer repeating
the candidate.

We do not delete anything. Steno (stenolabs/stenoai) drops the quieter side of
each matched pair, which is right for a notepad rewriting its own file after
the fact; here a segment is already on screen and already in the analysis
window by the time the pair can be seen, and an automatic decision on ambiguous
evidence would cost more than a doubled line. So this only says it once, in the
UI, while there is still time to put headphones on.

The pairing test is Steno's cheap one: a token Jaccard between the two sides
inside a short window. What is deliberately not copied is their RMS comparison
as evidence — a microphone's gain and a loopback's digital level are not on the
same scale, so the levels go into the log for diagnosis and name the quieter
side in the warning, but the match itself is what decides. Two channels holding
the same sentence at the same moment is the anomaly; people do not chorus.
"""
from __future__ import annotations

import logging
import re
from collections import deque
from dataclasses import dataclass

from ..domain import Speaker

log = logging.getLogger("ilh.audio")

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)

# Below this a match means nothing: "ага", "да, понятно" legitimately land on
# both channels within a second of each other all the time.
MIN_CHARS = 16
MIN_TOKENS = 3


def tokens(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower()))


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


@dataclass
class _Entry:
    speaker: Speaker
    t0: float
    t1: float
    tokens: set[str]
    level: float
    text: str


class EchoDetector:
    """Fed every recognised segment; returns the channel picking up the echo
    exactly once, on the hit that reaches `min_hits`.

    Returning it once and only once is the point: the controller turns it into
    a warning, and a warning that repeats every other sentence is a warning
    nobody reads.
    """

    def __init__(
        self,
        window_s: float = 1.5,
        threshold: float = 0.6,
        min_hits: int = 2,
    ):
        self.window_s = window_s
        self.threshold = threshold
        self.min_hits = min_hits
        self.hits = 0
        self._recent: deque[_Entry] = deque()
        self._reported = False
        self._horizon = 0.0  # the latest moment any observed segment reached

    def observe(
        self, speaker: Speaker, t0: float, t1: float, text: str, level: float = 0.0
    ) -> Speaker | None:
        entry = _Entry(speaker, t0, t1, tokens(text), level, text)
        # Segments arrive in the order ASR finished them, which is only roughly
        # the order they were spoken in, so the window is measured against the
        # furthest moment seen so far. A segment that comes back badly out of
        # order can find its partner already forgotten — one missed hit, and
        # the next pair will do just as well.
        self._horizon = max(self._horizon, t1)
        self._forget_before(self._horizon - self.window_s)
        quiet = self._match(entry)
        if len(text) >= MIN_CHARS and len(entry.tokens) >= MIN_TOKENS:
            self._recent.append(entry)
        if quiet is None:
            return None
        self.hits += 1
        if self._reported or self.hits < self.min_hits:
            return None
        self._reported = True
        return quiet

    def _forget_before(self, cutoff: float) -> None:
        while self._recent and self._recent[0].t1 < cutoff:
            self._recent.popleft()

    def _match(self, entry: _Entry) -> Speaker | None:
        """The quieter speaker of the best cross-channel match, if there is one."""
        if len(entry.text) < MIN_CHARS or len(entry.tokens) < MIN_TOKENS:
            return None
        best: tuple[float, _Entry] | None = None
        for other in self._recent:
            if other.speaker is entry.speaker:
                continue
            if abs(other.t0 - entry.t0) > self.window_s:
                continue
            score = jaccard(entry.tokens, other.tokens)
            if score >= self.threshold and (best is None or score > best[0]):
                best = (score, other)
        if best is None:
            return None
        score, other = best
        quiet = entry.speaker if entry.level <= other.level else other.speaker
        log.info(
            "Echo between the channels (Jaccard %.2f): %s %.1f s %r (level %.4f) vs "
            "%s %.1f s %r (level %.4f)",
            score, entry.speaker.value, entry.t0, entry.text[:60], entry.level,
            other.speaker.value, other.t0, other.text[:60], other.level,
        )
        return quiet
