"""Streaming detection of utterance boundaries: an audio stream -> speech events.

The difference from `vad.py`: that one re-annotates speech across the whole
buffer from scratch on every poll (the batch `speech_regions`), whereas here each
frame is processed exactly once and the end of an utterance becomes known
`vad_redemption_s` after the person fell silent — without waiting for the buffer
to grow to `max_chunk_s`. This is exactly what keeps the transcript latency low.

The module deliberately depends on neither onnxruntime nor audio hardware: the
state machine and the energy event source are testable in isolation. The Silero
implementation lives in `silero_stream.py` and is imported lazily.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

log = logging.getLogger("ilh.vad")

SAMPLE_RATE = 16000
# The native frame size of Silero VAD at 16 kHz (32 ms). The energy source uses
# the same frame so that sample accounting in the state machine stays uniform.
FRAME_SAMPLES = 512


@dataclass(frozen=True)
class SpeechStart:
    """The start of an utterance. `timestamp_samples` is a position in the stream
    (not wall-clock)."""

    timestamp_samples: int


@dataclass(frozen=True)
class SpeechEnd:
    """The end of an utterance: silence held for `vad_redemption_s`.

    `end_timestamp_samples` points at the moment speech stopped — the redemption
    window has already been subtracted from it, so the consumer needs its own
    post-pad or the trailing consonant gets clipped.
    """

    start_timestamp_samples: int
    end_timestamp_samples: int


SpeechEvent = SpeechStart | SpeechEnd


@runtime_checkable
class SpeechEventSource(Protocol):
    """The contract of a speech-event source. Implementations: Silero (onnx) and energy."""

    @property
    def in_speech(self) -> bool:
        """Whether an utterance is in progress right now."""

    def process(self, samples: np.ndarray) -> list[SpeechEvent]:
        """Feed in the next block of audio, of arbitrary length."""

    def flush(self) -> list[SpeechEvent]:
        """Close the unfinished utterance when the stream stops."""

    def reset(self) -> None:
        ...


class SpeechStateMachine:
    """Hysteresis plus a redemption window plus a minimum-duration filter.

    Takes a speech probability frame by frame and emits events. The logic and the
    defaults follow the well-worn scheme of the Silero wrappers (silero-rs,
    FluidAudio):

    - two thresholds: entering speech is harder (`positive`) than staying in it
      (`negative`) — without this the detector flickers at the boundary;
    - `redemption` survives pauses within a phrase: an utterance is not torn
      apart every time the person takes a breath;
    - `min_speech` filters out the clicks and lip smacks on which Silero
      reliably produces 40-100 ms spikes.
    """

    def __init__(
        self,
        positive_threshold: float = 0.50,
        negative_threshold: float = 0.35,
        min_speech_s: float = 0.25,
        redemption_s: float = 0.6,
        frame_samples: int = FRAME_SAMPLES,
        sample_rate: int = SAMPLE_RATE,
    ):
        self.positive_threshold = positive_threshold
        self.negative_threshold = negative_threshold
        self.frame_samples = frame_samples
        self._min_speech_samples = int(sample_rate * min_speech_s)
        self._redemption_samples = int(sample_rate * redemption_s)
        self.reset()

    def reset(self) -> None:
        self._in_speech = False
        self._cursor = 0  # position in the stream: samples processed
        self._start: int | None = None
        self._silence_run = 0
        self._speech_run = 0

    @property
    def in_speech(self) -> bool:
        return self._in_speech

    @property
    def cursor_samples(self) -> int:
        return self._cursor

    def advance(self, prob: float) -> list[SpeechEvent]:
        """Process one frame given its speech probability."""
        events: list[SpeechEvent] = []
        if self._in_speech:
            self._speech_run += self.frame_samples
            if prob >= self.negative_threshold:
                self._silence_run = 0
            else:
                self._silence_run += self.frame_samples
                if self._silence_run >= self._redemption_samples:
                    # Speech ended where the silence started, not here.
                    end_at = self._cursor + self.frame_samples - self._silence_run
                    voiced = self._speech_run - self._silence_run
                    if voiced >= self._min_speech_samples and self._start is not None:
                        events.append(SpeechEnd(self._start, end_at))
                    self._reset_run()
        elif prob >= self.positive_threshold:
            self._in_speech = True
            # The start is the beginning of the frame that fired; anything
            # earlier the consumer picks up with its own pre-pad.
            self._start = self._cursor
            self._silence_run = 0
            self._speech_run = self.frame_samples
            events.append(SpeechStart(self._start))
        self._cursor += self.frame_samples
        return events

    def flush(self, extra_samples: int = 0) -> list[SpeechEvent]:
        """Close the current utterance at the end of the stream.

        `extra_samples` is the tail that did not make up a full frame: the
        consumer has already accumulated it, so it must be part of the timestamp.
        """
        events: list[SpeechEvent] = []
        if self._in_speech and self._start is not None:
            events.append(SpeechEnd(self._start, self._cursor + extra_samples))
        self._reset_run()
        return events

    def _reset_run(self) -> None:
        self._in_speech = False
        self._start = None
        self._silence_run = 0
        self._speech_run = 0


class _FramedProcessor:
    """The shared wrapper: arbitrary-length blocks -> frames of `FRAME_SAMPLES`.

    A subclass must implement `_prob(frame)`. The remainder that does not make up
    a frame is held back until the next call so no audio is lost at the seams.
    """

    def __init__(self, machine: SpeechStateMachine):
        self._machine = machine
        self._tail = np.zeros(0, dtype=np.float32)

    @property
    def in_speech(self) -> bool:
        return self._machine.in_speech

    def reset(self) -> None:
        self._machine.reset()
        self._tail = np.zeros(0, dtype=np.float32)

    def process(self, samples: np.ndarray) -> list[SpeechEvent]:
        if samples.dtype != np.float32:
            samples = samples.astype(np.float32)
        if samples.ndim > 1:
            samples = samples.reshape(-1)
        if self._tail.size:
            samples = np.concatenate([self._tail, samples])
            self._tail = np.zeros(0, dtype=np.float32)

        events: list[SpeechEvent] = []
        full = (len(samples) // FRAME_SAMPLES) * FRAME_SAMPLES
        for i in range(0, full, FRAME_SAMPLES):
            events.extend(self._machine.advance(self._prob(samples[i : i + FRAME_SAMPLES])))
        if full < len(samples):
            self._tail = samples[full:].copy()
        return events

    def flush(self) -> list[SpeechEvent]:
        events = self._machine.flush(extra_samples=len(self._tail))
        self._tail = np.zeros(0, dtype=np.float32)
        return events

    def _prob(self, frame: np.ndarray) -> float:
        raise NotImplementedError


class EnergyStreamProcessor(_FramedProcessor):
    """A fallback event source built on RMS — no model, no onnxruntime.

    The threshold adapts to the noise: while there is no speech, a running mean
    of the RMS estimates the noise floor. The probability comes in three steps so
    that the hysteresis in the state machine means something: 1.0 — confident
    speech, 0.4 — the grey zone (continues an utterance but does not start one),
    0.0 — silence.
    """

    NOISE_ALPHA = 0.05      # how fast the noise floor adapts
    SPEECH_FACTOR = 3.5     # how many times louder than the floor speech is (as in EnergyDetector)
    SUSTAIN_FACTOR = 0.5    # the fraction of the threshold at which an utterance is not yet over

    def __init__(self, machine: SpeechStateMachine, abs_floor: float = 0.006):
        super().__init__(machine)
        self.abs_floor = abs_floor
        self._noise = abs_floor

    def reset(self) -> None:
        super().reset()
        self._noise = self.abs_floor

    def _prob(self, frame: np.ndarray) -> float:
        rms = float(np.sqrt(np.mean(np.square(frame, dtype=np.float64))))
        threshold = max(self.abs_floor, self._noise * self.SPEECH_FACTOR)
        if not self._machine.in_speech:
            self._noise = (1 - self.NOISE_ALPHA) * self._noise + self.NOISE_ALPHA * rms
        if rms >= threshold:
            return 1.0
        if rms >= threshold * self.SUSTAIN_FACTOR:
            return 0.4
        return 0.0


def create_stream_processor(cfg, kind: str | None = None) -> SpeechEventSource:
    """The speech-event source per the config: `auto`/`silero-stream` -> Silero,
    otherwise energy."""
    kind = kind or cfg.vad
    machine = SpeechStateMachine(
        positive_threshold=cfg.vad_positive_threshold,
        negative_threshold=cfg.vad_negative_threshold,
        min_speech_s=cfg.vad_min_speech_s,
        redemption_s=cfg.vad_redemption_s,
        sample_rate=cfg.sample_rate,
    )
    if kind in ("auto", "silero-stream", "silero"):
        try:
            from .silero_stream import SileroStreamProcessor

            proc = SileroStreamProcessor(machine)
            log.info("VAD: Silero (streaming, onnx)")
            return proc
        except Exception as e:
            if kind != "auto":
                raise
            log.warning("Silero VAD is unavailable (%s) — switching to the energy one", e)
    log.info("VAD: energy (fallback, streaming)")
    return EnergyStreamProcessor(machine)
