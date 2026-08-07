"""Потоковое детектирование границ реплик: поток аудио -> события речи.

Отличие от `vad.py`: тот пересчитывает разметку речи по всему буферу заново на
каждом опросе (батчевый `speech_regions`), здесь каждый кадр обрабатывается
ровно один раз, а конец реплики становится известен через `vad_redemption_s`
после того, как человек замолчал — не дожидаясь, пока буфер дорастёт до
`max_chunk_s`. Именно на этом держится низкая задержка транскрипта.

Модуль намеренно не зависит ни от onnxruntime, ни от аудио-железа: машина
состояний и энергетический источник событий тестируются в чистом виде.
Silero-реализация живёт в `silero_stream.py` и подключается лениво.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

log = logging.getLogger("ilh.vad")

SAMPLE_RATE = 16000
# Родной размер кадра Silero VAD на 16 kHz (32 мс). Тот же кадр используется и
# энергетическим источником, чтобы учёт сэмплов в машине состояний был единым.
FRAME_SAMPLES = 512


@dataclass(frozen=True)
class SpeechStart:
    """Начало реплики. `timestamp_samples` — позиция в потоке (не wall-clock)."""

    timestamp_samples: int


@dataclass(frozen=True)
class SpeechEnd:
    """Конец реплики: тишина держалась `vad_redemption_s`.

    `end_timestamp_samples` указывает на момент, когда речь смолкла, — окно
    «искупления» из него уже вычтено, поэтому потребителю нужен собственный
    post-pad, иначе срежется хвостовой согласный.
    """

    start_timestamp_samples: int
    end_timestamp_samples: int


SpeechEvent = SpeechStart | SpeechEnd


@runtime_checkable
class SpeechEventSource(Protocol):
    """Контракт источника событий речи. Реализации: Silero (onnx) и энергия."""

    @property
    def in_speech(self) -> bool:
        """Идёт ли реплика прямо сейчас."""

    def process(self, samples: np.ndarray) -> list[SpeechEvent]:
        """Скормить очередной блок аудио произвольной длины."""

    def flush(self) -> list[SpeechEvent]:
        """Закрыть незавершённую реплику при остановке потока."""

    def reset(self) -> None:
        ...


class SpeechStateMachine:
    """Гистерезис + окно «искупления» + фильтр минимальной длительности.

    Принимает вероятность речи покадрово, отдаёт события. Логика и дефолты
    повторяют обкатанную схему Silero-обвязок (silero-rs, FluidAudio):

    - два порога: войти в речь труднее (`positive`), чем в ней остаться
      (`negative`) — без этого детектор «мигает» на границе;
    - `redemption` переживает паузы внутри фразы: реплика не рвётся на части
      каждый раз, когда человек переводит дыхание;
    - `min_speech` отсекает щелчки и причмокивания, на которых Silero
      исправно выдаёт всплески по 40–100 мс.
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
        self._cursor = 0  # позиция в потоке: сэмплов обработано
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
        """Обработать один кадр по его вероятности речи."""
        events: list[SpeechEvent] = []
        if self._in_speech:
            self._speech_run += self.frame_samples
            if prob >= self.negative_threshold:
                self._silence_run = 0
            else:
                self._silence_run += self.frame_samples
                if self._silence_run >= self._redemption_samples:
                    # Речь кончилась там, где началась тишина, а не здесь.
                    end_at = self._cursor + self.frame_samples - self._silence_run
                    voiced = self._speech_run - self._silence_run
                    if voiced >= self._min_speech_samples and self._start is not None:
                        events.append(SpeechEnd(self._start, end_at))
                    self._reset_run()
        elif prob >= self.positive_threshold:
            self._in_speech = True
            # Началом считаем начало сработавшего кадра; всё, что раньше,
            # потребитель добирает собственным pre-pad'ом.
            self._start = self._cursor
            self._silence_run = 0
            self._speech_run = self.frame_samples
            events.append(SpeechStart(self._start))
        self._cursor += self.frame_samples
        return events

    def flush(self, extra_samples: int = 0) -> list[SpeechEvent]:
        """Закрыть текущую реплику концом потока.

        `extra_samples` — хвост, не набравший полного кадра: потребитель его
        уже накопил, поэтому в таймштамп он входить обязан.
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
    """Общая обвязка: блоки произвольной длины -> кадры по `FRAME_SAMPLES`.

    Наследник обязан реализовать `_prob(frame)`. Остаток, не набравший кадра,
    придерживается до следующего вызова, чтобы аудио не терялось на стыках.
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
    """Запасной источник событий на RMS — без модели и без onnxruntime.

    Порог адаптируется к шуму: пока речи нет, скользящее среднее RMS даёт
    оценку шумового пола. Вероятность отдаётся тремя ступенями, чтобы
    гистерезис в машине состояний имел смысл: 1.0 — уверенная речь, 0.4 —
    «серая зона» (реплику продолжает, но не начинает), 0.0 — тишина.
    """

    NOISE_ALPHA = 0.05      # скорость адаптации шумового пола
    SPEECH_FACTOR = 3.5     # во сколько раз речь громче пола (как в EnergyDetector)
    SUSTAIN_FACTOR = 0.5    # доля порога, на которой реплика ещё не считается законченной

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
    """Источник событий речи по конфигу: `auto`/`silero-stream` -> Silero, иначе энергия."""
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
            log.info("VAD: Silero (потоковый, onnx)")
            return proc
        except Exception as e:
            if kind != "auto":
                raise
            log.warning("Silero VAD недоступен (%s) — переключаюсь на энергетический", e)
    log.info("VAD: энергетический (запасной, потоковый)")
    return EnergyStreamProcessor(machine)
