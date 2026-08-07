"""Silero VAD v5 через onnxruntime — покадровая инференция для потокового VAD.

Почему не `faster_whisper.vad`, откуда Silero берёт `vad.py`: оттуда доступен
только батчевый `get_speech_timestamps` по готовому буферу, а машине состояний
нужна вероятность на каждый кадр по мере поступления аудио. Модель (~2.3 МБ,
MIT) лежит рядом в `data/`, onnxruntime и так есть в окружении.

Происхождение файла `data/silero_vad.onnx`:
https://github.com/snakers4/silero-vad -> `src/silero_vad/data/silero_vad.onnx`
sha256 1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from .speech_events import FRAME_SAMPLES, SAMPLE_RATE, SpeechStateMachine, _FramedProcessor

log = logging.getLogger("ilh.vad")

MODEL_PATH = Path(__file__).parent / "data" / "silero_vad.onnx"

# Перед каждым кадром модель v5 ждёт хвостовые сэмплы предыдущего — так её
# свёрточный вход видит непрерывный сигнал на стыке кадров. Ровно это делает
# и официальная обвязка silero-vad. Стоит один np.concatenate на кадр, так что
# отступать от эталонного поведения ради экономии смысла нет.
CONTEXT_SAMPLES = 64


class SileroStreamProcessor(_FramedProcessor):
    def __init__(self, machine: SpeechStateMachine, model_path: Path | None = None):
        super().__init__(machine)
        import onnxruntime as ort

        path = Path(model_path) if model_path else MODEL_PATH
        if not path.exists():
            raise FileNotFoundError(f"Модель Silero VAD не найдена: {path}")
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        # Граф крошечный: любой ускоритель тут дороже самого вычисления.
        self._sess = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
        self._sr = np.array(SAMPLE_RATE, dtype=np.int64)
        self._reset_model_state()

    def reset(self) -> None:
        super().reset()
        self._reset_model_state()

    def _reset_model_state(self) -> None:
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, CONTEXT_SAMPLES), dtype=np.float32)

    def _prob(self, frame: np.ndarray) -> float:
        window = np.concatenate([self._context, frame.reshape(1, FRAME_SAMPLES)], axis=1)
        prob, self._state = self._sess.run(
            None, {"input": window, "state": self._state, "sr": self._sr}
        )
        self._context = window[:, -CONTEXT_SAMPLES:]
        return float(prob[0, 0])
