"""Parakeet TDT v3 через onnx-asr — быстрый бэкенд по умолчанию.

Whisper large-v3-turbo декодирует авторегрессионно и на реплике в несколько
секунд стоит секунды же. Parakeet TDT 0.6B — примерно на порядок быстрее при
сопоставимом качестве (на русском WER чуть хуже Whisper large-v3, в том же
классе), и int8-версия уверенно работает на CPU, освобождая VRAM для LLM.

Отличия от Whisper, важные для вызывающего кода:

- нет conditioning промптом, поэтому `asr.vocabulary` с этим бэкендом не
  работает — при непустом словаре предупреждаем в лог и в статус UI;
- нет `no_speech_prob`: модель на тишине отдаёт пустую строку, а не
  галлюцинирует «Продолжение следует…», так что фильтр по вероятности не нужен
  (`ASRWorker` пропускает `None`);
- нет детекции языка: Parakeet многоязычен и языко-агностичен на инференсе,
  возвращаем то, что явно задано в конфиге.

Тем же API onnx-asr отдаёт и другие модели (например gigaam-v2-rnnt для
русского) — достаточно поменять `asr.parakeet_model`.
"""
from __future__ import annotations

import logging

import numpy as np

from ..config import ASRConfig
from .base import ASRBackend, ASRResult

log = logging.getLogger("ilh.asr")

SAMPLE_RATE = 16000
# Короче этого onnx-asr отдаёт мусор: кадров не хватает даже на один шаг энкодера.
MIN_AUDIO_S = 0.1


class ParakeetOnnxBackend(ASRBackend):
    def __init__(self, cfg: ASRConfig):
        self.cfg = cfg
        self.name = "parakeet"
        self.model = None

    def load(self) -> None:
        import onnx_asr

        log.info(
            "Загружаю Parakeet %s (%s, %s)…",
            self.cfg.parakeet_model, self.cfg.parakeet_quantization,
            ", ".join(self.cfg.providers),
        )
        self.model = onnx_asr.load_model(
            self.cfg.parakeet_model,
            quantization=self.cfg.parakeet_quantization or None,
            providers=list(self.cfg.providers),
        )
        # Прогрев: первая настоящая реплика не должна ждать инициализацию сессии.
        self.model.recognize(np.zeros(SAMPLE_RATE, dtype=np.float32), sample_rate=SAMPLE_RATE)
        log.info("Parakeet готов")
        if self.cfg.vocabulary.strip():
            log.warning(
                "Словарь терминов задан, но Parakeet не поддерживает подсказку промптом — "
                "он будет проигнорирован. Для словаря переключите asr.backend на faster или mlx."
            )

    def transcribe(self, audio: np.ndarray) -> ASRResult:
        if len(audio) < MIN_AUDIO_S * SAMPLE_RATE:
            return ASRResult(text="")
        arr = np.asarray(audio, dtype=np.float32).reshape(-1)
        result = self.model.recognize(arr, sample_rate=SAMPLE_RATE)
        return ASRResult(text=_extract_text(result), language=self.cfg.language)

    def describe(self) -> str:
        return f"parakeet {self.cfg.parakeet_model} ({', '.join(self.cfg.providers)})"

    def warnings(self) -> list[str]:
        if self.cfg.vocabulary.strip():
            return ["словарь терминов не поддерживается Parakeet и проигнорирован"]
        return []


def _extract_text(result) -> str:
    """onnx-asr в разных версиях отдаёт то объект с `.text`, то саму строку."""
    if isinstance(result, str):
        return result.strip()
    text = getattr(result, "text", None)
    return text.strip() if isinstance(text, str) else ""
