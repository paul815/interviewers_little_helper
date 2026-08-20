"""Parakeet TDT v3 through onnx-asr — the fast default backend.

Whisper large-v3-turbo decodes autoregressively and takes seconds over an
utterance a few seconds long. Parakeet TDT 0.6B is roughly an order of magnitude
faster at comparable quality (on Russian its WER is a little worse than Whisper
large-v3, in the same class), and the int8 version copes comfortably on CPU,
freeing the VRAM for the LLM.

The differences from Whisper that matter to the calling code:

- there is no prompt conditioning, so `asr.vocabulary` does not work with this
  backend — when the vocabulary is non-empty we warn in the log and in the UI status;
- there is no `no_speech_prob`: on silence the model returns an empty string
  rather than hallucinating "To be continued…", so a probability filter is not
  needed (`ASRWorker` lets `None` through);
- there is no language detection: Parakeet is multilingual and language-agnostic
  at inference, so we return whatever the config states explicitly.

Despite the name, this backend is not Parakeet-specific — it runs whatever
`asr.parakeet_model` names, and for a Russian interview the router points that
at gigaam-v3-e2e-rnnt (app/asr/router.py, app/asr/catalog.py).
"""
from __future__ import annotations

import logging

import numpy as np

from ..config import ASRConfig
from .base import ASRBackend, ASRResult
from .weights import load_pinned_model

log = logging.getLogger("ilh.asr")

SAMPLE_RATE = 16000
# Shorter than this and onnx-asr returns garbage: there are not even enough
# frames for a single encoder step.
MIN_AUDIO_S = 0.1


class ParakeetOnnxBackend(ASRBackend):
    def __init__(self, cfg: ASRConfig):
        self.cfg = cfg
        self.name = "parakeet"
        self.model = None

    def load(self) -> None:
        log.info(
            "Loading Parakeet %s (%s, %s)…",
            self.cfg.parakeet_model, self.cfg.parakeet_quantization,
            ", ".join(self.cfg.providers),
        )
        self.model = load_pinned_model(self.cfg)
        # Warm-up: the first real utterance must not wait for session initialisation.
        self.model.recognize(np.zeros(SAMPLE_RATE, dtype=np.float32), sample_rate=SAMPLE_RATE)
        log.info("Parakeet is ready")
        if self.cfg.vocabulary.strip():
            log.warning(
                "A term vocabulary is set, but Parakeet does not support prompt hinting — "
                "it will be ignored. For the vocabulary, switch asr.backend to faster or mlx."
            )

    def transcribe(self, audio: np.ndarray) -> ASRResult:
        if len(audio) < MIN_AUDIO_S * SAMPLE_RATE:
            return ASRResult(text="")
        arr = np.asarray(audio, dtype=np.float32).reshape(-1)
        result = self.model.recognize(arr, sample_rate=SAMPLE_RATE)
        return ASRResult(text=_extract_text(result), language=self.cfg.language)

    def describe(self) -> str:
        # The model name, not "parakeet": this line goes on screen, and on a
        # Russian interview the model running is GigaAM.
        return f"onnx-asr {self.cfg.parakeet_model} ({', '.join(self.cfg.providers)})"

    def warnings(self) -> list[str]:
        if self.cfg.vocabulary.strip():
            return ["the term vocabulary is not supported by Parakeet and was ignored"]
        return []


def _extract_text(result) -> str:
    """Different versions of onnx-asr return either an object with `.text` or the string itself."""
    if isinstance(result, str):
        return result.strip()
    text = getattr(result, "text", None)
    return text.strip() if isinstance(text, str) else ""
