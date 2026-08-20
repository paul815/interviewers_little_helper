"""Choosing an ASR model for the language of an interview.

The researcher names the language when starting a session; this turns that into
an `ASRConfig` for that session alone. Three outcomes:

- a language one of our catalogue models specialises in -> that model. Russian
  goes to GigaAM rather than Parakeet, which covers Russian too but less well;
- any other language Parakeet v3 was trained on -> Parakeet, the fast default;
- anything else -> Whisper, which knows about a hundred languages, with the
  language pinned so it stops guessing per chunk.

That last point is the quiet win. Whisper detects the language from the first
30 seconds of what it is given, and what it is given here is one VAD chunk —
often a second and a half of "mhm". Naming the language up front removes a coin
flip that, when it lands wrong, produces confident text in the wrong alphabet.

The language is recorded in `ASRConfig.language` in every branch, even where it
changes no decoding at all (Parakeet and GigaAM are language-agnostic at
inference). It travels on into the transcript so a session can be read back
knowing what it was recognised as.
"""
from __future__ import annotations

import dataclasses
import logging

from ..config import ASRConfig
from .catalog import MODELS

log = logging.getLogger("ilh.asr")

# Most specialised first: the first model that covers the language wins. GigaAM
# ahead of Parakeet is the whole of the policy — one Russian-only model that
# beats the generalist on Russian, and the generalist for everything else.
_PREFERENCE = ("gigaam-v3-e2e-rnnt", "nemo-parakeet-tdt-0.6b-v3")

# Backends that reach Whisper, which serves any language.
_WHISPER_BACKENDS = frozenset({"mlx", "faster", "faster-cpu", "whisper"})


def normalise(language: str | None) -> str | None:
    """"ru-RU" -> "ru". None for "not chosen", which is not an error.

    The UI sends plain ISO 639-1 codes; this exists so a config file or an old
    session written by hand does not have to.
    """
    code = (language or "").strip().lower()
    if not code or code == "auto":
        return None
    return code.split("-", 1)[0].split("_", 1)[0]


def model_for_language(language: str | None) -> str | None:
    """The onnx-asr model that serves this language, or None if none of ours does."""
    code = normalise(language)
    if code is None:
        return None
    for name in _PREFERENCE:
        if code in MODELS[name].languages:
            return name
    return None


def languages() -> dict[str, str]:
    """Language code -> the model that would be used, for every language the
    catalogue covers. The setup screen needs this to know what to download."""
    return {
        code: name
        for name in reversed(_PREFERENCE)          # preferred entries overwrite
        for code in MODELS[name].languages
    }


def route(cfg: ASRConfig, language: str | None) -> ASRConfig:
    """`cfg` adjusted for an interview in `language`.

    Returned unchanged when no language was chosen: that is what "let config.json
    decide" looks like, and it is the behaviour every session had before the
    selector existed.
    """
    code = normalise(language)
    if code is None:
        return cfg

    model = model_for_language(code)
    if model:
        # `backend` is deliberately left alone. On "auto" the factory still gets
        # to fall back to Whisper when onnx-asr is not installed, and a researcher
        # who pinned a backend in config.json keeps it.
        log.info("Interview language %s: recognising with %s", code, model)
        return dataclasses.replace(cfg, parakeet_model=model, language=code)

    if cfg.backend in _WHISPER_BACKENDS:
        log.info("Interview language %s: no onnx-asr model of ours covers it, "
                 "using the configured %s", code, cfg.backend)
        return dataclasses.replace(cfg, language=code)

    # Parakeet does not fail on a language it was not trained on — it produces
    # fluent nonsense. Whisper is the only honest option here, so it overrides
    # even an explicitly configured backend.
    log.warning("Interview language %s is outside %s — switching to Whisper for "
                "this session", code, cfg.parakeet_model)
    return dataclasses.replace(cfg, backend="whisper", language=code)
