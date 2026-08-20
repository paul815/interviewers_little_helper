"""The onnx-asr models the application ships with.

onnx-asr will load dozens of models, and `asr.parakeet_model` accepts any name
it knows. Only the ones listed here, though, come with three things the rest of
the application needs: a weights commit to pin, a download size for the setup
screen, and the interview languages the model is actually trained on. An
unlisted model still runs — it just arrives from the current main of its
repository, and the setup screen cannot say in advance how big it is.

Updating an entry: take the commit from the repository page on HuggingFace,
then recompute `download_mb`. It is the sum of the files onnx-asr really
fetches for `quantization`, which is far less than the repository holds — the
repositories keep an fp32 copy of every head next to the int8 one, and
`app/asr/weights.py` filters the download down to what gets loaded.
"""
from __future__ import annotations

from dataclasses import dataclass

# The 25 European languages Parakeet TDT v3 was trained on (ISO 639-1). Russian
# and Ukrainian are in the list: choosing GigaAM for Russian below is an upgrade
# in accuracy, not a fix for missing coverage.
_PARAKEET_V3_LANGUAGES = frozenset(
    "bg cs da de el en es et fi fr hr hu it lt lv mt nl pl pt ro ru sk sl sv uk".split()
)


@dataclass(frozen=True)
class ASRModelSpec:
    """What we know about a model beyond its onnx-asr name."""

    # Commit of the weights repository. Without it the current main is
    # downloaded, i.e. the model contents change without our knowledge.
    revision: str
    # Megabytes fetched on first use at the default quantization (int8).
    # Shown on the setup screen before the download starts.
    download_mb: int
    # Interview languages the model serves. The router refuses to send anything
    # else at it (app/asr/router.py).
    languages: frozenset[str]
    # One line for the setup screen and the logs — what this model is for.
    summary: str


MODELS: dict[str, ASRModelSpec] = {
    "nemo-parakeet-tdt-0.6b-v3": ASRModelSpec(
        revision="8f23f0c03c8761650bdb5b40aaf3e40d2c15f1ce",
        download_mb=671,
        languages=_PARAKEET_V3_LANGUAGES,
        summary="Parakeet TDT v3 — 25 European languages, the fast default",
    ),
    # Russian-only, and markedly more accurate on it than Parakeet. The "e2e"
    # head also restores punctuation and normalises numbers, which the other
    # onnx-asr models do not do at all — their output is one unpunctuated
    # stream. The catch is that it was trained on whole utterances while the
    # chunker feeds it a second or three at a time, so the punctuation is worth
    # eyeballing on a real recording before this becomes anyone's default.
    "gigaam-v3-e2e-rnnt": ASRModelSpec(
        revision="322c3b29492673eb7d0b434bfa9dfb8653e34d02",
        download_mb=227,
        languages=frozenset({"ru"}),
        summary="GigaAM v3 E2E — Russian only, with punctuation",
    ),
}


def spec(model_name: str) -> ASRModelSpec | None:
    """The catalogue entry, or None for a model we ship no knowledge about."""
    return MODELS.get(model_name)
