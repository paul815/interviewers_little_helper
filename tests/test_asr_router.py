"""Turning the interview language into an ASR config.

Nothing here loads a model: the router is a lookup table, and these tests are
about which table row wins.
"""
from __future__ import annotations

import pytest

from app.asr import catalog, router
from app.config import ASRConfig

RUSSIAN_MODEL = "gigaam-v3-e2e-rnnt"
DEFAULT_MODEL = "nemo-parakeet-tdt-0.6b-v3"


# ------------------------------------------------------------------ normalising


@pytest.mark.parametrize("given", [None, "", "   ", "auto", "AUTO"])
def test_no_language_chosen(given):
    assert router.normalise(given) is None


@pytest.mark.parametrize("given", ["ru", "RU", " ru ", "ru-RU", "ru_RU", "RU-ru"])
def test_region_and_case_are_stripped(given):
    assert router.normalise(given) == "ru"


# --------------------------------------------------------------- choosing a model


def test_russian_prefers_the_specialist():
    """Parakeet covers Russian too. GigaAM is chosen because it covers it better —
    if that preference ever silently flips, this is what catches it."""
    assert "ru" in catalog.MODELS[DEFAULT_MODEL].languages
    assert router.model_for_language("ru") == RUSSIAN_MODEL


@pytest.mark.parametrize("code", ["en", "de", "fr", "uk", "pl"])
def test_other_european_languages_go_to_parakeet(code):
    assert router.model_for_language(code) == DEFAULT_MODEL


@pytest.mark.parametrize("code", ["ja", "zh", "ar", "he", "kk"])
def test_languages_outside_the_catalogue(code):
    assert router.model_for_language(code) is None


def test_language_table_covers_every_catalogue_language():
    table = router.languages()
    for name, entry in catalog.MODELS.items():
        for code in entry.languages:
            assert code in table, f"{code} ({name}) is missing from the table"
    assert table["ru"] == RUSSIAN_MODEL
    assert table["en"] == DEFAULT_MODEL


# ------------------------------------------------------------------- routing


def test_no_language_leaves_the_config_alone():
    """The behaviour every session had before the selector existed."""
    cfg = ASRConfig(parakeet_model="whatever-model", backend="faster")
    assert router.route(cfg, None) is cfg
    assert router.route(cfg, "auto") is cfg


def test_russian_switches_the_model_but_not_the_backend():
    cfg = ASRConfig()
    routed = router.route(cfg, "ru")
    assert routed.parakeet_model == RUSSIAN_MODEL
    assert routed.language == "ru"
    # "auto" must survive: it is what lets the factory fall back to Whisper on a
    # machine where onnx-asr was never installed.
    assert routed.backend == "auto"
    assert cfg.parakeet_model == DEFAULT_MODEL, "the base config was mutated"


def test_covered_language_still_reaches_a_pinned_whisper():
    """A researcher who chose Whisper in config.json keeps it — and now Whisper
    stops guessing the language per chunk, which is the whole point."""
    routed = router.route(ASRConfig(backend="faster"), "de")
    assert routed.backend == "faster"
    assert routed.language == "de"


def test_uncovered_language_forces_whisper():
    """Parakeet does not fail on Japanese, it invents. Whisper is the only
    honest option, so it overrides even an explicit backend."""
    routed = router.route(ASRConfig(backend="parakeet"), "ja")
    assert routed.backend == "whisper"
    assert routed.language == "ja"


@pytest.mark.parametrize("backend", ["mlx", "faster", "faster-cpu"])
def test_uncovered_language_keeps_a_whisper_backend(backend):
    """Forcing "whisper" over "mlx" would cost Apple Silicon its Metal path."""
    routed = router.route(ASRConfig(backend=backend), "ja")
    assert routed.backend == backend
    assert routed.language == "ja"


def test_every_routed_model_is_in_the_catalogue():
    """The router must never name a model with no pinned revision behind it."""
    for code in router.languages():
        assert router.route(ASRConfig(), code).parakeet_model in catalog.MODELS
