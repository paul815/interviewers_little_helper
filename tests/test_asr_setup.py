"""Choosing the ASR backend and pinning the weights revision.

The backends themselves are never loaded here: their classes are replaced with
stubs, so the tests run without a GPU, without onnxruntime and without a trip to
HuggingFace.
"""
from __future__ import annotations

import logging
import sys
import types

import pytest

from app.asr import catalog, factory, weights
from app.config import ASRConfig

# ------------------------------------------------------------- backend choice


@pytest.fixture()
def fake_backends(monkeypatch):
    """Stub out the backend modules: what matters is the choice, not loading a model."""
    names = {}
    for mod_name, cls_name in (
        ("app.asr.parakeet_backend", "ParakeetOnnxBackend"),
        ("app.asr.mlx_backend", "MLXWhisperBackend"),
        ("app.asr.faster_backend", "FasterWhisperBackend"),
    ):
        module = types.ModuleType(mod_name)

        def make(label):
            class Fake:
                def __init__(self, cfg, device=None):
                    self.label, self.device = label, device

            return Fake

        setattr(module, cls_name, make(cls_name))
        monkeypatch.setitem(sys.modules, mod_name, module)
        names[cls_name] = getattr(module, cls_name)
    return names


def _auto(monkeypatch, *, onnx: bool, mlx: bool, platform_name: str, machine: str):
    monkeypatch.setattr(factory, "_onnx_asr_available", lambda: onnx)
    monkeypatch.setattr(factory, "_mlx_available", lambda: mlx)
    monkeypatch.setattr(factory.sys, "platform", platform_name)
    monkeypatch.setattr(factory.platform, "machine", lambda: machine)
    return factory.create_asr_backend(ASRConfig(backend="auto"))


def test_auto_prefers_parakeet(monkeypatch, fake_backends):
    b = _auto(monkeypatch, onnx=True, mlx=True, platform_name="darwin", machine="arm64")
    assert b.label == "ParakeetOnnxBackend"


def test_auto_falls_back_to_mlx_on_apple_silicon(monkeypatch, fake_backends):
    b = _auto(monkeypatch, onnx=False, mlx=True, platform_name="darwin", machine="arm64")
    assert b.label == "MLXWhisperBackend"


@pytest.mark.parametrize("platform_name,machine", [
    ("win32", "AMD64"),
    ("linux", "x86_64"),
    ("darwin", "x86_64"),   # an Intel Mac: mlx is not for it
])
def test_auto_falls_back_to_faster(monkeypatch, fake_backends, platform_name, machine):
    b = _auto(monkeypatch, onnx=False, mlx=True, platform_name=platform_name, machine=machine)
    assert b.label == "FasterWhisperBackend"


def test_auto_uses_faster_when_mlx_missing(monkeypatch, fake_backends):
    b = _auto(monkeypatch, onnx=False, mlx=False, platform_name="darwin", machine="arm64")
    assert b.label == "FasterWhisperBackend"


@pytest.mark.parametrize("backend,expected,device", [
    ("parakeet", "ParakeetOnnxBackend", None),
    ("mlx", "MLXWhisperBackend", None),
    ("faster", "FasterWhisperBackend", "auto"),
    ("faster-cpu", "FasterWhisperBackend", "cpu"),
])
def test_explicit_backend_wins(fake_backends, backend, expected, device):
    b = factory.create_asr_backend(ASRConfig(backend=backend))
    assert b.label == expected and b.device == device


def test_whisper_backend_never_reaches_parakeet(monkeypatch, fake_backends):
    """What the router asks for when the interview is in a language Parakeet was
    not trained on: the best Whisper this machine has, and never Parakeet — even
    though onnx-asr is installed and `auto` would have chosen it."""
    monkeypatch.setattr(factory, "_onnx_asr_available", lambda: True)
    monkeypatch.setattr(factory, "_mlx_available", lambda: True)
    monkeypatch.setattr(factory.sys, "platform", "darwin")
    monkeypatch.setattr(factory.platform, "machine", lambda: "arm64")

    b = factory.create_asr_backend(ASRConfig(backend="whisper"))
    assert b.label == "MLXWhisperBackend"


def test_whisper_backend_falls_back_to_faster(monkeypatch, fake_backends):
    monkeypatch.setattr(factory, "_onnx_asr_available", lambda: True)
    monkeypatch.setattr(factory, "_mlx_available", lambda: False)
    monkeypatch.setattr(factory.sys, "platform", "win32")
    monkeypatch.setattr(factory.platform, "machine", lambda: "AMD64")

    b = factory.create_asr_backend(ASRConfig(backend="whisper"))
    assert b.label == "FasterWhisperBackend" and b.device == "auto"


def test_unknown_backend_is_an_error():
    with pytest.raises(ValueError, match="Unknown ASR backend"):
        factory.create_asr_backend(ASRConfig(backend="whisper.cpp"))


def test_availability_probes_survive_missing_packages(monkeypatch):
    """A missing onnx-asr/mlx is not a crash but a warning and a fallback."""
    real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __import__

    def deny(name, *a, **kw):
        if name in ("onnx_asr", "mlx_whisper"):
            raise ImportError(name)
        return real_import(name, *a, **kw)

    monkeypatch.setattr("builtins.__import__", deny)
    assert factory._onnx_asr_available() is False
    assert factory._mlx_available() is False


# ------------------------------------------------------------ revision pinning

PIN = "8f23f0c03c8761650bdb5b40aaf3e40d2c15f1ce"


def test_pinned_snapshot_passes_revision(monkeypatch):
    """The main point: we download the pinned commit, not the current main."""
    seen = {}

    def fake_download(**kw):
        seen.update(kw)
        return "/cache/models--istupakov--parakeet"

    monkeypatch.setattr(weights, "asr_repo_id", lambda name: "istupakov/parakeet-tdt-0.6b-v3-onnx")
    monkeypatch.setitem(
        sys.modules, "huggingface_hub",
        types.SimpleNamespace(snapshot_download=fake_download),
    )
    path = weights.pinned_snapshot(ASRConfig(parakeet_revision=PIN))

    assert seen["revision"] == PIN
    assert seen["repo_id"] == "istupakov/parakeet-tdt-0.6b-v3-onnx"
    assert path == "/cache/models--istupakov--parakeet"


def test_empty_revision_warns_and_delegates(monkeypatch, caplog):
    monkeypatch.setattr(weights, "asr_repo_id", lambda name: "org/repo")
    with caplog.at_level(logging.WARNING, logger="ilh.asr"):
        assert weights.pinned_snapshot(ASRConfig(parakeet_revision="")) is None
    assert "is unpinned" in caplog.text


def test_local_model_needs_no_snapshot(monkeypatch):
    """A model that is not from the hub: nothing to download, the revision is irrelevant."""
    monkeypatch.setattr(weights, "asr_repo_id", lambda name: None)
    assert weights.pinned_snapshot(ASRConfig(parakeet_revision=PIN)) is None


def test_load_pinned_model_forwards_path(monkeypatch):
    seen = {}

    def fake_load(model, **kw):
        seen["model"] = model
        seen.update(kw)
        return "model-object"

    monkeypatch.setattr(weights, "pinned_snapshot", lambda cfg: "/cache/dir")
    monkeypatch.setitem(sys.modules, "onnx_asr", types.SimpleNamespace(load_model=fake_load))

    cfg = ASRConfig(parakeet_quantization="int8")
    assert weights.load_pinned_model(cfg) == "model-object"
    assert seen["path"] == "/cache/dir"
    assert seen["model"] == cfg.parakeet_model
    assert seen["quantization"] == "int8"
    assert seen["providers"] == ["CPUExecutionProvider"]


def test_config_ships_a_pinned_revision():
    """The default is pinned: otherwise build reproducibility is lost silently."""
    cfg = ASRConfig()
    assert cfg.parakeet_revision == "auto"
    assert weights.pinned_revision(cfg) == catalog.MODELS[cfg.parakeet_model].revision


# ---------------------------------------------------- "auto" against the catalogue


def test_auto_revision_follows_the_model(monkeypatch):
    """The bug this replaced: one revision string was applied to every repository,
    so any model other than Parakeet died on RevisionNotFound."""
    for name, entry in catalog.MODELS.items():
        cfg = ASRConfig(parakeet_model=name)  # parakeet_revision defaults to "auto"
        assert weights.pinned_revision(cfg) == entry.revision


def test_auto_revision_on_an_unlisted_model_warns(caplog):
    cfg = ASRConfig(parakeet_model="gigaam-multilingual-ctc")
    with caplog.at_level(logging.WARNING, logger="ilh.asr"):
        assert weights.pinned_revision(cfg) is None
    assert "not in the model catalogue" in caplog.text


def test_explicit_revision_overrides_the_catalogue():
    other = "0" * 40
    assert weights.pinned_revision(ASRConfig(parakeet_revision=other)) == other


def test_empty_revision_stays_unpinned():
    assert weights.pinned_revision(ASRConfig(parakeet_revision="")) is None


# -------------------------------------------------------- narrowing the download


def _stub_onnx_asr(monkeypatch, files: dict[str, str]):
    """onnx-asr is absent from the test environment, and its private file table is
    what we mirror — so the mirror is what gets tested, against a stub of it."""
    model_type = types.SimpleNamespace(_get_model_files=lambda quantization=None: files)
    loader = types.SimpleNamespace(
        create_asr_resolver=lambda name: types.SimpleNamespace(model_type=model_type)
    )
    monkeypatch.setitem(sys.modules, "onnx_asr", types.ModuleType("onnx_asr"))
    monkeypatch.setitem(sys.modules, "onnx_asr.loader", loader)


def test_file_patterns_cover_weights_config_and_sidecars(monkeypatch):
    _stub_onnx_asr(monkeypatch, {
        "encoder": "encoder-model?int8.onnx",
        "decoder": "decoder_joint-model?int8.onnx",
        "vocab": "vocab.txt",
    })
    patterns = weights._model_file_patterns("nemo-parakeet-tdt-0.6b-v3", "int8")

    assert "config.json" in patterns
    assert "vocab.txt" in patterns
    # Weights over 2 GB are split into a .onnx + .onnx.data pair; missing the
    # sidecar leaves an unloadable graph on disk.
    assert "encoder-model?int8.onnx?data" in patterns


def test_file_patterns_exclude_the_other_quantization(monkeypatch):
    """Why this function exists at all: the repositories keep an fp32 copy of
    every head, and a bare snapshot_download takes 3.2 GB to use 0.7 GB."""
    import fnmatch

    _stub_onnx_asr(monkeypatch, {"encoder": "encoder-model?int8.onnx", "vocab": "vocab.txt"})
    patterns = weights._model_file_patterns("nemo-parakeet-tdt-0.6b-v3", "int8")

    def wanted(filename: str) -> bool:
        return any(fnmatch.fnmatch(filename, p) for p in patterns)

    assert wanted("encoder-model.int8.onnx")
    assert not wanted("encoder-model.onnx")
    assert not wanted("encoder-model.onnx.data")


def test_custom_repository_takes_the_whole_snapshot(monkeypatch):
    """A "org/name" model would need a network round trip to learn its type."""
    assert weights._model_file_patterns("some-org/private-asr", "int8") is None


def test_missing_onnx_asr_falls_back_to_the_whole_repository(monkeypatch):
    real_import = __import__

    def deny(name, *a, **kw):
        if name.startswith("onnx_asr"):
            raise ImportError(name)
        return real_import(name, *a, **kw)

    monkeypatch.setattr("builtins.__import__", deny)
    assert weights._model_file_patterns("nemo-parakeet-tdt-0.6b-v3", "int8") is None


def test_snapshot_narrows_the_download(monkeypatch):
    """End to end: pinned_snapshot hands allow_patterns to huggingface_hub."""
    seen = {}
    _stub_onnx_asr(monkeypatch, {"encoder": "encoder-model?int8.onnx"})
    monkeypatch.setattr(weights, "asr_repo_id", lambda name: "istupakov/parakeet-tdt-0.6b-v3-onnx")
    monkeypatch.setitem(
        sys.modules, "huggingface_hub",
        types.SimpleNamespace(snapshot_download=lambda **kw: seen.update(kw) or "/cache/dir"),
    )

    weights.pinned_snapshot(ASRConfig())
    assert "encoder-model?int8.onnx" in seen["allow_patterns"]


# -------------------------------------------------------------------- catalogue


def test_catalogue_entries_are_usable():
    assert catalog.MODELS, "an empty catalogue would silently unpin every model"
    for name, entry in catalog.MODELS.items():
        assert len(entry.revision) == 40, f"{name}: not a commit sha"
        assert int(entry.revision, 16) >= 0, f"{name}: not hexadecimal"
        assert entry.download_mb > 0, f"{name}: the setup screen shows this"
        assert entry.languages, f"{name}: the router has nothing to match on"
        assert entry.summary


def test_catalogue_names_are_known_to_onnx_asr():
    """A typo in a catalogue key would only surface at the start of an interview."""
    model_repos = pytest.importorskip("onnx_asr.resolver").model_repos
    for name in catalog.MODELS:
        assert name in model_repos, f"{name} is not an onnx-asr model name"
