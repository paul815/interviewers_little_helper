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

from app.asr import factory, weights
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
    assert len(ASRConfig().parakeet_revision) == 40
