"""Loading ASR weights from a pinned revision.

`onnx_asr.load_model("nemo-parakeet-tdt-0.6b-v3")` pulls files from a HuggingFace
repository on the main branch — that is, whatever the repository owner uploaded
last arrives on the researcher's machine. load_model has no revision parameter,
but it does have `path`, and huggingface_hub has `snapshot_download(revision=...)`.
Hence the order: first download the pinned commit, then ask onnx-asr to read the
already-local folder — the network takes no part in resolution at all.

Doing the download ourselves costs us something onnx-asr does for free, and it
has to be paid back here: onnx-asr asks for the handful of files the model
really loads, while a bare `snapshot_download` takes the repository whole. That
is not a rounding error — parakeet-v3 is 3.2 GB of repository around 0.7 GB of
int8 weights, because the fp32 copy sits right next to it. `_model_file_patterns`
rebuilds the same file list onnx-asr would have asked for.

Which commit to pin comes from `asr.parakeet_revision`:

    "auto"  the commit recorded for this model in app/asr/catalog.py (default)
    ""      deliberately unpinned — track the repository's main, lose reproducibility
    <sha>   this exact commit, whatever the catalogue says
"""
from __future__ import annotations

import logging
from pathlib import Path

from ..config import ASRConfig
from .catalog import spec

log = logging.getLogger("ilh.asr")


def asr_repo_id(model_name: str) -> str | None:
    """An onnx-asr model name -> a HuggingFace repository. None for local paths."""
    try:
        from onnx_asr.resolver import model_repos
    except ImportError:
        return None
    return model_repos.get(model_name)


def pinned_revision(cfg: ASRConfig) -> str | None:
    """The commit to download, or None to track the repository's main.

    "auto" against a model the catalogue does not know is not an error — it is
    how you try a model out. It does mean the download is no longer reproducible,
    so it is worth a line in the log.
    """
    configured = (cfg.parakeet_revision or "").strip()
    if configured != "auto":
        return configured or None

    entry = spec(cfg.parakeet_model)
    if entry is None:
        log.warning(
            "%s is not in the model catalogue — its weights come from the current "
            "main and may change under you. Pin a commit in asr.parakeet_revision "
            "to make the download reproducible.",
            cfg.parakeet_model,
        )
        return None
    return entry.revision


def _model_file_patterns(model_name: str, quantization: str | None) -> list[str] | None:
    """The files onnx-asr will look for, as `allow_patterns` for the download.

    Rebuilt the way `onnx_asr.resolver.Resolver._download_model` builds it, from
    the same `_get_model_files` table, so the pinned download and onnx-asr's own
    download fetch exactly the same set. None means "could not work it out" —
    the caller then takes the whole repository, which is wasteful but correct.
    """
    if "/" in model_name:
        # A custom repository: onnx-asr would have to read its config.json off
        # the network to learn the model type, and this function must not block.
        return None
    try:
        from onnx_asr.loader import create_asr_resolver

        model_type = create_asr_resolver(model_name).model_type
        files = list(model_type._get_model_files(quantization).values())
    except Exception as e:  # unknown name, or onnx-asr moved the private table
        log.debug("Could not derive the file list for %s: %s", model_name, e)
        return None

    files += [f.removeprefix("**/") for f in files if f.startswith("**/")]
    return [
        "config.json",
        "config.yaml",
        *files,
        # Weights above 2 GB live in a sidecar next to the .onnx graph.
        *(str(p.with_suffix(".onnx?data")) for f in files if (p := Path(f)).suffix == ".onnx"),
    ]


def pinned_snapshot(cfg: ASRConfig) -> str | None:
    """The local folder holding the weights of the pinned revision.

    None means "let onnx-asr work it out itself": either the model is not from the
    hub, or the revision is deliberately unpinned.
    """
    repo = asr_repo_id(cfg.parakeet_model)
    revision = pinned_revision(cfg)
    if not repo or not revision:
        if repo and not revision:
            log.warning("The weights revision of %s is unpinned — the current main "
                        "will be downloaded", repo)
        return None

    from huggingface_hub import snapshot_download

    patterns = _model_file_patterns(cfg.parakeet_model, cfg.parakeet_quantization or None)
    if patterns is None:
        log.warning("The file list for %s is unknown — downloading the whole of %s",
                    cfg.parakeet_model, repo)
    path = snapshot_download(repo_id=repo, revision=revision, allow_patterns=patterns)
    log.info("Weights %s@%s: %s", repo, revision[:12], path)
    return path


def load_pinned_model(cfg: ASRConfig):
    """`onnx_asr.load_model`, but from the weights of the pinned revision."""
    import onnx_asr

    return onnx_asr.load_model(
        cfg.parakeet_model,
        path=pinned_snapshot(cfg),
        quantization=cfg.parakeet_quantization or None,
        providers=list(cfg.providers),
    )
