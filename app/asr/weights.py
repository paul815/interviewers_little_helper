"""Loading ASR weights from a pinned revision.

`onnx_asr.load_model("nemo-parakeet-tdt-0.6b-v3")` pulls files from a HuggingFace
repository on the main branch — that is, whatever the repository owner uploaded
last arrives on the researcher's machine. load_model has no revision parameter,
but it does have `path`, and huggingface_hub has `snapshot_download(revision=...)`.
Hence the order: first download the pinned commit, then ask onnx-asr to read the
already-local folder — the network takes no part in resolution at all.

The revision lives in the config (`asr.parakeet_revision`): it can be unpinned by
setting an empty string, but then build reproducibility is lost.
"""
from __future__ import annotations

import logging

from ..config import ASRConfig

log = logging.getLogger("ilh.asr")


def asr_repo_id(model_name: str) -> str | None:
    """An onnx-asr model name -> a HuggingFace repository. None for local paths."""
    try:
        from onnx_asr.resolver import model_repos
    except ImportError:
        return None
    return model_repos.get(model_name)


def pinned_snapshot(cfg: ASRConfig) -> str | None:
    """The local folder holding the weights of the pinned revision.

    None means "let onnx-asr work it out itself": either the model is not from the
    hub, or the revision is deliberately unpinned.
    """
    repo = asr_repo_id(cfg.parakeet_model)
    revision = (cfg.parakeet_revision or "").strip()
    if not repo or not revision:
        if repo and not revision:
            log.warning("The weights revision of %s is unpinned — the current main "
                        "will be downloaded", repo)
        return None

    from huggingface_hub import snapshot_download

    path = snapshot_download(repo_id=repo, revision=revision)
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
