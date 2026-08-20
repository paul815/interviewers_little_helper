"""Downloading the missing models from the application window, not the console.

The first run must not run into a terminal: the application sees for itself what
is missing — the ASR weights or the LLM model — downloads them and shows the
progress. Exactly what the first-run wizard in a desktop dictation app does.

On the accuracy of the progress. Ollama has a streaming /api/pull that reports
completed/total in bytes — the percentages there are honest. HuggingFace, on the
other hand, is downloaded through onnx-asr, which decides for itself which files
of the repository it needs for the chosen quantisation; poking into that logic
for the sake of percentages means depending on someone else's internals. So for
the ASR weights we show the megabytes downloaded: the size of the repository
folder in the cache grows live, including partially fetched .incomplete blobs.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path

import httpx

from .asr import router
from .asr.catalog import spec
from .asr.weights import asr_repo_id as _asr_repo_id
from .asr.weights import load_pinned_model
from .config import AppConfig, ASRConfig

log = logging.getLogger("ilh.provision")

# Below this size the repository folder is the residue of a failed attempt, not a model.
_ASR_MIN_BYTES = 32 * 1024 * 1024


def asr_repo_id(model_name: str) -> str | None:
    """An onnx-asr model name -> a HuggingFace repository. None for local paths.

    A wrapper over app/asr/weights.py: the name is kept for the calling code and the tests.
    """
    return _asr_repo_id(model_name)


def _repo_dir(repo_id: str) -> Path | None:
    try:
        from huggingface_hub.constants import HF_HUB_CACHE
    except ImportError:
        return None
    return Path(HF_HUB_CACHE) / ("models--" + repo_id.replace("/", "--"))


def _dir_bytes(path: Path) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.stat(os.path.join(root, name)).st_size
            except OSError:
                continue
    return total


def asr_cached_bytes(repo_id: str) -> int:
    """How much already sits in the HF cache. Grows as the download proceeds."""
    path = _repo_dir(repo_id)
    if path is None or not path.exists():
        return 0
    return _dir_bytes(path)


def asr_ready(repo_id: str | None) -> bool:
    """Whether the weights look as though they have already been downloaded.

    The estimate is deliberately crude: the cost of being wrong is small either
    way. A false "no" and a repeat load_model finishes in seconds off the cache.
    A false "yes" and the behaviour is what it was before the wizard existed:
    whatever is missing is fetched at startup.
    """
    if repo_id is None:
        return True  # a local path or an unfamiliar model — we do not presume to judge
    path = _repo_dir(repo_id)
    if path is None or not path.exists():
        return False
    blobs = path / "blobs"
    if blobs.exists() and any(f.suffix == ".incomplete" for f in blobs.iterdir()):
        return False
    return _dir_bytes(path) >= _ASR_MIN_BYTES


def language_options() -> list[dict]:
    """Every interview language the catalogue covers, the model behind each one,
    and whether its weights are already on disk.

    The start screen needs that last flag. The ASR model is loaded when a session
    starts, so choosing a language whose weights are missing would open the
    interview with a several-hundred-megabyte download — with the respondent
    already talking. Better to find out while the Start button is still grey.
    """
    return [
        {
            "code": code,
            "model": model,
            "summary": entry.summary if (entry := spec(model)) else model,
            "download_mb": entry.download_mb if entry else None,
            "ready": asr_ready(asr_repo_id(model)),
        }
        for code, model in sorted(router.languages().items())
    ]


class Provisioner:
    """Background downloading, with the progress relayed to the UI."""

    ITEMS = ("asr", "llm")

    def __init__(self, cfg: AppConfig, hub, llm):
        self.cfg = cfg
        self.hub = hub
        self.llm = llm
        self._task: asyncio.Task | None = None
        # Which model the screen is currently talking about. The interview
        # language decides it, so it changes as the researcher picks one.
        self._asr_cfg: ASRConfig = cfg.asr
        self._items: dict[str, dict] = {
            "asr": self._item("Speech recognition model"),
            "llm": self._item(f"Language model {cfg.llm.model}"),
        }

    @staticmethod
    def _item(title: str) -> dict:
        return {"title": title, "state": "unknown", "done_bytes": 0,
                "total_bytes": None, "message": ""}

    # ----------------------------------------------------------------- state

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    def snapshot(self) -> dict:
        return {"running": self.running, "items": {k: dict(v) for k, v in self._items.items()}}

    def _set(self, key: str, **fields) -> None:
        self._items[key].update(fields)
        self.hub.broadcast_threadsafe("setup_progress", self.snapshot())

    def _use_language(self, language: str | None) -> ASRConfig:
        """Point the ASR item at the model that language would be recognised with."""
        self._asr_cfg = router.route(self.cfg.asr, language)
        return self._asr_cfg

    async def refresh(self, language: str | None = None) -> dict:
        """What is there and what is not. Downloads nothing, only looks."""
        asr_cfg = self._use_language(language)
        entry = spec(asr_cfg.parakeet_model)
        repo = asr_repo_id(asr_cfg.parakeet_model)
        self._items["asr"]["title"] = (
            entry.summary if entry else f"Speech recognition model {asr_cfg.parakeet_model}"
        )
        if asr_ready(repo):
            self._items["asr"].update(state="ok", message="downloaded", done_bytes=0)
        else:
            self._items["asr"].update(
                state="missing",
                # An unlisted model has no measured size — say so rather than
                # quote the default model's, which is what used to happen.
                message=f"about {entry.download_mb} MB" if entry else "size unknown",
                done_bytes=0,
            )

        st = await self.llm.check()
        if st.get("model_found"):
            self._items["llm"].update(state="ok", message="downloaded")
        elif st.get("server_up"):
            self._items["llm"].update(state="missing", message="a few GB")
        else:
            # There is nowhere to download the model to: the server is not there.
            # This is the one thing the application cannot fix by itself — Ollama
            # is installed into the system.
            self._items["llm"].update(
                state="blocked",
                message="Ollama is not running — install it from ollama.com and open it",
            )
        return self.snapshot()

    # ------------------------------------------------------------ downloading

    def start(self, items: list[str] | None = None, language: str | None = None) -> dict:
        if self.running:
            return self.snapshot()
        if language is not None:
            self._use_language(language)
        wanted = [k for k in (items or self.ITEMS) if k in self._items]
        self._task = asyncio.create_task(self._run(wanted))
        return self.snapshot()

    async def _run(self, wanted: list[str]) -> None:
        for key in wanted:
            if self._items[key]["state"] in ("ok", "blocked"):
                continue
            try:
                if key == "asr":
                    await self._fetch_asr()
                elif key == "llm":
                    await self._pull_llm()
            except asyncio.CancelledError:
                self._set(key, state="error", message="cancelled")
                raise
            except Exception as e:
                log.exception("Could not download «%s»", key)
                self._set(key, state="error", message=str(e))
        self.hub.broadcast_threadsafe("setup_progress", self.snapshot())

    async def _fetch_asr(self) -> None:
        repo = asr_repo_id(self._asr_cfg.parakeet_model)
        self._set("asr", state="downloading", message="downloading…", done_bytes=0)

        started = asr_cached_bytes(repo) if repo else 0
        done = asyncio.Event()

        async def report() -> None:
            while not done.is_set():
                if repo:
                    grown = max(0, asr_cached_bytes(repo) - started)
                    self._set("asr", done_bytes=grown)
                try:
                    await asyncio.wait_for(done.wait(), timeout=1.0)
                except TimeoutError:
                    pass

        reporter = asyncio.create_task(report())
        try:
            await asyncio.to_thread(self._load_asr_blocking)
        finally:
            done.set()
            await reporter
        self._set("asr", state="ok", message="downloaded")

    def _load_asr_blocking(self) -> None:
        load_pinned_model(self._asr_cfg)

    async def _pull_llm(self) -> None:
        model = self.cfg.llm.model
        self._set("llm", state="downloading", message="downloading…", done_bytes=0)
        url = f"{self.cfg.llm.base_url}/api/pull"
        async with httpx.AsyncClient(timeout=httpx.Timeout(None, connect=10.0)) as client:
            async with client.stream("POST", url, json={"model": model, "stream": True}) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        msg = json.loads(line)
                    except ValueError:
                        continue
                    if msg.get("error"):
                        raise RuntimeError(msg["error"])
                    self._set(
                        "llm",
                        done_bytes=int(msg.get("completed") or 0),
                        total_bytes=int(msg["total"]) if msg.get("total") else None,
                        message=str(msg.get("status") or ""),
                    )
        self._set("llm", state="ok", message="downloaded")
