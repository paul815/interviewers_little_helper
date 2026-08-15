"""All application configuration in one place.

Defaults live here; local overrides go in config.json at the repo root
(see config.example.json). Unknown keys are ignored with a warning.
"""
from __future__ import annotations

import dataclasses
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
log = logging.getLogger("ilh.config")


@dataclass
class ServerConfig:
    host: str = "127.0.0.1"  # loopback only: nothing is exposed outside
    port: int = 8756


@dataclass
class AnalysisConfig:
    interval_s: int = 240
    max_recommendations: int = 6
    max_probes: int = 2           # probe questions per cycle (0 = disable)
    # Every Nth cycle re-checks the window since the last reconcile (0 = off).
    reconcile_every: int = 4
    final_sweep: bool = True      # full-transcript sweep on "Stop"
    report: bool = True           # build report.md after the session
    default_duration_min: int = 60
    # Language of the analysis output — recommendations, evidence, findings and
    # the report summary. "auto" follows the guide's own language (as detected
    # when the guide was parsed); anything else is passed to the model verbatim,
    # so both an ISO 639-1 code ("en", "ru") and a plain name ("English") work.
    output_language: str = "auto"


@dataclass
class AudioConfig:
    sample_rate: int = 16000
    ring_seconds: float = 120.0
    # How often the chunker pulls audio out of the ring buffer. With streaming
    # VAD this is also the granularity at which the end of an utterance is
    # noticed at all — a value well above vad_redemption_s would eat the entire
    # latency win.
    poll_interval_s: float = 0.2
    max_chunk_s: float = 25.0
    min_pause_s: float = 0.7   # batch chunking only (vad: silero | energy)
    min_speech_s: float = 0.3  # batch chunking only
    pad_s: float = 0.2         # post-pad: trailing consonants and breath
    pre_pad_s: float = 0.3     # VAD always fires after the first syllable
    # auto | silero-stream | energy-stream — streaming chunking driven by speech
    # events; silero | energy — the original batch chunking by pauses (fallback).
    vad: str = "auto"
    vad_positive_threshold: float = 0.50  # enter speech
    vad_negative_threshold: float = 0.35  # ...harder than it is to stay in it
    vad_min_speech_s: float = 0.25        # shorter than this is a click, not an utterance
    vad_redemption_s: float = 0.6         # this much silence ends an utterance
    watchdog_silence_s: float = 12.0  # no samples for longer — the channel counts as dead


@dataclass
class ASRConfig:
    # auto | parakeet | mlx | faster | faster-cpu
    backend: str = "auto"
    model: str = "large-v3-turbo"
    mlx_model: str = "mlx-community/whisper-large-v3-turbo"
    # Model for the parakeet backend (onnx-asr). The same key can point at the
    # Russian gigaam-v2-rnnt — at the cost of mixed RU/EN speech support.
    parakeet_model: str = "nemo-parakeet-tdt-0.6b-v3"
    parakeet_quantization: str = "int8"
    # Commit of the weights repository on HuggingFace. Without it the current
    # main is downloaded, i.e. the model contents change without our knowledge.
    # An empty string unpins it (see app/asr/weights.py).
    parakeet_revision: str = "8f23f0c03c8761650bdb5b40aaf3e40d2c15f1ce"
    # CPU on purpose: int8 Parakeet copes on CPU with room to spare, and all the
    # VRAM stays with the LLM. CUDAExecutionProvider needs onnxruntime-gpu, which
    # conflicts with the CPU build of onnxruntime (pulled in by faster-whisper).
    providers: list[str] = field(default_factory=lambda: ["CPUExecutionProvider"])
    compute_type: str = "auto"
    beam_size: int = 1
    language: str | None = None  # None = auto-detect per chunk
    drop_no_speech_prob: float = 0.85
    # Project terms (brands, jargon, names) separated by commas — a hint for
    # Whisper that markedly improves recognition of exactly those words.
    # Parakeet does not support prompt conditioning: the setting does nothing
    # there (the backend warns about it).
    vocabulary: str = ""


@dataclass
class LLMConfig:
    base_url: str = "http://127.0.0.1:11434"
    model: str = "qwen3:8b"
    num_ctx: int = 8192
    temperature: float = 0.2
    keep_alive: int | str = -1
    request_timeout_s: float = 300.0
    max_delta_chars: int = 12000
    num_predict: int = 2048


@dataclass
class StorageConfig:
    # Sessions outside a project (including ones recorded before projects existed).
    sessions_dir: str = "sessions"
    guides_dir: str = "guides"
    projects_dir: str = "projects"
    # Prompt edits made in "Settings" (see app/storage/prompt_store.py).
    prompts_file: str = "prompts.json"
    # Stereo recording of the interview into audio.wav next to the transcript
    # (~230 MB/hour). Turn it off if you don't need the recording: the
    # transcript and the report do not depend on it.
    save_audio: bool = True


@dataclass
class WindowConfig:
    """Native window. The working pose is Zoom on the left, the helper on the
    right, so by default the window takes the right half of the screen at full
    height."""

    # The half is measured against the real monitor; width/height are the
    # fallback if the screen cannot be queried (sized for 2560x1440).
    half_screen: bool = True
    width: int = 1280
    height: int = 1440
    # Below this the interface starts to break: the two header columns collapse.
    min_width: int = 420
    min_height: int = 300
    resizable: bool = True
    on_top: bool = True
    # The size the user stretched the window to survives a restart.
    remember_size: bool = True


@dataclass
class SecurityConfig:
    """Who is allowed to knock on the local server.

    Binding to 127.0.0.1 is not enough. First, WebSocket does not obey CORS: any
    site open in the browser during the interview can connect to /ws and read
    the transcript in real time. Second, DNS rebinding substitutes its own
    domain, which resolves to 127.0.0.1, and bypasses the binding entirely —
    only a Host header check defends against that.
    """

    allowed_hosts: list[str] = field(
        default_factory=lambda: ["127.0.0.1", "localhost", "::1"]
    )
    # Requests without Origin (curl, scripts) are allowed through by default:
    # a browser always sends Origin, and a local process reads projects/
    # straight off the disk anyway.
    require_origin: bool = False


@dataclass
class AppConfig:
    server: ServerConfig = field(default_factory=ServerConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    asr: ASRConfig = field(default_factory=ASRConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    window: WindowConfig = field(default_factory=WindowConfig)
    log_level: str = "INFO"

    @property
    def sessions_path(self) -> Path:
        p = Path(self.storage.sessions_dir)
        return p if p.is_absolute() else ROOT / p

    @property
    def guides_path(self) -> Path:
        p = Path(self.storage.guides_dir)
        return p if p.is_absolute() else ROOT / p

    @property
    def projects_path(self) -> Path:
        p = Path(self.storage.projects_dir)
        return p if p.is_absolute() else ROOT / p

    @property
    def window_state_path(self) -> Path:
        """Remembered window geometry — not config, a human does not edit it."""
        return ROOT / ".window.json"

    @property
    def prompts_path(self) -> Path:
        """Prompt edits from "Settings". Kept apart from config.json: this file
        is written by the application, while config.json is written by a human,
        and rewriting it is not our business."""
        p = Path(self.storage.prompts_file)
        return p if p.is_absolute() else ROOT / p

    @staticmethod
    def load(path: Path | None = None) -> AppConfig:
        cfg = AppConfig()
        cfg_path = path or ROOT / "config.json"
        if cfg_path.exists():
            try:
                data = json.loads(cfg_path.read_text(encoding="utf-8"))
                _apply_overrides(cfg, data, source=str(cfg_path))
                log.info("Config loaded from %s", cfg_path)
            except (OSError, json.JSONDecodeError) as e:
                log.error("Could not read %s: %s — falling back to defaults", cfg_path, e)
        return cfg


def _apply_overrides(obj, data: dict, source: str, prefix: str = "") -> None:
    for key, val in data.items():
        if not hasattr(obj, key):
            log.warning("Unknown config key %s%s in %s — ignoring", prefix, key, source)
            continue
        cur = getattr(obj, key)
        if dataclasses.is_dataclass(cur) and isinstance(val, dict):
            _apply_overrides(cur, val, source, prefix=f"{prefix}{key}.")
        else:
            setattr(obj, key, val)
