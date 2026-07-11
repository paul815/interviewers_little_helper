"""Вся конфигурация приложения в одном месте.

Дефолты заданы здесь; локальные переопределения — в config.json в корне репо
(см. config.example.json). Неизвестные ключи игнорируются с предупреждением.
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
    host: str = "127.0.0.1"  # только loopback: наружу ничего не отдаём
    port: int = 8756


@dataclass
class AnalysisConfig:
    interval_s: int = 240
    max_recommendations: int = 6


@dataclass
class AudioConfig:
    sample_rate: int = 16000
    ring_seconds: float = 120.0
    poll_interval_s: float = 0.5
    max_chunk_s: float = 25.0
    min_pause_s: float = 0.7
    min_speech_s: float = 0.3
    pad_s: float = 0.2
    vad: str = "auto"  # auto | silero | energy


@dataclass
class ASRConfig:
    backend: str = "auto"  # auto | mlx | faster | faster-cpu
    model: str = "large-v3-turbo"
    mlx_model: str = "mlx-community/whisper-large-v3-turbo"
    compute_type: str = "auto"
    beam_size: int = 1
    language: str | None = None  # None = автоопределение на каждый чанк
    drop_no_speech_prob: float = 0.85


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
    sessions_dir: str = "sessions"
    save_audio_chunks: bool = False  # WAV-чанки для отладки


@dataclass
class AppConfig:
    server: ServerConfig = field(default_factory=ServerConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    asr: ASRConfig = field(default_factory=ASRConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    log_level: str = "INFO"

    @property
    def sessions_path(self) -> Path:
        p = Path(self.storage.sessions_dir)
        return p if p.is_absolute() else ROOT / p

    @staticmethod
    def load(path: Path | None = None) -> "AppConfig":
        cfg = AppConfig()
        cfg_path = path or ROOT / "config.json"
        if cfg_path.exists():
            try:
                data = json.loads(cfg_path.read_text(encoding="utf-8"))
                _apply_overrides(cfg, data, source=str(cfg_path))
                log.info("Конфиг загружен из %s", cfg_path)
            except (OSError, json.JSONDecodeError) as e:
                log.error("Не удалось прочитать %s: %s — использую дефолты", cfg_path, e)
        return cfg


def _apply_overrides(obj, data: dict, source: str, prefix: str = "") -> None:
    for key, val in data.items():
        if not hasattr(obj, key):
            log.warning("Неизвестный ключ конфига %s%s в %s — игнорирую", prefix, key, source)
            continue
        cur = getattr(obj, key)
        if dataclasses.is_dataclass(cur) and isinstance(val, dict):
            _apply_overrides(cur, val, source, prefix=f"{prefix}{key}.")
        else:
            setattr(obj, key, val)
