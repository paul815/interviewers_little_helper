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
    max_probes: int = 2           # пробинг-вопросов за цикл (0 = выключить)
    reconcile_every: int = 4      # каждый N-й цикл — сверка окна с прошлой сверки (0 = выкл)
    final_sweep: bool = True      # финальный проход по всему транскрипту на «Стоп»
    report: bool = True           # собирать report.md после сессии
    default_duration_min: int = 60


@dataclass
class AudioConfig:
    sample_rate: int = 16000
    ring_seconds: float = 120.0
    # Как часто чанкер забирает аудио из кольца. При потоковом VAD это ещё и
    # гранулярность, с которой вообще замечается конец реплики, — значение
    # заметно больше vad_redemption_s съело бы весь выигрыш в задержке.
    poll_interval_s: float = 0.2
    max_chunk_s: float = 25.0
    min_pause_s: float = 0.7   # только для батчевой нарезки (vad: silero | energy)
    min_speech_s: float = 0.3  # только для батчевой нарезки
    pad_s: float = 0.2         # post-pad: хвостовые согласные и дыхание
    pre_pad_s: float = 0.3     # VAD всегда срабатывает позже первого слога
    # auto | silero-stream | energy-stream — потоковая нарезка по событиям речи;
    # silero | energy — исходная батчевая нарезка по паузам (запасной путь).
    vad: str = "auto"
    vad_positive_threshold: float = 0.50  # войти в речь
    vad_negative_threshold: float = 0.35  # ...труднее, чем в ней остаться
    vad_min_speech_s: float = 0.25        # короче — щелчок, а не реплика
    vad_redemption_s: float = 0.6         # столько тишины = конец реплики
    watchdog_silence_s: float = 12.0  # нет сэмплов дольше — канал считается умершим


@dataclass
class ASRConfig:
    # auto | parakeet | mlx | faster | faster-cpu
    backend: str = "auto"
    model: str = "large-v3-turbo"
    mlx_model: str = "mlx-community/whisper-large-v3-turbo"
    # Модель для бэкенда parakeet (onnx-asr). Тем же ключом можно взять
    # русскоязычную gigaam-v2-rnnt — ценой поддержки смешанной RU/EN речи.
    parakeet_model: str = "nemo-parakeet-tdt-0.6b-v3"
    parakeet_quantization: str = "int8"
    # CPU намеренно: int8-Parakeet на CPU справляется с запасом, а вся VRAM
    # остаётся LLM. CUDAExecutionProvider требует onnxruntime-gpu, который
    # конфликтует с CPU-сборкой onnxruntime (её тянет faster-whisper).
    providers: list[str] = field(default_factory=lambda: ["CPUExecutionProvider"])
    compute_type: str = "auto"
    beam_size: int = 1
    language: str | None = None  # None = автоопределение на каждый чанк
    drop_no_speech_prob: float = 0.85
    # Термины проекта (бренды, жаргон, имена) через запятую — подсказка Whisper,
    # заметно улучшает распознавание именно этих слов. Parakeet conditioning
    # промптом не поддерживает: с ним настройка не работает (бэкенд предупредит).
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
    sessions_dir: str = "sessions"
    guides_dir: str = "guides"
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

    @property
    def guides_path(self) -> Path:
        p = Path(self.storage.guides_dir)
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
