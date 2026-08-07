"""Замер скорости ASR-бэкендов на одном файле.

Отвечает на вопрос «успевает ли распознавание за речью»: печатает RTF —
во сколько раз декодирование быстрее реального времени (RTF 0.1 = секунда
речи за 0.1 c). Для живого транскрипта нужен RTF заметно меньше 1; на глаз
разница между Parakeet и Whisper видна сразу.

    python -m tools.bench_asr запись.wav
    python -m tools.bench_asr запись.wav --backends parakeet faster
    python -m tools.bench_asr запись.wav --chunk-s 8   # как в реальной нарезке

Аудио берётся любым WAV; моно 16 kHz используется как есть, остальное
приводится (нужен soxr, он и так в зависимостях).
"""
from __future__ import annotations

import argparse
import sys
import time
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.asr.factory import create_asr_backend  # noqa: E402
from app.config import AppConfig  # noqa: E402
from app.logging_setup import configure  # noqa: E402

SR = 16000
ALL_BACKENDS = ["parakeet", "mlx", "faster", "faster-cpu"]


def load_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as wf:
        if wf.getsampwidth() != 2:
            raise SystemExit(f"{path}: поддерживается только 16-битный PCM WAV")
        channels, rate = wf.getnchannels(), wf.getframerate()
        raw = wf.readframes(wf.getnframes())
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    if rate != SR:
        import soxr

        audio = soxr.resample(audio, rate, SR).astype(np.float32)
    return np.ascontiguousarray(audio, dtype=np.float32)


def split(audio: np.ndarray, chunk_s: float) -> list[np.ndarray]:
    """Резать на куски по chunk_s, чтобы мерить в том же режиме, в каком
    бэкенд работает в сессии: короткие реплики, а не файл целиком."""
    if chunk_s <= 0:
        return [audio]
    step = int(chunk_s * SR)
    return [audio[i : i + step] for i in range(0, len(audio), step)]


def bench(name: str, cfg: AppConfig, chunks: list[np.ndarray], total_s: float) -> None:
    import dataclasses

    asr_cfg = dataclasses.replace(cfg.asr, backend=name)
    print(f"\n=== {name} ===")
    try:
        backend = create_asr_backend(asr_cfg)
        t = time.monotonic()
        backend.load()
        print(f"загрузка: {time.monotonic() - t:.1f} c — {backend.describe()}")
    except Exception as e:
        print(f"недоступен: {e}")
        return
    for w in backend.warnings():
        print(f"внимание: {w}")

    parts, spent = [], 0.0
    for chunk in chunks:
        t = time.monotonic()
        result = backend.transcribe(chunk)
        spent += time.monotonic() - t
        if result.text.strip():
            parts.append(result.text.strip())

    line = f"декодирование: {spent:.2f} c на {total_s:.1f} c аудио"
    if total_s > 0 and spent > 0:
        rtf = spent / total_s
        line += f" — RTF {rtf:.3f} ({1 / rtf:.0f}x реального времени)"
    print(line)
    print(f"среднее на чанк: {spent / max(len(chunks), 1):.2f} c")
    print("текст:", " ".join(parts)[:600] or "(пусто)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Сравнение скорости ASR-бэкендов")
    ap.add_argument("wav", type=Path)
    ap.add_argument("--backends", nargs="+", default=ALL_BACKENDS,
                    help=f"по умолчанию все: {' '.join(ALL_BACKENDS)}")
    ap.add_argument("--chunk-s", type=float, default=8.0,
                    help="длина куска в секундах (0 — файл целиком); по умолчанию 8")
    args = ap.parse_args()

    cfg = AppConfig.load()
    configure(cfg.log_level)
    audio = load_wav(args.wav)
    chunks = split(audio, args.chunk_s)
    total_s = len(audio) / SR
    print(f"{args.wav}: {total_s:.1f} c аудио, {len(chunks)} кусков")

    for name in args.backends:
        bench(name, cfg, chunks, total_s)


if __name__ == "__main__":
    main()
