"""Measuring the speed of the ASR backends on one file.

Answers the question "does recognition keep up with speech": it prints the RTF —
how many times faster than real time the decoding is (RTF 0.1 = a second of
speech in 0.1 s). A live transcript needs an RTF well below 1; the difference
between Parakeet and Whisper is obvious at a glance.

    python -m tools.bench_asr recording.wav
    python -m tools.bench_asr recording.wav --backends parakeet faster
    python -m tools.bench_asr recording.wav --chunk-s 8   # as in the real chunking

Any WAV will do; mono 16 kHz is used as is, anything else is converted (soxr is
needed, and it is in the dependencies anyway).
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
            raise SystemExit(f"{path}: only 16-bit PCM WAV is supported")
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
    """Cut into chunk_s pieces so the measurement runs in the same mode the
    backend works in during a session: short utterances, not the whole file."""
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
        print(f"load: {time.monotonic() - t:.1f} s — {backend.describe()}")
    except Exception as e:
        print(f"unavailable: {e}")
        return
    for w in backend.warnings():
        print(f"note: {w}")

    parts, spent = [], 0.0
    for chunk in chunks:
        t = time.monotonic()
        result = backend.transcribe(chunk)
        spent += time.monotonic() - t
        if result.text.strip():
            parts.append(result.text.strip())

    line = f"decoding: {spent:.2f} s for {total_s:.1f} s of audio"
    if total_s > 0 and spent > 0:
        rtf = spent / total_s
        line += f" — RTF {rtf:.3f} ({1 / rtf:.0f}x real time)"
    print(line)
    print(f"average per chunk: {spent / max(len(chunks), 1):.2f} s")
    print("text:", " ".join(parts)[:600] or "(empty)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Comparing the speed of the ASR backends")
    ap.add_argument("wav", type=Path)
    ap.add_argument("--backends", nargs="+", default=ALL_BACKENDS,
                    help=f"all of them by default: {' '.join(ALL_BACKENDS)}")
    ap.add_argument("--chunk-s", type=float, default=8.0,
                    help="chunk length in seconds (0 — the whole file); 8 by default")
    args = ap.parse_args()

    cfg = AppConfig.load()
    configure(cfg.log_level)
    audio = load_wav(args.wav)
    chunks = split(audio, args.chunk_s)
    total_s = len(audio) / SR
    print(f"{args.wav}: {total_s:.1f} s of audio, {len(chunks)} chunks")

    for name in args.backends:
        bench(name, cfg, chunks, total_s)


if __name__ == "__main__":
    main()
