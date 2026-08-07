"""Предзагрузка весов ASR, чтобы первый старт сессии не ждал скачивания.

Parakeet в int8 — около 670 МБ с HuggingFace. Без этого шага первое интервью
начнётся с многоминутной паузы на загрузку прямо в момент, когда респондент
уже говорит.

    python -m tools.fetch_asr_model               # модель из config.json
    python -m tools.fetch_asr_model --model gigaam-v2-rnnt
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import AppConfig  # noqa: E402


def main() -> int:
    cfg = AppConfig.load()
    ap = argparse.ArgumentParser(description="Скачать веса ASR заранее")
    ap.add_argument("--model", default=cfg.asr.parakeet_model)
    ap.add_argument("--quantization", default=cfg.asr.parakeet_quantization)
    args = ap.parse_args()

    try:
        import onnx_asr
    except ImportError:
        print("onnx-asr не установлен: pip install -r requirements-common.txt", file=sys.stderr)
        return 1

    print(f"Загружаю {args.model} ({args.quantization})… это может занять несколько минут")
    t = time.monotonic()
    try:
        onnx_asr.load_model(
            args.model,
            quantization=args.quantization or None,
            providers=list(cfg.asr.providers),
        )
    except Exception as e:
        print(f"Не удалось: {e}", file=sys.stderr)
        return 1
    print(f"Готово за {time.monotonic() - t:.0f} c — дальше модель берётся из кэша офлайн")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
