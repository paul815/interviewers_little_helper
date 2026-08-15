"""Pre-fetching the ASR weights so the first session start does not wait on a download.

Parakeet in int8 is about 670 MB from HuggingFace. Without this step the first
interview begins with a multi-minute pause for the download, right at the moment
the respondent is already talking.

    python -m tools.fetch_asr_model               # the model from config.json
    python -m tools.fetch_asr_model --model gigaam-v2-rnnt
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.asr.weights import load_pinned_model  # noqa: E402
from app.config import AppConfig  # noqa: E402


def main() -> int:
    cfg = AppConfig.load()
    ap = argparse.ArgumentParser(description="Download the ASR weights up front")
    ap.add_argument("--model", default=cfg.asr.parakeet_model)
    ap.add_argument("--quantization", default=cfg.asr.parakeet_quantization)
    args = ap.parse_args()

    try:
        import onnx_asr  # noqa: F401 — check it is there before the long download
    except ImportError:
        print("onnx-asr is not installed: pip install -r requirements-common.txt", file=sys.stderr)
        return 1

    # The flags override the config, but the revision stays pinned (asr/weights.py).
    cfg.asr.parakeet_model = args.model
    cfg.asr.parakeet_quantization = args.quantization

    print(f"Downloading {args.model} ({args.quantization})… this may take a few minutes")
    t = time.monotonic()
    try:
        load_pinned_model(cfg.asr)
    except Exception as e:
        print(f"Failed: {e}", file=sys.stderr)
        return 1
    print(f"Done in {time.monotonic() - t:.0f} s — from now on the model comes "
          "from the cache, offline")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
