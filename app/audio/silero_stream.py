"""Silero VAD v5 through onnxruntime — frame-by-frame inference for the streaming VAD.

Why not `faster_whisper.vad`, where `vad.py` gets its Silero from: only the batch
`get_speech_timestamps` over a finished buffer is available there, while the
state machine needs a probability per frame as the audio arrives. The model
(~2.3 MB, MIT) sits next door in `data/`, and onnxruntime is in the environment
anyway.

Provenance of the file `data/silero_vad.onnx`:
https://github.com/snakers4/silero-vad -> `src/silero_vad/data/silero_vad.onnx`
sha256 1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from .speech_events import FRAME_SAMPLES, SAMPLE_RATE, SpeechStateMachine, _FramedProcessor

log = logging.getLogger("ilh.vad")

MODEL_PATH = Path(__file__).parent / "data" / "silero_vad.onnx"

# Before each frame the v5 model expects the trailing samples of the previous
# one — that way its convolutional input sees a continuous signal across the
# frame seam. This is exactly what the official silero-vad wrapper does. It costs
# one np.concatenate per frame, so departing from the reference behaviour to save
# that is pointless.
CONTEXT_SAMPLES = 64


class SileroStreamProcessor(_FramedProcessor):
    def __init__(self, machine: SpeechStateMachine, model_path: Path | None = None):
        super().__init__(machine)
        import onnxruntime as ort

        path = Path(model_path) if model_path else MODEL_PATH
        if not path.exists():
            raise FileNotFoundError(f"The Silero VAD model was not found: {path}")
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        # The graph is tiny: any accelerator here costs more than the computation.
        self._sess = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
        self._sr = np.array(SAMPLE_RATE, dtype=np.int64)
        self._reset_model_state()

    def reset(self) -> None:
        super().reset()
        self._reset_model_state()

    def _reset_model_state(self) -> None:
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, CONTEXT_SAMPLES), dtype=np.float32)

    def _prob(self, frame: np.ndarray) -> float:
        window = np.concatenate([self._context, frame.reshape(1, FRAME_SAMPLES)], axis=1)
        prob, self._state = self._sess.run(
            None, {"input": window, "state": self._state, "sr": self._sr}
        )
        self._context = window[:, -CONTEXT_SAMPLES:]
        return float(prob[0, 0])
