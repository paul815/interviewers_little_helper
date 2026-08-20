"""Conditioning of the audio on its way from capture to the VAD and to ASR.

None of this reaches the session recording: audio/recorder.py taps the stream
in ChannelCapture, before the chunker, and keeps whatever the devices gave us.

Two steps, deliberately different in kind:

- `HighPass` runs on the whole stream, ahead of the VAD, so the speech detector
  and the model see the same signal. It takes out the DC offset some USB
  microphones carry and the rumble a room supplies for free — fans, air
  conditioning, a knock on the desk, a chair. That energy sits below the voice,
  adds nothing a model can recognise, and does inflate the frame energy the VAD
  measures.
- `normalise_for_asr` runs on a finished chunk, and only lifts quiet ones. A
  streaming AGC would have to keep adapting through silence, which walks the
  noise floor up into the VAD's face; a chunk is a whole utterance, so it can be
  scaled once, by its own RMS, with no state to drift. The gain is never applied
  downwards: attenuation cannot un-clip a channel that was recorded too hot, and
  the mel front-ends of both Parakeet and Whisper cope with loud audio by
  themselves. The case worth fixing is the built-in laptop microphone two feet
  away from the speaker, which arrives 20-30 dB below a headset.

The filter is a centred moving average subtracted from the signal, not a
biquad: it is linear phase (a biquad would smear the utterance boundaries the
VAD has just measured), it costs two cumsums per block instead of a per-sample
Python loop, and numpy does all of it without scipy in the dependency list.
"""
from __future__ import annotations

import numpy as np

# Below this a human voice has nothing to say: a male fundamental starts around
# 85 Hz. Anything lower is the room, the desk or the microphone's own DC.
DEFAULT_HIGHPASS_HZ = 80.0

# The -3 dB point of a length-N moving average is at ~0.443*sr/N; the window
# length is derived from the requested cutoff through it.
_MA_CUTOFF_FACTOR = 0.443

TARGET_RMS = 0.05    # ~ -26 dBFS, where the training corpora of both engines sit
MAX_GAIN = 8.0       # +18 dB. Past this we would be amplifying the room, not the voice
PEAK_CEILING = 0.97  # leave the ceiling alone: gain must not create clipping
SILENT_RMS = 1e-4    # quieter than this there is nothing to lift, only noise
_GAIN_DEADBAND = 0.15  # a 1 dB correction is not worth copying the array for


def rms(audio: np.ndarray) -> float:
    """Root mean square of a block, in the 0..1 scale of float32 audio."""
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))


def window_length(cutoff_hz: float, sample_rate: int) -> int:
    """The odd moving-average length whose -3 dB point lands on `cutoff_hz`."""
    n = int(round(_MA_CUTOFF_FACTOR * sample_rate / max(cutoff_hz, 1e-6)))
    n = max(n, 3)
    return n if n % 2 else n + 1


class HighPass:
    """A linear-phase high-pass, stateful across blocks.

    `process()` returns exactly as many samples as it was given, delayed by half
    the window (2.8 ms at the default cutoff) — the look-ahead a centred window
    needs. Sample counts stay one to one, so every position the chunker counts
    in stays where it was; only the content shifts, by a fraction of one VAD
    frame.
    """

    def __init__(self, cutoff_hz: float, sample_rate: int):
        self.cutoff_hz = cutoff_hz
        self.sample_rate = sample_rate
        self.taps = window_length(cutoff_hz, sample_rate)
        self._half = (self.taps - 1) // 2
        self._tail = np.zeros(self.taps - 1, dtype=np.float32)

    def reset(self) -> None:
        self._tail = np.zeros(self.taps - 1, dtype=np.float32)

    def process(self, block: np.ndarray) -> np.ndarray:
        if block.size == 0:
            return block
        buf = np.concatenate([self._tail, np.asarray(block, dtype=np.float32)])
        # Trailing average over `taps` samples, for every position that has a
        # full window behind it — one value per input sample.
        cum = np.cumsum(buf, dtype=np.float64)
        cum = np.concatenate([[0.0], cum])
        window_sum = cum[self.taps:] - cum[: -self.taps]
        moving_avg = window_sum / self.taps
        # The trailing average ending at i is the centred average at i - half.
        centres = buf[self.taps - 1 - self._half : buf.size - self._half]
        out = (centres - moving_avg).astype(np.float32)
        self._tail = buf[-(self.taps - 1):]
        return out


def create_highpass(cutoff_hz: float, sample_rate: int) -> HighPass | None:
    """None when filtering is switched off — the caller then does nothing at all."""
    if cutoff_hz <= 0:
        return None
    if cutoff_hz >= sample_rate / 2:
        raise ValueError(f"highpass_hz {cutoff_hz} is above the Nyquist of {sample_rate} Hz")
    return HighPass(cutoff_hz, sample_rate)


def normalise_for_asr(
    audio: np.ndarray,
    target_rms: float = TARGET_RMS,
    level: float | None = None,
) -> np.ndarray:
    """Lift a quiet utterance towards `target_rms`; return the input untouched
    when it is loud enough, silent, or would clip.

    `level` is the chunk's RMS if the caller has already measured it.
    `target_rms <= 0` switches the whole step off.
    """
    if target_rms <= 0 or audio.size == 0:
        return audio
    current = rms(audio) if level is None else level
    if current < SILENT_RMS:
        return audio
    gain = min(target_rms / current, MAX_GAIN)
    if gain <= 1.0 + _GAIN_DEADBAND:
        return audio
    peak = float(np.max(np.abs(audio)))
    if peak > 0:
        gain = min(gain, PEAK_CEILING / peak)
    if gain <= 1.0 + _GAIN_DEADBAND:
        return audio
    return (audio * gain).astype(np.float32)
