"""Recording the session audio track: the file format and channel alignment.

No hardware needed — the blocks are handed to SessionRecorder directly, the way
ChannelCapture does from the audio callback.
"""
from __future__ import annotations

import wave

import numpy as np
import pytest

from app.audio.capture import ChannelCapture
from app.audio.recorder import SessionRecorder, WavWriter
from app.config import AudioConfig
from app.domain import Speaker

SR = 16000


def _read(path):
    """WAV -> (interviewer frames, respondent frames) in float32."""
    with wave.open(str(path), "rb") as wf:
        assert wf.getnchannels() == 2
        assert wf.getsampwidth() == 2
        assert wf.getframerate() == SR
        raw = wf.readframes(wf.getnframes())
    data = np.frombuffer(raw, dtype="<i2").astype(np.float32).reshape(-1, 2) / 32767.0
    return data[:, 0], data[:, 1]


def _rec(tmp_path, **kw) -> SessionRecorder:
    rec = SessionRecorder(tmp_path / "audio.wav", SR, **kw)
    rec.start()
    return rec


def _tone(n: int, value: float = 0.5) -> np.ndarray:
    return np.full(n, value, dtype=np.float32)


def test_channels_land_in_own_wav_channels(tmp_path):
    rec = _rec(tmp_path)
    rec.submit(Speaker.INTERVIEWER, _tone(SR, 0.5))
    rec.submit(Speaker.RESPONDENT, _tone(SR, -0.25))
    info = rec.stop()

    left, right = _read(tmp_path / "audio.wav")
    assert len(left) == SR
    assert np.allclose(left, 0.5, atol=1e-4)
    assert np.allclose(right, -0.25, atol=1e-4)
    assert info["audio_seconds"] == 1.0
    assert info["audio_file"] == "audio.wav"


def test_blocks_are_concatenated_in_order(tmp_path):
    rec = _rec(tmp_path)
    for i in range(5):
        rec.submit(Speaker.INTERVIEWER, _tone(100, i / 10))
        rec.submit(Speaker.RESPONDENT, _tone(100, -i / 10))
    rec.stop()

    left, _ = _read(tmp_path / "audio.wav")
    assert len(left) == 500
    for i in range(5):
        assert np.allclose(left[i * 100:(i + 1) * 100], i / 10, atol=1e-4)


def test_silent_channel_is_padded_not_stretched(tmp_path):
    """A device that dropped out must not shift the other track."""
    rec = _rec(tmp_path, stall_s=0.0)  # a channel with no blocks counts as silent at once
    rec.submit(Speaker.INTERVIEWER, _tone(SR, 0.5))
    info = rec.stop()

    left, right = _read(tmp_path / "audio.wav")
    assert len(left) == len(right) == SR
    assert np.allclose(left, 0.5, atol=1e-4)
    assert np.all(right == 0.0)
    assert info["audio_silence_padded_s"] == {"respondent": 1.0}


def test_late_channel_starts_at_its_own_position(tmp_path):
    """A channel that started later lands in its own place, not at the start of the file."""
    rec = _rec(tmp_path, stall_s=0.0)
    rec.submit(Speaker.INTERVIEWER, _tone(SR, 0.5))
    rec._drain(final=False)  # the writer managed to write the first second
    rec.submit(Speaker.INTERVIEWER, _tone(SR, 0.5))
    rec.submit(Speaker.RESPONDENT, _tone(SR, -0.5))
    rec.stop()

    left, right = _read(tmp_path / "audio.wav")
    assert len(left) == len(right) == 2 * SR
    assert np.all(right[:SR] == 0.0)               # the first second is silence
    assert np.allclose(right[SR:], -0.5, atol=1e-4)  # the second is real audio
    assert np.allclose(left, 0.5, atol=1e-4)


def test_loud_signal_is_clipped_not_wrapped(tmp_path):
    """An overload must hit the ceiling rather than flip the phase."""
    rec = _rec(tmp_path)
    rec.submit(Speaker.INTERVIEWER, _tone(10, 3.0))
    rec.submit(Speaker.RESPONDENT, _tone(10, -3.0))
    rec.stop()

    left, right = _read(tmp_path / "audio.wav")
    assert np.all(left > 0.99)
    assert np.all(right < -0.99)


def test_empty_session_leaves_valid_wav(tmp_path):
    rec = _rec(tmp_path)
    info = rec.stop()

    left, right = _read(tmp_path / "audio.wav")
    assert len(left) == len(right) == 0
    assert info["audio_seconds"] == 0.0


def test_stop_is_idempotent(tmp_path):
    rec = _rec(tmp_path)
    rec.submit(Speaker.INTERVIEWER, _tone(100))
    assert rec.stop()["audio_seconds"] == pytest.approx(100 / SR, abs=0.1)
    assert rec.stop() == {}


def test_overflow_drops_audio_instead_of_memory(tmp_path):
    rec = SessionRecorder(tmp_path / "audio.wav", SR)
    rec._max_pending = 1000  # as if the writer could not keep up
    for _ in range(5):
        rec.submit(Speaker.INTERVIEWER, _tone(400))
    assert rec._available[Speaker.INTERVIEWER] <= 1000 + 400
    assert rec._dropped[Speaker.INTERVIEWER] > 0


def test_capture_forwards_the_same_audio_it_feeds_to_asr():
    """The recording tap sits where the chunker's ring is filled."""
    cap = ChannelCapture(0, Speaker.INTERVIEWER, AudioConfig())
    got: list[np.ndarray] = []
    cap.on_audio = got.append
    cap._ingest(_tone(64), False)

    assert [len(b) for b in got] == [64]
    assert len(cap.ring.pop_all()) == 64


def test_broken_recorder_does_not_kill_capture():
    cap = ChannelCapture(0, Speaker.INTERVIEWER, AudioConfig())

    def boom(_block):
        raise OSError("the disk ran out of space")

    cap.on_audio = boom
    cap._ingest(_tone(64), False)  # the audio callback must survive a recording failure

    assert cap.on_audio is None  # the tap is disabled and capture goes on
    assert cap.stats.samples == 64


def test_header_is_repaired_before_close(tmp_path):
    """After a process crash the file must open in a player."""
    path = tmp_path / "crash.wav"
    writer = WavWriter(path, SR, 2)
    writer.write(np.zeros(SR * 2, dtype="<i2").tobytes(), SR)
    writer.sync()  # the process "dies" here, close() is never called

    with wave.open(str(path), "rb") as wf:
        assert wf.getnframes() == SR
        assert wf.getnchannels() == 2
