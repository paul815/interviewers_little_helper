"""Conditioning of the audio before the VAD and ASR, and the echo detector."""
from __future__ import annotations

import queue
import threading

import numpy as np

from app.audio.capture import RingBuffer
from app.audio.chunker import ChunkerThread
from app.audio.echo import EchoDetector, jaccard, tokens
from app.audio.preprocess import (
    MAX_GAIN,
    PEAK_CEILING,
    HighPass,
    create_highpass,
    normalise_for_asr,
    rms,
    window_length,
)
from app.config import AudioConfig
from app.domain import AudioChunk, Speaker

SR = 16000


def sine(freq: float, seconds: float = 1.0, amp: float = 0.3, sr: int = SR) -> np.ndarray:
    t = np.arange(int(seconds * sr), dtype=np.float32) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def steady_amplitude(x: np.ndarray, sr: int = SR) -> float:
    """RMS of the middle of the signal, past the filter's edge transient."""
    return rms(x[sr // 4 : -sr // 4])


# ------------------------------------------------------------------ high-pass

def test_window_length_is_odd_and_tracks_the_cutoff():
    assert window_length(80.0, SR) % 2 == 1
    # Halving the cutoff doubles the window.
    assert window_length(40.0, SR) > 1.8 * window_length(80.0, SR)


def test_highpass_keeps_the_sample_count():
    hp = HighPass(80.0, SR)
    for size in (0, 1, 160, 3200):
        assert len(hp.process(np.zeros(size, dtype=np.float32))) == size


def test_highpass_removes_dc():
    hp = HighPass(80.0, SR)
    out = hp.process(np.full(SR, 0.4, dtype=np.float32))
    assert abs(float(np.mean(out[SR // 4 :]))) < 0.005


def test_highpass_cuts_rumble_and_spares_speech():
    rumble = steady_amplitude(HighPass(80.0, SR).process(sine(30)))
    voice = steady_amplitude(HighPass(80.0, SR).process(sine(300)))
    high = steady_amplitude(HighPass(80.0, SR).process(sine(1000)))
    reference = steady_amplitude(sine(300))
    assert rumble < 0.1 * reference          # ~ -20 dB at 30 Hz
    assert voice > 0.95 * reference          # the voice band passes untouched
    assert high > 0.98 * reference


def test_highpass_is_continuous_across_blocks():
    """State has to survive the block boundaries: the chunker feeds it whatever
    the ring buffer happened to hold."""
    signal = sine(300, seconds=0.5) + sine(20, seconds=0.5)
    whole = HighPass(80.0, SR).process(signal)
    chunked = HighPass(80.0, SR)
    parts = [chunked.process(signal[i : i + 777]) for i in range(0, len(signal), 777)]
    assert np.allclose(whole, np.concatenate(parts), atol=1e-6)


def test_create_highpass_off():
    assert create_highpass(0.0, SR) is None


# --------------------------------------------------------------- chunk levels

def test_normalise_lifts_a_quiet_chunk():
    quiet = sine(300, amp=0.02)  # RMS ~0.014, a gain of 3.5 — inside the cap
    out = normalise_for_asr(quiet, target_rms=0.05)
    assert rms(out) > 3 * rms(quiet)
    assert abs(rms(out) - 0.05) < 0.005


def test_normalise_leaves_a_loud_chunk_alone():
    loud = sine(300, amp=0.5)
    assert normalise_for_asr(loud, target_rms=0.05) is loud


def test_normalise_never_clips():
    # Quiet on average, but with a peak close to the ceiling already.
    audio = sine(300, amp=0.004)
    audio[100] = 0.9
    out = normalise_for_asr(audio, target_rms=0.05)
    assert float(np.max(np.abs(out))) <= PEAK_CEILING + 1e-6


def test_normalise_caps_the_gain_on_near_silence():
    faint = sine(300, amp=0.001)
    out = normalise_for_asr(faint, target_rms=0.05)
    assert rms(out) <= MAX_GAIN * rms(faint) + 1e-9


def test_normalise_disabled_and_silence():
    audio = sine(300, amp=0.005)
    assert normalise_for_asr(audio, target_rms=0.0) is audio
    silence = np.zeros(SR, dtype=np.float32)
    assert normalise_for_asr(silence, target_rms=0.05) is silence


def test_chunk_level_is_the_raw_rms():
    audio = sine(300, amp=0.2)
    chunk = AudioChunk(speaker=Speaker.INTERVIEWER, audio=audio, t0=0.0, t1=1.0)
    assert abs(chunk.level - rms(audio)) < 1e-9
    assert AudioChunk(Speaker.INTERVIEWER, np.zeros(0, dtype=np.float32), 0.0, 0.0).level == 0.0


# ----------------------------------------------------- the chunker's own path

class _SpyAssembler:
    """Records what the chunker hands the assembler, emits nothing."""

    def __init__(self):
        self.seen = np.zeros(0, dtype=np.float32)

    def feed(self, audio):
        self.seen = np.concatenate([self.seen, audio])
        return []

    def flush(self):
        return None


def pump_once(highpass_hz: float) -> np.ndarray:
    cfg = AudioConfig(highpass_hz=highpass_hz)
    ring = RingBuffer(cfg.ring_seconds, cfg.sample_rate)
    ring.append(sine(300, seconds=0.5) + 0.2)  # a voice sitting on a DC offset
    spy = _SpyAssembler()
    chunker = ChunkerThread(
        Speaker.INTERVIEWER, ring, spy, queue.Queue(), cfg, threading.Event()
    )
    chunker._pump()
    return spy.seen


def test_chunker_filters_what_the_assembler_sees():
    filtered = pump_once(80.0)
    assert len(filtered) == SR // 2
    assert abs(float(np.mean(filtered[SR // 4 :]))) < 0.005  # the offset is gone


def test_chunker_leaves_the_audio_alone_when_the_filter_is_off():
    raw = pump_once(0.0)
    assert abs(float(np.mean(raw)) - 0.2) < 0.01


# ------------------------------------------------------------ echo between the channels

PHRASE_A = "мы использовали кафку для событий и постгрес для основного хранилища"
PHRASE_B = "потом переписали половину сервисов на го и стало заметно быстрее"


def test_tokens_and_jaccard():
    assert jaccard(tokens("Кафка и Постгрес!"), tokens("кафка, и постгрес")) == 1.0
    assert jaccard(tokens("совсем другое"), tokens("кафка и постгрес")) == 0.0


def feed_pair(det: EchoDetector, phrase: str, t0: float) -> Speaker | None:
    """The respondent speaks; the microphone hears the speakers a beat later."""
    det.observe(Speaker.RESPONDENT, t0, t0 + 2.0, phrase, level=0.20)
    return det.observe(Speaker.INTERVIEWER, t0 + 0.2, t0 + 2.2, phrase, level=0.03)


def test_echo_reported_once_after_min_hits():
    det = EchoDetector(min_hits=2)
    assert feed_pair(det, PHRASE_A, 10.0) is None      # one coincidence is a coincidence
    assert feed_pair(det, PHRASE_B, 20.0) is Speaker.INTERVIEWER
    assert feed_pair(det, PHRASE_A, 30.0) is None      # said once, never again
    assert det.hits == 3


def test_echo_names_the_quieter_channel():
    det = EchoDetector(min_hits=1)
    det.observe(Speaker.INTERVIEWER, 5.0, 7.0, PHRASE_A, level=0.30)
    assert det.observe(Speaker.RESPONDENT, 5.1, 7.1, PHRASE_A, level=0.02) is Speaker.RESPONDENT


def test_different_speech_is_not_an_echo():
    det = EchoDetector(min_hits=1)
    det.observe(Speaker.RESPONDENT, 1.0, 3.0, PHRASE_A, level=0.2)
    assert det.observe(Speaker.INTERVIEWER, 1.2, 3.2, PHRASE_B, level=0.02) is None


def test_the_same_channel_repeating_itself_is_not_an_echo():
    det = EchoDetector(min_hits=1)
    det.observe(Speaker.RESPONDENT, 1.0, 3.0, PHRASE_A, level=0.2)
    assert det.observe(Speaker.RESPONDENT, 1.2, 3.2, PHRASE_A, level=0.02) is None


def test_a_late_copy_is_outside_the_window():
    det = EchoDetector(window_s=1.5, min_hits=1)
    det.observe(Speaker.RESPONDENT, 1.0, 3.0, PHRASE_A, level=0.2)
    assert det.observe(Speaker.INTERVIEWER, 40.0, 42.0, PHRASE_A, level=0.02) is None


def test_short_agreements_are_ignored():
    det = EchoDetector(min_hits=1)
    for i in range(5):
        t = 1.0 + i
        det.observe(Speaker.RESPONDENT, t, t + 0.4, "да, понятно", level=0.2)
        assert det.observe(Speaker.INTERVIEWER, t + 0.1, t + 0.5, "да, понятно", level=0.02) is None
