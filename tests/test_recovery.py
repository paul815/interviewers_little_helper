"""What is left on disk after a process crash or a power cut.

Every test has the same scenario: the session was cut short, there was no proper
"Stop" — we check that the audio and the transcript are rebuilt from what has
already been written.
"""
from __future__ import annotations

import json
import wave

import numpy as np
import pytest

from app.audio.capture import ChannelCapture
from app.audio.recorder import SessionRecorder, WavWriter
from app.config import AudioConfig
from app.domain import Speaker
from app.storage.recovery import recover_session, recover_sessions, repair_wav

from .conftest import tmp_cfg

SR = 16000


def _crashed_wav(path, seconds: float, synced_seconds: float = 0.0):
    """A WAV whose header lags behind the data: the process died between syncs."""
    writer = WavWriter(path, SR, 2)
    if synced_seconds:
        writer.write(np.zeros(int(SR * synced_seconds) * 2, dtype="<i2").tobytes(),
                     int(SR * synced_seconds))
        writer.sync()
    rest = int(SR * (seconds - synced_seconds))
    writer.write(np.zeros(rest * 2, dtype="<i2").tobytes(), rest)
    writer._f.flush()  # the power went here: sync() and close() never happened
    writer._f.close()
    return path


def _duration(path) -> float:
    with wave.open(str(path), "rb") as wf:
        return wf.getnframes() / wf.getframerate()


def _session(tmp_path, name="2026-07-11_14-30-00", **meta):
    d = tmp_path / "sessions" / name
    d.mkdir(parents=True)
    (d / "meta.json").write_text(
        json.dumps({"session_id": name, "started_at": "2026-07-11T14:30:00+03:00", **meta}),
        encoding="utf-8",
    )
    return d


def _segments(d, n=3):
    rows = [
        {"id": i + 1, "speaker": "RESPONDENT" if i % 2 else "INTERVIEWER",
         "t0": i * 5.0, "t1": i * 5.0 + 4.0, "text": f"utterance {i + 1}", "language": "en"}
        for i in range(n)
    ]
    (d / "transcript.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
    )
    return rows


# --------------------------------------------------------------------- header


def test_repair_wav_returns_tail_lost_by_crash(tmp_path):
    path = _crashed_wav(tmp_path / "audio.wav", seconds=30.0, synced_seconds=10.0)
    assert _duration(path) == pytest.approx(10.0)  # the header only knows about the sync

    info = repair_wav(path)

    assert info["audio_recovered_s"] == pytest.approx(20.0)
    assert info["audio_seconds"] == pytest.approx(30.0)
    assert _duration(path) == pytest.approx(30.0)


def test_repair_wav_trims_header_when_data_did_not_reach_disk(tmp_path):
    """The opposite case: the header promises more than survived."""
    path = _crashed_wav(tmp_path / "audio.wav", seconds=20.0, synced_seconds=20.0)
    with open(path, "r+b") as f:  # the power went, the tail never made it
        f.truncate(44 + SR * 10 * 4)

    info = repair_wav(path)

    assert info["audio_lost_s"] == pytest.approx(10.0)
    assert _duration(path) == pytest.approx(10.0)


def test_repair_wav_is_idempotent_and_skips_foreign_files(tmp_path):
    path = _crashed_wav(tmp_path / "audio.wav", seconds=5.0)
    repair_wav(path)
    assert repair_wav(path) is None  # there is nothing left to repair

    alien = tmp_path / "notes.txt"
    alien.write_text("this is not a wav" * 20, encoding="utf-8")
    assert repair_wav(alien) is None


def test_repair_wav_drops_incomplete_frame(tmp_path):
    """A cut exactly mid-frame must not leave a broken last sample."""
    path = _crashed_wav(tmp_path / "audio.wav", seconds=1.0)
    with open(path, "ab") as f:
        f.write(b"\x01\x02\x03")  # an incomplete frame (4 bytes are needed)

    info = repair_wav(path)

    assert info["audio_seconds"] == pytest.approx(1.0)
    _duration(path)  # wave does not choke on the file


# ------------------------------------------------------------------- session


def test_recover_session_rebuilds_transcript_and_meta(tmp_path):
    d = _session(tmp_path)
    _segments(d)
    (d / "guide.json").write_text(json.dumps({"title": "The onboarding guide"}), encoding="utf-8")
    (d / "flags.jsonl").write_text(
        json.dumps({"t": 6.0, "note": "lying", "anchor": ""}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _crashed_wav(d / "audio.wav", seconds=42.0, synced_seconds=40.0)

    info = recover_session(d)

    assert info["segments"] == 3
    assert info["audio_seconds"] == pytest.approx(42.0)
    md = (d / "transcript.md").read_text(encoding="utf-8")
    assert "The onboarding guide" in md and "utterance 3" in md and "lying" in md
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    assert meta["crashed"] is True and meta["recovered_at"]
    assert meta["segments"] == 3 and "stopped_at" not in meta


def test_recover_session_survives_torn_last_line(tmp_path):
    """A jsonl line was cut in the middle — the other segments are not lost."""
    d = _session(tmp_path)
    _segments(d, n=2)
    with open(d / "transcript.jsonl", "a", encoding="utf-8") as f:
        f.write('{"id": 3, "speaker": "INTERV')

    info = recover_session(d)

    assert info["segments"] == 2
    assert "utterance 2" in (d / "transcript.md").read_text(encoding="utf-8")


def test_recover_session_skips_closed_and_repeated(tmp_path):
    closed = _session(tmp_path, name="closed", stopped_at="2026-07-11T15:30:00+03:00")
    _segments(closed)
    assert recover_session(closed) is None
    assert not (closed / "transcript.md").exists()

    crashed = _session(tmp_path, name="crashed")
    _segments(crashed)
    assert recover_session(crashed) is not None
    assert recover_session(crashed) is None  # a second run rewrites nothing


def test_recover_session_ignores_empty_folder(tmp_path):
    assert recover_session(_session(tmp_path, name="empty")) is None


def test_recover_sessions_covers_projects(tmp_path):
    cfg = tmp_cfg(tmp_path)
    loose = tmp_path / "sessions" / "loose"
    loose.mkdir(parents=True)
    (loose / "meta.json").write_text(json.dumps({"session_id": "loose"}), encoding="utf-8")
    _segments(loose, n=1)
    inside = tmp_path / "projects" / "p1" / "sessions" / "inside"
    inside.mkdir(parents=True)
    (inside / "meta.json").write_text(json.dumps({"session_id": "inside"}), encoding="utf-8")
    _segments(inside, n=1)

    ids = {r["session_id"] for r in recover_sessions(cfg)}

    assert ids == {"loose", "inside"}


def test_recovery_runs_on_server_start(tmp_path):
    """The live scenario: the application came up after a crash — the folder is already repaired."""
    from .conftest import local_client

    cfg = tmp_cfg(tmp_path)
    d = tmp_path / "sessions" / "2026-07-11_14-30-00"
    d.mkdir(parents=True)
    (d / "meta.json").write_text(json.dumps({"session_id": "2026-07-11_14-30-00"}),
                                 encoding="utf-8")
    _segments(d, n=2)
    _crashed_wav(d / "audio.wav", seconds=12.0)

    with local_client(cfg) as client:
        state = client.get("/api/state").json()

    assert (d / "transcript.md").exists()
    assert _duration(d / "audio.wav") == pytest.approx(12.0)
    assert [r["session_id"] for r in state["recovered"]] == ["2026-07-11_14-30-00"]


# ------------------------------------------------------- recording warnings


def test_recorder_warns_when_disk_falls_behind(tmp_path):
    rec = SessionRecorder(tmp_path / "audio.wav", SR)
    warnings: list[str] = []
    rec.on_warning = warnings.append
    # The writer is not running: the queue piles up as it would on a stalled disk.
    rec._max_pending = SR

    for _ in range(4):
        rec.submit(Speaker.INTERVIEWER, np.zeros(SR, dtype=np.float32))

    assert warnings and "cannot keep up" in warnings[0]
    assert len(warnings) == 1  # the same problem does not spam the UI


def test_capture_reports_disabled_recording():
    cap = ChannelCapture(0, Speaker.INTERVIEWER, AudioConfig())
    reported: list[str] = []
    cap.on_audio = lambda _b: (_ for _ in ()).throw(OSError("no space left"))
    cap.on_audio_failed = reported.append

    cap._ingest(np.zeros(64, dtype=np.float32), False)
    cap._ingest(np.zeros(64, dtype=np.float32), False)

    assert reported == ["no space left"]  # exactly once, and capture goes on
    assert cap.on_audio is None
