"""Recovery of sessions that were never closed properly.

A Python crash, killing the process from the task manager, a power cut — after
any of them the session folder is left in an intermediate state:

* `audio.wav` holds more audio than its header declares (the writer refreshes
  the sizes once every `SessionRecorder.HEADER_SYNC_S`), and a player shows the
  recording as shorter than it is;
* `transcript.md` was never assembled — the readable transcript exists only as
  `transcript.jsonl`;
* `meta.json` has no `stopped_at`, no recording duration and no segment count.

All of this is fixable from data that is already on disk: neither ASR nor the
LLM is needed here. The module is called once at application startup (see
`server/app.py`), walks every session folder and repairs the ones left open.
The operation is idempotent: a repaired session is marked `recovered_at` and is
not touched again.

The report (`report.md`) is not recovered — it needs the LLM; in the UI summary
such a session is simply marked as interrupted.
"""
from __future__ import annotations

import json
import logging
import os
import struct
from datetime import datetime
from pathlib import Path

from ..config import AppConfig
from .session_store import build_transcript_markdown

log = logging.getLogger("ilh.recovery")

_HEADER_BYTES = 44


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _read_jsonl(path: Path) -> list[dict]:
    """JSONL lines; a broken tail is skipped.

    On a power cut the last line is cut off halfway — losing the whole file over
    it is not an option.
    """
    rows: list[dict] = []
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return rows
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except ValueError:
            log.warning("%s: skipping a truncated line", path.name)
    return rows


def repair_wav(path: Path) -> dict | None:
    """Bring the sizes in the WAV header in line with the actual file size.

    Returns a summary if anything changed, otherwise None. Foreign files (not
    our canonical 44-byte header) are left alone.
    """
    try:
        size = path.stat().st_size
    except OSError:
        return None
    if size <= _HEADER_BYTES:
        return None
    with open(path, "r+b") as f:
        head = f.read(_HEADER_BYTES)
        if len(head) < _HEADER_BYTES or head[:4] != b"RIFF" or head[8:12] != b"WAVE":
            return None
        if head[12:16] != b"fmt " or head[36:40] != b"data":
            log.warning("%s: unfamiliar header layout — leaving it alone", path)
            return None
        channels, rate = struct.unpack_from("<HI", head, 22)
        block_align = struct.unpack_from("<H", head, 32)[0]
        declared = struct.unpack_from("<I", head, 40)[0]
        if not block_align or not rate:
            return None
        # An incomplete frame at the end (a cut exactly mid-sample) is dropped.
        actual = ((size - _HEADER_BYTES) // block_align) * block_align
        if actual == declared:
            return None
        f.seek(4)
        f.write(struct.pack("<I", 36 + actual))
        f.seek(40)
        f.write(struct.pack("<I", actual))
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:
            pass
    delta_s = round((actual - declared) / block_align / rate, 1)
    info = {
        "audio_seconds": round(actual / block_align / rate, 1),
        "channels": channels,
    }
    if delta_s > 0:
        info["audio_recovered_s"] = delta_s
        log.info("%s: the header lagged by %.1f s — extended it", path, delta_s)
    else:
        # The header promises more than there is: the data never reached the disk.
        info["audio_lost_s"] = abs(delta_s)
        log.warning("%s: %.1f s of tail missing — trimmed the header", path, abs(delta_s))
    return info


def recover_session(session_dir: Path) -> dict | None:
    """Repair one session folder. None means there was nothing to repair."""
    meta_path = session_dir / "meta.json"
    meta = _read_json(meta_path)
    if meta.get("stopped_at") or meta.get("recovered_at"):
        return None
    transcript = session_dir / "transcript.jsonl"
    audio = session_dir / "audio.wav"
    if not transcript.exists() and not audio.exists():
        return None  # an empty stub: the session died before it started

    info: dict = {"session_id": meta.get("session_id") or session_dir.name,
                  "path": str(session_dir)}
    if audio.exists():
        repaired = repair_wav(audio)
        if repaired is not None:
            meta.update({k: v for k, v in repaired.items() if k != "channels"})
            info.update({k: v for k, v in repaired.items() if k != "channels"})
        else:  # the header is already correct — meta still needs the duration
            meta.setdefault("audio_seconds", _wav_seconds(audio))
            info["audio_seconds"] = meta.get("audio_seconds")

    segments = _read_jsonl(transcript)
    info["segments"] = len(segments)
    if segments:
        guide = _read_json(session_dir / "guide.json")
        text = build_transcript_markdown(
            info["session_id"],
            guide.get("title") or meta.get("guide_title") or "—",
            segments,
            _read_jsonl(session_dir / "flags.jsonl"),
            _read_jsonl(session_dir / "questions.jsonl"),
        )
        _write(session_dir / "transcript.md", text)

    meta.update({
        "session_id": info["session_id"],
        "segments": len(segments),
        "crashed": True,  # there was no proper "Stop": the report was never built
        "recovered_at": datetime.now().astimezone().isoformat(),
    })
    _write(meta_path, json.dumps(meta, ensure_ascii=False, indent=2))
    log.info("Recovered an interrupted session %s: %d segments, %.1f s of audio",
             info["session_id"], info["segments"], info.get("audio_seconds") or 0.0)
    return info


def recover_sessions(cfg: AppConfig) -> list[dict]:
    """Walk every session folder and repair the ones left open."""
    out = []
    for d in session_dirs(cfg):
        try:
            info = recover_session(d)
        except Exception:
            log.exception("Could not recover session %s", d)
            continue
        if info:
            out.append(info)
    if out:
        log.warning("Interrupted sessions recovered: %d", len(out))
    return out


def session_dirs(cfg: AppConfig) -> list[Path]:
    """Session folders: both the shared ones and those inside projects."""
    dirs: list[Path] = []
    if cfg.sessions_path.is_dir():
        dirs += [d for d in cfg.sessions_path.iterdir() if d.is_dir()]
    if cfg.projects_path.is_dir():
        for project in cfg.projects_path.iterdir():
            sessions = project / "sessions"
            if sessions.is_dir():
                dirs += [d for d in sessions.iterdir() if d.is_dir()]
    return sorted(dirs)


def _wav_seconds(path: Path) -> float:
    try:
        with open(path, "rb") as f:
            head = f.read(_HEADER_BYTES)
        rate = struct.unpack_from("<I", head, 24)[0]
        block_align = struct.unpack_from("<H", head, 32)[0]
        data = struct.unpack_from("<I", head, 40)[0]
        return round(data / block_align / rate, 1) if rate and block_align else 0.0
    except (OSError, struct.error, ZeroDivisionError):
        return 0.0


def _write(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)
