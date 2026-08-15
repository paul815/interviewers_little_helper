"""A project is a series of interviews run off one guide.

A research series lives for weeks: an interview for one project in the morning,
for another one in the evening. A project holds the preset (guide, ASR
vocabulary, duration, extra instructions for the analysis) so that switching
does not mean reassembling the settings, and it keeps its sessions next to it —
in one folder that can be opened, archived or handed to a colleague.

projects/b2b-onboarding/
  project.json              — the project preset and metadata
  sessions/
    2026-08-08_10-15-00/    — an ordinary session folder (see session_store)
    2026-08-08_19-40-00/

Sessions recorded before projects existed stay in sessions/ at the repo root:
they belong to nobody, and there is no reason to touch them.
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
from datetime import datetime
from pathlib import Path

from ..coverage.schemas import CoverageState
from ..guide.schemas import Guide
from ..paths import UnsafeName, safe_child

log = logging.getLogger("ilh.projects")

# Cyrillic stays in the whitelist on purpose: the interface is English, but a
# project may well be named in another language, and the folder name should
# stay readable.
_SLUG_RE = re.compile(r"[^0-9a-zA-Zа-яА-ЯёЁ]+")
_STATUSES = ("covered", "partial", "not_covered")


def slugify(title: str, max_len: int = 40) -> str:
    slug = _SLUG_RE.sub("-", title).strip("-").lower()
    return slug[:max_len] or "project"


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


class ProjectError(Exception):
    """An error that must be shown to the user verbatim."""


class ProjectStore:
    def __init__(self, root: Path):
        self.root = root
        self._lock = threading.Lock()

    # ------------------------------------------------------------- paths

    def _dir(self, project_id: str) -> Path:
        # The identifier arrives over HTTP: a folder name, not a path (see app/paths.py).
        try:
            return safe_child(self.root, project_id)
        except UnsafeName as e:
            raise ProjectError("Invalid project identifier") from e

    def dir(self, project_id: str) -> Path:
        path = self._dir(project_id)
        if not (path / "project.json").exists():
            raise ProjectError(f"Project «{project_id}» not found")
        return path

    def sessions_root(self, project_id: str) -> Path:
        """Where SessionStore puts this project's session folders."""
        return self.dir(project_id) / "sessions"

    def session_dir(self, project_id: str, session_id: str) -> Path:
        # The session identifier also arrives over HTTP — same guard as the project.
        try:
            path = safe_child(self.sessions_root(project_id), session_id)
        except UnsafeName as e:
            raise ProjectError("Invalid interview identifier") from e
        if not path.is_dir():
            raise ProjectError(f"Interview folder not found: {session_id}")
        return path

    # ------------------------------------------------------------- CRUD

    def list(self) -> list[dict]:
        if not self.root.exists():
            return []
        items = []
        for path in self.root.iterdir():
            if not (path / "project.json").exists():
                continue
            try:
                data = self._read(path)
            except ProjectError as e:
                log.warning("Skipping project %s: %s", path.name, e)
                continue
            sessions = self._session_dirs(path)
            items.append(
                {
                    "project_id": path.name,
                    "title": data.get("title", path.name),
                    "guide_title": (data.get("guide") or {}).get("title", ""),
                    "topics_count": _topics_count(data.get("guide")),
                    "duration_min": data.get("duration_min"),
                    "sessions_count": len(sessions),
                    "last_session_at": sessions[-1].name if sessions else None,
                    "created_at": data.get("created_at"),
                    "updated_at": data.get("updated_at"),
                }
            )
        # Freshest first: sort by last activity, not by folder name.
        items.sort(key=lambda p: (p["updated_at"] or "", p["title"]), reverse=True)
        return items

    def create(
        self,
        title: str,
        guide: Guide | None = None,
        asr_vocabulary: str = "",
        duration_min: int | None = None,
        guide_text: str = "",
        llm_instructions: str = "",
    ) -> dict:
        title = title.strip()
        if not title:
            raise ProjectError("A project must have a name")
        now = datetime.now().astimezone().isoformat()
        with self._lock:
            project_id = self._free_id(slugify(title))
            path = self.root / project_id
            (path / "sessions").mkdir(parents=True, exist_ok=True)
            data = {
                "project_id": project_id,
                "title": title,
                "created_at": now,
                "updated_at": now,
                "guide": guide.model_dump() if guide else None,
                "guide_text": guide_text,
                "asr_vocabulary": asr_vocabulary,
                "duration_min": duration_min,
                "llm_instructions": llm_instructions,
            }
            _atomic_write(path / "project.json", json.dumps(data, ensure_ascii=False, indent=2))
        log.info("Project created: %s (%s)", title, project_id)
        return data

    def load(self, project_id: str) -> dict:
        return self._read(self.dir(project_id))

    def update(self, project_id: str, **fields) -> dict:
        """A targeted preset update; None means "leave this field alone"."""
        allowed = {
            "title", "guide", "guide_text", "asr_vocabulary",
            "duration_min", "llm_instructions",
        }
        unknown = set(fields) - allowed
        if unknown:
            raise ProjectError(f"These fields cannot be changed: {', '.join(sorted(unknown))}")
        path = self.dir(project_id)
        with self._lock:
            data = self._read(path)
            for key, value in fields.items():
                if value is None:
                    continue
                if key == "title":
                    value = str(value).strip()
                    if not value:
                        raise ProjectError("A project must have a name")
                if key == "guide" and isinstance(value, Guide):
                    value = value.model_dump()
                data[key] = value
            data["updated_at"] = datetime.now().astimezone().isoformat()
            _atomic_write(path / "project.json", json.dumps(data, ensure_ascii=False, indent=2))
        return data

    def delete(self, project_id: str) -> None:
        """Only an empty project is deleted.

        Recorded interviews cannot be brought back, and the "delete" button next
        to the project list sits too close to "select" — let a human remove the
        folder with the sessions by hand, deliberately.
        """
        path = self.dir(project_id)
        sessions = self._session_dirs(path)
        if sessions:
            raise ProjectError(
                f"The project holds {len(sessions)} recorded interviews — "
                f"delete the folder by hand: {path}"
            )
        with self._lock:
            (path / "project.json").unlink(missing_ok=True)
            (path / "sessions").rmdir()
            path.rmdir()
        log.info("Empty project deleted: %s", project_id)

    # -------------------------------------------------------- sessions

    def sessions(self, project_id: str) -> list[dict]:
        """The project's recorded interviews, freshest first."""
        path = self.dir(project_id)
        out = []
        for sdir in self._session_dirs(path):
            meta = _read_json(sdir / "meta.json") or {}
            state = _read_coverage(sdir)
            out.append(
                {
                    "session_id": sdir.name,
                    "started_at": meta.get("started_at"),
                    "duration_s": meta.get("duration_s"),
                    "segments": meta.get("segments"),
                    "counts": state.counts() if state else None,
                    "has_report": (sdir / "report.md").exists(),
                    "has_transcript": (sdir / "transcript.jsonl").exists(),
                    "has_audio": (sdir / "audio.wav").exists(),
                    "path": str(sdir),
                }
            )
        out.reverse()
        return out

    def session_detail(self, project_id: str, session_id: str) -> dict:
        """Notes from a past interview: flags, "ask" questions, coverage.

        Read from disk rather than from the controller's memory: what people
        usually look at is an old interview that this run of the application
        never saw.
        """
        sdir = self.session_dir(project_id, session_id)
        meta = _read_json(sdir / "meta.json") or {}
        state = _read_coverage(sdir)
        flags = _read_jsonl(sdir / "flags.jsonl")
        flags.sort(key=lambda f: f.get("t") or 0)
        return {
            "session_id": session_id,
            "started_at": meta.get("started_at"),
            "duration_s": meta.get("duration_s"),
            "segments": meta.get("segments"),
            "counts": state.counts() if state else None,
            "flags": flags,
            "questions": _read_jsonl(sdir / "questions.jsonl"),
            "has_report": (sdir / "report.md").exists(),
            "has_transcript": (sdir / "transcript.md").exists(),
            "has_audio": (sdir / "audio.wav").exists(),
            "path": str(sdir),
        }

    def session_transcript(self, project_id: str, session_id: str) -> dict:
        """The recorded interview as it is read afterwards: utterances and flags.

        transcript.jsonl rather than transcript.md — the viewer needs the
        timecodes to drive the audio, and markdown has lost them.
        """
        sdir = self.session_dir(project_id, session_id)
        meta = _read_json(sdir / "meta.json") or {}
        segments = _read_jsonl(sdir / "transcript.jsonl")
        segments.sort(key=lambda s: s.get("t0") or 0)
        flags = _read_jsonl(sdir / "flags.jsonl")
        flags.sort(key=lambda f: f.get("t") or 0)
        return {
            "session_id": session_id,
            "started_at": meta.get("started_at"),
            "duration_s": meta.get("duration_s"),
            "segments": segments,
            "flags": flags,
            "has_audio": (sdir / "audio.wav").exists(),
        }

    def session_audio(self, project_id: str, session_id: str) -> Path:
        """The interview recording. Absent if audio saving was switched off."""
        path = self.session_dir(project_id, session_id) / "audio.wav"
        if not path.is_file():
            raise ProjectError("This interview has no recording")
        return path

    def coverage(self, project_id: str) -> dict:
        """Aggregate coverage across the series: in how many interviews a topic
        was covered.

        Only sessions whose guide contained the topic at all are counted: if a
        topic was added to the guide after the third interview, it has its own
        denominator.
        """
        path = self.dir(project_id)
        data = self._read(path)
        guide = data.get("guide")
        if not guide:
            return {"sessions_count": 0, "sections": [], "guide_title": ""}

        tally: dict[str, dict[str, int]] = {}
        sessions = 0
        for sdir in self._session_dirs(path):
            state = _read_coverage(sdir)
            if state is None:
                continue
            sessions += 1
            for topic_id, topic in state.topics.items():
                row = tally.setdefault(topic_id, {s: 0 for s in _STATUSES})
                row[topic.status] = row.get(topic.status, 0) + 1

        sections = []
        for section in guide.get("sections", []):
            topics = []
            for topic in section.get("topics", []):
                row = tally.get(topic["id"], {s: 0 for s in _STATUSES})
                seen = sum(row.values())
                topics.append(
                    {
                        "topic_id": topic["id"],
                        "question": topic.get("question", ""),
                        "priority": topic.get("priority", "must"),
                        "covered": row["covered"],
                        "partial": row["partial"],
                        "not_covered": row["not_covered"],
                        "sessions_with_topic": seen,
                    }
                )
            sections.append({"title": section.get("title", ""), "topics": topics})
        return {
            "sessions_count": sessions,
            "guide_title": guide.get("title", ""),
            "sections": sections,
        }

    # -------------------------------------------------------- internals

    def _read(self, path: Path) -> dict:
        try:
            return json.loads((path / "project.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            raise ProjectError(f"Could not read project.json: {e}") from e

    def _free_id(self, slug: str) -> str:
        """Two projects with similar names must not share one folder.

        It also makes sure the generated id passes safe_child: otherwise the
        project would be created but could no longer be opened ("CON", say).
        """
        try:
            safe_child(self.root, slug)
        except UnsafeName:
            slug = f"p-{slug}"  # a word like "con" — the name of a DOS device
        candidate, n = slug, 2
        while (self.root / candidate / "project.json").exists():
            candidate = f"{slug}-{n}"
            n += 1
        return candidate

    @staticmethod
    def _session_dirs(path: Path) -> list[Path]:
        sessions = path / "sessions"
        if not sessions.exists():
            return []
        return sorted(d for d in sessions.iterdir() if d.is_dir())


def _topics_count(guide: dict | None) -> int:
    if not guide:
        return 0
    return sum(len(s.get("topics", [])) for s in guide.get("sections", []))


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _read_jsonl(path: Path) -> list[dict]:
    """A broken line is skipped: a torn write must not hide the other notes."""
    out: list[dict] = []
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return out
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            log.warning("Broken line in %s", path)
    return out


def _read_coverage(session_dir: Path) -> CoverageState | None:
    data = _read_json(session_dir / "coverage_state.json")
    if data is None:
        return None
    try:
        return CoverageState.model_validate(data)
    except Exception as e:  # a broken file must not take the series summary down
        log.warning("Broken coverage in %s: %s", session_dir.name, e)
        return None
