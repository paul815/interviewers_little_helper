"""Guide library: a series of interviews runs off a single guide, so confirmed
structures are saved and reused across sessions."""
from __future__ import annotations

import json
import logging
import re
import threading
from datetime import datetime
from pathlib import Path

from ..paths import UnsafeName, safe_child
from .schemas import Guide

log = logging.getLogger("ilh.guide")

# Cyrillic stays in the whitelist on purpose: the interface is English, but a
# guide may still be written in any language, and the slug should stay readable.
_SLUG_RE = re.compile(r"[^0-9a-zA-Zа-яА-ЯёЁ]+")


def _slugify(title: str, max_len: int = 40) -> str:
    slug = _SLUG_RE.sub("-", title).strip("-").lower()
    return slug[:max_len] or "guide"


class GuideLibrary:
    def __init__(self, root: Path):
        self.root = root
        self._lock = threading.Lock()

    def _path(self, file_id: str) -> Path:
        # The identifier comes from the URL: see app/paths.py for why a whitelist.
        try:
            return safe_child(self.root, file_id, ".json")
        except UnsafeName as e:
            raise ValueError("Invalid guide identifier") from e

    def list(self) -> list[dict]:
        if not self.root.exists():
            return []
        items = []
        for path in sorted(self.root.glob("*.json"), reverse=True):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                guide = data.get("guide", {})
                topics = sum(len(s.get("topics", [])) for s in guide.get("sections", []))
                items.append(
                    {
                        "file_id": path.stem,
                        "title": guide.get("title", path.stem),
                        "topics_count": topics,
                        "saved_at": data.get("saved_at"),
                    }
                )
            except (OSError, json.JSONDecodeError) as e:
                log.warning("Broken guide file %s: %s", path.name, e)
        return items

    def save(self, guide: Guide, source_text: str = "") -> str:
        self.root.mkdir(parents=True, exist_ok=True)
        file_id = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{_slugify(guide.title)}"
        payload = {
            "saved_at": datetime.now().astimezone().isoformat(),
            "source_text": source_text,
            "guide": guide.model_dump(),
        }
        with self._lock:
            self._path(file_id).write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        log.info("Guide saved to the library: %s", file_id)
        return file_id

    def load(self, file_id: str) -> dict:
        path = self._path(file_id)
        data = json.loads(path.read_text(encoding="utf-8"))
        Guide.model_validate(data["guide"])  # validate before handing it to the UI
        return {"file_id": file_id, "guide": data["guide"],
                "source_text": data.get("source_text", "")}

    def delete(self, file_id: str) -> None:
        with self._lock:
            self._path(file_id).unlink(missing_ok=True)
        log.info("Guide deleted from the library: %s", file_id)
