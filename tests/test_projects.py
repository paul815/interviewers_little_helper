"""Project tests: the series preset, the folder layout, the aggregate coverage.

No hardware or LLM required: what is checked is the storage and the REST surface,
not an interview.
"""
from __future__ import annotations

import json

import pytest

from app.config import AppConfig
from app.guide.schemas import Guide, Section, Topic
from app.storage.project_store import ProjectError, ProjectStore
from app.storage.session_store import SessionStore
from tests.conftest import local_client, tmp_cfg

GUIDE = Guide(
    guide_id="g",
    title="Onboarding",
    sections=[
        Section(id="s1", title="Introductions", topics=[
            Topic(id="t1", question="Role?"), Topic(id="t2", question="Years of experience?")]),
        Section(id="s2", title="Pain points", topics=[
            Topic(id="t3", question="What gets in the way?")]),
    ],
)


@pytest.fixture()
def store(tmp_path) -> ProjectStore:
    return ProjectStore(tmp_path / "projects")


@pytest.fixture()
def client(tmp_path):
    with local_client(tmp_cfg(tmp_path)) as c:
        yield c


def add_session(store: ProjectStore, project_id: str, name: str, statuses: dict[str, str]) -> None:
    """A ready interview folder with coverage — as a real run would leave it."""
    sdir = store.sessions_root(project_id) / name
    sdir.mkdir(parents=True)
    (sdir / "meta.json").write_text(
        json.dumps({"session_id": name, "started_at": "2026-08-08T10:00:00+03:00",
                    "segments": 42, "duration_s": 1800}),
        encoding="utf-8",
    )
    (sdir / "coverage_state.json").write_text(
        json.dumps({"session_id": name,
                    "topics": {tid: {"status": st} for tid, st in statuses.items()}}),
        encoding="utf-8",
    )


# -------------------------------------------------------------------- store

def test_create_makes_folder_and_preset(store):
    p = store.create("B2B onboarding", GUIDE, "CRM, LTV", 45)
    assert (store.root / p["project_id"] / "sessions").is_dir()
    assert store.load(p["project_id"])["asr_vocabulary"] == "CRM, LTV"
    assert store.load(p["project_id"])["duration_min"] == 45


def test_similar_titles_get_separate_folders(store):
    first = store.create("Onboarding")
    second = store.create("Onboarding")
    assert first["project_id"] != second["project_id"]
    assert len(store.list()) == 2


def test_create_requires_title(store):
    with pytest.raises(ProjectError):
        store.create("   ")


def test_update_leaves_untouched_fields_alone(store):
    p = store.create("Series", GUIDE, "CRM", 45)
    store.update(p["project_id"], title="Spring series")
    data = store.load(p["project_id"])
    assert data["title"] == "Spring series"
    assert data["asr_vocabulary"] == "CRM" and data["duration_min"] == 45


@pytest.mark.parametrize("bad", ["../etc", "a/b", "a\\b", ""])
def test_path_traversal_rejected(store, bad):
    with pytest.raises(ProjectError):
        store.dir(bad)


def test_session_id_traversal_rejected(store):
    p = store.create("Series")
    with pytest.raises(ProjectError):
        store.session_dir(p["project_id"], "../../secrets")


def test_delete_refuses_project_with_recorded_interviews(store):
    p = store.create("Series", GUIDE)
    add_session(store, p["project_id"], "2026-08-08_10-00-00", {"t1": "covered"})
    with pytest.raises(ProjectError, match="by hand"):
        store.delete(p["project_id"])
    assert store.list()  # the project and its recordings are still there

    empty = store.create("Empty series")
    store.delete(empty["project_id"])
    assert [x["project_id"] for x in store.list()] == [p["project_id"]]


# ------------------------------------------------- sessions inside a project

def test_session_store_writes_inside_project(store):
    p = store.create("Series", GUIDE)
    s = SessionStore.create(AppConfig(), {}, root=store.sessions_root(p["project_id"]))
    s.close()
    assert s.dir.parent == store.sessions_root(p["project_id"])
    assert store.sessions(p["project_id"])[0]["session_id"] == s.session_id


def test_sessions_listed_newest_first_with_counts(store):
    p = store.create("Series", GUIDE)
    add_session(store, p["project_id"], "2026-08-05_10-00-00", {"t1": "covered", "t2": "partial"})
    add_session(store, p["project_id"], "2026-08-07_19-00-00", {"t1": "covered"})
    sessions = store.sessions(p["project_id"])
    assert [s["session_id"] for s in sessions] == [
        "2026-08-07_19-00-00", "2026-08-05_10-00-00"]
    assert sessions[1]["counts"] == {"covered": 1, "partial": 1, "not_covered": 0, "total": 2}
    assert sessions[0]["segments"] == 42


def test_session_detail_returns_notes_in_time_order(store):
    p = store.create("Series", GUIDE)
    add_session(store, p["project_id"], "d1", {"t1": "covered"})
    sdir = store.sessions_root(p["project_id"]) / "d1"
    (sdir / "flags.jsonl").write_text(
        json.dumps({"t": 120.0, "note": "lying here", "anchor": "Role?",
                    "anchor_section": "Introductions"}, ensure_ascii=False) + "\n"
        + "{broken line\n"
        + json.dumps({"t": 10.0, "note": ""}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (sdir / "questions.jsonl").write_text(
        json.dumps({"id": 1, "t": 30.0, "text": "Ask about the CRM", "done": True},
                   ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    d = store.session_detail(p["project_id"], "d1")
    # A broken line must not hide the other notes.
    assert [f["t"] for f in d["flags"]] == [10.0, 120.0]
    assert d["flags"][1]["note"] == "lying here"
    assert d["questions"][0]["done"] is True
    assert d["counts"]["covered"] == 1


def test_session_detail_without_notes_is_empty_not_error(store):
    p = store.create("Series", GUIDE)
    add_session(store, p["project_id"], "d1", {"t1": "covered"})
    d = store.session_detail(p["project_id"], "d1")
    assert d["flags"] == [] and d["questions"] == [] and d["has_report"] is False


# ------------------------------------------------------- aggregate coverage

def test_coverage_counts_statuses_across_series(store):
    p = store.create("Series", GUIDE)
    add_session(store, p["project_id"], "d1",
                {"t1": "covered", "t2": "partial", "t3": "not_covered"})
    add_session(store, p["project_id"], "d2",
                {"t1": "covered", "t2": "covered", "t3": "not_covered"})
    cov = store.coverage(p["project_id"])

    assert cov["sessions_count"] == 2
    topics = {t["topic_id"]: t for sec in cov["sections"] for t in sec["topics"]}
    assert topics["t1"]["covered"] == 2
    assert (topics["t2"]["covered"], topics["t2"]["partial"]) == (1, 1)
    # A topic no interview covered is the very reason for looking at the series.
    assert topics["t3"]["covered"] == 0 and topics["t3"]["not_covered"] == 2


def test_coverage_denominator_is_per_topic(store):
    """A topic added to the guide later is counted only across its own interviews."""
    p = store.create("Series", GUIDE)
    add_session(store, p["project_id"], "d1", {"t1": "covered"})
    add_session(store, p["project_id"], "d2", {"t1": "covered", "t3": "covered"})
    topics = {t["topic_id"]: t for sec in store.coverage(p["project_id"])["sections"]
              for t in sec["topics"]}
    assert topics["t1"]["sessions_with_topic"] == 2
    assert topics["t3"]["sessions_with_topic"] == 1


def test_coverage_survives_broken_session(store):
    p = store.create("Series", GUIDE)
    add_session(store, p["project_id"], "d1", {"t1": "covered"})
    broken = store.sessions_root(p["project_id"]) / "d2"
    broken.mkdir()
    (broken / "coverage_state.json").write_text("{not json", encoding="utf-8")
    assert store.coverage(p["project_id"])["sessions_count"] == 1


def test_coverage_without_guide_is_empty(store):
    p = store.create("Series without a guide")
    assert store.coverage(p["project_id"])["sections"] == []


# ------------------------------------------------------------------- API

def test_projects_crud_over_http(client):
    assert client.get("/api/projects").json() == {"projects": []}

    created = client.post("/api/projects", json={"title": "Morning", "duration_min": 45}).json()
    pid = created["project_id"]
    assert client.get(f"/api/projects/{pid}").json()["title"] == "Morning"

    client.patch(f"/api/projects/{pid}", json={"title": "Morning — B2B"})
    assert client.get(f"/api/projects/{pid}").json()["title"] == "Morning — B2B"
    # An empty PATCH must not wipe the preset.
    client.patch(f"/api/projects/{pid}", json={})
    assert client.get(f"/api/projects/{pid}").json()["duration_min"] == 45

    assert client.delete(f"/api/projects/{pid}").status_code == 200
    assert client.get("/api/projects").json()["projects"] == []


def test_missing_project_returns_409_with_message(client):
    r = client.get("/api/projects/no-such-thing")
    assert r.status_code == 409 and "not found" in r.json()["detail"]


def test_open_rejects_traversal(client):
    pid = client.post("/api/projects", json={"title": "Morning"}).json()["project_id"]
    r = client.post(f"/api/projects/{pid}/open", json={"session_id": "../../.."})
    assert r.status_code == 409


def test_session_detail_over_http(client, tmp_path):
    pid = client.post("/api/projects", json={"title": "Morning"}).json()["project_id"]
    sdir = tmp_path / "projects" / pid / "sessions" / "d1"
    sdir.mkdir(parents=True)
    (sdir / "flags.jsonl").write_text(
        json.dumps({"t": 5.0, "note": "important"}, ensure_ascii=False) + "\n", encoding="utf-8")

    r = client.get(f"/api/projects/{pid}/sessions/d1")
    assert r.status_code == 200 and r.json()["flags"][0]["note"] == "important"
    # The series list is still served from its own address.
    assert client.get(f"/api/projects/{pid}/sessions").json()["sessions"][0]["session_id"] == "d1"
    assert client.get(f"/api/projects/{pid}/sessions/no-such-thing").status_code == 409


# ------------------------------------ a past interview: transcript and audio

def _wav(frames: int = 800) -> bytes:
    """A stereo 16-bit 16 kHz WAV — enough of one for the range checks."""
    data = b"\x00\x01\x02\x03" * frames
    header = (b"RIFF" + (36 + len(data)).to_bytes(4, "little") + b"WAVEfmt "
              + (16).to_bytes(4, "little") + (1).to_bytes(2, "little")
              + (2).to_bytes(2, "little") + (16000).to_bytes(4, "little")
              + (64000).to_bytes(4, "little") + (4).to_bytes(2, "little")
              + (16).to_bytes(2, "little") + b"data" + len(data).to_bytes(4, "little"))
    return header + data


def test_session_transcript_merges_utterances_and_flags_in_time_order(store):
    p = store.create("Series", GUIDE)
    add_session(store, p["project_id"], "d1", {"t1": "covered"})
    sdir = store.sessions_root(p["project_id"]) / "d1"
    (sdir / "transcript.jsonl").write_text(
        json.dumps({"id": 2, "speaker": "RESPONDENT", "t0": 12.0, "t1": 15.0,
                    "text": "Twice a week"}, ensure_ascii=False) + "\n"
        + json.dumps({"id": 1, "speaker": "INTERVIEWER", "t0": 3.5, "t1": 6.0,
                      "text": "How often?"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (sdir / "flags.jsonl").write_text(
        json.dumps({"t": 13.0, "note": "come back to this"}, ensure_ascii=False) + "\n",
        encoding="utf-8")

    d = store.session_transcript(p["project_id"], "d1")
    assert [s["t0"] for s in d["segments"]] == [3.5, 12.0]
    assert d["segments"][0]["speaker"] == "INTERVIEWER"
    assert d["flags"][0]["note"] == "come back to this"
    # No recording was written for this one — the viewer shows the text alone.
    assert d["has_audio"] is False and d["duration_s"] == 1800


def test_session_audio_is_reported_and_refused_when_absent(store):
    p = store.create("Series", GUIDE)
    add_session(store, p["project_id"], "d1", {"t1": "covered"})
    with pytest.raises(ProjectError):
        store.session_audio(p["project_id"], "d1")

    (store.sessions_root(p["project_id"]) / "d1" / "audio.wav").write_bytes(_wav())
    assert store.session_audio(p["project_id"], "d1").name == "audio.wav"
    assert store.sessions(p["project_id"])[0]["has_audio"] is True
    assert store.session_transcript(p["project_id"], "d1")["has_audio"] is True


def test_transcript_over_http(client, tmp_path):
    pid = client.post("/api/projects", json={"title": "Morning"}).json()["project_id"]
    sdir = tmp_path / "projects" / pid / "sessions" / "d1"
    sdir.mkdir(parents=True)
    (sdir / "transcript.jsonl").write_text(
        json.dumps({"id": 1, "speaker": "INTERVIEWER", "t0": 1.0, "t1": 2.0,
                    "text": "Hello"}, ensure_ascii=False) + "\n", encoding="utf-8")

    r = client.get(f"/api/projects/{pid}/sessions/d1/transcript")
    assert r.status_code == 200 and r.json()["segments"][0]["text"] == "Hello"
    assert client.get(f"/api/projects/{pid}/sessions/d1/audio").status_code == 409
    escape = client.get(f"/api/projects/{pid}/sessions/../../etc/transcript")
    assert escape.status_code in (404, 409)


def test_audio_served_whole_and_by_range(client, tmp_path):
    pid = client.post("/api/projects", json={"title": "Morning"}).json()["project_id"]
    sdir = tmp_path / "projects" / pid / "sessions" / "d1"
    sdir.mkdir(parents=True)
    blob = _wav()
    (sdir / "audio.wav").write_bytes(blob)
    url = f"/api/projects/{pid}/sessions/d1/audio"

    whole = client.get(url)
    assert whole.status_code == 200 and whole.content == blob
    assert whole.headers["accept-ranges"] == "bytes"

    # <audio> cannot seek without 206 — this is what makes the timecodes work.
    part = client.get(url, headers={"Range": "bytes=100-199"})
    assert part.status_code == 206 and part.content == blob[100:200]
    assert part.headers["content-range"] == f"bytes 100-199/{len(blob)}"

    open_ended = client.get(url, headers={"Range": "bytes=100-"})
    assert open_ended.status_code == 206 and open_ended.content == blob[100:]

    tail = client.get(url, headers={"Range": "bytes=-64"})
    assert tail.status_code == 206 and tail.content == blob[-64:]

    past_end = client.get(url, headers={"Range": f"bytes={len(blob)}-"})
    assert past_end.status_code == 416


def test_start_with_unknown_project_is_refused(client):
    r = client.post("/api/session/start", json={
        "mic_index": 0, "system_index": 1,
        "guide": GUIDE.model_dump(), "project_id": "no-such-thing",
    })
    assert r.status_code == 409 and "not found" in r.json()["detail"]


def test_state_snapshot_exposes_project(client):
    snap = client.get("/api/state").json()
    assert snap["project_id"] is None and snap["project_title"] == ""
