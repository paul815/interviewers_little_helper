"""Tests of the editable prompts and the project's extra instructions.

No hardware, no LLM: what is checked is that an edit reaches the system prompt
and that a broken template never survives to a session.
"""
from __future__ import annotations

import json

import pytest

from app.coverage import prompts as P
from app.coverage.engine import CoverageEngine
from app.guide.schemas import Guide, Section, Topic
from app.storage.prompt_store import PromptError, PromptStore, defaults
from tests.conftest import local_client, tmp_cfg

GUIDE = Guide(
    guide_id="g",
    language="en",
    title="Onboarding",
    sections=[Section(id="s1", title="Introductions", topics=[
        Topic(id="s1.t1", question="Role?")])],
)

CUSTOM_LIVE = "My own prompt.\nGuide:\n{guide_block}\nProbes: {probes_rule}\nJSON."


@pytest.fixture()
def store(tmp_path) -> PromptStore:
    return PromptStore(tmp_path / "prompts.json")


@pytest.fixture()
def client(tmp_path):
    with local_client(tmp_cfg(tmp_path)) as c:
        yield c


# -------------------------------------------------------------------- store

def test_defaults_when_nothing_saved(store):
    assert store.templates() == defaults()
    assert not any(p["customized"] for p in store.describe())
    assert not store.path.exists()


def test_save_and_reset(store):
    store.save("system_live", CUSTOM_LIVE)
    assert store.templates()["system_live"] == CUSTOM_LIVE
    assert [p["customized"] for p in store.describe() if p["key"] == "system_live"] == [True]

    store.reset("system_live")
    assert store.templates() == defaults()


def test_text_equal_to_default_is_not_stored(store):
    """Otherwise an edited default in a new version would never reach an old project."""
    store.save("system_final", P.SYSTEM_FINAL)
    assert not store.path.exists()


@pytest.mark.parametrize(
    "text, expect",
    [
        ("", "cannot be empty"),
        ("No substitutions at all", "will not work"),
        ("{guide_block} {probes_rule} {invention}", "Unknown substitutions"),
        ("{guide_block} {probes_rule} {", "curly braces"),
    ],
)
def test_broken_template_rejected(store, text, expect):
    with pytest.raises(PromptError, match=expect):
        store.save("system_live", text)
    assert not store.path.exists()


def test_unknown_key_rejected(store):
    with pytest.raises(PromptError, match="Unknown prompt"):
        store.save("no-such-thing", "text")


def test_corrupted_file_falls_back_to_defaults(store):
    """A broken prompts.json must not take startup down in the middle of a workday."""
    store.path.write_text('{"system_live": "broken {invention}"}', encoding="utf-8")
    assert store.templates() == defaults()

    store.path.write_text("not json at all", encoding="utf-8")
    assert store.templates() == defaults()


# ------------------------------------------------------------ prompt assembly

def test_custom_template_reaches_system_prompt():
    text = P.system_prompt_live(GUIDE, 6, 2, templates={"system_live": CUSTOM_LIVE})
    assert text.startswith("My own prompt.")
    assert "(s1.t1) Role?" in text        # the guide was substituted in
    assert "quote (a short trigger" in text  # so was the probe rule


def test_instructions_appended_to_both_prompts():
    live = P.system_prompt_live(GUIDE, 6, 2, extra="Do not ask about the price.")
    final = P.system_prompt_final(GUIDE, extra="Do not ask about the price.")
    for text in (live, final):
        assert text.endswith("Do not ask about the price.")
        assert "ADDITIONAL INSTRUCTIONS" in text
    assert "ADDITIONAL INSTRUCTIONS" not in P.system_prompt_live(GUIDE, 6, 2)


def test_instructions_survive_rewritten_template():
    """The instructions are appended after the template, not substituted into it."""
    text = P.system_prompt_live(
        GUIDE, 6, 2, templates={"system_live": CUSTOM_LIVE}, extra="Context: a bank."
    )
    assert text.endswith("Context: a bank.")


@pytest.mark.asyncio
async def test_engine_uses_templates_and_instructions():
    async def notify(kind, payload):
        pass

    from app.config import AnalysisConfig, LLMConfig

    engine = CoverageEngine(
        guide=GUIDE, transcript=None, llm=None,
        analysis_cfg=AnalysisConfig(), llm_cfg=LLMConfig(), notify=notify,
        templates={"system_live": CUSTOM_LIVE}, instructions="Onboarding only.",
    )
    assert engine._system_live.startswith("My own prompt.")
    assert engine._system_live.endswith("Onboarding only.")
    assert engine._system_final.endswith("Onboarding only.")


# ------------------------------------------------------------------- REST

def test_api_list_save_reset(client):
    items = client.get("/api/prompts").json()["prompts"]
    keys = {p["key"] for p in items}
    assert keys == set(defaults())
    assert all(p["text"] == p["default"] for p in items)

    saved = client.put("/api/prompts/system_live", json={"text": CUSTOM_LIVE}).json()
    assert saved["customized"] is True
    after = {p["key"]: p for p in client.get("/api/prompts").json()["prompts"]}
    assert after["system_live"]["text"] == CUSTOM_LIVE
    assert after["system_live"]["default"] == P.SYSTEM_LIVE

    assert client.delete("/api/prompts/system_live").status_code == 200
    after = {p["key"]: p for p in client.get("/api/prompts").json()["prompts"]}
    assert after["system_live"]["customized"] is False


def test_api_rejects_broken_template(client):
    resp = client.put("/api/prompts/system_live", json={"text": "no substitutions"})
    assert resp.status_code == 400
    assert "will not work" in resp.json()["detail"]


def test_api_rejects_unknown_key(client):
    assert client.put("/api/prompts/invention", json={"text": "x"}).status_code == 400


# ------------------------------------------ project-level extra instructions

def test_project_keeps_instructions(client, tmp_path):
    project = client.post("/api/projects", json={
        "title": "Onboarding", "llm_instructions": "The respondents are doctors.",
    }).json()
    pid = project["project_id"]
    loaded = client.get(f"/api/projects/{pid}").json()
    assert loaded["llm_instructions"] == "The respondents are doctors."

    client.patch(f"/api/projects/{pid}",
                 json={"llm_instructions": "Narrowed down: paediatricians."})
    on_disk = json.loads(
        (tmp_path / "projects" / pid / "project.json").read_text(encoding="utf-8")
    )
    assert on_disk["llm_instructions"] == "Narrowed down: paediatricians."

    # An empty string means "there are no instructions any more", not "leave the field alone".
    client.patch(f"/api/projects/{pid}", json={"llm_instructions": ""})
    assert client.get(f"/api/projects/{pid}").json()["llm_instructions"] == ""
