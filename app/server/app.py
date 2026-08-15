"""FastAPI application: REST for control plus a WebSocket for live updates.
Listens on 127.0.0.1 only; the front-end assets are served locally (no CDN)."""
from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from ..config import AppConfig
from ..provision import Provisioner
from ..storage.project_store import ProjectError
from ..storage.prompt_store import PromptError
from .controller import AppController, ControllerError
from .hub import WsHub
from .security import OriginGuardMiddleware

log = logging.getLogger("ilh.server")
WEB_DIR = Path(__file__).resolve().parents[1] / "web"


def open_in_file_manager(path: Path) -> None:
    """Open the folder in Explorer/Finder — the report and transcript are read outside."""
    import subprocess
    import sys

    try:
        if sys.platform == "win32":
            os.startfile(path)  # noqa: S606 — the path is ours, not straight from a request
        elif sys.platform == "darwin":
            subprocess.run(["open", str(path)], check=True)  # noqa: S603,S607 — fixed argv
        else:
            subprocess.run(["xdg-open", str(path)], check=True)  # noqa: S603,S607
    except (OSError, subprocess.SubprocessError) as e:
        raise ProjectError(f"Could not open the folder {path}: {e}") from e


class GuideParseRequest(BaseModel):
    text: str


class GuideSaveRequest(BaseModel):
    guide: dict
    source_text: str = ""


class SessionStartRequest(BaseModel):
    mic_index: int
    system_index: int
    guide: dict
    duration_min: int | None = None
    asr_vocabulary: str = ""
    project_id: str | None = None
    guide_text: str = ""  # the original guide text — this is what the interview screen shows


class ProjectCreateRequest(BaseModel):
    title: str
    guide: dict | None = None
    asr_vocabulary: str = ""
    duration_min: int | None = None
    guide_text: str = ""
    llm_instructions: str = ""


class ProjectUpdateRequest(BaseModel):
    """None in a field means "leave it as it was", not "clear it"."""

    title: str | None = None
    guide: dict | None = None
    asr_vocabulary: str | None = None
    duration_min: int | None = None
    guide_text: str | None = None
    llm_instructions: str | None = None


class PromptSaveRequest(BaseModel):
    text: str


class ProjectOpenRequest(BaseModel):
    session_id: str | None = None  # without it we open the project folder itself


class TopicStatusRequest(BaseModel):
    topic_id: str
    status: Literal["covered", "partial", "not_covered"] | None = None


class DismissRequest(BaseModel):
    topic_id: str


class MonitorRequest(BaseModel):
    mic_index: int | None = None
    system_index: int | None = None


class FlagRequest(BaseModel):
    note: str = ""
    anchor: str = ""          # the guide question selected at the moment of typing
    anchor_section: str = ""  # the guide section that question belongs to
    topic_id: str | None = None


class QuestionAddRequest(BaseModel):
    text: str


class QuestionDoneRequest(BaseModel):
    question_id: int
    done: bool = True


class SetupDownloadRequest(BaseModel):
    items: list[str] | None = None


def create_app(cfg: AppConfig) -> FastAPI:
    hub = WsHub()
    controller = AppController(cfg, hub)
    provisioner = Provisioner(cfg, hub, controller.llm)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        hub.set_loop(asyncio.get_running_loop())
        # The previous run may have been cut short mid-interview: repair its
        # folder before the user starts a new one (see storage/recovery).
        try:
            await asyncio.to_thread(controller.recover_crashed_sessions)
        except Exception:
            log.exception("Recovery of interrupted sessions failed")
        yield
        await controller.stop_monitor()
        if controller.state in ("running", "starting"):
            try:
                await controller.stop_session()
            except Exception:
                log.exception("Error stopping the session while shutting the server down")

    app = FastAPI(title="Interviewer's Little Helper", lifespan=lifespan)
    app.state.controller = controller
    # Ahead of routing, so it covers REST and /ws alike (see security.py).
    app.add_middleware(OriginGuardMiddleware, cfg=cfg)

    @app.exception_handler(ControllerError)
    async def controller_error_handler(request, exc: ControllerError):
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    @app.exception_handler(ProjectError)
    async def project_error_handler(request, exc: ProjectError):
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    @app.exception_handler(PromptError)
    async def prompt_error_handler(request, exc: PromptError):
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    # ---------------------------------------------------------------- static

    @app.get("/")
    async def index():
        return FileResponse(WEB_DIR / "index.html")

    @app.get("/app.css")
    async def css():
        return FileResponse(WEB_DIR / "app.css", media_type="text/css")

    @app.get("/app.js")
    async def js():
        return FileResponse(WEB_DIR / "app.js", media_type="text/javascript")

    # ------------------------------------------------------------------- API

    @app.get("/api/state")
    async def state():
        return controller.snapshot()

    @app.get("/api/devices")
    async def devices():
        from ..audio.capture import list_input_devices

        try:
            return {"devices": await asyncio.to_thread(list_input_devices), "error": None}
        except Exception as e:
            log.exception("Could not enumerate the audio devices")
            return {"devices": [], "error": f"Audio devices are unavailable: {e}"}

    @app.get("/api/llm/status")
    async def llm_status():
        return await controller.llm.check()

    # --------------------------------------------------------------- prompts

    @app.get("/api/prompts")
    async def prompts_list():
        return {"prompts": controller.prompts.describe()}

    @app.put("/api/prompts/{key}")
    async def prompts_save(key: str, req: PromptSaveRequest):
        return controller.prompts.save(key, req.text)

    @app.delete("/api/prompts/{key}")
    async def prompts_reset(key: str):
        controller.prompts.reset(key)
        return {"ok": True}

    # ------------------------------------------------------ first-run wizard

    @app.get("/api/setup/status")
    async def setup_status():
        return await provisioner.refresh()

    @app.post("/api/setup/download")
    async def setup_download(req: SetupDownloadRequest):
        return provisioner.start(req.items)

    @app.post("/api/guide/parse")
    async def guide_parse(req: GuideParseRequest):
        guide = await controller.parse_guide(req.text)
        return guide.model_dump()

    # --------------------------------------------------------- guide library

    @app.get("/api/guides")
    async def guides_list():
        return {"guides": controller.library.list()}

    @app.post("/api/guides")
    async def guides_save(req: GuideSaveRequest):
        from ..guide.schemas import Guide

        try:
            guide = Guide.model_validate(req.guide)
        except Exception as e:
            raise ControllerError(f"Invalid guide structure: {e}") from e
        file_id = controller.library.save(guide, req.source_text)
        return {"file_id": file_id}

    @app.get("/api/guides/{file_id}")
    async def guides_load(file_id: str):
        try:
            return controller.library.load(file_id)
        except FileNotFoundError as e:
            raise ControllerError("The guide was not found in the library") from e
        except (ValueError, KeyError) as e:
            raise ControllerError(f"Could not load the guide: {e}") from e

    @app.delete("/api/guides/{file_id}")
    async def guides_delete(file_id: str):
        try:
            controller.library.delete(file_id)
        except ValueError as e:
            raise ControllerError(str(e)) from e
        return {"ok": True}

    # ------------------------------------------------------------- projects

    @app.get("/api/projects")
    async def projects_list():
        return {"projects": controller.projects.list()}

    @app.post("/api/projects")
    async def projects_create(req: ProjectCreateRequest):
        from ..guide.schemas import Guide

        guide = None
        if req.guide:
            try:
                guide = Guide.model_validate(req.guide)
            except Exception as e:
                raise ControllerError(f"Invalid guide structure: {e}") from e
        return controller.projects.create(
            req.title, guide, req.asr_vocabulary, req.duration_min,
            req.guide_text, req.llm_instructions,
        )

    @app.get("/api/projects/{project_id}")
    async def projects_load(project_id: str):
        return controller.projects.load(project_id)

    @app.patch("/api/projects/{project_id}")
    async def projects_update(project_id: str, req: ProjectUpdateRequest):
        return controller.projects.update(
            project_id,
            **{k: v for k, v in req.model_dump().items() if v is not None},
        )

    @app.delete("/api/projects/{project_id}")
    async def projects_delete(project_id: str):
        controller.projects.delete(project_id)
        return {"ok": True}

    @app.get("/api/projects/{project_id}/sessions")
    async def projects_sessions(project_id: str):
        return {"sessions": controller.projects.sessions(project_id)}

    @app.get("/api/projects/{project_id}/sessions/{session_id}")
    async def projects_session_detail(project_id: str, session_id: str):
        return controller.projects.session_detail(project_id, session_id)

    @app.get("/api/projects/{project_id}/sessions/{session_id}/transcript")
    async def projects_session_transcript(project_id: str, session_id: str):
        return controller.projects.session_transcript(project_id, session_id)

    @app.get("/api/projects/{project_id}/sessions/{session_id}/audio")
    async def projects_session_audio(project_id: str, session_id: str):
        """The recording, for the <audio> element in the interview viewer.

        FileResponse answers Range requests with a 206 by itself — without that
        the player could not seek, and clicking a timecode is the whole point.
        The file is still being written during a live interview, so no caching.
        """
        path = controller.projects.session_audio(project_id, session_id)
        return FileResponse(path, media_type="audio/wav", headers={"Cache-Control": "no-store"})

    @app.get("/api/projects/{project_id}/coverage")
    async def projects_coverage(project_id: str):
        return controller.projects.coverage(project_id)

    @app.post("/api/projects/{project_id}/open")
    async def projects_open(project_id: str, req: ProjectOpenRequest):
        """Reveal the folder: the transcript and report are read outside the app."""
        target = (
            controller.projects.session_dir(project_id, req.session_id)
            if req.session_id
            else controller.projects.dir(project_id)
        )
        await asyncio.to_thread(open_in_file_manager, target)
        return {"ok": True, "path": str(target)}

    # ---------------------------------------------------------- level monitor

    @app.post("/api/monitor/start")
    async def monitor_start(req: MonitorRequest):
        return await controller.start_monitor(req.mic_index, req.system_index)

    @app.post("/api/monitor/stop")
    async def monitor_stop():
        await controller.stop_monitor()
        return {"ok": True}

    # ---------------------------------------------------------------- session

    @app.post("/api/session/start")
    async def session_start(req: SessionStartRequest):
        return await controller.start_session(
            req.mic_index, req.system_index, req.guide, req.duration_min,
            req.asr_vocabulary, req.project_id, req.guide_text,
        )

    @app.post("/api/session/flag")
    async def session_flag(req: FlagRequest):
        return await controller.add_flag(
            req.note, req.anchor, req.anchor_section, req.topic_id
        )

    @app.post("/api/session/question")
    async def session_question(req: QuestionAddRequest):
        return await controller.add_question(req.text)

    @app.post("/api/session/question/done")
    async def session_question_done(req: QuestionDoneRequest):
        return await controller.set_question_done(req.question_id, req.done)

    @app.post("/api/topics/status")
    async def topic_status(req: TopicStatusRequest):
        await controller.set_topic_status(req.topic_id, req.status)
        return {"ok": True}

    @app.post("/api/recommendations/dismiss")
    async def rec_dismiss(req: DismissRequest):
        await controller.dismiss_recommendation(req.topic_id)
        return {"ok": True}

    @app.post("/api/session/stop")
    async def session_stop():
        return await controller.stop_session()

    @app.post("/api/session/analyze")
    async def analyze_now():
        controller.analyze_now()
        return {"ok": True}

    # ------------------------------------------------------------- WebSocket

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket):
        await hub.register(ws)
        try:
            await ws.send_json({"type": "snapshot", "payload": controller.snapshot()})
            while True:
                await ws.receive_text()  # incoming messages are not used
        except WebSocketDisconnect:
            pass
        except Exception:
            log.debug("A WS client dropped", exc_info=True)
        finally:
            hub.unregister(ws)

    return app
