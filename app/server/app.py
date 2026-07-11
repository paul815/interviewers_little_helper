"""FastAPI-приложение: REST для управления + WebSocket для живых обновлений.
Слушает только 127.0.0.1, статика фронтенда отдаётся локально (без CDN)."""
from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from ..config import AppConfig
from .controller import AppController, ControllerError
from .hub import WsHub

log = logging.getLogger("ilh.server")
WEB_DIR = Path(__file__).resolve().parents[1] / "web"


class GuideParseRequest(BaseModel):
    text: str


class SessionStartRequest(BaseModel):
    mic_index: int
    system_index: int
    guide: dict


def create_app(cfg: AppConfig) -> FastAPI:
    hub = WsHub()
    controller = AppController(cfg, hub)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        hub.set_loop(asyncio.get_running_loop())
        yield
        if controller.state in ("running", "starting"):
            try:
                await controller.stop_session()
            except Exception:
                log.exception("Ошибка остановки сессии при выключении сервера")

    app = FastAPI(title="Interviewer's Little Helper", lifespan=lifespan)
    app.state.controller = controller

    @app.exception_handler(ControllerError)
    async def controller_error_handler(request, exc: ControllerError):
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    # --------------------------------------------------------------- статика

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
            log.exception("Не удалось перечислить аудиоустройства")
            return {"devices": [], "error": f"Аудиоустройства недоступны: {e}"}

    @app.get("/api/llm/status")
    async def llm_status():
        return await controller.llm.check()

    @app.post("/api/guide/parse")
    async def guide_parse(req: GuideParseRequest):
        guide = await controller.parse_guide(req.text)
        return guide.model_dump()

    @app.post("/api/session/start")
    async def session_start(req: SessionStartRequest):
        return await controller.start_session(req.mic_index, req.system_index, req.guide)

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
                await ws.receive_text()  # входящие сообщения не используются
        except WebSocketDisconnect:
            pass
        except Exception:
            log.debug("WS-клиент отвалился", exc_info=True)
        finally:
            hub.unregister(ws)

    return app
