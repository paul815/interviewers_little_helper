"""Host and Origin checks for the local server.

Binding to 127.0.0.1 is not enough, for two reasons.

1. WebSocket does not obey CORS. While an interview is running, any tab open in
   the browser can do `new WebSocket("ws://127.0.0.1:8756/ws")`, get a snapshot
   back and read the transcript in real time — exactly the data whose privacy
   the application was made fully local for.
2. DNS rebinding: an attacker serves a domain that resolves to 127.0.0.1, and
   for the browser their page becomes "ours". Binding to loopback does not catch
   that — only checking the Host header does.

Both cases are closed by a single ASGI-level layer, ahead of routing, so it
works the same for REST and for /ws.
"""
from __future__ import annotations

import logging

from ..config import AppConfig

log = logging.getLogger("ilh.security")


def _host_only(value: str) -> str:
    """The host without the port. `[::1]:8756` -> `::1`, `127.0.0.1:8756` -> `127.0.0.1`."""
    value = value.strip()
    if value.startswith("["):  # IPv6 in brackets
        end = value.find("]")
        return value[1:end] if end > 0 else value
    return value.rsplit(":", 1)[0] if ":" in value else value


def _allowed_origins(cfg: AppConfig) -> set[str]:
    """Origins that our own page may send."""
    origins = set()
    for host in cfg.security.allowed_hosts:
        netloc = f"[{host}]" if ":" in host else host
        origins.add(f"http://{netloc}:{cfg.server.port}")
    return origins


class OriginGuardMiddleware:
    """ASGI layer: lets through only requests from our own page."""

    def __init__(self, app, cfg: AppConfig):
        self.app = app
        self.cfg = cfg
        self.allowed_hosts = {h.lower() for h in cfg.security.allowed_hosts}
        self.allowed_origins = _allowed_origins(cfg)

    async def __call__(self, scope, receive, send):
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return

        reason = self._reject_reason(scope)
        if reason is None:
            await self.app(scope, receive, send)
            return

        log.warning("Request rejected (%s): %s", reason, scope.get("path", ""))
        if scope["type"] == "websocket":
            await self._deny_ws(receive, send)
        else:
            await self._deny_http(send)

    # --------------------------------------------------------------- checks

    def _reject_reason(self, scope) -> str | None:
        headers = {k.decode("latin-1").lower(): v.decode("latin-1")
                   for k, v in scope.get("headers", [])}

        host = headers.get("host")
        if not host or _host_only(host).lower() not in self.allowed_hosts:
            return f"foreign Host: {host!r}"

        origin = headers.get("origin")
        if origin is None:
            # Not a browser: curl, a script, the developer themselves. Such a
            # client can read projects/ off the disk without HTTP access anyway,
            # so there is nothing to forbid.
            return "no Origin" if self.cfg.security.require_origin else None
        if origin not in self.allowed_origins:
            return f"foreign Origin: {origin!r}"
        return None

    # --------------------------------------------------------------- denials

    @staticmethod
    async def _deny_http(send) -> None:
        body = b'{"detail":"Origin not allowed"}'
        await send({
            "type": "http.response.start",
            "status": 403,
            "headers": [(b"content-type", b"application/json"),
                        (b"content-length", str(len(body)).encode())],
        })
        await send({"type": "http.response.body", "body": body})

    @staticmethod
    async def _deny_ws(receive, send) -> None:
        # Per the ASGI spec, websocket.connect must be answered with accept or
        # close; a close before accept turns into a rejected handshake.
        await receive()
        await send({"type": "websocket.close", "code": 1008})
