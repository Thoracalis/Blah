"""
websocket.py – WebSocket connection manager.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from fastapi import WebSocket

logger = logging.getLogger(__name__)


class ConnectionManager:
    def __init__(self) -> None:
        self._active: list[WebSocket] = []

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self._active.append(ws)
        logger.info("WS connected. Active: %d", len(self._active))

    def disconnect(self, ws: WebSocket) -> None:
        if ws in self._active:
            self._active.remove(ws)
        logger.info("WS disconnected. Active: %d", len(self._active))

    async def broadcast(self, data: Any) -> None:
        if not self._active:
            return
        payload = json.dumps(data) if not isinstance(data, str) else data
        dead = []
        for ws in list(self._active):
            try:
                await ws.send_text(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)

    async def send_to(self, ws: WebSocket, data: Any) -> None:
        payload = json.dumps(data) if not isinstance(data, str) else data
        try:
            await ws.send_text(payload)
        except Exception as exc:
            logger.warning("Failed to send to WS: %s", exc)
            self.disconnect(ws)

    @property
    def n_connected(self) -> int:
        return len(self._active)
