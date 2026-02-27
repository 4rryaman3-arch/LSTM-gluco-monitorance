from __future__ import annotations

import asyncio
import hashlib
import json
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from app.cache import InMemoryTTLCache, RedisJsonCache
from app.config import settings
from app.schemas import ForecastRequest, ForecastResponse
from app.services.forecast import ForecastEngine
from app.services.model_adapter import OptionalLSTMAdapter


class ConnectionManager:
    def __init__(self) -> None:
        self.active_connections: set[WebSocket] = set()

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self.active_connections.add(websocket)

    def disconnect(self, websocket: WebSocket) -> None:
        self.active_connections.discard(websocket)

    async def broadcast(self, message: dict[str, Any]) -> None:
        dead: list[WebSocket] = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                dead.append(connection)
        for connection in dead:
            self.disconnect(connection)


def create_app() -> FastAPI:
    app = FastAPI(title="CGM Forecast API", version="1.0.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins if settings.cors_origins != ["*"] else ["*"],
        allow_methods=["*"],
        allow_headers=["*"],
        allow_credentials=True,
    )

    adapter = OptionalLSTMAdapter(
        checkpoint_path=settings.model_checkpoint, seq_len=settings.model_seq_len
    )
    engine = ForecastEngine(model_adapter=adapter)
    manager = ConnectionManager()

    if settings.redis_url:
        cache_backend = RedisJsonCache(settings.redis_url)
    else:
        cache_backend = InMemoryTTLCache()

    def cache_key(req: ForecastRequest) -> str:
        payload = req.model_dump(mode="json")
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
        return f"forecast:{digest}"

    async def resolve_forecast(req: ForecastRequest) -> ForecastResponse:
        key = cache_key(req)
        cached = await cache_backend.get(key)
        if cached is not None:
            return ForecastResponse.model_validate(cached)

        response = engine.forecast(req)
        await cache_backend.set(
            key, response.model_dump(mode="json"), ttl_seconds=settings.cache_ttl_seconds
        )
        return response

    @app.get("/api/v1/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "redis_enabled": bool(settings.redis_url),
            "model_active": adapter.model_active,
            "model_notes": adapter.notes,
        }

    @app.post("/api/v1/forecast", response_model=ForecastResponse)
    async def forecast(payload: ForecastRequest) -> ForecastResponse:
        response = await resolve_forecast(payload)
        await manager.broadcast(
            {
                "event": "broadcast_forecast",
                "request_id": response.request_id,
                "model": response.model.model_dump(),
            }
        )
        return response

    @app.websocket("/ws/forecast")
    async def forecast_ws(websocket: WebSocket) -> None:
        await manager.connect(websocket)
        try:
            while True:
                body = await websocket.receive_json()
                try:
                    req = ForecastRequest.model_validate(body)
                except ValidationError as exc:
                    await websocket.send_json({"event": "error", "detail": exc.errors()})
                    continue

                response = await resolve_forecast(req)
                await websocket.send_json(
                    {
                        "event": "start",
                        "request_id": response.request_id,
                        "model": response.model.model_dump(),
                    }
                )

                max_len = max(
                    len(response.without_insulin or []),
                    len(response.with_insulin or []),
                )
                for idx in range(max_len):
                    await websocket.send_json(
                        {
                            "event": "tick",
                            "index": idx,
                            "without_insulin": (
                                response.without_insulin[idx].model_dump(mode="json")
                                if response.without_insulin and idx < len(response.without_insulin)
                                else None
                            ),
                            "with_insulin": (
                                response.with_insulin[idx].model_dump(mode="json")
                                if response.with_insulin and idx < len(response.with_insulin)
                                else None
                            ),
                        }
                    )
                    await asyncio.sleep(0.08)

                await websocket.send_json(
                    {
                        "event": "complete",
                        "response": response.model_dump(mode="json"),
                    }
                )
        except WebSocketDisconnect:
            manager.disconnect(websocket)
        except Exception as exc:
            manager.disconnect(websocket)
            raise HTTPException(status_code=500, detail=f"WebSocket error: {exc}") from exc

    @app.exception_handler(ValidationError)
    async def validation_exception_handler(_, exc: ValidationError) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": exc.errors()})

    return app


app = create_app()
