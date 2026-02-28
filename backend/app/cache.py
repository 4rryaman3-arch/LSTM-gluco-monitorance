from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from typing import Any

from redis.asyncio import Redis


class CacheBackend(ABC):
    @abstractmethod
    async def get(self, key: str) -> dict[str, Any] | None:
        raise NotImplementedError

    @abstractmethod
    async def set(self, key: str, value: dict[str, Any], ttl_seconds: int) -> None:
        raise NotImplementedError


class InMemoryTTLCache(CacheBackend):
    def __init__(self) -> None:
        self._store: dict[str, tuple[float, dict[str, Any]]] = {}

    async def get(self, key: str) -> dict[str, Any] | None:
        value = self._store.get(key)
        if value is None:
            return None
        expires_at, payload = value
        if time.time() > expires_at:
            self._store.pop(key, None)
            return None
        return payload

    async def set(self, key: str, value: dict[str, Any], ttl_seconds: int) -> None:
        self._store[key] = (time.time() + ttl_seconds, value)


class RedisJsonCache(CacheBackend):
    def __init__(self, redis_url: str) -> None:
        self.client = Redis.from_url(redis_url, decode_responses=True)

    async def get(self, key: str) -> dict[str, Any] | None:
        payload = await self.client.get(key)
        if payload is None:
            return None
        return json.loads(payload)

    async def set(self, key: str, value: dict[str, Any], ttl_seconds: int) -> None:
        await self.client.setex(key, ttl_seconds, json.dumps(value))
