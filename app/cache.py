import hashlib
import os
from functools import lru_cache


def make_key(prompt: str, steps: int, seed: int) -> str:
    raw = f"{prompt}|{steps}|{seed}"
    return "img:" + hashlib.sha256(raw.encode()).hexdigest()


class MemoryCache:
    """Used when REDIS_URL is not set (local dev and tests)."""

    name = "memory"

    def __init__(self) -> None:
        self._data: dict[str, bytes] = {}

    def get(self, key: str):
        return self._data.get(key)

    def set(self, key: str, value: bytes, ttl: int = 3600) -> None:
        self._data[key] = value


class RedisCache:
    name = "redis"

    def __init__(self, url: str) -> None:
        import redis

        self._client = redis.Redis.from_url(
            url, socket_connect_timeout=1, socket_timeout=1
        )

    def get(self, key: str):
        try:
            return self._client.get(key)
        except Exception:
            return None  # if Redis is down, act as a cache miss

    def set(self, key: str, value: bytes, ttl: int = 3600) -> None:
        try:
            self._client.set(key, value, ex=ttl)
        except Exception:
            pass  # never fail a request because the cache is down


@lru_cache(maxsize=1)
def get_cache():
    url = os.getenv("REDIS_URL")
    return RedisCache(url) if url else MemoryCache()
