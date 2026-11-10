import os
import time
from functools import lru_cache

LIMIT = int(os.getenv("RATE_LIMIT_PER_MIN", "10"))
WINDOW_SECONDS = 60


class MemoryLimiter:
    """Used when REDIS_URL is not set (local dev and tests)."""

    def __init__(self) -> None:
        self._window = 0
        self._hits: dict[str, int] = {}

    def allow(self, client_id: str) -> bool:
        window = int(time.time() // WINDOW_SECONDS)
        if window != self._window:  # a new minute started, forget the old counts
            self._window = window
            self._hits = {}
        self._hits[client_id] = self._hits.get(client_id, 0) + 1
        return self._hits[client_id] <= LIMIT


class RedisLimiter:
    def __init__(self, url: str) -> None:
        import redis

        self._client = redis.Redis.from_url(
            url, socket_connect_timeout=1, socket_timeout=1
        )

    def allow(self, client_id: str) -> bool:
        window = int(time.time() // WINDOW_SECONDS)
        key = f"rl:{client_id}:{window}"
        try:
            count = self._client.incr(key)
            if count == 1:
                self._client.expire(key, WINDOW_SECONDS * 2)
            return count <= LIMIT
        except Exception:
            return True  # if Redis is down, let the request through


@lru_cache(maxsize=1)
def get_limiter():
    url = os.getenv("REDIS_URL")
    return RedisLimiter(url) if url else MemoryLimiter()
