import os
from functools import lru_cache

from app.tasks import generate_job


@lru_cache(maxsize=1)
def get_queue():
    """Returns an RQ queue backed by Redis. With no REDIS_URL, returns None."""
    url = os.getenv("REDIS_URL")
    if not url:
        return None
    from redis import Redis
    from rq import Queue

    return Queue("images", connection=Redis.from_url(url))


def enqueue(prompt: str, steps: int, seed: int) -> str:
    """Adds a job and returns its id."""
    queue = get_queue()
    job = queue.enqueue(
        generate_job, prompt, steps, seed, result_ttl=3600, job_timeout=300
    )
    return job.id
