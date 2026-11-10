import io

from app.cache import get_cache, make_key
from app.engine import get_engine


def generate_job(prompt: str, steps: int, seed: int) -> str:
    """Runs inside the worker process. Returns the cache key of the finished image."""
    cache = get_cache()
    key = make_key(prompt, steps, seed)

    if cache.get(key) is None:  # skip the model if this exact image already exists
        image = get_engine().generate(prompt, steps, seed)
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        cache.set(key, buf.getvalue())

    return key
