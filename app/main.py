import io
import os
import time
from typing import Literal

from fastapi import Depends, FastAPI, Request
from fastapi.responses import Response
from pydantic import BaseModel, Field

from app.auth import require_api_key
from app.cache import get_cache, make_key
from app.engine import get_engine
from app.ratelimit import get_limiter

app = FastAPI(title="LCM Image API")


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=200)
    steps: Literal[1, 2, 4] = 4
    seed: int = Field(default=42, ge=0, le=2**31 - 1)


@app.get("/health")
def health():
    return {"status": "ok", "engine": os.getenv("ENGINE", "mock")}


@app.post("/generate", response_class=Response, dependencies=[Depends(require_api_key)])
def generate(req: GenerateRequest, request: Request):
    client_id = request.client.host if request.client else "unknown"
    if not get_limiter().allow(client_id):
        return Response(
            content='{"detail":"Rate limit exceeded. Try again in a minute."}',
            media_type="application/json",
            status_code=429,
            headers={"Retry-After": "60"},
        )

    cache = get_cache()
    key = make_key(req.prompt, req.steps, req.seed)

    cached = cache.get(key)
    if cached is not None:
        return Response(
            content=cached,
            media_type="image/png",
            headers={"X-Cache": "HIT", "X-Generation-Ms": "0"},
        )

    engine = get_engine()
    start = time.perf_counter()
    image = engine.generate(req.prompt, req.steps, req.seed)
    elapsed_ms = int((time.perf_counter() - start) * 1000)

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    png = buf.getvalue()
    cache.set(key, png)

    return Response(
        content=png,
        media_type="image/png",
        headers={
            "X-Engine": engine.name,
            "X-Generation-Ms": str(elapsed_ms),
            "X-Cache": "MISS",
        },
    )
