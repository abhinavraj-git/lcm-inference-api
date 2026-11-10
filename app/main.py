import io
import os
import time
from typing import Literal

from fastapi import FastAPI
from fastapi.responses import Response
from pydantic import BaseModel, Field

from app.engine import get_engine

app = FastAPI(title="LCM Image API")


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=200)
    steps: Literal[1, 2, 4] = 4
    seed: int = Field(default=42, ge=0, le=2**31 - 1)


@app.get("/health")
def health():
    return {"status": "ok", "engine": os.getenv("ENGINE", "mock")}


@app.post("/generate", response_class=Response)
def generate(req: GenerateRequest):
    engine = get_engine()
    start = time.perf_counter()
    image = engine.generate(req.prompt, req.steps, req.seed)
    elapsed_ms = int((time.perf_counter() - start) * 1000)

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return Response(
        content=buf.getvalue(),
        media_type="image/png",
        headers={"X-Engine": engine.name, "X-Generation-Ms": str(elapsed_ms)},
    )
