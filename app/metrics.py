import time

from fastapi import Request
from prometheus_client import Counter, Histogram

REQUESTS = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "path", "status"],
)
LATENCY = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency in seconds",
    ["path"],
)
CACHE_RESULTS = Counter(
    "image_cache_total",
    "Image cache lookups",
    ["result"],
)


async def track_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = time.perf_counter() - start

    # Use the route pattern (like /jobs/{job_id}) so each job id isn't its own series
    route = request.scope.get("route")
    path = route.path if route else request.url.path

    REQUESTS.labels(request.method, path, str(response.status_code)).inc()
    LATENCY.labels(path).observe(elapsed)
    return response
