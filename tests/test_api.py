import os

os.environ["ENGINE"] = "mock"  # tests never load the real model

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_generate_returns_png():
    r = client.post(
        "/generate", json={"prompt": "a red lighthouse", "steps": 4, "seed": 1}
    )
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"
    assert r.content[:8] == b"\x89PNG\r\n\x1a\n"
    assert r.headers["x-engine"] == "mock"


def test_same_seed_gives_same_image():
    body = {"prompt": "a red lighthouse", "steps": 2, "seed": 7}
    assert (
        client.post("/generate", json=body).content
        == client.post("/generate", json=body).content
    )


def test_rejects_invalid_steps():
    r = client.post("/generate", json={"prompt": "x", "steps": 3, "seed": 1})
    assert r.status_code == 422


def test_rejects_empty_prompt():
    r = client.post("/generate", json={"prompt": "", "steps": 4, "seed": 1})
    assert r.status_code == 422


def test_rejects_too_long_prompt():
    r = client.post("/generate", json={"prompt": "a" * 201, "steps": 4, "seed": 1})
    assert r.status_code == 422


def test_second_identical_request_is_a_cache_hit():
    body = {"prompt": "cache test unique prompt", "steps": 2, "seed": 7}

    first = client.post("/generate", json=body)
    second = client.post("/generate", json=body)

    assert first.headers["x-cache"] == "MISS"
    assert second.headers["x-cache"] == "HIT"
    assert first.content == second.content


def test_rate_limit_returns_429(monkeypatch):
    from app import ratelimit

    monkeypatch.setattr(ratelimit, "LIMIT", 3)
    ratelimit.get_limiter.cache_clear()  # start with fresh counters

    codes = []
    for i in range(5):
        body = {"prompt": f"rate limit test {i}", "steps": 1, "seed": i}
        codes.append(client.post("/generate", json=body).status_code)

    ratelimit.get_limiter.cache_clear()  # don't leave the low limit behind
    assert codes == [200, 200, 200, 429, 429]


def test_api_key_required_when_configured(monkeypatch):
    monkeypatch.setenv("API_KEY", "secret123")
    body = {"prompt": "auth test prompt", "steps": 1, "seed": 99}

    no_key = client.post("/generate", json=body)
    wrong_key = client.post("/generate", json=body, headers={"X-API-Key": "nope"})
    right_key = client.post("/generate", json=body, headers={"X-API-Key": "secret123"})

    assert no_key.status_code == 401
    assert wrong_key.status_code == 401
    assert right_key.status_code == 200
