import numpy as np
import pytest

import app as service


@pytest.fixture
def client():
    return service.app.test_client()


@pytest.mark.parametrize(
    "body, expected_error",
    [
        (["not", "an", "object"], "Query parameter required"),
        ({"query": 123}, "Query must be a string"),
        ({"query": "   "}, "Query cannot be empty"),
        ({"query": "hello", "forceRefresh": "yes"}, "forceRefresh must be a boolean"),
    ],
)
def test_invalid_query_requests_return_400(client, body, expected_error):
    response = client.post("/api/query", json=body)

    assert response.status_code == 400
    assert response.get_json()["error"] == expected_error


def test_provider_failure_is_not_cached(client, monkeypatch):
    class FakeRedis:
        def hincrby(self, key, field, amount):
            return 1

    monkeypatch.setattr(service, "redis_client", FakeRedis())
    monkeypatch.setattr(
        service, "get_embedding", lambda query: np.zeros(384, dtype=np.float32)
    )
    monkeypatch.setattr(service, "search_cache", lambda *args: (None, 0.0, None))
    monkeypatch.setattr(
        service,
        "call_llm",
        lambda query: (_ for _ in ()).throw(service.LLMUnavailable()),
    )
    monkeypatch.setattr(
        service,
        "store_in_cache",
        lambda *args: pytest.fail("failed provider responses must not be cached"),
    )

    response = client.post("/api/query", json={"query": "What is caching?"})

    assert response.status_code == 503
    assert response.get_json() == {"error": "LLM provider unavailable"}
