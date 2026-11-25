import concurrent.futures
import threading
import time

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


def test_query_length_is_bounded(client, monkeypatch):
    monkeypatch.setattr(service, "MAX_QUERY_CHARS", 3)

    response = client.post("/api/query", json={"query": "long"})

    assert response.status_code == 400
    assert response.get_json() == {"error": "Query must be at most 3 characters"}


def test_provider_failure_is_not_cached(client, monkeypatch):
    class FakeRedis:
        def hincrby(self, key, field, amount):
            return 1

        def set(self, *args, **kwargs):
            return True

        def eval(self, *args):
            return 1

    monkeypatch.setattr(service, "redis_client", FakeRedis())
    monkeypatch.setattr(
        service, "get_embedding", lambda query: np.ones(384, dtype=np.float32)
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


def test_identical_concurrent_misses_share_one_provider_call(monkeypatch):
    class FakeRedis:
        def __init__(self):
            self.lock = threading.Lock()
            self.locks = {}
            self.stats = {}

        def hincrby(self, key, field, amount):
            self.stats[field] = self.stats.get(field, 0) + amount
            return self.stats[field]

        def get(self, key):
            return None

        def set(self, key, value, nx=False, ex=None):
            with self.lock:
                if nx and key in self.locks:
                    return False
                self.locks[key] = value
                return True

        def exists(self, key):
            with self.lock:
                return key in self.locks

        def eval(self, script, key_count, key, token):
            with self.lock:
                if self.locks.get(key) == token:
                    del self.locks[key]
                    return 1
                return 0

    fake_redis = FakeRedis()
    cached_response = {}
    provider_calls = []
    monkeypatch.setattr(service, "redis_client", fake_redis)
    monkeypatch.setattr(service, "get_embedding", lambda query: np.ones(384))
    monkeypatch.setattr(
        service,
        "search_cache",
        lambda embedding: (
            (cached_response["response"], 1.0, {"hit_count": 1})
            if "response" in cached_response
            else (None, 0.0, None)
        ),
    )

    def call_llm(query):
        provider_calls.append(query)
        time.sleep(0.15)
        return "shared response"

    def store_in_cache(query, response, embedding):
        cached_response["response"] = response
        return True

    monkeypatch.setattr(service, "call_llm", call_llm)
    monkeypatch.setattr(service, "store_in_cache", store_in_cache)

    def send_request():
        with service.app.test_client() as test_client:
            return test_client.post("/api/query", json={"query": "same query"})

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(send_request)
        time.sleep(0.02)
        second = executor.submit(send_request)
        responses = [first.result(), second.result()]

    assert all(response.status_code == 200 for response in responses)
    assert [response.get_json()["response"] for response in responses] == [
        "shared response",
        "shared response",
    ]
    assert sorted(response.get_json()["metadata"]["source"] for response in responses) == [
        "cache",
        "llm",
    ]
    assert len(provider_calls) == 1


def test_query_reports_provider_and_cache_write_status(client, monkeypatch):
    monkeypatch.setattr(service, "increment_stat", lambda key: None)
    monkeypatch.setattr(
        service, "get_embedding", lambda query: np.ones(384, dtype=np.float32)
    )
    monkeypatch.setattr(
        service,
        "_resolve_query",
        lambda *args, **kwargs: (
            "An answer",
            "llm",
            0.0,
            None,
            False,
            2.5,
            12.0,
        ),
    )

    response = client.post("/api/query", json={"query": "What is caching?"})

    assert response.status_code == 200
    body = response.get_json()
    assert body["response"] == "An answer"
    assert body["metadata"]["source"] == "llm"
    assert body["metadata"]["cache_stored"] is False
    assert body["metadata"]["performance"]["search_ms"] == 2.5
    assert body["metadata"]["performance"]["llm_ms"] == 12.0


def test_cache_clear_requires_configured_admin_key(client, monkeypatch):
    monkeypatch.setattr(service, "ADMIN_API_KEY", "")

    response = client.post("/api/cache/clear")

    assert response.status_code == 503
    assert response.get_json()["error"] == "Administrative operations are not configured"


def test_cache_clear_rejects_invalid_admin_key(client, monkeypatch):
    monkeypatch.setattr(service, "ADMIN_API_KEY", "expected-secret")

    response = client.post(
        "/api/cache/clear",
        headers={"Authorization": "Bearer wrong-secret"},
    )

    assert response.status_code == 401
    assert response.get_json() == {"error": "Unauthorized"}


def test_cache_clear_accepts_valid_admin_key(client, monkeypatch):
    class FakeCache:
        def clear(self):
            return 3

    monkeypatch.setattr(service, "ADMIN_API_KEY", "expected-secret")
    monkeypatch.setattr(service, "cache", FakeCache())

    response = client.post(
        "/api/cache/clear",
        headers={"Authorization": "Bearer expected-secret"},
    )

    assert response.status_code == 200
    assert response.get_json()["entries_deleted"] == 3


def test_stats_uses_completed_cache_lookups_and_has_no_fake_costs(client, monkeypatch):
    class FakeCache:
        def entry_count(self):
            return 4

    monkeypatch.setattr(service, "cache", FakeCache())
    monkeypatch.setattr(
        service,
        "get_stats",
        lambda: {
            "total_queries": 10,
            "cache_hits": 3,
            "cache_misses": 5,
            "llm_calls": 5,
            "errors": 2,
        },
    )

    response = client.get("/api/stats")

    assert response.status_code == 200
    body = response.get_json()
    assert body["query_statistics"]["cache_hit_rate_percent"] == 37.5
    assert body["cache_statistics"]["entries"] == 4
    assert "cost_analysis" not in body


def test_time_sensitive_classification_uses_word_boundaries():
    assert service.is_time_sensitive("What is the weather today?")
    assert not service.is_time_sensitive("I know how to build a cache")


def test_category_classification_does_not_match_substrings():
    assert service.categorize_query("I was curious about the topic") == "general"
    assert service.categorize_query("What is the forecast?") == "weather"
    assert service.categorize_query("How do I build a forecasting model?") == "general"
