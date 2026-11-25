import os
import uuid

import numpy as np
import pytest
import redis

from semantic_cache.redis_cache import RedisSemanticCache


def make_cache(
    client,
    *,
    namespace="tests",
    max_entries=10,
    threshold=0.8,
    index_name=None,
):
    return RedisSemanticCache(
        client,
        namespace=namespace,
        embedding_model="test-embedding-model",
        embedding_dimensions=2,
        llm_provider="mock",
        llm_model="mock",
        prompt_version="test-v1",
        similarity_threshold=threshold,
        max_entries=max_entries,
        max_ttl_seconds=60,
        index_name=index_name,
    )


@pytest.fixture
def redis_cache():
    redis_url = os.getenv("REDIS_TEST_URL")
    if not redis_url:
        pytest.skip("Set REDIS_TEST_URL to run Redis Stack integration tests")

    client = redis.Redis.from_url(redis_url, decode_responses=False)
    client.ping()
    index_name = f"semantic_cache_test_{uuid.uuid4().hex}"
    cache = make_cache(client, index_name=index_name)
    cache.ensure_index()
    try:
        yield client, cache
    finally:
        cache.clear()
        client.ft(index_name).dropindex(delete_documents=True)
        client.close()


def test_cache_scope_changes_with_model_and_prompt_version():
    client = object()
    base = make_cache(client)
    different_llm = RedisSemanticCache(
        client,
        namespace="tests",
        embedding_model="test-embedding-model",
        embedding_dimensions=2,
        llm_provider="mock",
        llm_model="another-model",
        prompt_version="test-v1",
        similarity_threshold=0.8,
        max_entries=10,
        max_ttl_seconds=60,
    )
    different_prompt = RedisSemanticCache(
        client,
        namespace="tests",
        embedding_model="test-embedding-model",
        embedding_dimensions=2,
        llm_provider="mock",
        llm_model="mock",
        prompt_version="test-v2",
        similarity_threshold=0.8,
        max_entries=10,
        max_ttl_seconds=60,
    )

    assert base.scope != different_llm.scope
    assert base.scope != different_prompt.scope
    assert "private question" not in base.entry_key("private question")


def test_redis_stack_search_is_scoped_and_tracks_hits(redis_cache):
    client, cache = redis_cache
    cache.store(
        "capital of France",
        "Paris",
        np.array([1.0, 0.0], dtype=np.float32),
        metadata={"category": "facts", "timestamp": "now"},
        ttl_seconds=60,
    )

    response, similarity, metadata = cache.search(np.array([1.0, 0.0]))

    assert response == "Paris"
    assert similarity == pytest.approx(1.0, abs=1e-5)
    assert metadata["category"] == "facts"
    assert metadata["hit_count"] == 1
    assert cache.entry_count() == 1
    assert client.ttl(cache.entry_key("capital of France")) > 0

    other_scope = make_cache(
        client,
        namespace="another-deployment",
        index_name=cache.index_name,
    )
    other_response, _, _ = other_scope.search(np.array([1.0, 0.0]))
    assert other_response is None


def test_similarity_threshold_rejects_unrelated_vectors(redis_cache):
    _, cache = redis_cache
    cache.store(
        "one topic",
        "response",
        np.array([1.0, 0.0], dtype=np.float32),
        metadata={},
        ttl_seconds=60,
    )

    response, similarity, metadata = cache.search(np.array([0.0, 1.0]))

    assert response is None
    assert similarity == 0.0
    assert metadata is None


def test_lru_limit_evicts_oldest_entry(redis_cache):
    client, original_cache = redis_cache
    cache = make_cache(
        client,
        namespace="bounded",
        max_entries=1,
        index_name=original_cache.index_name,
    )
    cache.store(
        "first",
        "first response",
        np.array([1.0, 0.0], dtype=np.float32),
        metadata={},
        ttl_seconds=60,
    )
    cache.store(
        "second",
        "second response",
        np.array([0.0, 1.0], dtype=np.float32),
        metadata={},
        ttl_seconds=60,
    )

    assert cache.entry_count() == 1
    response, _, _ = cache.search(np.array([1.0, 0.0]))
    assert response is None
