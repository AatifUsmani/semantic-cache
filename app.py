from __future__ import annotations

import hmac
import logging
import os
import re
import secrets
import threading
import time
from datetime import datetime, timezone

import numpy as np
import redis
from flask import Flask, jsonify, request
from redis.exceptions import RedisError

from semantic_cache.redis_cache import RedisSemanticCache


logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)


def _positive_int(name: str, default: int) -> int:
    value = int(os.getenv(name, default))
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.75"))
if not 0.0 <= SIMILARITY_THRESHOLD <= 1.0:
    raise ValueError("SIMILARITY_THRESHOLD must be between 0 and 1")

REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = _positive_int("REDIS_PORT", 6379)
CACHE_TTL_EVERGREEN = _positive_int("CACHE_TTL_EVERGREEN_SECONDS", 86400)
CACHE_TTL_TIME_SENSITIVE = _positive_int("CACHE_TTL_TIME_SENSITIVE_SECONDS", 3600)
MAX_CACHE_SIZE = _positive_int("MAX_CACHE_SIZE", 10000)
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
if not EMBEDDING_MODEL:
    raise ValueError("EMBEDDING_MODEL must not be empty")
EMBEDDING_DIMENSIONS = _positive_int("EMBEDDING_DIMENSIONS", 384)
CACHE_NAMESPACE = os.getenv("CACHE_NAMESPACE", "local").strip()
if not CACHE_NAMESPACE or len(CACHE_NAMESPACE) > 64:
    raise ValueError("CACHE_NAMESPACE must contain between 1 and 64 characters")

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "mock").strip().lower()
if LLM_PROVIDER not in {"mock", "openai"}:
    raise ValueError("LLM_PROVIDER must be either 'mock' or 'openai'")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
if LLM_PROVIDER == "openai" and not OPENAI_MODEL:
    raise ValueError("OPENAI_MODEL must not be empty")
if LLM_PROVIDER == "openai" and not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is required when LLM_PROVIDER=openai")

ADMIN_API_KEY = os.getenv("ADMIN_API_KEY", "")
MAX_QUERY_CHARS = _positive_int("MAX_QUERY_CHARS", 4000)
PROMPT_VERSION = "assistant-v1"
LLM_LOCK_WAIT_SECONDS = 65
LLM_LOCK_TTL_SECONDS = 90
STATS_KEY = "semantic-cache:v1:stats"
LLM_FAILURE_COUNT_KEY = "semantic-cache:v1:llm:failure-count"
LLM_CIRCUIT_OPEN_KEY = "semantic-cache:v1:llm:circuit-open"
MAX_FAILURES = 5
CIRCUIT_RESET_TIME = 60

redis_client = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    decode_responses=False,
    socket_connect_timeout=5,
    socket_timeout=5,
    retry_on_timeout=True,
)
cache = RedisSemanticCache(
    redis_client,
    namespace=CACHE_NAMESPACE,
    embedding_model=EMBEDDING_MODEL,
    embedding_dimensions=EMBEDDING_DIMENSIONS,
    llm_provider=LLM_PROVIDER,
    llm_model=OPENAI_MODEL if LLM_PROVIDER == "openai" else "mock",
    prompt_version=PROMPT_VERSION,
    similarity_threshold=SIMILARITY_THRESHOLD,
    max_entries=MAX_CACHE_SIZE,
    max_ttl_seconds=max(CACHE_TTL_EVERGREEN, CACHE_TTL_TIME_SENSITIVE),
)
STATS_KEY = f"{STATS_KEY}:{cache.scope}"
LLM_FAILURE_COUNT_KEY = f"{LLM_FAILURE_COUNT_KEY}:{cache.scope}"
LLM_CIRCUIT_OPEN_KEY = f"{LLM_CIRCUIT_OPEN_KEY}:{cache.scope}"

model = None
openai_client = None
_model_lock = threading.Lock()
_openai_client_lock = threading.Lock()


class LLMUnavailable(Exception):
    """Raised when the configured provider cannot produce a response."""


def increment_stat(key: str) -> None:
    redis_client.hincrby(STATS_KEY, key, 1)


def get_stats() -> dict[str, int]:
    stats_data = redis_client.hgetall(STATS_KEY)
    return {
        key.decode("utf-8") if isinstance(key, bytes) else key: int(value)
        for key, value in stats_data.items()
    }


def is_time_sensitive(query: str) -> bool:
    terms = (
        "today",
        "now",
        "current",
        "latest",
        "recent",
        "this week",
        "this month",
        "tonight",
        "yesterday",
        "tomorrow",
        "weather",
        "news",
        "breaking",
        "score",
        "scores",
        "stock",
        "price",
        "prices",
        "live",
        "update",
        "forecast",
    )
    normalized = query.casefold()
    return any(
        re.search(rf"(?<!\w){re.escape(term)}(?!\w)", normalized)
        for term in terms
    )


def categorize_query(query: str) -> str:
    normalized = query.casefold()
    categories = {
        "weather": ("weather", "temperature", "forecast", "rain", "snow", "climate"),
        "news": ("news", "breaking", "headline", "latest", "announcement"),
        "sports": ("score", "scores", "game", "match", "team", "player", "championship"),
        "finance": ("stock", "price", "market", "trading", "crypto", "bitcoin"),
        "facts": ("what is", "define", "explain", "meaning", "definition"),
        "history": (
            "history",
            "historical",
            "when was",
            "when did",
            "world war",
            "ancient",
            "century",
            "first person",
        ),
    }
    for category, terms in categories.items():
        if any(
            re.search(rf"(?<!\w){re.escape(term.strip())}(?!\w)", normalized)
            for term in terms
        ):
            return category
    return "general"


def get_embedding(text: str) -> np.ndarray:
    global model
    if model is None:
        with _model_lock:
            if model is None:
                from sentence_transformers import SentenceTransformer

                logger.info("Loading embedding model %s", EMBEDDING_MODEL)
                model = SentenceTransformer(EMBEDDING_MODEL)

    embedding = model.encode(
        text,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
    if vector.size != EMBEDDING_DIMENSIONS or not np.isfinite(vector).all():
        raise ValueError(
            "Embedding output does not match configured dimensions or contains non-finite values"
        )
    return vector


def search_cache(
    query_embedding: np.ndarray,
) -> tuple[str | None, float, dict | None]:
    return cache.search(query_embedding)


def store_in_cache(query: str, response: str, embedding: np.ndarray) -> bool:
    time_sensitive = is_time_sensitive(query)
    ttl = CACHE_TTL_TIME_SENSITIVE if time_sensitive else CACHE_TTL_EVERGREEN
    metadata = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "category": categorize_query(query),
        "time_sensitive": time_sensitive,
        "ttl_seconds": ttl,
    }
    cache.store(query, response, embedding, metadata=metadata, ttl_seconds=ttl)
    return True


def _get_openai_client():
    global openai_client
    if openai_client is None:
        with _openai_client_lock:
            if openai_client is None:
                from openai import OpenAI

                openai_client = OpenAI(
                    api_key=OPENAI_API_KEY,
                    timeout=30.0,
                    max_retries=1,
                )
    return openai_client


def call_llm(query: str) -> str:
    if redis_client.get(LLM_CIRCUIT_OPEN_KEY):
        raise LLMUnavailable("LLM circuit breaker is open")

    increment_stat("llm_calls")
    try:
        if LLM_PROVIDER == "openai":
            response = _get_openai_client().chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": query},
                ],
            )
            result = response.choices[0].message.content
            if not result or not result.strip():
                raise ValueError("LLM provider returned an empty response")
            result = result.strip()
        else:
            time.sleep(0.2)
            result = "Mock provider response. Configure the OpenAI provider for generated answers."

        redis_client.delete(LLM_FAILURE_COUNT_KEY)
        return result
    except Exception as error:
        logger.error("LLM provider failed: %s", type(error).__name__)
        failures = redis_client.incr(LLM_FAILURE_COUNT_KEY)
        if failures == 1:
            redis_client.expire(LLM_FAILURE_COUNT_KEY, CIRCUIT_RESET_TIME)
        if failures >= MAX_FAILURES:
            redis_client.setex(LLM_CIRCUIT_OPEN_KEY, CIRCUIT_RESET_TIME, "1")
            logger.error("LLM circuit breaker opened after %s failures", failures)
        raise LLMUnavailable("LLM provider failed") from error


def _release_refresh_lock(lock_key: str, token: str) -> None:
    redis_client.eval(
        """
        if redis.call('GET', KEYS[1]) == ARGV[1] then
            return redis.call('DEL', KEYS[1])
        end
        return 0
        """,
        1,
        lock_key,
        token,
    )


def _cached_result(embedding: np.ndarray):
    response_text, similarity, metadata = search_cache(embedding)
    if response_text is None:
        return None
    increment_stat("cache_hits")
    return response_text, similarity, metadata


def _generate_and_cache(
    query: str, embedding: np.ndarray, *, force_refresh: bool
) -> tuple[str, str, float, dict | None, bool, float]:
    if not force_refresh:
        increment_stat("cache_misses")
    llm_start = time.perf_counter()
    response_text = call_llm(query)
    llm_ms = (time.perf_counter() - llm_start) * 1000
    cache_stored = False
    try:
        cache_stored = store_in_cache(query, response_text, embedding)
    except RedisError:
        logger.exception("Could not store the generated response in Redis")
    return response_text, "llm", 0.0, None, cache_stored, llm_ms


def _resolve_query(
    query: str, embedding: np.ndarray, *, force_refresh: bool
) -> tuple[str, str, float, dict | None, bool, float, float]:
    if force_refresh:
        generated = _generate_and_cache(query, embedding, force_refresh=True)
        return (*generated[:5], 0.0, generated[5])

    search_ms = 0.0
    lookup_start = time.perf_counter()
    cached = _cached_result(embedding)
    search_ms += (time.perf_counter() - lookup_start) * 1000
    if cached is not None:
        return cached[0], "cache", cached[1], cached[2], False, search_ms, 0.0

    lock_key = cache.refresh_lock_key(query)
    lock_token = secrets.token_hex(16)
    deadline = time.monotonic() + LLM_LOCK_WAIT_SECONDS
    delay = 0.05
    while True:
        if redis_client.set(
            lock_key,
            lock_token,
            nx=True,
            ex=LLM_LOCK_TTL_SECONDS,
        ):
            try:
                lookup_start = time.perf_counter()
                cached = _cached_result(embedding)
                search_ms += (time.perf_counter() - lookup_start) * 1000
                if cached is not None:
                    return (
                        cached[0],
                        "cache",
                        cached[1],
                        cached[2],
                        False,
                        search_ms,
                        0.0,
                    )
                generated = _generate_and_cache(
                    query, embedding, force_refresh=False
                )
                return (*generated[:5], search_ms, generated[5])
            finally:
                _release_refresh_lock(lock_key, lock_token)

        if time.monotonic() >= deadline:
            logger.warning("Timed out waiting for an identical in-flight query")
            generated = _generate_and_cache(query, embedding, force_refresh=False)
            return (*generated[:5], search_ms, generated[5])

        time.sleep(delay)
        lookup_start = time.perf_counter()
        cached = _cached_result(embedding)
        search_ms += (time.perf_counter() - lookup_start) * 1000
        if cached is not None:
            return cached[0], "cache", cached[1], cached[2], False, search_ms, 0.0
        delay = min(delay * 2, 0.5)


@app.route("/api/query", methods=["POST"])
def handle_query():
    start_time = time.perf_counter()
    try:
        data = request.get_json(silent=True)
        if not isinstance(data, dict) or "query" not in data:
            return jsonify({"error": "Query parameter required"}), 400
        if not isinstance(data["query"], str):
            return jsonify({"error": "Query must be a string"}), 400

        query = data["query"].strip()
        if not query:
            return jsonify({"error": "Query cannot be empty"}), 400
        if len(query) > MAX_QUERY_CHARS:
            return jsonify({"error": f"Query must be at most {MAX_QUERY_CHARS} characters"}), 400

        force_refresh = data.get("forceRefresh", False)
        if not isinstance(force_refresh, bool):
            return jsonify({"error": "forceRefresh must be a boolean"}), 400

        increment_stat("total_queries")
        logger.info(
            "Processing query with %s characters (force_refresh=%s)",
            len(query),
            force_refresh,
        )

        embedding_start = time.perf_counter()
        query_embedding = get_embedding(query)
        embedding_ms = (time.perf_counter() - embedding_start) * 1000

        try:
            (
                response_text,
                source,
                similarity,
                cache_metadata,
                cache_stored,
                search_ms,
                llm_ms,
            ) = (
                _resolve_query(query, query_embedding, force_refresh=force_refresh)
            )
        except LLMUnavailable:
            increment_stat("errors")
            return jsonify({"error": "LLM provider unavailable"}), 503
        total_latency_ms = (time.perf_counter() - start_time) * 1000

        metadata = {
            "source": source,
            "latency_ms": round(total_latency_ms, 2),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "performance": {
                "embedding_ms": round(embedding_ms, 2),
                "search_ms": round(search_ms, 2) if search_ms else None,
                "llm_ms": round(llm_ms, 2) if llm_ms else None,
            },
        }
        if source == "cache":
            metadata["similarity_score"] = round(similarity, 4)
            metadata["cache_info"] = {
                "category": (cache_metadata or {}).get("category"),
                "hit_count": (cache_metadata or {}).get("hit_count"),
                "cached_at": (cache_metadata or {}).get("timestamp"),
            }
        else:
            metadata["cache_stored"] = cache_stored
            metadata["query_info"] = {
                "category": categorize_query(query),
                "time_sensitive": is_time_sensitive(query),
            }

        logger.info(
            "Request completed: source=%s latency_ms=%.2f",
            source,
            total_latency_ms,
        )
        return jsonify({"response": response_text, "metadata": metadata})
    except Exception:
        logger.exception("Error processing query")
        try:
            increment_stat("errors")
        except RedisError:
            logger.exception("Could not record request error")
        return jsonify({"error": "Internal server error"}), 500


@app.route("/api/stats", methods=["GET"])
def get_stats_endpoint():
    try:
        stats_data = get_stats()
        cache_hits = stats_data.get("cache_hits", 0)
        cache_misses = stats_data.get("cache_misses", 0)
        completed_cache_lookups = cache_hits + cache_misses
        hit_rate = (
            cache_hits / completed_cache_lookups * 100
            if completed_cache_lookups
            else 0.0
        )
        return jsonify(
            {
                "query_statistics": {
                    "total_queries": stats_data.get("total_queries", 0),
                    "cache_hits": cache_hits,
                    "cache_misses": cache_misses,
                    "cache_hit_rate_percent": round(hit_rate, 2),
                    "llm_calls": stats_data.get("llm_calls", 0),
                    "errors": stats_data.get("errors", 0),
                },
                "cache_statistics": {
                    "entries": cache.entry_count(),
                    "max_entries": MAX_CACHE_SIZE,
                    "ttl_evergreen_seconds": CACHE_TTL_EVERGREEN,
                    "ttl_time_sensitive_seconds": CACHE_TTL_TIME_SENSITIVE,
                },
                "matching": {
                    "similarity_threshold": SIMILARITY_THRESHOLD,
                    "embedding_model": EMBEDDING_MODEL,
                    "embedding_dimensions": EMBEDDING_DIMENSIONS,
                },
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )
    except Exception:
        logger.exception("Error retrieving statistics")
        return jsonify({"error": "Unable to retrieve statistics"}), 500


@app.route("/api/config", methods=["GET"])
def get_config():
    return jsonify(
        {
            "cache_namespace": CACHE_NAMESPACE,
            "embedding_model": EMBEDDING_MODEL,
            "embedding_dimensions": EMBEDDING_DIMENSIONS,
            "llm_provider": LLM_PROVIDER,
            "llm_model": OPENAI_MODEL if LLM_PROVIDER == "openai" else "mock",
            "similarity_threshold": SIMILARITY_THRESHOLD,
            "cache_ttl_evergreen_seconds": CACHE_TTL_EVERGREEN,
            "cache_ttl_time_sensitive_seconds": CACHE_TTL_TIME_SENSITIVE,
            "max_cache_entries": MAX_CACHE_SIZE,
        }
    )


@app.route("/api/cache/clear", methods=["POST"])
def clear_cache():
    if not ADMIN_API_KEY:
        return jsonify({"error": "Administrative operations are not configured"}), 503

    authorization = request.headers.get("Authorization", "")
    supplied_key = authorization.removeprefix("Bearer ").strip()
    if not hmac.compare_digest(
        supplied_key.encode("utf-8"), ADMIN_API_KEY.encode("utf-8")
    ):
        return jsonify({"error": "Unauthorized"}), 401

    try:
        deleted_count = cache.clear()
        return jsonify(
            {"message": "Cache cleared successfully", "entries_deleted": deleted_count}
        )
    except RedisError:
        logger.exception("Error clearing cache")
        return jsonify({"error": "Unable to clear cache"}), 500


@app.route("/health", methods=["GET"], endpoint="internal_health_check")
def health():
    try:
        redis_client.ping()
        cache.ensure_index()
        return jsonify(
            {
                "status": "healthy",
                "redis": "connected",
                "cache_index": "ready",
                "model_loaded": model is not None,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )
    except RedisError:
        logger.exception("Health check failed")
        return jsonify({"status": "unhealthy", "error": "Redis unavailable"}), 503


if __name__ == "__main__":
    logger.info(
        "Starting semantic cache: redis=%s:%s provider=%s threshold=%.3f",
        REDIS_HOST,
        REDIS_PORT,
        LLM_PROVIDER,
        SIMILARITY_THRESHOLD,
    )
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)
