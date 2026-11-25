from __future__ import annotations

import hashlib
import json
import threading
import time
from typing import Any

import numpy as np
from redis.commands.search.field import TagField, VectorField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query
from redis.exceptions import ResponseError


_EVICT_LRU_SCRIPT = """
redis.call('ZADD', KEYS[1], ARGV[1], ARGV[2])
redis.call('EXPIRE', KEYS[1], ARGV[4])
local overflow = redis.call('ZCARD', KEYS[1]) - tonumber(ARGV[3])
if overflow > 0 then
    local victims = redis.call('ZRANGE', KEYS[1], 0, overflow - 1)
    for _, victim in ipairs(victims) do
        redis.call('DEL', victim)
    end
    redis.call('ZREMRANGEBYRANK', KEYS[1], 0, overflow - 1)
end
return overflow
"""

_RECORD_HIT_SCRIPT = """
if redis.call('EXISTS', KEYS[1]) == 0 then
    return -1
end
local hits = redis.call('HINCRBY', KEYS[1], 'hit_count', 1)
redis.call('ZADD', KEYS[2], ARGV[1], KEYS[1])
return hits
"""


def _decode(value: bytes | str) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else value


class RedisSemanticCache:
    """Redis Stack semantic cache scoped to one deployment and model prompt."""

    def __init__(
        self,
        redis_client: Any,
        *,
        namespace: str,
        embedding_model: str,
        embedding_dimensions: int,
        llm_provider: str,
        llm_model: str,
        prompt_version: str,
        similarity_threshold: float,
        max_entries: int,
        max_ttl_seconds: int,
        index_name: str | None = None,
    ) -> None:
        if embedding_dimensions <= 0:
            raise ValueError("embedding_dimensions must be positive")
        if not 0.0 <= similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be between 0 and 1")
        if max_entries <= 0:
            raise ValueError("max_entries must be positive")
        if max_ttl_seconds <= 0:
            raise ValueError("max_ttl_seconds must be positive")

        self.redis = redis_client
        self.embedding_dimensions = embedding_dimensions
        self.similarity_threshold = similarity_threshold
        self.max_entries = max_entries
        self.max_ttl_seconds = max_ttl_seconds
        self.scope = hashlib.sha256(
            "\0".join(
                (
                    namespace,
                    embedding_model,
                    str(embedding_dimensions),
                    llm_provider,
                    llm_model,
                    prompt_version,
                )
            ).encode("utf-8")
        ).hexdigest()
        self.entry_prefix = f"semantic-cache:v1:d{embedding_dimensions}:entry:{self.scope}:"
        self.lru_key = f"semantic-cache:v1:lru:{self.scope}"
        self.index_name = index_name or f"semantic_cache_v1_d{embedding_dimensions}"
        self._index_ready = False
        self._index_lock = threading.Lock()

    def entry_key(self, query: str) -> str:
        query_id = hashlib.sha256(query.strip().encode("utf-8")).hexdigest()
        return f"{self.entry_prefix}{query_id}"

    def refresh_lock_key(self, query: str) -> str:
        query_id = hashlib.sha256(query.strip().encode("utf-8")).hexdigest()
        return f"semantic-cache:v1:lock:{self.scope}:{query_id}"

    def ensure_index(self) -> None:
        if self._index_ready:
            return

        with self._index_lock:
            if self._index_ready:
                return

            index = self.redis.ft(self.index_name)
            try:
                index.info()
            except ResponseError as error:
                if not str(error).casefold().startswith("unknown index"):
                    raise
                fields = [
                    TagField("scope"),
                    VectorField(
                        "embedding",
                        "HNSW",
                        {
                            "TYPE": "FLOAT32",
                            "DIM": self.embedding_dimensions,
                            "DISTANCE_METRIC": "COSINE",
                            "M": 16,
                            "EF_CONSTRUCTION": 200,
                        },
                    ),
                ]
                definition = IndexDefinition(
                    prefix=[f"semantic-cache:v1:d{self.embedding_dimensions}:entry:"],
                    index_type=IndexType.HASH,
                )
                try:
                    index.create_index(fields, definition=definition)
                except ResponseError as create_error:
                    if "index already exists" not in str(create_error).casefold():
                        raise

            self._index_ready = True

    def search(self, embedding: np.ndarray) -> tuple[str | None, float, dict | None]:
        vector = self._vector_bytes(embedding)
        self.ensure_index()
        query = (
            Query(
                f"(@scope:{{{self.scope}}})=>[KNN 1 @embedding $vector AS distance]"
            )
            .return_fields("response", "metadata", "distance")
            .sort_by("distance")
            .paging(0, 1)
            .dialect(2)
        )
        result = self.redis.ft(self.index_name).search(
            query, query_params={"vector": vector}
        )
        if not result.docs:
            return None, 0.0, None

        document = result.docs[0]
        distance = float(_decode(document.distance))
        similarity = float(np.clip(1.0 - distance, -1.0, 1.0))
        if similarity < self.similarity_threshold:
            return None, 0.0, None

        key = _decode(document.id)
        metadata = json.loads(_decode(document.metadata))
        hit_count = self.redis.eval(
            _RECORD_HIT_SCRIPT,
            2,
            key,
            self.lru_key,
            time.time(),
        )
        if hit_count < 0:
            return None, 0.0, None
        metadata["hit_count"] = hit_count
        return _decode(document.response), similarity, metadata

    def store(
        self,
        query: str,
        response: str,
        embedding: np.ndarray,
        *,
        metadata: dict,
        ttl_seconds: int,
    ) -> None:
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")

        self.ensure_index()
        vector = self._vector_bytes(embedding)
        entry_key = self.entry_key(query)
        fields = {
            "scope": self.scope,
            "embedding": vector,
            "response": response,
            "metadata": json.dumps(metadata, separators=(",", ":")),
            "hit_count": 0,
        }
        with self.redis.pipeline(transaction=True) as pipeline:
            pipeline.hset(entry_key, mapping=fields)
            pipeline.expire(entry_key, ttl_seconds)
            pipeline.execute()

        self.redis.eval(
            _EVICT_LRU_SCRIPT,
            1,
            self.lru_key,
            time.time(),
            entry_key,
            self.max_entries,
            self.max_ttl_seconds,
        )

    def entry_count(self) -> int:
        self.ensure_index()
        query = (
            Query(f"@scope:{{{self.scope}}}")
            .no_content()
            .paging(0, 0)
            .dialect(2)
        )
        return int(self.redis.ft(self.index_name).search(query).total)

    def clear(self) -> int:
        deleted_entries = 0
        batch: list[bytes | str] = []
        for key in self.redis.scan_iter(match=f"{self.entry_prefix}*", count=500):
            batch.append(key)
            if len(batch) == 500:
                deleted_entries += self.redis.delete(*batch)
                batch.clear()
        if batch:
            deleted_entries += self.redis.delete(*batch)
        self.redis.delete(self.lru_key)
        return deleted_entries

    def _vector_bytes(self, embedding: np.ndarray) -> bytes:
        vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if vector.size != self.embedding_dimensions:
            raise ValueError(
                f"Expected an embedding with {self.embedding_dimensions} dimensions"
            )
        if not np.isfinite(vector).all():
            raise ValueError("Embedding must contain only finite values")

        norm = float(np.linalg.norm(vector))
        if norm == 0:
            raise ValueError("Embedding must not be a zero vector")
        return np.ascontiguousarray(vector / norm, dtype=np.float32).tobytes()
