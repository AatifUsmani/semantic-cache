# Semantic Cache

A single-tenant Flask service that reuses LLM responses for semantically similar queries. It combines local sentence embeddings with Redis Stack vector search, bounded TTL storage, and an offline evaluation set for studying the cache's precision/recall tradeoff.

This is an experimental portfolio project, not a hosted or internet-facing service. It favors a clear, testable design over claims of production readiness or unmeasured performance.

## How it works

1. The service embeds a validated query with `sentence-transformers/all-MiniLM-L6-v2`.
2. Redis Stack's HNSW index finds the nearest cached embedding in the current cache scope.
3. If cosine similarity meets `SIMILARITY_THRESHOLD`, the cached response is returned; otherwise the configured provider is called.
4. Successful responses are stored with a time-to-live and metadata. A per-scope LRU list caps the number of entries.

Cache scope includes the deployment namespace, embedding model and dimensions, LLM provider/model, and prompt version. This prevents reuse across those configuration changes. Identical concurrent misses share a short-lived Redis lock and recheck the cache before making a provider call.

Time-sensitive TTL selection uses simple keyword heuristics, not a freshness guarantee. Similarity is also not a proof that two questions have the same intent; measure false positives on your own workload before lowering the threshold.

## Run locally

Requirements: Docker Engine and Docker Compose v2 (`docker compose`).

```bash
cp .env.example .env
docker compose up --build
```

By default, the service uses a mock LLM and requires no API key. The embedding model downloads from Hugging Face when the first query arrives, so that request may be slower and requires outbound network access. The API is bound to `127.0.0.1:3000`; Redis is available only on the Compose network and persists data in a named volume.

For live OpenAI responses, set `LLM_PROVIDER=openai`, `OPENAI_API_KEY`, and optionally `OPENAI_MODEL` in `.env`, then rebuild/restart the service. Requests sent in this mode are sent to OpenAI. Cache entries contain an embedding, response, and metadata, not a raw-prompt field; provider responses may quote user input. Request logs contain no prompt content.

## Try the API

```bash
curl -sS http://localhost:3000/health

curl -sS -X POST http://localhost:3000/api/query \
  -H 'Content-Type: application/json' \
  -d '{"query":"What is the capital of France?"}'
```

Send a paraphrase to exercise semantic matching. To bypass the cache and request a fresh response:

```bash
curl -sS -X POST http://localhost:3000/api/query \
  -H 'Content-Type: application/json' \
  -d '{"query":"What is the capital of France?","forceRefresh":true}'
```

`POST /api/cache/clear` is disabled unless `ADMIN_API_KEY` is configured. When enabled, it requires that key as a bearer token:

```bash
curl -sS -X POST http://localhost:3000/api/cache/clear \
  -H "Authorization: Bearer $ADMIN_API_KEY"
```

The query API itself has no caller authentication and is intentionally bound to localhost by the provided Compose file. Do not expose it publicly without adding authentication, authorization, rate limiting, and an appropriate tenant boundary. This implementation is single-tenant; `CACHE_NAMESPACE` isolates deployments, it is not an end-user identity mechanism.

## API

| Method | Endpoint | Purpose |
|---|---|---|
| `POST` | `/api/query` | Search the semantic cache or call the LLM |
| `GET` | `/api/stats` | Read request counts, hit rate, and cache capacity |
| `GET` | `/api/config` | Read non-secret runtime configuration |
| `POST` | `/api/cache/clear` | Clear this cache scope; requires `ADMIN_API_KEY` |
| `GET` | `/health` | Check Redis and vector-index readiness |

`/api/query` accepts a non-empty string `query` and an optional boolean `forceRefresh`. Queries longer than `MAX_QUERY_CHARS` return `400`; invalid requests are not sent to the embedding model or provider. Provider failures return `503` and are not cached. If Redis rejects a successful cache write, the LLM response is still returned with `metadata.cache_stored: false`.

The statistics endpoint reports request counts and observed cache hit rate only. It deliberately does not estimate dollar savings because real provider cost depends on model, prompt length, completion length, and pricing.

## Configuration

Copy `.env.example` to `.env`. Compose supplies Redis host and port; do not publish Redis outside the Compose network.

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `mock` | `mock` or `openai` |
| `OPENAI_API_KEY` | unset | Required when using the OpenAI provider |
| `OPENAI_MODEL` | `gpt-4o-mini` | OpenAI chat-completions model |
| `ADMIN_API_KEY` | unset | Enables cache clearing when set; use a high-entropy secret |
| `SIMILARITY_THRESHOLD` | `0.75` | Minimum cosine similarity in `[0, 1]`; not calibrated for every workload |
| `CACHE_TTL_EVERGREEN_SECONDS` | `86400` | TTL for queries not detected as time-sensitive |
| `CACHE_TTL_TIME_SENSITIVE_SECONDS` | `3600` | TTL for keyword-detected time-sensitive queries |
| `MAX_CACHE_SIZE` | `10000` | Maximum entries per cache scope; least-recently-used entries are evicted |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Sentence-Transformers model |
| `EMBEDDING_DIMENSIONS` | `384` | Must match the selected embedding model |
| `CACHE_NAMESPACE` | `local` | Deployment-level cache namespace, not tenant authentication |
| `MAX_QUERY_CHARS` | `4000` | Maximum accepted query length |

The prompt version is maintained in `app.py`. Increment it whenever the prompt or response-generation behavior changes so old responses do not cross that boundary.

## Evaluation and tests

The labeled pair set includes paraphrases and hard negatives (similar wording with a different intent). Run the offline threshold sweep with the configured embedding model:

```bash
python -m pip install -r requirements.txt
python evaluation/evaluate_semantic_cache.py
```

The output reports precision, recall, F1, accuracy, false positives, and false negatives at several thresholds. The dataset is small and hand-labeled; it is a reproducible starting point, not a benchmark or evidence that a threshold is safe for another domain.

Run unit tests:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-test.txt
python -m pytest -q
```

Redis Stack integration tests are skipped unless `REDIS_TEST_URL` is set. CI runs them against the same pinned Redis Stack image used by Compose. To run them locally, start a temporary Redis Stack container and point pytest at it:

```bash
docker run -d --rm --name semantic-cache-test-redis \
  -p 127.0.0.1:6379:6379 redis/redis-stack-server:7.4.0-v8
until docker exec semantic-cache-test-redis redis-cli ping; do sleep 1; done
REDIS_TEST_URL=redis://localhost:6379/0 python -m pytest -q
docker stop semantic-cache-test-redis
```

The interactive demo and load test call the running API. From the service container:

```bash
docker compose exec -e SEMANTIC_CACHE_API_URL=http://127.0.0.1:5000 semantic-cache python demo.py
docker compose exec -e SEMANTIC_CACHE_API_URL=http://127.0.0.1:5000 semantic-cache python load_test.py
```

Load-test results are workload- and machine-dependent; record your environment and do not treat a local run as a production capacity claim.

## Project layout

```text
app.py                          Flask API and provider flow
semantic_cache/redis_cache.py   Redis Stack vector index and bounded cache
semantic_cache/evaluation.py    Threshold metrics
evaluation/                     Labeled query pairs and offline evaluator
tests/                          Unit and Redis Stack integration tests
Dockerfile                      Non-root application image
docker-compose.yml              Local API and Redis Stack services
```

## License

MIT. See [LICENSE](LICENSE).
