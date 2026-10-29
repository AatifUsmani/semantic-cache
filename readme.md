# Semantic Cache

A small Flask service that caches LLM responses by meaning rather than exact wording. It creates sentence embeddings for incoming queries, searches Redis for a sufficiently similar cached query, and calls the configured LLM when there is no match.

## What it does

- Embeds queries with `sentence-transformers/all-MiniLM-L6-v2`.
- Searches cached embeddings with cosine similarity (default threshold: `0.75`).
- Returns a cached response when a match is found; `forceRefresh` can bypass the cache.
- Assigns shorter TTLs to queries identified as time-sensitive and longer TTLs to other queries.
- Tracks cache and request statistics through an API endpoint.
- Includes a circuit breaker for LLM failures.

## Run locally with Docker

Requirements: Docker, Docker Compose, and an OpenAI API key for live LLM responses. The service can also run in mock mode without a key.

```bash
export OPENAI_API_KEY="your-key"
docker-compose -f docker-compose.yml up --build
```

The API listens on `http://localhost:3000`.

## Try it

```bash
curl -X POST http://localhost:3000/api/query \
  -H "Content-Type: application/json" \
  -d '{"query":"What is the capital of France?"}'
```

Send the same query again to check the cache path. To bypass the cache and request a fresh response:

```bash
curl -X POST http://localhost:3000/api/query \
  -H "Content-Type: application/json" \
  -d '{"query":"What is the capital of France?","forceRefresh":true}'
```

## API

| Method | Endpoint | Purpose |
|---|---|---|
| `POST` | `/api/query` | Query the cache or LLM |
| `GET` | `/api/stats` | View cache and request statistics |
| `GET` | `/api/config` | View active configuration |
| `POST` | `/api/cache/clear` | Clear cache entries |
| `GET` | `/health` | Check service health |

Example request:

```json
{
  "query": "What is the capital of France?",
  "forceRefresh": false
}
```

## Configuration

Set these variables in `docker-compose.yml`:

| Variable | Default | Purpose |
|---|---:|---|
| `SIMILARITY_THRESHOLD` | `0.75` | Minimum cosine similarity for a cache hit |
| `CACHE_TTL_HOURS` | `24` | TTL for queries treated as evergreen |
| `MAX_CACHE_SIZE` | `10000` | Maximum cache entries |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformer model |
| `LLM_PROVIDER` | `openai` | LLM provider (`openai` or `mock`) |
| `OPENAI_API_KEY` | — | API key for OpenAI responses |

Time-sensitive queries use a shorter TTL (one hour). Classification is keyword-based, so it can miss context and may treat some queries incorrectly.

## Implementation notes

Redis stores query embeddings, responses, metadata, and hit counts. Similarity search currently scans cached embeddings, so search work increases as the cache grows. The similarity threshold is a tradeoff: a lower value can increase cache hits but may return a response that is not appropriate for the new query.

## Tests and demos

Run the automated unit tests locally:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-test.txt
python -m pytest
```

The demo and load-test scripts exercise a running service:

```bash
docker-compose exec semantic-cache python demo.py
docker-compose exec semantic-cache python load_test.py
```

## License

MIT. See [LICENSE](LICENSE).
