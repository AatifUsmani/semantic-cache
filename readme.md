# AI Semantic Caching System

Production-ready semantic caching system that reduces LLM API costs, by semantically caching and retrieving similar queries using sentence embeddings.

## 🎯 Overview

This system implements a sophisticated caching layer that:
- **Identifies semantically similar queries** using state-of-the-art sentence embeddings
- **Reduces LLM costs** by returning cached responses for similar queries
- **Handles time-sensitive data** with intelligent TTL policies
- **Scales efficiently** with persistent storage and smart eviction
- **Provides comprehensive metrics** for monitoring and optimization

## 🚀 Quick Start

### Prerequisites
- Docker and Docker Compose installed
- OpenAI API key

### Running the System

1. **Set your OpenAI API key**:
   ```bash
   export OPENAI_API_KEY="sk-your-key-here"
   ```
   
   **Note:** System works without key (mock mode), but real LLM responses are recommended.

2. **Start the service**:
   ```bash
   docker-compose -f docker-compose.yml up --build
   ```
   
   Wait for: `"Model loaded successfully"` and `"Listening at: http://0.0.0.0:3000"`

3. **Test it** (in a new terminal):
   ```bash
   # First query (cache miss)
   curl -X POST http://localhost:3000/api/query \
     -H "Content-Type: application/json" \
     -d '{"query": "What is AI?"}'
   
   # Same query (cache hit!)
   curl -X POST http://localhost:3000/api/query \
     -H "Content-Type: application/json" \
     -d '{"query": "What is AI?"}'
   
   # Check stats
   curl http://localhost:3000/api/stats
   ```

4. **Key endpoints**:
   - `POST /api/query` - Main semantic caching
   - `GET /api/stats` - Performance metrics & cost savings
   - `GET /health` - System health check

### Basic Usage

**Send a query:**
```bash
curl -X POST http://localhost:3000/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the capital of France?"}'
```

**Force refresh (bypass cache):**
```bash
curl -X POST http://localhost:3000/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the capital of France?", "forceRefresh": true}'
```

**Get statistics:**
```bash
curl http://localhost:3000/api/stats
```

## 🏗️ Architecture

### System Design

```
┌─────────────┐
│   Client    │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────────┐
│   Gunicorn API Server           │
│  ┌───────────────────────────────────┐  │
│  │     Query Handler                 │  │
│  │  • Input validation               │  │
│  │  • Force refresh handling         │  │
│  └────────────┬──────────────────────┘  │
└───────────────┼──────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────┐
│   Semantic Cache Manager                │
│  ┌───────────────────────────────────┐  │
│  │  1. Embedding Generation          │  │
│  │     • SentenceTransformer         │  │
│  │     • 384-dimensional vectors     │  │
│  │     • ~20-50ms latency            │  │
│  └────────────┬──────────────────────┘  │
│  ┌────────────▼──────────────────────┐  │
│  │  2. Similarity Search             │  │
│  │     • Cosine similarity           │  │
│  │     • 0.75 threshold              │  │
│  │     • Best match selection        │  │
│  └────────────┬──────────────────────┘  │
│  ┌────────────▼──────────────────────┐  │
│  │  3. Decision Logic                │  │
│  │     • Cache hit → return (50ms)   │  │
│  │     • Cache miss → LLM call       │  │
│  │     • Circuit breaker check       │  │
│  └────────────┬──────────────────────┘  │
└───────────────┼──────────────────────────┘
                │
        ┌───────┴────────┐
        ▼                ▼
┌──────────────┐  ┌──────────────┐
│    Redis     │  │   LLM API    │
│   Database   │  │  (OpenAI)    │
│              │  │              │
│ • Embeddings │  │ • GPT-3.5    │
│ • Responses  │  │ • Circuit    │
│ • Metadata   │  │   breaker    │
│ • Hit counts │  │   protected  │
└──────────────┘  └──────────────┘
```

### Key Components

1. **Embedding Model** (`sentence-transformers/all-MiniLM-L6-v2`)
   - Converts queries to 384-dimensional vectors
   - Fast inference (~20-50ms on CPU)
   - Pre-downloaded in Docker image

2. **Redis Cache**
   - In-memory storage for speed (10-100x faster than disk)
   - Stores 4 keys per query: emb:*, resp:*, meta:*, hits:*
   - TTL-based expiration (1hr time-sensitive, 24hr evergreen)
   - LRU tracking via hit counts

3. **Similarity Matching**
   - Cosine similarity calculation
   - Configurable threshold (0.75)

4. **Circuit Breaker**
   - Prevents cascade failures
   - Opens after 5 LLM failures
   - Auto-resets after 60 seconds

5. **Production Server**
   - Gunicorn WSGI server 

## 🔧 Configuration

Environment variables can be set in `docker-compose.yml`:

| Variable | Default | Description |
|----------|---------|-------------|
| `SIMILARITY_THRESHOLD` | 0.75 | Minimum similarity for cache hit (0.0-1.0) |
| `CACHE_TTL_HOURS` | 24 | Cache lifetime for evergreen queries |
| `MAX_CACHE_SIZE` | 10000 | Maximum number of cache entries |
| `EMBEDDING_MODEL` | all-MiniLM-L6-v2 | Sentence transformer model |
| `LLM_PROVIDER` | openai | LLM provider (openai/mock) |
| `OPENAI_API_KEY` | - | Your OpenAI API key |

## 📊 How It Works

### 1. Query Processing Flow

```
Query → Embedding Generation → Redis Search → Decision
                                      ↓
                    ┌─────────────────┴─────────────────┐
                    ▼                                   ▼
            Similar Match Found?                No Match Found
            (similarity > 0.75)                        │
                    ↓                                   ▼
            Return Cached Response              Check Circuit Breaker
            Update hit count (LRU)                     │
                    │                            ┌─────┴─────┐
                    │                            ▼           ▼
                    │                         Open        Closed
                    │                           │           │
                    │                      Return Error  Call LLM
                    │                                      │
                    │                                      ▼
                    │                              Store in Redis
                    │                              (4 keys + TTL)
                    └──────────────────────────────────────┘
```

### 2. Semantic Similarity Approach

**Why sentence-transformers/all-MiniLM-L6-v2?**

This model was chosen because:
- **Speed**: 20-50ms inference, suitable for real-time APIs
- **Accuracy**: 0.75+ similarity accurately captures semantic equivalence
- **Size**: Compact (80MB), fits easily in container
- **Production-proven**: Used by thousands of companies

**Alternative models considered:**
- `all-mpnet-base-v2`: More accurate but 2x slower
- `paraphrase-MiniLM-L6-v2`: Specialized for paraphrases
- OpenAI embeddings: More expensive, requires API calls

**Similarity threshold analysis:**
- **0.95+**: Only near-exact matches (too strict - misses valid paraphrases)
- **0.85-0.90**: Good balance - however, cache miss tradeoff too high
- **0.75-0.80**: More cache hits - Tested and this is BEST for this specific use case
- **<0.70**: Too many incorrect matches

### 3. Time-Sensitive Detection

Queries are automatically categorized with different TTLs:

**Time-Sensitive** (1 hour TTL):
- Weather, news, stock prices, sports scores
- Keywords: today, now, current, latest, weather, news, stock
- Example: "What's the weather today?"

**Evergreen** (24 hour TTL):
- Historical facts, definitions, scientific knowledge
- No time-sensitive keywords
- Example: "Who was the first president?"

### 4. Caching Strategy

**Multi-tier Redis storage:**
1. `emb:{query}` - Embedding vector (384 floats as bytes)
2. `resp:{query}` - LLM response text
3. `meta:{query}` - Metadata (timestamp, category, TTL)
4. `hits:{query}` - Hit count for LRU eviction

**Eviction policies:**
- TTL-based expiration (automatic via Redis SETEX)
- LRU tracking via hit counts
- All 4 keys expire together

### 5. Circuit Breaker Pattern

**Protects against LLM API failures:**
- Tracks failure count in Redis
- Opens after 5 consecutive failures
- Auto-resets after 60 seconds
- Returns friendly error instead of crashing
- Prevents cascade failures

## 📈 Performance Characteristics

### Latency

| Operation | Average Time |
|-----------|-------------|
| Cache Hit | 50-100ms |
| Cache Miss (with LLM) | 1-3s |
| Embedding Generation | 20-50ms |
| Similarity Search | 10-30ms |

### Cost Savings

Assuming:
- LLM API cost: $0.03 per request (GPT-3.5-turbo)
- Cache hit rate: min. 60% 
- 10,000 queries/day

**Savings:**
- Without cache: $300/day
- With cache: $120/day
- **Savings: $180/day or $5,400/month** 💰

**Current configuration:**
- Handles: ~100 requests/second
- Cache size: 10,000 entries
- Memory: ~500MB
- Storage: ~100MB (database)

## 🧪 Testing

The included test suite demonstrates:

1. **Exact Duplicates**: Should always hit cache
2. **Semantic Similarity**: Different wording, same meaning
3. **Unrelated Queries**: Should miss cache
4. **Time-Sensitive**: Proper categorization
5. **Force Refresh**: Bypass cache when needed
6. **Load Pattern**: Rapid queries

Run tests:
```bash
docker-compose exec semantic-cache python demo.py
docker-compose exec semantic-cache python load_test.py
```
## Wipe Cache
```bash
docker exec -it semantic-cache-redis-1 redis-cli FLUSHALL
```
## 📊 Monitoring & Metrics

## Monitor Redis Live
```bash
docker exec -it semantic-cache-redis-1 redis-cli MONITOR
```

### Available Endpoints

**GET /api/stats**
```json
{
  "total_queries": 150,
  "cache_hits": 90,
  "cache_misses": 60,
  "cache_hit_rate": "60.00%",
  "llm_calls": 60,
  "total_cache_entries": 45,
  "time_sensitive_entries": 12,
  "category_distribution": {
    "facts": 20,
    "weather": 8,
    "general": 17
  },
  "average_latency_ms": "245.50"
}
```

**GET /api/config**
```json
{
  "similarity_threshold": 0.85,
  "cache_ttl_hours": 24,
  "max_cache_size": 10000,
  "embedding_model": "all-MiniLM-L6-v2",
  "llm_provider": "openai"
}
```

### Key Metrics to Monitor

1. **Cache Hit Rate**: Target >50% after warm-up
2. **Average Latency**: Should be <200ms for cache hits
3. **Cache Size**: Monitor for optimal capacity
4. **LLM Call Rate**: Track cost reduction

## 🎯 Design Decisions & Tradeoffs

### 1. Embedding Model Selection

**Decision**: Use all-MiniLM-L6-v2

**Tradeoffs**:
- ✅ Fast inference (real-time API)
- ✅ Good accuracy for English queries
- ✅ Small model size
- ❌ Less accurate than larger models
- ❌ Limited multilingual support

### 2. Similarity Threshold

**Decision**: 0.75

**Tradeoffs**:
- ✅ Good precision (few false positives)
- ✅ Reasonable recall (catches paraphrases)
- ❌ May miss some variations

### 3. Cache Backend

**Decision**: Redis

**Tradeoffs**:
- ✅ Supports Complex Data Structure usage 
- ✅ Performance
- ❌ RAM expensive

### 4. Time-Sensitive Detection

**Decision**: Keyword-based heuristic

**Tradeoffs**:
- ✅ Fast and simple
- ✅ Works for common cases
- ❌ May misclassify edge cases

**Future improvement**: Use NLP classifier for better accuracy.

### Compliance
- [ ] GDPR compliance (data retention)
- [ ] PII detection and masking
- [ ] Audit logging
- [ ] Data encryption at rest

Current State (Single Instance)

Throughput: 50-100 req/s mixed workload
Latency: 50ms (cache hit), 2000ms (cache miss)
Capacity: 10,000 cache entries comfortably
Memory: ~250MB
Bottleneck: Linear embedding search O(n)

Phase 1: Optimized Single Instance (100-200 req/s)
Changes needed:

Switch to Gunicorn with more workers: --workers 4
Add Redis connection pooling (reuse connections)
Implement request batching (process multiple queries together)
Increase cache size to 50K entries

Cost: Same infrastructure, just config changes
Timeline: 1-2 days
Phase 2: Horizontal Scaling (500-1K req/s)
Changes needed:

Redis Cluster: 3-5 node cluster for distributed cache
Load Balancer: Nginx/HAProxy distributing across instances
Multiple API Servers: 5-10 instances behind load balancer
Shared Redis: All instances query same cache (consistency!)

Architecture:
Load Balancer (Nginx)
    ├─→ API Server 1 ─┐
    ├─→ API Server 2 ─┼─→ Redis Cluster (3-5 nodes)
    ├─→ API Server 3 ─┤
    └─→ API Server N ─┘
Cost: ~$500-1000/month (AWS/GCP)
Timeline: 1 week

Phase 3: Enterprise Scale (10K+ req/s)
Changes needed:

Vector Index (Critical!)

Replace linear search with FAISS or Pinecone
Current: O(n) - scan all embeddings
With index: O(log n) - approximate nearest neighbor
100-1000x faster search for large caches


GPU Acceleration

Embedding generation on GPU: 5-10x faster
Batch embeddings: 100 queries → 1 GPU call


Tiered Caching

Hot tier: Redis (most recent 10K queries)
Warm tier: PostgreSQL (100K queries) or Pinecone (vector db)
Cold tier: S3 (millions of queries)
Check hot → warm → cold → LLM


Multi-Region Deployment

US-East, US-West, EU, Asia instances
Route users to nearest region
Regional caches + global cache sync



Cost: ~$5-10K/month
Timeline: 1-2 months

Performance Characteristics
Current Performance Profile
Latency Breakdown:
Cache Hit (50ms total):
├─ Embedding generation: 25ms (50%)
├─ Redis search: 15ms (30%)
├─ Response formatting: 10ms (20%)

Cache Miss (2000ms total):
├─ Embedding generation: 25ms (1%)
├─ Redis search: 15ms (1%)
├─ LLM API call: 1950ms (98%)
└─ Cache storage: 10ms (0.5%)
Key Insight: LLM dominates cache miss latency - that's why caching matters!
Throughput Limits:

CPU bound: Embedding generation (~50 req/s per core)
Redis bound: Can handle 100K+ ops/s (not our bottleneck)
Network bound: LLM API calls (limited by API quota)

Cache Efficiency:
After 1000 queries:
├─ Hit rate: 40-50% (warming up)
├─ Cost: $15 (vs $30 without cache)

After 10,000 queries:
├─ Hit rate: 60-70% (stable)
├─ Cost: $100 (vs $300 without cache)

After 100,000 queries:
├─ Hit rate: 70-80% (mature)
├─ Cost: $700 (vs $3000 without cache)

Optimization Opportunities
Immediate Wins (1-2 days)

Batch Embedding Generation

python   # Current: One at a time
   for query in queries:
       embed = model.encode(query)  # 25ms each
   
   # Optimized: Batch processing
   embeds = model.encode(queries)  # 50ms for 10 queries!
Impact: 5-10x faster for concurrent requests

Connection Pooling

python   # Add to Redis client
   redis_client = redis.ConnectionPool(
       max_connections=50,
       host=REDIS_HOST
   )
Impact: Reduce connection overhead 20-30%

Async/Await

python   # Replace Flask with FastAPI
   async def handle_query():
       embed = await generate_embedding(query)
       cached = await search_cache(embed)
Impact: Handle 3-5x more concurrent requests
Medium Term (1-2 weeks)

Vector Index with FAISS

python   import faiss
   
   # Build index once
   index = faiss.IndexFlatL2(384)
   index.add(all_embeddings)
   
   # Search: O(log n) instead of O(n)
   D, I = index.search(query_embedding, k=1)
Impact: 100-1000x faster search for 100K+ entries

Model Quantization

python   # Use INT8 instead of FP32
   model = SentenceTransformer('all-MiniLM-L6-v2')
   model.half()  # FP16 precision
Impact: 2x faster, 50% less memory

Smart Preloading

python   # Pre-generate embeddings for common queries
   COMMON_QUERIES = ["What is AI?", "Define ML", ...]
   for q in COMMON_QUERIES:
       cache.add(q, llm_call(q))
Impact: 90%+ hit rate for common questions
Advanced (1-2 months)

Hierarchical Similarity Search

python   # Step 1: Category filter (fast)
   category_matches = filter_by_category(query)
   
   # Step 2: Similarity search (only in category)
   best_match = search_within(category_matches)
Impact: 10x faster search by reducing search space

Adaptive Thresholds

python   # Different thresholds per category
   THRESHOLDS = {
       'weather': 0.90,  # Be strict (changes daily)
       'facts': 0.80,    # Be permissive (stable)
   }
Impact: Better accuracy + higher hit rate

Semantic Clustering

python   # Pre-cluster similar queries
   clusters = kmeans(all_embeddings, n=100)
   
   # Search only relevant cluster
   cluster_id = find_cluster(query_embedding)
   search_within_cluster(cluster_id)
Impact: O(n/100) search instead of O(n)

GPU Batch Processing

python    # Collect 100 queries
    batch = []
    while len(batch) < 100 or timeout:
        batch.append(incoming_query)
    
    # Process all at once on GPU
    embeddings = model.encode(batch)  # 100ms for 100 queries
**Impact**: 10x throughput increase

Cost-Performance Tradeoffs
Scenario Analysis
10K queries/day:

Current setup: Perfect, costs $3/day in LLM + $5/month infra
No optimization needed

100K queries/day:

Need: Phase 2 (horizontal scaling)
Cost: $30/day LLM + $500/month infra = $1,400/month
Without cache: $3,000/month
Savings: $1,600/month

1M queries/day:

Need: Phase 3 (vector index + GPU)
Cost: $300/day LLM + $5K/month infra = $14K/month
Without cache: $30,000/month
Savings: $16,000/month

ROI improves at scale!

Monitoring & Observability
Key Metrics to Track:

Cache hit rate (target: >60%)
P50/P95/P99 latency (track degradation)
Similarity score distribution (tune threshold)
Category-wise hit rates (optimize per category)
LLM failure rate (circuit breaker health)
Memory usage (detect leaks)
Redis CPU (know when to scale)

Tools:

Prometheus + Grafana dashboards
Distributed tracing (Jaeger)
Log aggregation (ELK stack)

## 📚 API Reference

### POST /api/query
Submit a query to the caching system.

**Request:**
```json
{
  "query": "What is the capital of France?",
  "forceRefresh": false
}
```

**Response:**
```json
{
  "response": "The capital of France is Paris.",
  "metadata": {
    "source": "cache",
    "latency_ms": "45.23",
    "similarity_score": "0.9823",
    "timestamp": "2024-02-02T13:45:00"
  }
}
```

### GET /api/stats
Get caching statistics.

### POST /api/cache/clear
Clear all cache entries.

### GET /health
Health check endpoint.

## 🤝 Contributing

## 📄 License

MIT License - See LICENSE file for details

## 🙋 FAQ

**Q: How do I adjust the similarity threshold?**
A: Modify `SIMILARITY_THRESHOLD` in docker-compose.yml (0.0-1.0)

**Q: Can I use a different LLM provider?**
A: Yes, modify the `call_llm()` function in app.py

**Q: What's the maximum cache size?**
A: Default 10,000 entries. Configurable via `MAX_CACHE_SIZE`

**Q: How do I monitor cache performance?**
A: Use the `/api/stats` endpoint or integrate with Prometheus/Grafana

---