# Features

A complete tour of what SimpleVecDB offers, grouped by capability. For
quick install + first query, start with the [README](../README.md). For
release-by-release detail, see the [Changelog](CHANGELOG.md).

## Storage & schema

- **Single-file SQLite** — one `.db` file (or `:memory:`) holds everything:
  documents, vectors, FTS5 index, edges, events, TTL, clusters.
- **Multi-collection** — isolated namespaces per database via
  `db.collection("name")`. Each collection has its own quantization,
  distance metric, and (optional) embedding storage.
- **WAL mode + 5 s busy timeout** — concurrent readers don't block writers,
  and `PRAGMA busy_timeout=5000` cuts `DatabaseLockedError` pressure under
  contention.
- **Foreign keys cascade** — deleting a doc cascade-cleans its pending
  vectors, edges, and TTL rows. The events feed is intentionally FK-less so
  the audit trail survives deletions.
- **Encryption (SQLCipher AES-256)** — `VectorDB(path, encryption_key=…)`,
  via `[encryption]` extra. Salt and key derivation hardened in v2.6.0.
- **Cross-process safety** — advisory file locking on `usearch` index files
  prevents two processes from corrupting the same index.
- **Vacuum** — `db.vacuum()` reclaims disk space; truncates WAL by default.

## Indexing & vectors

- **HNSW via usearch** — 10–100× faster than brute force on collections >10k.
- **Adaptive search** — brute-force for <10k vectors (perfect recall), HNSW
  above that. Override per query with `exact=True` / `exact=False`.
- **Quantization** — `FLOAT32`, `FLOAT16` (2× compression), `INT8` (4×),
  `BIT` (32×).
- **Distance metrics** — `COSINE`, `L2`. (L1 removed in v2.0.)
- **Hardware acceleration** — auto-detects CUDA / MPS / CPU + SIMD via
  `usearch`.
- **Index rebuild** — `collection.rebuild_index(connectivity=, expansion_add=,
  expansion_search=)` for tuning; `collection.maintenance.rebuild_if_needed(
  max_pending=, max_deleted=)` triggers only when thresholds are crossed.

## Search

- **Vector similarity** — `collection.similarity_search(vec | text, k=, filter=,
  exact=, threads=)`.
- **Batch vector search** — `similarity_search_batch(queries, k=)` for
  ~10× throughput.
- **Keyword (BM25)** — `collection.keyword_search(query, k=, filter=)`
  backed by SQLite FTS5.
- **Hybrid (BM25 + vector)** — `collection.hybrid_search(query, k=,
  query_vector=, vector_k=, keyword_k=, rrf_k=)` using Reciprocal Rank
  Fusion.
- **Max-Marginal-Relevance** — `collection.max_marginal_relevance_search(
  query, k=, fetch_k=, lambda_mult=)` for diversified results.
- **Cross-collection search** — `db.search_collections(query, collections=,
  k=, filter=, normalize_scores=, parallel=)` merges and re-ranks across
  collections.
- **Range / set filters (v2.6.1)** — Mongo-style operators in `filter=`:
  `$eq $ne $gt $gte $lt $lte $in $nin $exists $between`. Tuple shorthand
  (`(">", 0.5)`, `("range", lo, hi)`) is normalised into the operator-dict
  form. Works on `similarity_search`, `keyword_search`, `hybrid_search`,
  `edges.get_edges`, and `events.read`.

## Mutation & updates

- **`add_texts`** — batch insert with optional metadata, embeddings,
  parent IDs, and explicit thread count.
- **`add_texts_streaming`** — generator-driven ingestion for large
  datasets, with progress callbacks.
- **`delete_by_ids` / `remove_texts(filter=)`** — point and bulk deletes.
- **`update_metadata([(id, patch), …])`** — shallow-merge metadata batch.
- **`update_embedding(id, vector)` (v2.6.1)** — buffers a vector update in
  a per-collection `_pending_vectors` overlay. New vector becomes visible
  to reads immediately; promoted to HNSW on `pending.flush()`. Removes
  the HNSW remove+re-add churn previously required for in-place edits.
- **Bulk vector math (v2.6.1)** — `collection.pending.update_many([(id,
  vec), …])` and `collection.pending.blend_toward(ids, centroid, alpha)`.
- **Atomic counters (v2.6.1)** — `collection.increment_metadata(id,
  {"hits": 1, "drift": 0.02})` applies dict-of-deltas to JSON metadata in
  one statement; WAL-atomic and safe under concurrent writers.
- **Transactions (v2.6.1)** — `with db.transaction() as tx: …` and
  `with collection.tx(): …` wrap a SAVEPOINT around catalog writes
  (metadata, counters, edges, events, TTL, and `update_embedding`'s
  pending overlay). A raised exception rolls all SQL writes back. Coarse
  vector mutations (`add_texts`, `delete_by_ids`) are NOT rolled back —
  use `update_embedding` + `pending.flush()` for vector changes that
  must be commit-gated.

## Relationships

- **Document hierarchies (v2.1+)** — `add_texts(..., parent_ids=…)`,
  `set_parent`, `get_children`, `get_parent`, `get_descendants`,
  `get_ancestors`. Useful for chunked-doc retrieval where children are
  the search target but the parent provides context.
- **Weighted directed edges (v2.6.1)** — `collection.edges` namespace:
  - `add_edge(src, dst, kind=, weight=, bonus=, hits=, metadata=)`
  - `update_edge(src, dst, kind=, dweight=, dbonus=, dhits=)` — deltas
    compile to a single atomic SQL UPDATE.
  - `get_edges(src=, dst=, kind=, filter=, limit=)` — supports
    range/set filters on numeric columns.
  - `delete_edge`, `prune_edges`. Edges have their own `last_touch`
    timestamp.

## Lifecycle & memory primitives (v2.6.1)

These primitives turn the database into a substrate for retrieval-with-
memory systems (frecency, decay, change feeds, expiry).

- **TTL / expiry** —
  - `collection.ttl.set(doc_id, seconds=… | expires_at=…, on_expire=
    "delete" | "callback")`
  - `collection.ttl.clear(doc_id)`
  - `collection.ttl.sweep(now=, limit=)` returns `(deleted_ids,
    callback_ids)`.
  - `collection.ttl.start_background(interval=…)` runs the sweep in a
    daemon thread (off by default).
- **Append-only event feed** — every mutating method appends one row to a
  per-collection `_events` table (kind, doc_id, payload, monotonic seq).
  - `collection.events.read(since=, kind=, limit=)`
  - `collection.events.subscribe(since=, poll_interval=)`
  - `collection.events.prune(before_seq=)`
  - `collection.events.last_seq()`
- **Incremental rebuild scheduler** — `collection.maintenance.rebuild_if_needed(
  max_pending=, max_deleted=)` triggers a full `rebuild_index()` only when
  the configured pending / tombstone / wall-time thresholds are crossed.

## Clustering (v2.2+)

- **Algorithms** — K-means, MiniBatch K-means, HDBSCAN.
- **Workflow** — `cluster() → auto_tag() → assign_cluster_metadata()`.
- **Auto-tag methods** — `keywords`, `tfidf`, or a `custom_callback`.
- **Persistence** — `save_cluster`, `load_cluster`, `list_clusters`,
  `delete_cluster`, `assign_to_cluster` for fast assignment of new
  documents.
- **Discovery** — `get_cluster_members(cluster_id)`.
- See the [Clustering Guide](guides/clustering.md) for tuning advice.

## Document management

- `collection.get_documents(filter_dict=, limit=, offset=)` —
  paginated catalog access.
- `collection.get_embeddings_by_ids([…])` — fetch stored embeddings (when
  `store_embeddings=True`).
- `collection.count()`, `collection.dim`.
- `db.list_collections()`, `db.delete_collection(name)`.

## Async API

- `AsyncVectorDB` and `AsyncVectorCollection` mirror the entire sync
  surface — every method listed above has an async equivalent.
- **Executor injection (v2.4+)** — pass `executor=ThreadPoolExecutor(...)`
  to share a pool across async instances (important for ONNX / usearch
  thread-safety).
- **Lifecycle** — `async with AsyncVectorDB(...)` drains the executor
  with `wait=True` before closing the SQLite connection, so pool threads
  finish before the underlying connection goes away.
- For a manual smoke runner that walks the entire async surface, see
  `scripts/exercise_async_collection.py` in the repository.

## Integrations

- **LangChain** — `db.as_langchain(embeddings, collection_name=…)` returns
  a `VectorStore`-compatible adapter. Supports all search methods.
- **LlamaIndex** — `db.as_llama_index(collection_name=…)` returns a
  `BasePydanticVectorStore`-compatible adapter.
- **FastAPI embeddings server** — `[server]` extra adds a local HTTP
  server with HuggingFace models, CORS, graceful shutdown, input
  validation, and model warm-up (v2.5+).

## Types & constants

- `simplevecdb.types`: `Document`, `DistanceStrategy`, `Quantization`,
  `Edge`, `Event`, `TTLEntry` (frozen dataclasses where applicable).
- `simplevecdb.constants` (v2.6.1) — tunables exposed as named
  constants:
  - `PENDING_FLUSH_DEFAULT_BATCH = 1000`
  - `EVENTS_POLL_INTERVAL_S = 0.1`
  - `EVENTS_RETENTION_LIMIT = 100_000`
  - `TTL_SWEEP_DEFAULT_INTERVAL_S = 60.0`
  - `REBUILD_PENDING_THRESHOLD = 5_000`
  - `REBUILD_TOMBSTONE_THRESHOLD = 5_000`
  - `REBUILD_MIN_INTERVAL_S = 3600.0`
  - `SQLITE_BUSY_TIMEOUT_MS = 5000`

## Performance snapshot

10,000 vectors, 384 dimensions, k=10 search:

| Quantization | Storage | Query  | Compression |
| :----------- | :------ | :----- | :---------- |
| FLOAT32      | 36.0 MB | 0.20 ms | 1×         |
| FLOAT16      | 28.7 MB | 0.20 ms | 2×         |
| INT8         | 25.0 MB | 0.16 ms | 4×         |
| BIT          | 21.8 MB | 0.08 ms | 32×        |

Full benchmarks and tuning guidance:
[Benchmarks](benchmarks.md).

## Roadmap

Implemented features track the [Changelog](CHANGELOG.md). Currently on the
near-term radar:

- Incremental clustering (online learning)
- Cluster visualization exports

Vote on these or propose new ones in
[GitHub Discussions](https://github.com/coderdayton/simplevecdb/discussions).
