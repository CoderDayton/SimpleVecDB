# Changelog

All notable changes to SimpleVecDB will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.6.0] - unreleased

### Fixed (concurrency & durability)

- **Atomic `UsearchIndex.save`** — now writes to a sibling `.tmp`, fsyncs, then `os.replace()`s onto the live path and fsyncs the parent directory. A crash mid-save can no longer corrupt the only copy of the index. Also moved the `_dirty` short-circuit inside `_write_lock` so a concurrent `add` cannot have its dirty flag silently cleared.
- **Atomic `rebuild_index`** — builds the new index at a sibling `.rebuild` path and atomically swaps it onto the live path; the old index remains the canonical copy until the swap succeeds.
- **Atomic encrypted save** — `encrypt_file` / `decrypt_file` now write to a sibling `.tmp`, fsync, set mode `0o600`, then `os.replace()`. `encrypt_index_file` only unlinks the plaintext after the encrypted output is durably on disk. A torn write can no longer leave the index unrecoverable.
- **`VectorDB`-level `RLock`** — a single re-entrant lock now serializes the `_collections` cache (no more check-then-insert TOCTOU on `collection()`) and is shared with every `CatalogManager` so all `with self.conn:` blocks across collections cannot interleave on the shared `sqlite3.Connection`. Reads remain lock-free at the SQLite level via WAL.
- **`AsyncVectorDB.close` drains** — switched from `executor.shutdown(wait=False)` to `wait=True` so in-flight pool tasks finish their cursors before the SQLite connection is closed. Pending (not-yet-started) work is still cancelled.
- **`set_parent` cycle check is transactional** — descendant lookup and parent UPDATE now run inside the same `with self._lock, self.conn:` block, closing a TOCTOU window where a concurrent edge could form a cycle.
- **Cluster persistence** — `_ensure_cluster_table`, `save_cluster_state`, `delete_cluster_state` now use `with self._lock, self.conn:` instead of bare `conn.commit()`; an exception during the execute is properly rolled back.
- **`add_documents` ID recovery is correct under upsert** — replaced the `last_insert_rowid()` arithmetic (which silently returned wrong IDs for batches mixing explicit and `None` IDs because UPSERTs do not advance the auto-increment counter) with a single `INSERT … RETURNING id` for the auto-ID rows. Explicit-ID rows still take the upsert path.
- **`delete_collection` closes cached indexes first** — any `VectorCollection` instances cached for the deleted name have their `UsearchIndex` closed before the file is unlinked, so a stale mmap view cannot race the unlink.

### Changed

- **`upsert_fts_rows` / `delete_fts_rows` are now `_upsert_fts_rows` / `_delete_fts_rows`** (private). The FTS shadow table must be updated inside the same transaction as the main table or it can desync on crash; the rename signals the contract.
- **`get_legacy_vectors`, `drop_legacy_vec_table`** now validate the supplied table name via `_validate_table_name` before interpolating into SQL.

### Added

- **Declared `python-dotenv` dependency** — `simplevecdb.config` already imported and called `load_dotenv` at package import; the missing dependency would `ImportError` on a clean install of the base package without optional extras.

### Fixed (correctness & quality)

- **RRF deduplication keys by document ID, not text** — `hybrid_search` previously deduped by `doc.page_content`, silently merging two distinct documents that happened to share text into one inflated-score result.
- **NaN/Inf guard at insert** — `add_texts` and `add_texts_streaming` reject non-finite vectors instead of feeding them to HNSW, which would produce undefined neighbours and could corrupt the graph.
- **`normalize_l2` handles subnormals** — replaced the exact `norm == 0` compare with a `< 1e-12` check (matching the existing usearch_index guard); subnormal floats no longer produce wildly large normalized vectors.
- **Silhouette score samples on large collections** — `silhouette_score` is O(n²); now caps the evaluation sample at `SILHOUETTE_MAX_SAMPLE = 10_000`. Large collections no longer OOM.
- **MMR maintains the selected matrix incrementally** — replaced per-iteration `np.stack(selected_embs)` with `np.vstack` of a running matrix. O(k²·d) wasted allocations dropped to O(k·d).
- **`_parse_bool_env` treats `KEY=` as unset** — empty strings now fall through to the default; previously they were truthy because `"".strip()` is not in the falsey set.
- **LangChain async methods use `asyncio.to_thread`** — `aadd_texts` / `asimilarity_search` / `amax_marginal_relevance_search` no longer block the event loop.
- **LlamaIndex `delete()` survives a process restart** — node IDs are persisted into document metadata under `_simplevecdb_node_id`; `delete()` falls back to a metadata query when the in-memory `_id_map` is empty.
- **LlamaIndex query results carry stable node IDs** — replaced `str(hash(page_content))` (process-randomized, collision-prone) with the persisted `_simplevecdb_node_id`.
- **`AsyncVectorDB.collection` accepts `store_embeddings`** — async callers can now enable embedding storage (required for `rebuild_index()`); previously they had no way to set it.

### Security

- **API key comparison uses `hmac.compare_digest`** — the prior `token not in allowed_keys` short-circuit leaked key prefixes via response time.
- **SQLCipher PRAGMA key always uses the `x'hex'` form** — every key path now goes through `_normalize_key` first, eliminating string interpolation of user-supplied passphrase characters into a quoted PRAGMA argument.
- **`is_database_encrypted` rejects zero-byte files** — previously a missing/empty DB looked like an unencrypted DB because `sqlite3.connect` would create a fresh one.

### Changed (tooling)

- **Ruff and mypy targets aligned with `requires-python>=3.10`** — both were `py312`, hiding 3.10/3.11 incompatibilities. Cleaned three resulting `F401` unused-import warnings (`signal` in models.py, `_batched` and `constants` re-imports).
- **Pre-commit version-sync hook** — `__init__.py` derives `__version__` dynamically via `importlib.metadata`, so `check_version_sync.py` was failing on every commit looking for a literal `__version__ = "x.y.z"` line that does not exist. The hook now validates only `pyproject.toml`'s version field. `bump_version.py` similarly stops trying to rewrite `__init__.py` and uses an anchored regex to update only the canonical version field.

### Added (hygiene & polish)

- **`ClusterResult` and `ClusterTagCallback` exported from `simplevecdb`** — they were return/argument types of public methods but had no public import path; users had to reach into `simplevecdb.types`.
- **`NullHandler` attached to the package's root logger** at import time, per the Python logging HOWTO. Idempotent — duplicate calls do not stack handlers.
- **`SimpleVecDBLlamaStore.delete_nodes` raises `NotImplementedError`** when called with `filters`, instead of silently dropping the filter portion and pretending the deletion succeeded.
- **Recursive CTE depth bound as a parameter** in `get_descendants` / `get_ancestors`. The previous f-string interpolation was safe due to `int()` coercion but is now one less line away from injection on a future refactor.
- **`Config.from_env()` documented** as returning the import-time-frozen instance; setting env vars after import does not refresh.
- **`ModelRegistry(allow_unlisted=...)` defaults to `False`** to match the secure-by-default config setting; programmatic instantiations no longer get an open registry by accident.
- **`/v1/usage` returns aggregated totals when auth is disabled** instead of leaking the per-IP buckets to anyone who hits the endpoint.
- **Server validates `EMBEDDING_SERVER_MAX_REQUEST_ITEMS <= _MAX_ENCODE_BATCH` at startup** so an out-of-range env var fails fast at boot rather than per request.
- **`pyproject.toml` gains `[project.urls]`, `classifiers`, and `keywords`** for a useful PyPI listing.
- **`.bandit` documents the B104 skip** and warns that any future `0.0.0.0` binding requires removing the skip.
- **Encrypted file format now carries a 3-byte header** (`'SV' + version`) so future format changes are detectable. `decrypt_file` accepts both the new v1 format and the v0 (pre-2.6.0) format, so existing encrypted indexes still load without re-encryption.

## [2.5.0] - 2026-04-07

### Added

- **`delete_collection(name)`** — drop a collection's SQLite tables, FTS index, and usearch file in one call. Available on both `VectorDB` and `AsyncVectorDB`.
- **`store_embeddings` parameter** on `collection()` — opt into storing embedding BLOBs in SQLite (default `False`). Saves ~2x storage; MMR transparently fetches vectors from the usearch index when BLOBs are absent.
- **`async_retry_on_lock` decorator** — async variant of `retry_on_lock` using `asyncio.sleep` instead of `time.sleep`, avoiding executor thread blocking.
- **`file_lock` context manager** — advisory cross-process file locking (`fcntl`/`msvcrt`) for usearch index files. Prevents corruption from concurrent processes.
- **`__repr__`** on `VectorDB`, `VectorCollection`, `AsyncVectorDB`, `AsyncVectorCollection` for debuggable string representations.
- **FLOAT16 quantization** fully implemented in `serialize()`/`deserialize()` — was previously defined in the enum but raised `ValueError` at runtime.
- **Pagination** on `get_documents(limit=, offset=)` and catalog methods (`find_ids_by_filter`, `find_ids_by_texts`) — previously returned unbounded result sets.
- **Embeddings server enhancements:**
  - Graceful shutdown with SIGTERM/SIGINT draining (10s timeout)
  - CORS middleware with configurable origins for browser-based clients
  - Model warm-up on startup (skip with `--no-warmup`)
  - Input validation: rejects empty strings (422) and texts exceeding 100k chars (413)
  - Proper `argparse` CLI with `--host`, `--port`, `--no-warmup`, `--help`
  - Startup banner logging config summary (host, port, model, auth, rate limits)
  - Nested token array normalization (`list[list[int]]` input format)
  - Async executor offload for `embed_texts` (non-blocking event loop)
  - OpenAPI version synced from package metadata
  - Module `__init__.py` exports (`embed_texts`, `get_embedder`, `load_model`, `app`, `run_server`)

### Fixed

- **`delete_by_ids` ordering** — SQLite deletion now happens first (transactional, can rollback), then usearch. Previously usearch removed first, leaving orphaned catalog entries on SQLite failure.
- **`_matches_filter` string semantics** — now uses exact equality, consistent with SQL `build_filter_clause`. Was using substring match (`value in str(meta_value)`).
- **`list_collections`** — scans `sqlite_master` for persisted collection tables instead of returning only session-cached names. Works across reopened databases.
- **WAL mode for encrypted databases** — `PRAGMA journal_mode=WAL` and `PRAGMA synchronous=NORMAL` now set for SQLCipher connections (was only set for unencrypted).
- **`collection()` cache key** — includes `distance_strategy` and `quantization` in cache key (sync version). Previously cached by name only, silently ignoring differing params on cache hit.
- **`_ensure_fts_table`** — retries up to 3 times on transient "database is locked" errors instead of permanently disabling FTS on first failure.
- **Connection health check** — `SELECT 1` probe after connection creation; raises `RuntimeError` immediately on corrupt databases.

### Improved

- **Usearch batch operations** — `add()`, `remove()`, and `get()` now use batch usearch APIs instead of per-key loops. Significant speedup for large operations.
- **Filtered search iterative deepening** — replaces fixed `k*3` overfetch with adaptive doubling (up to `k*30`). Highly selective filters now reliably return `k` results.
- **Memory-map heuristic** — uses file size threshold (50MB) instead of inaccurate `file_size // 100` vector count estimate for mmap vs load decision.
- **Apple chip detection** — uses `platform.processor()` instead of spawning a `sysctl` subprocess.

### Removed

- **Duplicate `_dim` property** — removed in favor of the public `dim` property.

### Breaking Changes

- String metadata filters now use exact equality (was substring match).
- `store_embeddings` defaults to `False` — `rebuild_index()` requires `store_embeddings=True` or re-adding documents.

## [2.4.0] - 2026-03-22

### Added

- **Public catalog API on VectorCollection + AsyncVectorCollection:**
  - `get_documents(filter_dict=)` — replaces private `_catalog` access
  - `get_embeddings_by_ids(ids)` — fetch stored embeddings
  - `update_metadata(updates)` — batch metadata merge
  - `count()`, `save()`, `dim` property — async wrappers
  - `add_texts(parent_ids=, threads=)` — full param support on async
  - `rebuild_index`, `get_children/parent/descendants/ancestors`, `set_parent` — async hierarchy API
- **Executor injection on AsyncVectorDB** — accept optional `executor` keyword argument so consumers can share a single-threaded executor for ONNX/usearch thread safety; `close()` only shuts down executor when `_owns_executor` is True
- **Safety constants** in `constants.py`: `SEARCH_COLLECTION_TIMEOUT`, `EXECUTOR_SHUTDOWN_TIMEOUT`, `MAX_HIERARCHY_DEPTH`

### Fixed

- **VectorDB.close()** now calls `conn.close()` — was leaking file descriptors when `save()` succeeded but connection was never closed
- **VectorDB.close()** wraps `save()` in `try/finally` so `conn.close()` always runs even if index serialization fails
- **add_documents ID recovery** uses `last_insert_rowid()` arithmetic instead of `ORDER BY id DESC LIMIT N`, which raced under concurrent inserts
- **String metadata filter** uses exact equality (`=`) instead of `LIKE` substring match — `{"type": "doc"}` no longer matches `"markdown_doc"`
- **update_metadata_batch** wrapped in single transaction (`with self.conn`) to prevent partial commits on crash
- **rebuild_index** uses `if x is not None` instead of `x or default` so passing `connectivity=0` no longer silently uses the default
- **search_collections** parallel futures now have a 30s timeout — one hung collection can no longer block the entire cross-collection search
- **AsyncVectorDB.close()** uses `shutdown(wait=False, cancel_futures=True)` instead of blocking `shutdown(wait=True)` which could hang forever on stuck tasks
- **Recursive CTE safety cap** — `get_descendants`/`get_ancestors` apply `MAX_HIERARCHY_DEPTH=100` when `max_depth=None` to prevent infinite recursion from parent_id cycles
- **RateLimiter cleanup** capped to 500 evictions per call to bound lock hold time under high bucket counts
- **HuggingFace download** now uses `etag_timeout=30` with local-cache fallback on network failure
- **embed_texts** rejects batches over 10,000 texts to prevent unbounded CPU time
- **retry_on_lock** adds `total_timeout=10s` budget — gives up early if cumulative sleep would exceed the budget

### Changed

- **`__version__`** now read from package metadata via `importlib.metadata` (single source of truth in `pyproject.toml`)
- **Upsert in usearch_index** separates conflict detection from removal for clearer flow

## [2.3.0] - 2026-03-08

### Breaking Changes

- **Integration dependencies are now optional.** LangChain and LlamaIndex packages are no longer installed by default. Install with `pip install simplevecdb[integrations]` to use them. Existing users upgrading from v2.2.x will see a clear ImportError with migration instructions.

### Added

- **`[integrations]` optional extra** — Install LangChain and LlamaIndex dependencies only when needed, reducing default install footprint
- **Runtime import guards** in integration modules with v2.3.0 migration messaging
- **Lazy `__getattr__` loading** in `integrations/__init__.py` — integration classes are only imported when accessed
- **Input validation guards** on search methods:
  - `similarity_search`, `similarity_search_batch`, `keyword_search`, `hybrid_search` now reject `k <= 0`
  - `add_texts` validates length consistency of `metadatas`, `embeddings`, `ids`, and `parent_ids` against `texts`
- **NaN/Inf validation** for float values in metadata filters (`utils.validate_filter`)
- **Empty list rejection** for list filter values
- **Double-close protection** on `VectorDB` with `_closed` flag
- **Context manager protocol** (`__enter__`/`__exit__`) on `VectorDB`
- **Table name validation** in `check_migration` (defense-in-depth against SQL injection)
- **Graceful per-future error handling** in `search_collections`
- **Adaptive batch search threshold** — queries below `USEARCH_BATCH_THRESHOLD` (10) use sequential search to avoid batch overhead

### Changed

- **Python dev target changed to 3.12** (`.python-version`), `requires-python` remains `>= "3.10"`
- **Version bumped to 2.3.0**
- **Performance: MMR search vectorized** — pre-normalize embeddings once, use `sel_matrix @ emb` matrix-vector multiply instead of Python inner loop, O(1) `list.pop` replaces O(n) `list.remove`, hoist `1 - lambda_mult` loop invariant
- **Performance: merged SQL round-trips in MMR** — new `get_documents_and_embeddings_by_ids` fetches text, metadata, and embeddings in a single query (previously two separate SELECTs)
- **Performance: `get_parent` collapsed** from 2 sequential SELECTs to 1 self-JOIN
- **Performance: `add_documents` ID recovery** — skip redundant `SELECT ORDER BY DESC` when explicit IDs are provided; removed unnecessary `list(texts)` copy
- **Performance: FLOAT serialization** — `np.asarray().tobytes()` replaces `struct.pack` with per-element Python loop (single C memcpy)
- **Performance: `np.array` → `np.asarray`** on every search and insert path to avoid unnecessary copies
- **Performance: SQL placeholder strings** — `",".join(["?"] * len(ids))` replaces generator expression across all 9 call sites
- **Performance: batched numpy conversion** in `add_texts` — single `np.asarray` call instead of per-item conversion
- **Performance: compact JSON separators** in catalog serialization
- **Performance: deduplicated `.tolist()` calls** in search engine
- **Performance: `np.unique(ravel())`** for batch key collection in `similarity_search_batch`
- **Performance: usearch upsert** — skip contains-check loop on empty index, cache `int(key)` once per iteration
- **Performance: cluster table DDL** — `_cluster_table_ready` flag skips `CREATE TABLE IF NOT EXISTS` on repeated calls; cached `_cluster_table_name`
- **`_normalize_key`** now delegates to `_derive_key` instead of duplicating PBKDF2 logic
- **HNSW defaults** in `usearch_index.py` now sourced from `constants.py` (removed local duplicates)
- **Collection name regex** uses `constants.COLLECTION_NAME_PATTERN` instead of hardcoded pattern
- **`VectorDB` defaults** for `distance_strategy` and `quantization` sourced from `constants.DEFAULT_DISTANCE_STRATEGY` / `constants.DEFAULT_QUANTIZATION`
- **`_batched` utility** moved from `core.py` to `utils.py` for reuse; now used in `catalog.py` batch updates
- **`auto_tag`** uses `defaultdict(list)` instead of manual if-not-in pattern
- **`import random`** hoisted to module level in `utils.py` (was inside retry loop)
- **Streaming placeholder bug fixed** — `_process_streaming_batch` now correctly detects `None` placeholders (previously used empty list `[]`, preventing auto-embedding replacement)
- **README updated** to document `pip install simplevecdb[integrations]` installation

### Removed

- LangChain and LlamaIndex packages from core `[project.dependencies]` (moved to `[project.optional-dependencies] integrations`)
- Duplicated HNSW default constants from `usearch_index.py` (now single source in `constants.py`)
- Unused `struct` import from `quantization.py`
- Unused `itertools` import from `core.py`

## [2.2.1] - 2026-01-27

### Changed

- Moved integration dependencies (langchain-core, langchain-openai, llama-index) from dev to main dependencies for easier installation
- Added bandit to dev dependencies for security linting in pre-commit
- Cleaned up duplicate dev dependency definitions

## [2.2.0] - 2026-01-26

### Added

- Version 2.2.0 release

## [2.1.0] - 2026-01-01

### Added

- **SQLCipher Encryption Support** - Full at-rest encryption for sensitive data:
  - `VectorDB(path, encryption_key="...")` enables AES-256 page-level database encryption
  - Uses SQLCipher for transparent SQLite encryption (PRAGMA key)
  - Usearch index files encrypted with AES-256-GCM (`.usearch.enc`)
  - Zero performance overhead during search (decrypt on load, encrypt on save only)
  - Key derivation: PBKDF2-SHA256 with 480,000 iterations for passphrases
  - Install with `pip install simplevecdb[encryption]`

- **New encryption module** (`simplevecdb.encryption`):
  - `create_encrypted_connection()` - SQLCipher connection factory
  - `is_database_encrypted()` - Check if a database file is encrypted
  - `encrypt_index_file()` / `decrypt_index_file()` - Index file encryption
  - `EncryptionError` / `EncryptionUnavailableError` - New exception types

- **Streaming Insert API** - Memory-efficient large-scale ingestion:
  - `collection.add_texts_streaming(iterable)` - Process from any iterator/generator
  - Configurable `batch_size` parameter (default: config.EMBEDDING_BATCH_SIZE)
  - Yields `StreamingProgress` after each batch for monitoring
  - Optional `on_progress` callback for custom logging/UI updates
  - New types: `StreamingProgress`, `ProgressCallback`

- **Hierarchical Document Relationships** - Parent/child document structure:
  - `parent_ids` parameter in `add_texts()` to link documents
  - `get_children(doc_id)` - Get direct child documents
  - `get_parent(doc_id)` - Get parent document
  - `get_descendants(doc_id, max_depth)` - Recursive children traversal
  - `get_ancestors(doc_id, max_depth)` - Path to root
  - `set_parent(doc_id, parent_id)` - Update relationships
  - Uses SQLite recursive CTE for efficient traversal
  - Auto-migrates existing databases (adds `parent_id` column)

### Changed

- `check_migration()` now gracefully handles encrypted databases (returns `needs_migration=False`)

### Dependencies

- New optional dependency group `[encryption]`: `sqlcipher3-binary>=0.5.0`, `cryptography>=41.0`

## [2.0.0] - 2025-12-23

### Breaking Changes

- **Backend Migration: sqlite-vec → usearch HNSW**
  - Vector search now uses usearch's high-performance HNSW algorithm
  - 10-100x faster similarity search for large collections
  - Vector data stored in separate `.usearch` files per collection (e.g., `mydb.db.default.usearch`)
  - SQLite still stores metadata, text, and FTS5 index
  
- **Removed `DistanceStrategy.L1`** - Manhattan distance not supported by usearch

- **Storage Format Change**
  - Embeddings now stored in both usearch index AND SQLite (for MMR support)
  - Existing sqlite-vec databases will auto-migrate on first open
  - Migration is one-way; backup before upgrading

### Added

- **`usearch_index.py`** - New UsearchIndex wrapper class:
  - Thread-safe HNSW index operations (lock on writes, lock-free reads)
  - Automatic persistence to `.usearch` files
  - Upsert support (removes existing keys before add)
  - BIT quantization using Hamming metric with bit packing
  - Configurable HNSW parameters (connectivity, expansion_add, expansion_search)

- **Proper MMR Implementation** - Max Marginal Relevance now computes actual pairwise similarity between candidates and selected documents using stored embeddings

- **Embedding Storage in SQLite** - Embeddings stored as BLOB for:
  - Accurate MMR diversity computation
  - Future index rebuild from SQLite backup
  - Schema auto-migrates existing tables

- **`VectorCollection.rebuild_index()`** - Reconstruct usearch HNSW index from SQLite embeddings:
  - Useful for index corruption recovery
  - Tune HNSW parameters (connectivity, expansion_add, expansion_search)
  - Reclaim space after many deletions

- **`VectorDB.check_migration(path)`** - Dry-run migration check:
  - Reports which collections need migration
  - Shows total vector count and estimated storage
  - Provides detailed rollback instructions

- **Adaptive Search** - Automatically optimizes search strategy based on collection size:
  - Collections < 10k vectors use brute-force (`exact=True`) for perfect recall
  - Collections ≥ 10k vectors use HNSW for faster approximate search
  - Threshold configurable via `constants.USEARCH_BRUTEFORCE_THRESHOLD`

- **`exact` parameter** - Force search mode in `similarity_search()`:
  - `None` (default): adaptive based on collection size
  - `True`: force brute-force for perfect recall
  - `False`: force HNSW approximate search

- **`Quantization.FLOAT16`** - Half-precision floating point:
  - 2x memory savings compared to FLOAT32
  - 1.5x faster search with minimal precision loss
  - Ideal for embeddings where full precision isn't needed

- **`threads` parameter** - Parallel execution control:
  - Added to `add_texts()` and `similarity_search()`
  - `0` (default): auto-detect optimal thread count
  - Explicit value: control parallelism for batch operations

- **Auto Memory-Mapping** - Large indexes automatically use memory-mapped mode:
  - Indexes >100k vectors use `view=True` for instant startup
  - Lower memory footprint for large collections
  - Transparent upgrade to writable mode on add operations
  - Configurable via `constants.USEARCH_MMAP_THRESHOLD`

- **`similarity_search_batch()`** - Multi-query batch search:
  - ~10x throughput for batch query workloads
  - Uses usearch's native batch search under the hood
  - Same parameters as `similarity_search()` but accepts list of queries

- **`examples/backend_benchmark.py`** - Benchmark script comparing usearch vs brute-force:
  - Measures speedup, recall, and storage efficiency
  - Supports all quantization levels
  - Validates 10-100x performance claims

### Changed

- **Dependencies**: Replaced `sqlite-vec>=0.1.6` with `usearch>=2.12`
- **CatalogManager**: Removed vec0 virtual table operations, added embedding column
- **SearchEngine**: Rewrote to use UsearchIndex for all vector operations
- **VectorCollection**: Creates usearch index at `{db_path}.{collection}.usearch`

### Migration Notes

1. **Backup your database** before upgrading
2. On first open, existing sqlite-vec data will be migrated automatically
3. New `.usearch` files will be created alongside your `.db` file
4. The legacy sqlite-vec table is dropped after successful migration

## [1.3.0] - 2025-12-07

### Added

- **Structured Logging Module** - New `simplevecdb.logging` module for production-grade observability
  - `get_logger(name)` - Get namespaced loggers under `simplevecdb.*`
  - `configure_logging(level, format, handler)` - One-call logging setup
  - `log_operation(name, **context)` - Context manager for operation timing and error tracking
  - `log_error(operation, error, **context)` - Consistent error logging with context

- **SQLite Lock Retry Logic** - Automatic retry with exponential backoff for database lock contention
  - `@retry_on_lock(max_retries, base_delay, max_delay, jitter)` decorator
  - `DatabaseLockedError` exception for exhausted retries with attempt/wait metrics
  - Applied to `add_texts()` and `delete_by_ids()` operations in CatalogManager

- **Filter Validation** - Early validation of metadata filter dictionaries
  - `validate_filter(filter_dict)` - Validates keys are strings, values are supported types
  - Clear error messages for invalid filter structures
  - Automatically called in `build_filter_clause()` before SQL generation

- **New Exports** - Added to `simplevecdb.__all__`:
  - `get_logger`, `configure_logging`, `log_operation`
  - `DatabaseLockedError`, `retry_on_lock`, `validate_filter`

### Changed

- **CatalogManager** internal refactoring:
  - `add_texts()` now delegates to `_insert_batch()` which has retry logic
  - `delete_by_ids()` now has retry logic for lock contention
  - `build_filter_clause()` validates filters before processing
- **`delete_by_ids()` no longer auto-vacuums** - Call `VectorDB.vacuum()` separately to reclaim disk space after large deletions. This improves performance for batch deletions.
- **RateLimiter** now includes TTL-based cleanup to prevent memory exhaustion on long-running servers with many unique clients (default: 1 hour TTL, 10k max buckets).
- **AsyncVectorDB.close()** now guarantees database connection is closed even if executor shutdown fails.

### Testing

- Added 25 new tests in `tests/unit/test_error_handling.py`:
  - 7 tests for `retry_on_lock` decorator behavior
  - 2 tests for `DatabaseLockedError` exception
  - 4 tests for `validate_filter` function
  - 8 tests for logging utilities
  - 4 integration tests for error handling in VectorDB operations

### Example

```python
import logging
from simplevecdb import (
    VectorDB,
    configure_logging,
    get_logger,
    log_operation,
    DatabaseLockedError,
)

# Enable debug logging
configure_logging(level=logging.DEBUG)

logger = get_logger(__name__)

try:
    with log_operation("bulk_insert", collection="docs", count=1000):
        db = VectorDB("data.db")
        collection = db.collection("docs")
        collection.add_texts(texts, embeddings=embeddings)
except DatabaseLockedError as e:
    logger.error(f"Insert failed after {e.attempts} attempts")
```

## [1.2.0] - 2025-11-25

### Added

- **Async API Support** - New `AsyncVectorDB` and `AsyncVectorCollection` classes
  - Full async/await support for all collection operations
  - Uses ThreadPoolExecutor to avoid blocking event loops
  - Async context manager support (`async with AsyncVectorDB(...)`)
  - All methods mirror sync API: `add_texts`, `similarity_search`, `keyword_search`, `hybrid_search`, `max_marginal_relevance_search`, `delete_by_ids`, `remove_texts`
  - Configurable thread pool size via `max_workers` parameter

### Changed

- Added `pytest-asyncio` to dev dependencies for async test support

### Example

```python
import asyncio
from simplevecdb import AsyncVectorDB

async def main():
    async with AsyncVectorDB("data.db") as db:
        collection = db.collection("docs")
        await collection.add_texts(["Hello"], embeddings=[[0.1]*384])
        results = await collection.similarity_search([0.1]*384, k=5)
        return results

asyncio.run(main())
```

## [1.1.1] - 2025-11-23

### Changed

- **Refactored configuration constants** into dedicated `constants.py` module
  - Extracted hardware batch size thresholds (VRAM, CPU cores, ARM variants)
  - Extracted search defaults (k=5, rrf_k=60, fetch_k=20)
  - Improved maintainability and centralized configuration

### Fixed

- **Updated dependencies**
  - Bumped `sentence-transformers[onnx]` from 3.3.1 to 5.1.2
  - All embeddings/server tests passing with new version

## [1.1.0] - 2025-11-23

### 🏗️ Architecture Refactoring

Major internal restructuring for better maintainability and extensibility while preserving backward compatibility.

### Changed

- **Refactored core.py** (879→216 lines, 75% reduction)
  - Extracted search operations to `engine/search.py` (SearchEngine)
  - Extracted quantization logic to `engine/quantization.py` (QuantizationStrategy)
  - Extracted catalog management to `engine/catalog.py` (CatalogManager)
  - Core now uses clean facade pattern with delegation
- **Improved documentation**
  - Added comprehensive Google-style docstrings to all public API methods
  - Reorganized MkDocs navigation with dedicated Engine section
  - Updated architecture documentation in AGENTS.md and CONTRIBUTING.md
  - Simplified CODE_OF_CONDUCT.md to be more approachable

### Added

- **Security infrastructure**
  - GitHub Actions workflow for weekly security scans (Bandit, Safety, Semgrep)
  - Dependabot configuration for automated dependency updates
  - Bandit configuration with validated false-positive suppressions
- **Automated publishing**
  - GitHub Actions workflow for PyPI publishing on releases
- **Test coverage improvements**
  - Added 11 new tests covering edge cases in search engine
  - Maintained 97% overall coverage across refactored modules

### Fixed

- Fixed unused `filter_builder` parameter in `_brute_force_search` method
- Simplified brute-force filtering to use proper filter builder delegation
- Fixed import paths for embeddings module in search engine

### Internal

- All modules now follow consistent interface patterns
- Engine components properly isolated with clear responsibilities
- No breaking changes to public API

## [1.0.0] - 2025-11-23

### 🎉 Initial Release

SimpleVecDB's first stable release brings production-ready local vector search to a single SQLite file.

### Added

#### Core Features

- **Multi-collection catalog system**: Organize documents in named collections within a single database
- **Vector search**: Cosine, L2 (Euclidean), and L1 (Manhattan) distance metrics
- **Quantization**: FLOAT32, INT8 (4x compression), and BIT (32x compression) support
- **Metadata filtering**: JSON-based filtering with SQL `WHERE` clauses
- **Batch processing**: Automatic batching for efficient bulk operations
- **Persistence**: Single `.db` file with WAL mode for concurrent reads

#### Hybrid Search

- **BM25 keyword search**: Full-text search using SQLite FTS5
- **Hybrid search**: Reciprocal Rank Fusion combining BM25 + vector similarity
- **Query vector reuse**: Pass pre-computed embeddings to avoid redundant embedding calls
- **Metadata filtering**: Works across all search modes (vector, keyword, hybrid)

#### Embeddings Server

- **OpenAI-compatible API**: `/v1/embeddings` endpoint for local embedding generation
- **Model registry**: Configure allowed models or allow arbitrary HuggingFace repos
- **Request limits**: Configurable max batch size per request
- **API key authentication**: Optional Bearer token / X-API-Key authentication
- **Usage tracking**: Per-key request and token metrics via `/v1/usage`
- **Model listing**: `/v1/models` endpoint for registry inspection
- **ONNX optimization**: Quantized ONNX runtime for fast CPU inference

#### Hardware Optimization

- **Auto-detection**: Automatically detects CUDA GPUs, Apple Silicon (MPS), ROCm, and CPU
- **Adaptive batching**: Optimal batch sizes based on:
  - NVIDIA GPUs: 64-512 (scaled by VRAM 4GB-24GB+)
  - AMD GPUs: 256 (ROCm)
  - Apple Silicon: 32-128 (M1/M2 vs M3/M4, base vs Max/Ultra)
  - ARM CPUs: 4-16 (mobile, Raspberry Pi, servers)
  - x86 CPUs: 8-64 (scaled by core count)
- **Manual override**: `EMBEDDING_BATCH_SIZE` environment variable

#### Integrations

- **LangChain**: `SimpleVecDBVectorStore` with async support and MMR
  - `similarity_search`, `similarity_search_with_score`
  - `max_marginal_relevance_search`
  - `keyword_search`, `hybrid_search`
  - `add_texts`, `add_documents`, `delete`
- **LlamaIndex**: `SimpleVecDBLlamaStore` with query mode support
  - `VectorStoreQueryMode.DEFAULT` (dense vector)
  - `VectorStoreQueryMode.SPARSE` / `TEXT_SEARCH` (BM25)
  - `VectorStoreQueryMode.HYBRID` / `SEMANTIC_HYBRID` (fusion)
  - Metadata filtering across all modes

#### Examples & Documentation

- **RAG notebooks**: LangChain, LlamaIndex, and Ollama integration examples
- **Performance benchmarks**: Insertion speed, query latency, storage efficiency
- **API documentation**: Full class and method reference via MkDocs
- **Setup guide**: Environment variables and configuration options
- **Contributing guide**: Development setup and testing instructions

### Configuration

- `EMBEDDING_MODEL`: HuggingFace model ID (default: `Snowflake/snowflake-arctic-embed-xs`)
- `EMBEDDING_CACHE_DIR`: Model cache directory (default: `~/.cache/simplevecdb`)
- `EMBEDDING_MODEL_REGISTRY`: Comma-separated `alias=repo_id` entries
- `EMBEDDING_MODEL_REGISTRY_LOCKED`: Enforce registry allowlist (default: `1`)
- `EMBEDDING_BATCH_SIZE`: Inference batch size (auto-detected if not set)
- `EMBEDDING_SERVER_MAX_REQUEST_ITEMS`: Max prompts per `/v1/embeddings` call
- `EMBEDDING_SERVER_API_KEYS`: Comma-separated API keys for authentication
- `DATABASE_PATH`: SQLite database path (default: `:memory:`)
- `SERVER_HOST`: Embeddings server host (default: `0.0.0.0`)
- `SERVER_PORT`: Embeddings server port (default: `8000`)

### Performance

Benchmarks on i9-13900K & RTX 4090 with 10k vectors (384-dim):

| Quantization | Storage  | Insert Speed | Query Time (k=10) |
| ------------ | -------- | ------------ | ----------------- |
| FLOAT32      | 15.50 MB | 15,585 vec/s | 3.55 ms           |
| INT8         | 4.23 MB  | 27,893 vec/s | 3.93 ms           |
| BIT          | 0.95 MB  | 32,321 vec/s | 0.27 ms           |

### Testing

- 177 unit and integration tests
- 97% code coverage
- Type-safe (mypy strict mode)
- CI/CD on Python 3.10, 3.11, 3.12, 3.13

### Dependencies

- Core: `sqlite-vec>=0.1.6`, `numpy>=2.0`, `python-dotenv>=1.2.1`, `psutil>=5.9.0`
- Server extras: `fastapi>=0.115`, `uvicorn[standard]>=0.30`, `sentence-transformers[onnx]==3.3.1`

### Notes

- Requires SQLite builds with FTS5 enabled for keyword/hybrid search (bundled with Python 3.10+)
- Works on Linux, macOS, Windows, and WASM environments
- Zero external dependencies beyond Python for core functionality

---

## Links

- **GitHub**: https://github.com/coderdayton/simplevecdb
- **PyPI**: https://pypi.org/project/simplevecdb/
- **Documentation**: https://coderdayton.github.io/simplevecdb/
- **License**: MIT

[2.4.0]: https://github.com/coderdayton/simplevecdb/releases/tag/v2.4.0
[2.3.0]: https://github.com/coderdayton/simplevecdb/releases/tag/v2.3.0
[2.2.1]: https://github.com/coderdayton/simplevecdb/releases/tag/v2.2.1
[2.2.0]: https://github.com/coderdayton/simplevecdb/releases/tag/v2.2.0
[2.1.0]: https://github.com/coderdayton/simplevecdb/releases/tag/v2.1.0
[2.0.0]: https://github.com/coderdayton/simplevecdb/releases/tag/v2.0.0
[1.3.0]: https://github.com/coderdayton/simplevecdb/releases/tag/v1.3.0
[1.2.0]: https://github.com/coderdayton/simplevecdb/releases/tag/v1.2.0
[1.1.1]: https://github.com/coderdayton/simplevecdb/releases/tag/v1.1.1
[1.1.0]: https://github.com/coderdayton/simplevecdb/releases/tag/v1.1.0
[1.0.0]: https://github.com/coderdayton/simplevecdb/releases/tag/v1.0.0
