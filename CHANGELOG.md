# Changelog

All notable changes to SimpleVecDB will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

[1.3.0]: https://github.com/coderdayton/simplevecdb/releases/tag/v1.3.0
[1.2.0]: https://github.com/coderdayton/simplevecdb/releases/tag/v1.2.0
[1.1.1]: https://github.com/coderdayton/simplevecdb/releases/tag/v1.1.1
[1.1.0]: https://github.com/coderdayton/simplevecdb/releases/tag/v1.1.0
[1.0.0]: https://github.com/coderdayton/simplevecdb/releases/tag/v1.0.0
