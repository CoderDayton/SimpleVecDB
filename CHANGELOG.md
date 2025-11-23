# Changelog

All notable changes to SimpleVecDB will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 11-23-2025

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

## [1.0.0] - 11-23-2025

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

[1.0.0]: https://github.com/coderdayton/simplevecdb/releases/tag/v1.0.0
