# TinyVecDB • A dead-simple, file-based vector database

[![CI](https://github.com/coderdayton/tinyvecdb/actions/workflows/ci.yml/badge.svg)](https://github.com/coderdayton/tinyvecdb/actions)
[![PyPI](https://img.shields.io/pypi/v/tinyvecdb?color=blue)](https://pypi.org/project/tinyvecdb/)
[![License: MIT](https://img.shields.io/github/license/coderdayton/tinyvecdb)](LICENSE)
[![GitHub Stars](https://img.shields.io/github/stars/coderdayton/tinyvecdb?style=social)](https://github.com/coderdayton/tinyvecdb)

**One SQLite file. Zero external servers. Full-text + vector search. OpenAI-compatible embeddings endpoint.**

TinyVecDB is a lightweight, dependency-minimal vector database built directly on top of **[sqlite-vec](https://github.com/asg017/sqlite-vec)** — a high-performance C vector search extension for SQLite. It is designed for indie developers, local-first RAG applications, and offline AI agents who want Chroma-like simplicity without Docker, Redis, or cloud bills.

- Single `.db` file (or `:memory:`)
- 10,000 × 384-dim vectors ≈ 14–18 MB on disk
- ~1–3 ms similarity queries on consumer laptops
- Runs everywhere SQLite runs: macOS · Linux · Windows · WASM · Android · iOS
- Optional FastAPI `/v1/embeddings` server (100% OpenAI compatible)
- First-class LangChain and LlamaIndex integrations

Perfect for private knowledge bases, local copilot tools, edge-device RAG, and anyone tired of Pinecone pricing.

## Quickstart

```bash
uv pip install tinyvecdb    # or: pip install tinyvecdb[server]
```

```python
from tinyvecdb import VectorDB

# Create or open a persistent database
db = VectorDB("my_knowledge.db")

# Add documents (embeddings auto-generated with a local model)
db.add_texts([
    "Paris is the capital of France.",
    "Berlin is the capital of Germany.",
    "The mitochondria is the powerhouse of the cell."
])

# Semantic search
results = db.similarity_search("What is the capital of France?", k=2)

for doc, score in results:
    print(f"{score:.4f} → {doc.page_content}")
# 0.0021 → Paris is the capital of France.
# 0.7214 → Berlin is the capital of Germany.
# 0.8453 → The mitochondria is the powerhouse of the cell.
```

## Features

| Feature                             | Status  | Notes                                       |
| ----------------------------------- | ------- | ------------------------------------------- |
| Persistent single-file storage      | ✅      | Plain SQLite file                           |
| Cosine / Euclidean search           | ✅      | Powered by sqlite-vec (HNSW coming soon)    |
| int8 & binary quantization          | ✅      | Up to 32× storage reduction                 |
| Metadata storage & filtering        | ✅      | Standard SQL WHERE clauses                  |
| Upsert / delete_by_id               | ✅      |                                             |
| OpenAI-compatible embeddings        | ✅      | Built-in FastAPI server + local models      |
| LangChain VectorStore               | ✅      | `db.as_langchain()`                         |
| LlamaIndex NodeStore                | ✅      | `db.as_llama_index()`                       |
| No-extension fallback (brute-force) | Planned | Pure-NumPy mode for restricted environments |

## Benchmarks (M2 MacBook Pro, sqlite-vec v0.1.2)

| Dataset                  | Dimensions | Vectors | File Size | Avg Query (k=10) |
| ------------------------ | ---------- | ------- | --------- | ---------------- |
| Random normalized floats | 384        | 10,000  | 14.7 MB   | 1.8 ms           |
| Random normalized floats | 1536       | 10,000  | 59.2 MB   | 2.4 ms           |

Real-world private knowledge bases (1–50k chunks) typically stay under 500 MB with int8 quantization.

## Installation

```bash
# Core (no embeddings server)
uv pip install tinyvecdb

# With OpenAI-compatible server + default local embedder
uv pip install "tinyvecdb[server]"
```

## Roadmap → v1.0

- [ ] Hybrid BM25 + vector search (FTS5 integration)
- [ ] Multi-collection support
- [ ] Built-in encryption (SQLCipher) in Pro version
- [ ] Desktop GUI (Tauri) in Pro version
- [ ] HNSW indexing when sqlite-vec adds it

## Contributing

Contributions are very welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) (will be added soon).

## Sponsors & Pro Version

Love TinyVecDB? Consider [sponsoring on GitHub ❤️](https://github.com/sponsors/coderdayton) — sponsors get priority feature requests, early access to the encrypted Pro build, and eternal gratitude.

A paid Pro tier (one-time or subscription) with encryption, GUI, and team sync is in active development.

## License

[MIT](./LICENSE)

---

Built because the world needs more local-first, privacy-preserving AI tools.
