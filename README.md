# SimpleVecDB

[![CI](https://github.com/coderdayton/simplevecdb/actions/workflows/ci.yml/badge.svg)](https://github.com/coderdayton/simplevecdb/actions)
[![PyPI](https://img.shields.io/pypi/v/simplevecdb?color=blue)](https://pypi.org/project/simplevecdb/)
[![License: MIT](https://img.shields.io/github/license/coderdayton/simplevecdb)](LICENSE)
[![GitHub Stars](https://img.shields.io/github/stars/coderdayton/simplevecdb?style=social)](https://github.com/coderdayton/simplevecdb)

<a href='https://ko-fi.com/U7U01WTJF9' target='_blank'><img height='36' style='border:0px;height:36px;' src='https://storage.ko-fi.com/cdn/kofi3.png?v=6' border='0' alt='Buy Me a Coffee at ko-fi.com' /></a>

**The dead-simple, local-first vector database.**

SimpleVecDB brings **Chroma-like simplicity** to a single **SQLite file**. Built on `usearch` HNSW indexing, it offers high-performance vector search, quantization, and zero infrastructure headaches. Perfect for local RAG, offline agents, and indie hackers who need production-grade vector search without the operational overhead.

## Why SimpleVecDB?

- **Zero Infrastructure** — Just a `.db` file. No Docker, no Redis, no cloud bills.
- **Blazing Fast** — 10-100x faster search via usearch HNSW. Adaptive: brute-force for <10k vectors (perfect recall), HNSW for larger collections.
- **Truly Portable** — Runs anywhere SQLite runs: Linux, macOS, Windows, even WASM.
- **Async Ready** — Full async/await support with optional executor injection for thread-safe ONNX/usearch sharing.
- **Batteries Included** — Optional FastAPI embeddings server + LangChain/LlamaIndex integrations via `[integrations]` extra.
- **Production Ready** — Hybrid search (BM25 + vector), metadata filtering, multi-collection support, and automatic hardware acceleration.

### When to Choose SimpleVecDB

| Use Case                       | SimpleVecDB           | Cloud Vector DB          |
| :----------------------------- | :-------------------- | :----------------------- |
| **Local RAG applications**     | ✅ Perfect fit        | ❌ Overkill + latency    |
| **Offline-first agents**       | ✅ No internet needed | ❌ Requires connectivity |
| **Prototyping & MVPs**         | ✅ Zero config        | ⚠️ Setup overhead        |
| **Multi-tenant SaaS at scale** | ⚠️ Consider sharding  | ✅ Built for this        |
| **Budget-conscious projects**  | ✅ $0/month           | ❌ $50-500+/month        |

## Prerequisites

**System Requirements:**

- Python 3.10+
- SQLite 3.35+ with FTS5 support (included in Python 3.8+ standard library)
- 50MB+ disk space for core library, 500MB+ with `[server]` extras

**Optional for GPU Acceleration:**

- CUDA 11.8+ for NVIDIA GPUs
- Metal Performance Shaders (MPS) for Apple Silicon

> **Note:** If using custom-compiled SQLite, ensure `-DSQLITE_ENABLE_FTS5` is enabled for full-text search support.

## Installation

```bash
# Standard installation (includes clustering, encryption)
pip install simplevecdb

# With LangChain & LlamaIndex integrations
pip install "simplevecdb[integrations]"

# With local embeddings server (adds 500MB+ models)
pip install "simplevecdb[server]"
```

**What's included by default:**
- Vector search with HNSW indexing
- Clustering (K-means, MiniBatch K-means, HDBSCAN)
- Encryption (SQLCipher AES-256)
- Async support

**Verify Installation:**

```bash
python -c "from simplevecdb import VectorDB; print('SimpleVecDB installed successfully!')"
```

## Quickstart

SimpleVecDB is just a storage and search layer — it doesn't ship an LLM
and won't generate embeddings for you. Bring whichever embedding source
you already use; three common ones below.

### Option 1: OpenAI embeddings

```python
from simplevecdb import VectorDB
from openai import OpenAI

client = OpenAI()
db = VectorDB("notes.db")
notes = db.collection("personal")

def embed(text: str) -> list[float]:
    return (
        client.embeddings
        .create(model="text-embedding-3-small", input=text)
        .data[0].embedding
    )

entries = [
    ("Cherry MX silent reds bottom out around 45g — quieter than browns", "keyboards"),
    ("Sourdough hydration sweet spot is ~75% with this flour",            "baking"),
    ("EXPLAIN ANALYZE showed seq scan; ANALYZE on the table fixed it",    "work"),
    ("Passport renewal took 3 weeks, not the advertised 6–8",             "admin"),
]

notes.add_texts(
    texts=[t for t, _ in entries],
    embeddings=[embed(t) for t, _ in entries],
    metadatas=[{"tag": tag} for _, tag in entries],
)

hits = notes.similarity_search(embed("how loud are silent reds"), k=2)
for doc, score in hits:
    print(f"{score:.3f}  {doc.page_content}")

work = notes.similarity_search(
    embed("query plan slow"),
    k=5,
    filter={"tag": "work"},
)
```

### Option 2: Fully local (no network, no API key)

```bash
pip install "simplevecdb[server]"
```

```python
from simplevecdb import VectorDB
from simplevecdb.embeddings.models import embed_texts

db = VectorDB("notes.db")
notes = db.collection("personal")

texts = [
    "Cherry MX silent reds bottom out around 45g",
    "Sourdough hydration sweet spot is ~75% with this flour",
    "EXPLAIN ANALYZE showed seq scan; ANALYZE on the table fixed it",
]
notes.add_texts(texts=texts, embeddings=embed_texts(texts))

vec = notes.similarity_search(embed_texts(["quieter switches"])[0], k=2)
mixed = notes.hybrid_search("postgres slow query", k=3)
```

If you'd rather hit an HTTP endpoint than import the embedding models
directly, the bundled server speaks the same shape as OpenAI's
embeddings API:

```bash
simplevecdb-server --port 8000                # default model, auto warm-up
simplevecdb-server --host 0.0.0.0 --port 9000
simplevecdb-server --no-warmup                # skip the model preload
simplevecdb-server --help
```

Server tuning (model registry, rate limits, API keys, CORS, CUDA) lives
in the [Setup Guide](ENV_SETUP.md).

### Option 3: LangChain or LlamaIndex

Already wired into one of the big RAG frameworks? Drop SimpleVecDB in
as the vector store:

```bash
pip install "simplevecdb[integrations]"
```

```python
from simplevecdb.integrations.langchain import SimpleVecDBVectorStore
from langchain_openai import OpenAIEmbeddings

store = SimpleVecDBVectorStore(
    db_path="notes.db",
    embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
)

store.add_texts([
    "Cherry MX silent reds bottom out around 45g",
    "EXPLAIN ANALYZE showed seq scan; ANALYZE on the table fixed it",
])
store.similarity_search("quieter switches", k=1)
store.hybrid_search("postgres performance", k=3)
```

LlamaIndex is the same shape:

```python
from simplevecdb.integrations.llamaindex import SimpleVecDBLlamaStore
from llama_index.embeddings.openai import OpenAIEmbedding

store = SimpleVecDBLlamaStore(
    db_path="notes.db",
    embedding=OpenAIEmbedding(model="text-embedding-3-small"),
)
```

End-to-end notebooks (including a fully local Ollama RAG) live in the
[examples gallery](https://coderdayton.github.io/SimpleVecDB/examples/).

## Feature Highlights

A few of the things SimpleVecDB does well — see
[`docs/Features.md`](docs/Features.md) for the comprehensive list.

- **Vector + keyword + hybrid search** — cosine / L2 similarity, BM25
  via SQLite FTS5, and Reciprocal Rank Fusion in one collection.
- **Adaptive HNSW** — brute-force for <10k vectors (perfect recall),
  `usearch` HNSW above that. Override per query with `exact=True/False`.
- **Quantization** — `FLOAT32`, `FLOAT16`, `INT8`, `BIT` for 1×–32×
  compression.
- **Multi-collection + cross-collection search** — isolated namespaces in
  one `.db` file, with merged ranked search across them.
- **Mongo-style filters** — `$eq $ne $gt $gte $lt $lte $in $nin $exists
  $between` on metadata, edges, and events.
- **Memory primitives (v2.6.1)** — pending-vector buffer with atomic
  flush, weighted directed edges, append-only event feed, TTL with
  delete/callback sweep, and a threshold-driven rebuild scheduler.
- **Atomic counters & transactions (v2.6.1)** — `increment_metadata` for
  JSON deltas in one statement; SAVEPOINT-backed `db.transaction()` /
  `collection.tx()` rolling all catalog writes back on error.
- **Async, encryption, clustering, hierarchies** — full async surface
  (with executor injection), SQLCipher AES-256, K-means / MiniBatch
  K-means / HDBSCAN, parent/child relationships.
- **Framework integrations** — drop-in `LangChain` and `LlamaIndex`
  adapters via the `[integrations]` extra; optional FastAPI embeddings
  server via `[server]`.

For full method-level coverage, see [the Features doc](docs/Features.md)
or the [API reference](https://coderdayton.github.io/SimpleVecDB/api/core).


## Performance Benchmarks

**10,000 vectors, 384 dimensions, k=10 search** — [Full benchmarks →](https://coderdayton.github.io/SimpleVecDB/benchmarks)

| Quantization | Storage  | Query Time | Compression |
| :----------- | :------- | :--------- | :---------- |
| FLOAT32      | 36.0 MB  | 0.20 ms    | 1x          |
| FLOAT16      | 28.7 MB  | 0.20 ms    | 2x          |
| INT8         | 25.0 MB  | 0.16 ms    | 4x          |
| BIT          | 21.8 MB  | 0.08 ms    | 32x         |

**Key highlights:**
- **3-34x faster** than brute-force for collections >10k vectors
- **Adaptive search**: perfect recall for small collections, HNSW for large
- **FLOAT16 recommended**: best balance of speed, memory, and precision

## Documentation

- **[Features](docs/Features.md)** — Comprehensive list of every capability, grouped by area
- **[Setup Guide](https://coderdayton.github.io/SimpleVecDB/ENV_SETUP)** — Environment variables, server configuration, authentication
- **[API Reference](https://coderdayton.github.io/SimpleVecDB/api/core)** — Complete class/method documentation with type signatures
- **[Benchmarks](https://coderdayton.github.io/SimpleVecDB/benchmarks)** — Quantization strategies, batch sizes, hardware optimization
- **[Integration Examples](https://coderdayton.github.io/SimpleVecDB/examples)** — RAG notebooks, Ollama workflows, production patterns
- **[Contributing Guide](CONTRIBUTING.md)** — Development setup, testing, PR guidelines

## Troubleshooting

**Import Error: `sqlite3.OperationalError: no such module: fts5`**

```bash
# Your Python's SQLite was compiled without FTS5
# Solution: Install Python from python.org (includes FTS5) or compile SQLite with:
# -DSQLITE_ENABLE_FTS5
```

**Dimension Mismatch Error**

```python
# Ensure all vectors in a collection have identical dimensions
collection = db.collection("docs", dim=384)  # Explicit dimension
```

**CUDA Not Detected (GPU Available)**

```bash
# Verify CUDA installation
python -c "import torch; print(torch.cuda.is_available())"

# Reinstall PyTorch with CUDA support
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

**Slow Queries on Large Datasets**

- Enable quantization: `collection = db.collection("docs", quantization=Quantization.INT8)`
- For >10k vectors, HNSW is automatic; tune with `rebuild_index(connectivity=32)`
- Use `exact=False` to force HNSW even on smaller collections
- Use metadata filtering to reduce search space

## Roadmap

What's on the near-term radar:

- [ ] Incremental clustering (online learning)
- [ ] Cluster visualization exports

For shipped capabilities, see [`docs/Features.md`](docs/Features.md) and the
release-by-release [Changelog](CHANGELOG.md). Vote on these or propose new
ideas in [GitHub Discussions](https://github.com/coderdayton/simplevecdb/discussions).

## Contributing

Contributions are welcome! Whether you're fixing bugs, improving documentation, or proposing new features:

1. Read [CONTRIBUTING.md](CONTRIBUTING.md) for development setup
2. Check existing [Issues](https://github.com/coderdayton/simplevecdb/issues) and [Discussions](https://github.com/coderdayton/simplevecdb/discussions)
3. Open a PR with clear description and tests

## Community & Support

**Get Help:**

- [GitHub Discussions](https://github.com/coderdayton/simplevecdb/discussions) — Q&A and feature requests
- [GitHub Issues](https://github.com/coderdayton/simplevecdb/issues) — Bug reports

**Stay Updated:**

- [GitHub Releases](https://github.com/coderdayton/simplevecdb/releases) — Changelog and updates
- [Examples Gallery](https://coderdayton.github.io/SimpleVecDB/examples/) — Community-contributed notebooks

## Other Ways to Support

- ☕ **[Buy me a coffee](https://ko-fi.com/xbbvii)** - One-time donation
- ⭐ **Star the repo** - Helps with visibility
- 🐛 **Report bugs** - Improve the project for everyone
- 📝 **Contribute** - See [CONTRIBUTING.md](CONTRIBUTING.md)

## License

[MIT License](LICENSE) — Free for personal and commercial use.
