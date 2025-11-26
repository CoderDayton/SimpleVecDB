# SimpleVecDB

[![CI](https://github.com/coderdayton/simplevecdb/actions/workflows/ci.yml/badge.svg)](https://github.com/coderdayton/simplevecdb/actions)
[![PyPI](https://img.shields.io/pypi/v/simplevecdb?color=blue)](https://pypi.org/project/simplevecdb/)
[![License: MIT](https://img.shields.io/github/license/coderdayton/simplevecdb)](LICENSE)
[![GitHub Stars](https://img.shields.io/github/stars/coderdayton/simplevecdb?style=social)](https://github.com/coderdayton/simplevecdb)

**The dead-simple, local-first vector database.**

SimpleVecDB brings **Chroma-like simplicity** to a single **SQLite file**. Built on `sqlite-vec`, it offers high-performance vector search, quantization, and zero infrastructure headaches. Perfect for local RAG, offline agents, and indie hackers who need production-grade vector search without the operational overhead.

## Why SimpleVecDB?

- **Zero Infrastructure** — Just a `.db` file. No Docker, no Redis, no cloud bills.
- **Blazing Fast** — ~2ms queries on consumer hardware with 32x storage efficiency via quantization.
- **Truly Portable** — Runs anywhere SQLite runs: Linux, macOS, Windows, even WASM.
- **Async Ready** — Full async/await support for web servers and concurrent workloads.
- **Batteries Included** — Optional FastAPI embeddings server + LangChain/LlamaIndex integrations.
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
# Core library only (lightweight, 50MB)
pip install simplevecdb

# With local embeddings server + HuggingFace models (500MB+)
pip install "simplevecdb[server]"
```

**Verify Installation:**

```bash
python -c "from simplevecdb import VectorDB; print('SimpleVecDB installed successfully!')"
```

## Quickstart

SimpleVecDB is **just a vector storage layer**—it doesn't include an LLM or generate embeddings. This design keeps it lightweight and flexible. Choose your integration path:

### Option 1: With OpenAI (Simplest)

Best for: Quick prototypes, production apps with OpenAI subscriptions.

```python
from simplevecdb import VectorDB
from openai import OpenAI

db = VectorDB("knowledge.db")
collection = db.collection("docs")
client = OpenAI()

texts = ["Paris is the capital of France.", "Mitochondria powers cells."]
embeddings = [
    client.embeddings.create(model="text-embedding-3-small", input=t).data[0].embedding
    for t in texts
]

collection.add_texts(
    texts=texts,
    embeddings=embeddings,
    metadatas=[{"category": "geography"}, {"category": "biology"}]
)

# Search
query_emb = client.embeddings.create(
    model="text-embedding-3-small",
    input="capital of France"
).data[0].embedding

results = collection.similarity_search(query_emb, k=1)
print(results[0][0].page_content)  # "Paris is the capital of France."

# Filter by metadata
filtered = collection.similarity_search(query_emb, k=10, filter={"category": "geography"})
```

### Option 2: Fully Local (Privacy-First)

Best for: Offline apps, sensitive data, zero API costs.

```bash
pip install "simplevecdb[server]"
```

```python
from simplevecdb import VectorDB
from simplevecdb.embeddings.models import embed_texts

db = VectorDB("local.db")
collection = db.collection("docs")

texts = ["Paris is the capital of France.", "Mitochondria powers cells."]
embeddings = embed_texts(texts)  # Local HuggingFace models

collection.add_texts(texts=texts, embeddings=embeddings)

# Search
query_emb = embed_texts(["capital of France"])[0]
results = collection.similarity_search(query_emb, k=1)

# Hybrid search (BM25 + vector)
hybrid = collection.hybrid_search("powerhouse cell", k=2)
```

**Optional: Run embeddings server (OpenAI-compatible)**

```bash
simplevecdb-server --port 8000
```

See [Setup Guide](ENV_SETUP.md) for configuration: model registry, rate limits, API keys, CUDA optimization.

### Option 3: With LangChain or LlamaIndex

Best for: Existing RAG pipelines, framework-based workflows.

```python
from simplevecdb.integrations.langchain import SimpleVecDBVectorStore
from langchain_openai import OpenAIEmbeddings

store = SimpleVecDBVectorStore(
    db_path="langchain.db",
    embedding=OpenAIEmbeddings(model="text-embedding-3-small")
)

store.add_texts(["Paris is the capital of France."])
results = store.similarity_search("capital of France", k=1)
hybrid = store.hybrid_search("France capital", k=3)  # BM25 + vector
```

**LlamaIndex:**

```python
from simplevecdb.integrations.llamaindex import SimpleVecDBLlamaStore
from llama_index.embeddings.openai import OpenAIEmbedding

store = SimpleVecDBLlamaStore(
    db_path="llama.db",
    embedding=OpenAIEmbedding(model="text-embedding-3-small")
)
```

See **[Examples](https://coderdayton.github.io/SimpleVecDB/examples/)** for complete RAG workflows with Ollama.

## Core Features

### Multi-Collection Support

Organize vectors by domain within a single database file:

```python
from simplevecdb import VectorDB, Quantization

db = VectorDB("app.db")
users = db.collection("users", quantization=Quantization.INT8)
products = db.collection("products", quantization=Quantization.BIT)

# Isolated namespaces
users.add_texts(["Alice likes hiking"], embeddings=[[0.1]*384])
products.add_texts(["Hiking boots"], embeddings=[[0.9]*384])
```

### Search Capabilities

```python
# Vector similarity (cosine/L2/inner product)
results = collection.similarity_search(query_vector, k=10)

# Keyword search (BM25)
results = collection.keyword_search("exact phrase", k=10)

# Hybrid (BM25 + vector fusion)
results = collection.hybrid_search("machine learning", k=10)
results = collection.hybrid_search("ML concepts", query_vector=my_vector, k=10)

# Metadata filtering
results = collection.similarity_search(
    query_vector,
    k=10,
    filter={"category": "technical", "verified": True}
)
```

> **Tip:** LangChain and LlamaIndex integrations support all search methods.

## Feature Matrix

| Feature                   | Status | Description                                                |
| :------------------------ | :----- | :--------------------------------------------------------- |
| **Single-File Storage**   | ✅     | SQLite `.db` file or in-memory mode                        |
| **Multi-Collection**      | ✅     | Isolated namespaces per database                           |
| **Vector Search**         | ✅     | Cosine, Euclidean, Inner Product metrics                   |
| **Hybrid Search**         | ✅     | BM25 + vector fusion (Reciprocal Rank Fusion)              |
| **Quantization**          | ✅     | FLOAT32, INT8, BIT (1-bit) for 4-32x compression           |
| **Metadata Filtering**    | ✅     | SQL `WHERE` clause support                                 |
| **Framework Integration** | ✅     | LangChain \& LlamaIndex adapters                           |
| **Hardware Acceleration** | ✅     | Auto-detects CUDA/MPS/CPU                                  |
| **Local Embeddings**      | ✅     | HuggingFace models via `[server]` extras                   |
| **HNSW Indexing**         | 🔜     | Approximate nearest neighbor (pending `sqlite-vec` update) |
| **Built-in Encryption**   | 🔜     | SQLCipher integration for at-rest encryption               |

## Performance Benchmarks

**Test Environment:** Intel i9-13900K, NVIDIA RTX 4090, `sqlite-vec` v0.1.6
**Dataset:** 10,000 vectors × 384 dimensions

| Quantization | Storage Size | Insert Speed | Query Latency (k=10) | Compression Ratio |
| :----------- | :----------- | :----------- | :------------------- | :---------------- |
| **FLOAT32**  | 15.50 MB     | 15,585 vec/s | 3.55 ms              | 1x (baseline)     |
| **INT8**     | 4.23 MB      | 27,893 vec/s | 3.93 ms              | 3.7x smaller      |
| **BIT**      | 0.95 MB      | 32,321 vec/s | 0.27 ms              | 16.3x smaller     |

**Key Takeaways:**

- BIT quantization delivers 13x faster queries with 16x storage reduction
- INT8 offers balanced performance (79% faster inserts, minimal query overhead)
- Sub-4ms query latency on consumer hardware

## Documentation

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
- Consider HNSW indexing when available (roadmap item)
- Use metadata filtering to reduce search space

## Roadmap

- [x] Hybrid Search (BM25 + Vector)
- [x] Multi-collection support
- [ ] HNSW indexing (pending `sqlite-vec` upstream)
- [ ] SQLCipher encryption (at-rest data protection)
- [ ] Streaming insert API for large-scale ingestion
- [ ] Graph-based metadata relationships

Vote on features or propose new ones in [GitHub Discussions](https://github.com/coderdayton/simplevecdb/discussions).

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

## Sponsors

SimpleVecDB is independently developed and maintained. If you or your company use it in production, please consider sponsoring to ensure its continued development and support.

**Company Sponsors**

_Become the first company sponsor!_ [Support on GitHub →](https://github.com/sponsors/coderdayton)

**Individual Supporters**

_Join the list of supporters!_ [Support on GitHub →](https://github.com/sponsors/coderdayton)

<!-- sponsors --><!-- sponsors -->

### Other Ways to Support

- 🍵 **[Buy me a coffee](https://www.buymeacoffee.com/coderdayton)** - One-time donation
- 💎 **[Get the Pro Pack](https://simplevecdb.lemonsqueezy.com/)** - Production deployment templates & recipes
- ⭐ **Star the repo** - Helps with visibility
- 🐛 **Report bugs** - Improve the project for everyone
- 📝 **Contribute** - See [CONTRIBUTING.md](CONTRIBUTING.md)

**Why sponsor?** Your support ensures SimpleVecDB stays maintained, secure, and compatible with the latest Python/SQLite versions.

## License

[MIT License](LICENSE) — Free for personal and commercial use.
