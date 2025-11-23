# TinyVecDB

[![CI](https://github.com/coderdayton/tinyvecdb/actions/workflows/ci.yml/badge.svg)](https://github.com/coderdayton/tinyvecdb/actions)
[![PyPI](https://img.shields.io/pypi/v/tinyvecdb?color=blue)](https://pypi.org/project/tinyvecdb/)
[![License: MIT](https://img.shields.io/github/license/coderdayton/tinyvecdb)](LICENSE)
[![GitHub Stars](https://img.shields.io/github/stars/coderdayton/tinyvecdb?style=social)](https://github.com/coderdayton/tinyvecdb)

**The dead-simple, local-first vector database.**

TinyVecDB brings **Chroma-like simplicity** to a single **SQLite file**. Built on `sqlite-vec`, it offers high-performance vector search, quantization, and zero infrastructure headaches. Perfect for local RAG, offline agents, and indie hackers who need production-grade vector search without the operational overhead.

---

## Why TinyVecDB?

- **Zero Infrastructure** — Just a `.db` file. No Docker, no Redis, no cloud bills.
- **Blazing Fast** — ~2ms queries on consumer hardware with 32x storage efficiency via quantization.
- **Truly Portable** — Runs anywhere SQLite runs: Linux, macOS, Windows, even WASM.
- **Batteries Included** — Optional FastAPI embeddings server + LangChain/LlamaIndex integrations.
- **Production Ready** — Hybrid search (BM25 + vector), metadata filtering, multi-collection support, and automatic hardware acceleration.

### When to Choose TinyVecDB

| Use Case                       | TinyVecDB             | Cloud Vector DB          |
| :----------------------------- | :-------------------- | :----------------------- |
| **Local RAG applications**     | ✅ Perfect fit        | ❌ Overkill + latency    |
| **Offline-first agents**       | ✅ No internet needed | ❌ Requires connectivity |
| **Prototyping \& MVPs**        | ✅ Zero config        | ⚠️ Setup overhead        |
| **Multi-tenant SaaS at scale** | ⚠️ Consider sharding  | ✅ Built for this        |
| **Budget-conscious projects**  | ✅ \$0/month          | ❌ \$50-500+/month       |

---

## Prerequisites

**System Requirements:**

- Python 3.10+
- SQLite 3.35+ with FTS5 support (included in Python 3.8+ standard library)
- 50MB+ disk space for core library, 500MB+ with `[server]` extras

**Optional for GPU Acceleration:**

- CUDA 11.8+ for NVIDIA GPUs
- Metal Performance Shaders (MPS) for Apple Silicon

> **Note:** If using custom-compiled SQLite, ensure `-DSQLITE_ENABLE_FTS5` is enabled for full-text search support.

---

## Installation

```bash
# Core library only (lightweight, 50MB)
pip install tinyvecdb

# With local embeddings server + HuggingFace models (500MB+)
pip install "tinyvecdb[server]"
```

**Verify Installation:**

```bash
python -c "from tinyvecdb import VectorDB; print('TinyVecDB installed successfully!')"
```

---

## Quickstart

TinyVecDB is **just a vector storage layer**—it doesn't include an LLM or generate embeddings. This design keeps it lightweight and flexible. Choose your integration path:

### Option 1: With OpenAI (Simplest)

Best for: Quick prototypes, production apps with OpenAI subscriptions.

```python
from tinyvecdb import VectorDB
from openai import OpenAI

# Initialize database
db = VectorDB("knowledge.db")
collection = db.collection("knowledge_base")

# Generate embeddings using OpenAI
client = OpenAI()
texts = [
    "Paris is the capital of France.",
    "The mitochondria is the powerhouse of the cell."
]

embeddings = [
    client.embeddings.create(
        model="text-embedding-3-small",
        input=t
    ).data[0].embedding
    for t in texts
]

# Store vectors with metadata
collection.add_texts(
    texts=texts,
    embeddings=embeddings,
    metadatas=[
        {"category": "geography", "verified": True},
        {"category": "biology", "verified": True}
    ]
)

# Semantic search
query_embedding = client.embeddings.create(
    model="text-embedding-3-small",
    input="capital of France"
).data[0].embedding

results = collection.similarity_search(query_embedding, k=1)
print(f"Top result: {results[0][0].page_content}")
# Output: Top result: Paris is the capital of France.

# Filter by metadata
geo_results = collection.similarity_search(
    query_embedding,
    k=10,
    filter={"category": "geography"}
)
print(f"Geography results: {len(geo_results)}")
```

---

### Option 2: Fully Local (Privacy-First)

Best for: Offline apps, sensitive data, zero API costs.

```bash
pip install "tinyvecdb[server]"
```

```python
from tinyvecdb import VectorDB
from tinyvecdb.embeddings.models import embed_texts

db = VectorDB("local.db")
collection = db.collection("local_docs")

texts = [
    "Paris is the capital of France.",
    "The mitochondria is the powerhouse of the cell."
]

# Generate embeddings locally (uses HuggingFace models)
embeddings = embed_texts(texts)

collection.add_texts(
    texts=texts,
    embeddings=embeddings,
    metadatas=[{"category": "geography"}, {"category": "biology"}]
)

# Search locally
query_embeddings = embed_texts(["capital of France"])
results = collection.similarity_search(query_embeddings[0], k=1)
print(f"Result: {results[0][0].page_content}")

# Hybrid search (BM25 + vector fusion)
hybrid_results = collection.hybrid_search("powerhouse cell", k=2)
for doc, score in hybrid_results:
    print(f"{doc.page_content} (score: {score:.3f})")
```

**Optional: Local Embeddings Server**

Run an OpenAI-compatible API endpoint locally:

```bash
tinyvecdb-server --port 8000
# Use http://localhost:8000/v1/embeddings with any OpenAI SDK
```

See the [Setup Guide](ENV_SETUP.md) for advanced configuration: model registry locking, rate limits, API key authentication, and CUDA optimization.

---

### Option 3: With LangChain or LlamaIndex

Best for: Existing RAG pipelines, framework-based workflows.

```python
from tinyvecdb.integrations.langchain import TinyVecDBVectorStore
from langchain_openai import OpenAIEmbeddings

# Use any LangChain embedding model
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
store = TinyVecDBVectorStore(
    db_path="langchain.db",
    embedding=embeddings
)

# Standard LangChain interface
store.add_texts(["Paris is the capital of France."])
results = store.similarity_search("capital of France", k=1)
print(results[0].page_content)

# Hybrid search available too
hybrid_results = store.hybrid_search("France capital", k=3)
```

**For LlamaIndex:**

```python
from tinyvecdb.integrations.llamaindex import TinyVecDBVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding

embedding = OpenAIEmbedding(model="text-embedding-3-small")
store = TinyVecDBVectorStore(db_path="llamaindex.db", embedding=embedding)
# Use with LlamaIndex's VectorStoreIndex
```

See complete RAG workflows with Ollama, LangChain, and LlamaIndex in the **[Examples](https://coderdayton.github.io/tinyvecdb/examples/)** page.

---

## Core Features

### Multi-Collection Support

Organize vectors by domain within a single database file:

```python
from tinyvecdb import VectorDB, Quantization

db = VectorDB("multi_tenant.db")

# Different collections can have different quantization strategies
users = db.collection("user_profiles", quantization=Quantization.INT8)
products = db.collection("product_catalog", quantization=Quantization.FLOAT)
documents = db.collection("knowledge_base", quantization=Quantization.BIT)

# Isolated namespaces
users.add_texts(["Alice likes hiking"], embeddings=[[0.1]*384])
products.add_texts(["Hiking boots"], embeddings=[[0.9]*384])

# Collections don't interfere with each other
assert len(users.similarity_search([0.1]*384, k=10)) == 1
assert len(products.similarity_search([0.9]*384, k=10)) == 1
```

---

### Search Capabilities

**Vector Similarity Search**

```python
# Cosine similarity (default)
results = collection.similarity_search(query_vector, k=10)
```

**Keyword Search (BM25 + FTS5)**

```python
# Full-text search with BM25 ranking
keyword_results = collection.keyword_search("exact phrase matching", k=10)
```

**Hybrid Search (Best of Both Worlds)**

```python
# Combines BM25 + vector similarity via Reciprocal Rank Fusion
hybrid_results = collection.hybrid_search(
    query_text="machine learning concepts",
    k=10
)

# Reuse your own embeddings (skips re-encoding)
hybrid_results = collection.hybrid_search(
    query_text="machine learning concepts",
    query_vector=my_precomputed_vector,
    k=10
)
```

**Metadata Filtering**

```python
# SQL WHERE clause syntax
filtered = collection.similarity_search(
    query_vector,
    k=10,
    filter={"category": "technical", "verified": True}
)
```

> **Pro tip:** Both LangChain and LlamaIndex integrations expose `keyword_search()` and `hybrid_search()` methods for framework-native hybrid retrieval.

---

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

---

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

See detailed benchmarks across different hardware configurations in the **[Benchmarks](benchmarks.md)** section.

---

## Documentation

- **[Setup Guide](ENV_SETUP.md)** — Environment variables, server configuration, authentication
- **[API Reference](api/core.md)** — Complete class/method documentation with type signatures
- **[Benchmarks](benchmarks.md)** — Quantization strategies, batch sizes, hardware optimization
- **[Integration Examples](examples.md)** — RAG notebooks, Ollama workflows, production patterns
- **[Contributing Guide](CONTRIBUTING.md)** — Development setup, testing, PR guidelines

---

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

Vote on features or propose new ones in [GitHub Discussions](https://github.com/coderdayton/tinyvecdb/discussions).

## Contributing

Contributions are welcome! Whether you're fixing bugs, improving documentation, or proposing new features:

1. Read [CONTRIBUTING.md](CONTRIBUTING.md) for development setup
2. Check existing [Issues](https://github.com/coderdayton/tinyvecdb/issues) and [Discussions](https://github.com/coderdayton/tinyvecdb/discussions)
3. Open a PR with clear description and tests

**Need help?** Join the conversation in [GitHub Discussions](https://github.com/coderdayton/tinyvecdb/discussions).

## Community & Support

**Get Help:**

- [GitHub Discussions](https://github.com/coderdayton/tinyvecdb/discussions) — Q\&A and feature requests
- [GitHub Issues](https://github.com/coderdayton/tinyvecdb/issues) — Bug reports

**Stay Updated:**

- [GitHub Releases](https://github.com/coderdayton/tinyvecdb/releases) — Changelog and updates
- [Examples Gallery](https://coderdayton.github.io/tinyvecdb/examples/) — Community-contributed notebooks

## Sponsors

TinyVecDB is an independent open-source project. If it's useful to you, consider supporting its development:

**Company Sponsors**
_Become the first company sponsor!_ [Support on GitHub →](https://github.com/sponsors/coderdayton)

**Individual Supporters**
_Join the list of supporters!_ [Support on GitHub →](https://github.com/sponsors/coderdayton)

<!-- sponsors --><!-- sponsors -->

## Other Ways to Support

- ☕ [Buy me a coffee](https://www.buymeacoffee.com/coderdayton) (one-time donation)
- 💎 [Get the Pro Pack](https://tinyvecdb.lemonsqueezy.com/) — Deployment templates \& production recipes (coming soon)
- 💖 [GitHub Sponsors](https://github.com/sponsors/coderdayton) (monthly support)

## License

[MIT License](LICENSE) — Free for personal and commercial use.
