# SimpleVecDB

[![CI](https://github.com/coderdayton/simplevecdb/actions/workflows/ci.yml/badge.svg)](https://github.com/coderdayton/simplevecdb/actions)
[![PyPI](https://img.shields.io/pypi/v/simplevecdb?color=blue)](https://pypi.org/project/simplevecdb/)
[![License: MIT](https://img.shields.io/github/license/coderdayton/simplevecdb)](LICENSE)
[![GitHub Stars](https://img.shields.io/github/stars/coderdayton/simplevecdb?style=social)](https://github.com/coderdayton/simplevecdb)

**The dead-simple, local-first vector database.**

SimpleVecDB brings **Chroma-like simplicity** to a single **SQLite file**. Built on `usearch` HNSW indexing, it offers high-performance vector search, quantization, and zero infrastructure headaches. Perfect for local RAG, offline agents, and indie hackers who need production-grade vector search without the operational overhead.

## Why SimpleVecDB?

- **Zero Infrastructure** — Just a `.db` file. No Docker, no Redis, no cloud bills.
- **Blazing Fast** — 10-100x faster search via usearch HNSW. Adaptive: brute-force for <10k vectors (perfect recall), HNSW for larger collections.
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

# With encryption support (SQLCipher)
pip install "simplevecdb[encryption]"
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
users = db.collection("users", quantization=Quantization.FLOAT16)  # 2x memory savings
products = db.collection("products", quantization=Quantization.BIT)  # 32x compression

# Isolated namespaces
users.add_texts(["Alice likes hiking"], embeddings=[[0.1]*384])
products.add_texts(["Hiking boots"], embeddings=[[0.9]*384])
```

### Search Capabilities

```python
# Vector similarity (cosine/L2) - adaptive search by default
results = collection.similarity_search(query_vector, k=10)

# Force exact search for perfect recall (brute-force)
results = collection.similarity_search(query_vector, k=10, exact=True)

# Force HNSW approximate search (faster, may miss some results)
results = collection.similarity_search(query_vector, k=10, exact=False)

# Parallel search with explicit thread count
results = collection.similarity_search(query_vector, k=10, threads=8)

# Batch search - 10x throughput for multiple queries
queries = [query1, query2, query3]  # List of embedding vectors
batch_results = collection.similarity_search_batch(queries, k=10)

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

### Encryption (v2.1+)

Protect sensitive data with AES-256 at-rest encryption:

```bash
pip install "simplevecdb[encryption]"
```

```python
from simplevecdb import VectorDB

# Create encrypted database
db = VectorDB("secure.db", encryption_key="your-secret-key")
collection = db.collection("confidential")

collection.add_texts(["sensitive data"], embeddings=[[0.1]*384])
db.close()

# Reopen requires same key
db = VectorDB("secure.db", encryption_key="your-secret-key")
```

### Streaming Insert (v2.1+)

Memory-efficient ingestion for large datasets:

```python
def load_documents():
    for line in open("large_file.jsonl"):
        doc = json.loads(line)
        yield (doc["text"], doc.get("metadata"), doc.get("embedding"))

for progress in collection.add_texts_streaming(load_documents(), batch_size=1000):
    print(f"Processed {progress['docs_processed']} documents")
```

### Document Hierarchies (v2.1+)

Organize documents in parent-child relationships:

```python
# Add parent document
parent_ids = collection.add_texts(["Main document"], embeddings=[[0.1]*384])

# Add children
child_ids = collection.add_texts(
    ["Chunk 1", "Chunk 2"],
    embeddings=[[0.11]*384, [0.12]*384],
    parent_ids=[parent_ids[0], parent_ids[0]]
)

# Navigate hierarchy
children = collection.get_children(parent_ids[0])
parent = collection.get_parent(child_ids[0])
descendants = collection.get_descendants(parent_ids[0])
```

## Feature Matrix

| Feature                   | Status | Description                                                  |
| :------------------------ | :----- | :----------------------------------------------------------- |
| **Single-File Storage**   | ✅     | SQLite `.db` file or in-memory mode                          |
| **Multi-Collection**      | ✅     | Isolated namespaces per database                             |
| **HNSW Indexing**         | ✅     | usearch HNSW for 10-100x faster search                       |
| **Adaptive Search**       | ✅     | Auto brute-force for <10k vectors, HNSW for larger           |
| **Vector Search**         | ✅     | Cosine, Euclidean metrics (L1 removed in v2.0)               |
| **Hybrid Search**         | ✅     | BM25 + vector fusion (Reciprocal Rank Fusion)                |
| **Quantization**          | ✅     | FLOAT32, FLOAT16, INT8, BIT for 2-32x compression            |
| **Parallel Operations**   | ✅     | `threads` parameter for add/search                           |
| **Metadata Filtering**    | ✅     | SQL `WHERE` clause support                                   |
| **Framework Integration** | ✅     | LangChain \& LlamaIndex adapters                             |
| **Hardware Acceleration** | ✅     | Auto-detects CUDA/MPS/CPU + SIMD via usearch                 |
| **Local Embeddings**      | ✅     | HuggingFace models via `[server]` extras                     |
| **Built-in Encryption**   | ✅     | SQLCipher AES-256 at-rest encryption via `[encryption]` extras |
| **Streaming Insert**      | ✅     | Memory-efficient large-scale ingestion with progress callbacks |
| **Document Hierarchies**  | ✅     | Parent/child relationships for chunked docs                  |

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

- [x] Hybrid Search (BM25 + Vector)
- [x] Multi-collection support
- [x] HNSW indexing (usearch backend)
- [x] Adaptive search (brute-force/HNSW)
- [x] SQLCipher encryption (at-rest data protection)
- [x] Streaming insert API for large-scale ingestion
- [x] Hierarchical document relationships (parent/child)
- [ ] Cross-collection search
- [ ] Vector clustering and auto-tagging

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
