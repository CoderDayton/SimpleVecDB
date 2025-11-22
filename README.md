# TinyVecDB

[![CI](https://github.com/coderdayton/tinyvecdb/actions/workflows/ci.yml/badge.svg)](https://github.com/coderdayton/tinyvecdb/actions)
[![PyPI](https://img.shields.io/pypi/v/tinyvecdb?color=blue)](https://pypi.org/project/tinyvecdb/)
[![License: MIT](https://img.shields.io/github/license/coderdayton/tinyvecdb)](LICENSE)
[![GitHub Stars](https://img.shields.io/github/stars/coderdayton/tinyvecdb?style=social)](https://github.com/coderdayton/tinyvecdb)

**The dead-simple, local-first vector database.**

TinyVecDB brings **Chroma-like simplicity** to a single **SQLite file**. Built on `sqlite-vec`, it offers high-performance vector search, quantization, and zero infrastructure headaches. Perfect for local RAG, offline agents, and indie hackers.

---

## 🚀 Why TinyVecDB?

- **Zero Infra**: Just a `.db` file. No Docker, no Redis, no cloud bills.
- **Fast**: ~2ms queries on consumer hardware.
- **Efficient**: 4x-32x storage reduction with INT8/BIT quantization.
- **Universal**: Runs anywhere SQLite runs (Linux, macOS, Windows, WASM).
- **Batteries Included**: Optional FastAPI embeddings server & LangChain/LlamaIndex integrations.

## 📦 Installation

```bash
# Core only (lightweight)
pip install tinyvecdb

# With server & local models
pip install "tinyvecdb[server]"
```

## ⚡ Quickstart

### Environment Setup (optional)

View the [Setup Guide](docs/ENV_SETUP.md) for configuring environment variables for local embedding models.

**Python API:**

```python
from tinyvecdb import VectorDB, Quantization

# Initialize persistent DB with INT8 quantization
db = VectorDB("knowledge.db", quantization=Quantization.INT8)

# Add documents (auto-embedded locally)
db.add_texts(
    ["Paris is the capital of France.", "The mitochondria is the powerhouse of the cell."],
    metadatas=[{"category": "geography"}, {"category": "biology"}]
)

# Search
results = db.similarity_search("capital of France", k=1)
print(f"Result: {results[0][0].page_content}")

# Search with metadata filter
geo_results = db.similarity_search(
    "capital",
    k=3,
    filter={"category": "geography"},
)
```

**Embeddings Server:**

```bash
# Start the OpenAI-compatible server
tinyvecdb-server --port 8000
```

This server runs **entirely locally** and exposes an OpenAI-compatible `/v1/embeddings` endpoint backed by your configured HuggingFace model. TinyVecDB never calls remote APIs on your behalf.

**Using Remote Embeddings (Optional):**

You can also generate embeddings with a remote provider (e.g. OpenAI) **in your own code** and store them directly in TinyVecDB. TinyVecDB remains LLM- and provider-agnostic – it just stores whatever vectors you give it:

```python
from tinyvecdb import VectorDB
from openai import OpenAI

client = OpenAI()

texts = [
    "TinyVecDB is a local-first vector database.",
    "SQLite with sqlite-vec can power fast semantic search.",
]

# Use your preferred embedding model (e.g. OpenAI, Gemini, etc.)
embeddings = [
    client.embeddings.create(
        model="text-embedding-3-small",
        input=t,
    ).data[0].embedding
    for t in texts
]

db = VectorDB("remote_embed.db")
db.add_texts(texts=texts, embeddings=embeddings)
```

For end-to-end RAG examples using TinyVecDB with different LLMs (Ollama, LangChain, LlamaIndex), see the **[Examples](https://coderdayton.github.io/tinyvecdb/examples/)** page.

## 🛠️ Features

| Feature          | Status | Description                                     |
| :--------------- | :----- | :---------------------------------------------- |
| **Storage**      | ✅     | Single SQLite file or in-memory.                |
| **Search**       | ✅     | Cosine, Euclidean, and IP distance metrics.     |
| **Quantization** | ✅     | FLOAT32, INT8, and BIT (1-bit) support.         |
| **Filtering**    | ✅     | Metadata filtering with SQL `WHERE` clauses.    |
| **Integrations** | ✅     | First-class LangChain & LlamaIndex support.     |
| **Hardware**     | ✅     | Auto-detects CUDA/MPS/CPU for optimal batching. |

### Integrations (at a glance)

TinyVecDB plugs into popular Python ecosystems without dictating your LLM provider:

```python
from tinyvecdb import VectorDB

db = VectorDB("knowledge.db")

# LangChain
from tinyvecdb.integrations.langchain import TinyVecDBVectorStore
lc_store = TinyVecDBVectorStore(db_path="knowledge.db", embedding=my_langchain_embeddings)

# LlamaIndex
from tinyvecdb.integrations.llamaindex import TinyVecDBLlamaStore
li_store = TinyVecDBLlamaStore(db_path="knowledge.db")
```

## 📊 Benchmarks

_Tested on i9-13900K & RTX 4090 with `sqlite-vec` v0.1.2 (10k vectors, 384-dim)_

| Type      | Storage  | Insert Speed | Query Time (k=10) |
| :-------- | :------- | :----------- | :---------------- |
| **FLOAT** | 15.50 MB | 13,241 vec/s | 4.29 ms           |
| **INT8**  | 4.23 MB  | 23,472 vec/s | 4.33 ms           |
| **BIT**   | 0.95 MB  | 25,299 vec/s | 0.30 ms           |

## 📚 Documentation

- **[Setup Guide](docs/ENV_SETUP.md)**: Configuration and environment variables.
- **[API Reference](https://coderdayton.github.io/tinyvecdb/api/core/)**: Full class and method documentation.
- **[Benchmarks](https://coderdayton.github.io/tinyvecdb/benchmarks/)**: Performance comparisons.
- **[Examples](https://coderdayton.github.io/tinyvecdb/examples/)**: RAG notebooks and integration demos.
- **[Contributing](CONTRIBUTING.md)**: How to build and test.

## 🗺️ Roadmap

- [ ] Hybrid Search (BM25 + Vector)
- [ ] Multi-collection support
- [ ] HNSW Indexing (via sqlite-vec updates)
- [ ] Built-in Encryption (SQLCipher)

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details on setting up your development environment.

## ❤️ Sponsors

TinyVecDB is an independent, open‑source project, built and maintained transparently in public. If it’s useful to you and you’d like to support it, there are a few simple ways to contribute.

### Company Sponsors

_Become our first company sponsor! [Support us on GitHub](https://github.com/sponsors/coderdayton)_

### Individual Supporters

_Join the list of supporters! [Support us on GitHub](https://github.com/sponsors/coderdayton)_

**Want to support the project?**

- 🍵 [Buy me a coffee](https://www.buymeacoffee.com/coderdayton) (One-time donation)
- 💎 [Get the Pro Pack](https://tinyvecdb.gumroad.com/l/pro-pack) (Deployment templates & production recipes)
- 💖 [GitHub Sponsors](https://github.com/sponsors/coderdayton) (Monthly support)

## 📄 License

[MIT](LICENSE)
