# Examples

## RAG with LangChain

[View Notebook](https://github.com/coderdayton/simplevecdb/blob/main/examples/rag/langchain_rag.ipynb)

## RAG with LlamaIndex

[View Notebook](https://github.com/coderdayton/simplevecdb/blob/main/examples/rag/llama_rag.ipynb)

## RAG with Ollama LLM

[View Notebook](https://github.com/coderdayton/simplevecdb/blob/main/examples/rag/ollama_rag.ipynb)

## Quick Start Examples

### Basic Usage

```python
from simplevecdb import VectorDB, Quantization

db = VectorDB("vectors.db")
collection = db.collection("docs", quantization=Quantization.FLOAT16)

# Add documents with embeddings
texts = ["Paris is the capital of France", "Berlin is in Germany"]
embeddings = [[0.1] * 384, [0.2] * 384]  # Your embedding model output
collection.add_texts(texts, embeddings=embeddings)

# Search
query_embedding = [0.1] * 384
results = collection.similarity_search(query_embedding, k=5)
for doc, score in results:
    print(f"{doc.page_content} (score: {score:.4f})")
```

### Keyword & Hybrid Search

```python
from simplevecdb import VectorDB

db = VectorDB("local.db")
collection = db.collection("default")
collection.add_texts(
    ["banana is yellow", "grapes are purple"],
    embeddings=[[0.1, 0.2] * 192, [0.3, 0.4] * 192]
)

# BM25 keyword search
bm25 = collection.keyword_search("banana", k=1)

# Hybrid search (BM25 + vectors with RRF)
hybrid = collection.hybrid_search("yellow fruit", k=2)
```

### Batch Search (v2.0+)

```python
from simplevecdb import VectorDB

db = VectorDB("vectors.db")
collection = db.collection("docs")

# Add some documents...
collection.add_texts(texts, embeddings=embeddings)

# Search multiple queries at once (~10x faster than sequential)
queries = [embedding1, embedding2, embedding3]
results = collection.similarity_search_batch(queries, k=10)

for i, query_results in enumerate(results):
    print(f"Query {i}: {len(query_results)} results")
```

### Force Exact Search

```python
# Adaptive (default): brute-force for <10k, HNSW for larger
results = collection.similarity_search(query, k=10)

# Force brute-force for perfect recall
results = collection.similarity_search(query, k=10, exact=True)

# Force HNSW for speed on small collections
results = collection.similarity_search(query, k=10, exact=False)
```

### Metadata Filtering

```python
collection.add_texts(
    ["doc1", "doc2", "doc3"],
    embeddings=[...],
    metadatas=[
        {"category": "tech", "year": 2024},
        {"category": "science", "year": 2023},
        {"category": "tech", "year": 2023},
    ]
)

# Filter by metadata
results = collection.similarity_search(
    query,
    k=10,
    filter={"category": "tech"}
)
```

### Encrypted Database (v2.1+)

```bash
pip install "simplevecdb[encryption]"
```

```python
from simplevecdb import VectorDB

# Create encrypted database with SQLCipher
db = VectorDB("secure.db", encryption_key="your-secret-key")
collection = db.collection("confidential")

collection.add_texts(
    ["sensitive financial data", "private health records"],
    embeddings=[[0.1]*384, [0.2]*384],
    metadatas=[{"type": "finance"}, {"type": "health"}]
)
db.close()

# Reopen requires the same encryption key
db = VectorDB("secure.db", encryption_key="your-secret-key")
results = db.collection("confidential").similarity_search([0.1]*384, k=1)

# Wrong key will fail
try:
    bad_db = VectorDB("secure.db", encryption_key="wrong-key")
except Exception as e:
    print(f"Access denied: {e}")
```

### Streaming Insert (v2.1+)

Memory-efficient ingestion for large datasets:

```python
import json
from simplevecdb import VectorDB

db = VectorDB("large_dataset.db")
collection = db.collection("docs")

# Generator function for memory efficiency
def load_documents():
    with open("large_file.jsonl") as f:
        for line in f:
            doc = json.loads(line)
            yield (
                doc["text"],
                doc.get("metadata", {}),
                doc["embedding"]
            )

# Stream with progress tracking
for progress in collection.add_texts_streaming(load_documents(), batch_size=1000):
    print(f"Batch {progress['batch_num']}: {progress['docs_processed']} total docs")

# With callback instead of iteration
def log_progress(p):
    if p['docs_processed'] % 10000 == 0:
        print(f"Milestone: {p['docs_processed']} docs")

list(collection.add_texts_streaming(
    load_documents(),
    batch_size=500,
    on_progress=log_progress
))
```

### Document Hierarchies (v2.1+)

Organize documents in parent-child trees for chunked documents, threads, or nested content:

```python
from simplevecdb import VectorDB

db = VectorDB("hierarchical.db")
collection = db.collection("docs")

# Add parent documents (e.g., full articles)
parent_ids = collection.add_texts(
    ["Chapter 1: Introduction to ML", "Chapter 2: Neural Networks"],
    embeddings=[[0.1]*384, [0.2]*384],
    metadatas=[{"type": "chapter", "num": 1}, {"type": "chapter", "num": 2}]
)

# Add children with parent references (e.g., chunks/sections)
child_ids = collection.add_texts(
    ["Section 1.1: What is ML?", "Section 1.2: History of ML", "Section 2.1: Perceptrons"],
    embeddings=[[0.11]*384, [0.12]*384, [0.21]*384],
    metadatas=[
        {"type": "section", "chapter": 1},
        {"type": "section", "chapter": 1},
        {"type": "section", "chapter": 2}
    ],
    parent_ids=[parent_ids[0], parent_ids[0], parent_ids[1]]
)

# Navigate the hierarchy
children = collection.get_children(parent_ids[0])
print(f"Chapter 1 has {len(children)} sections")

parent = collection.get_parent(child_ids[0])
print(f"Section belongs to: {parent.page_content}")

# Get all descendants (recursive)
all_descendants = collection.get_descendants(parent_ids[0])

# Get ancestors (path to root)
ancestors = collection.get_ancestors(child_ids[0])

# Reparent a document
collection.set_parent(child_ids[2], parent_ids[0])  # Move section to Chapter 1

# Search within a subtree
results = collection.similarity_search(
    [0.1]*384,
    k=10,
    filter={"type": "section", "chapter": 1}
)
```

### Async Usage

```python
import asyncio
from simplevecdb import AsyncVectorDB

async def main():
    async with AsyncVectorDB("vectors.db") as db:
        collection = db.collection("docs")
        
        # Add documents
        await collection.add_texts(texts, embeddings=embeddings)
        
        # Batch search
        results = await collection.similarity_search_batch(queries, k=10)
        
        return results

results = asyncio.run(main())
```

## Benchmark Scripts

### Backend Benchmark

Compare HNSW vs brute-force performance:

```bash
python examples/backend_benchmark.py
```

### Quantization Benchmark

Test different quantization levels:

```bash
python examples/quant_benchmark.py
```

### Embedding Performance

Benchmark local embedding generation:

```bash
python examples/embeddings/perf_benchmark.py
```
