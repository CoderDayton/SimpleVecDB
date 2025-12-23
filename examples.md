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
