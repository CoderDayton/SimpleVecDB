# Core API

## VectorDB

The main database class for managing vector collections.

::: simplevecdb.core.VectorDB
    options:
      members:
        - collection
        - vacuum
        - close
        - check_migration

## VectorCollection

A named collection of vectors within a database.

::: simplevecdb.core.VectorCollection
    options:
      members:
        - add_texts
        - similarity_search
        - similarity_search_batch
        - keyword_search
        - hybrid_search
        - max_marginal_relevance_search
        - delete_by_ids
        - remove_texts
        - rebuild_index

## Quick Reference

### Search Methods

| Method | Description | Use Case |
|--------|-------------|----------|
| `similarity_search()` | Vector similarity search | Single query, best match |
| `similarity_search_batch()` | Batch vector search | Multiple queries, ~10x throughput |
| `keyword_search()` | BM25 full-text search | Keyword matching |
| `hybrid_search()` | BM25 + vector fusion | Best of both worlds |
| `max_marginal_relevance_search()` | Diversity-aware search | Avoid redundant results |

### Search Parameters

```python
# Adaptive search (default) - auto-selects brute-force or HNSW
results = collection.similarity_search(query, k=10)

# Force exact brute-force search (perfect recall)
results = collection.similarity_search(query, k=10, exact=True)

# Force HNSW approximate search (faster)
results = collection.similarity_search(query, k=10, exact=False)

# Parallel search with explicit thread count
results = collection.similarity_search(query, k=10, threads=4)

# Batch search for multiple queries
results = collection.similarity_search_batch(queries, k=10)
```

### Quantization Options

```python
from simplevecdb import Quantization

# Full precision (default)
collection = db.collection("docs", quantization=Quantization.FLOAT)

# Half precision - 2x memory savings, 1.5x faster
collection = db.collection("docs", quantization=Quantization.FLOAT16)

# 8-bit quantization - 4x memory savings
collection = db.collection("docs", quantization=Quantization.INT8)

# 1-bit quantization - 32x memory savings
collection = db.collection("docs", quantization=Quantization.BIT)
```
