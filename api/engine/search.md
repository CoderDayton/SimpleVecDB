# Search Engine

The search module handles all vector and keyword search operations using the usearch HNSW backend.

## Overview

`SearchEngine` is the internal class that powers all search operations in SimpleVecDB. It manages:

- **HNSW Index**: Fast approximate nearest neighbor search via usearch
- **Brute-Force Fallback**: Exact search for small collections or when requested
- **Keyword Search**: BM25 full-text search via SQLite FTS5
- **Hybrid Search**: Reciprocal Rank Fusion combining vector and keyword results

## Search Behavior

### Adaptive Search Strategy

The search engine automatically selects the optimal strategy:

| Collection Size | Default Strategy | Override |
|-----------------|------------------|----------|
| < 10,000        | Brute-force      | `exact=False` for HNSW |
| ≥ 10,000        | HNSW             | `exact=True` for brute-force |

### Batch Search

For multiple queries, `similarity_search_batch()` provides ~10x throughput by using usearch's native batch search:

```python
# Single queries (slower for many queries)
results = [collection.similarity_search(q, k=10) for q in queries]

# Batch search (10x faster)
results = collection.similarity_search_batch(queries, k=10)
```

## API Reference

::: simplevecdb.engine.search.SearchEngine
    options:
      members:
        - similarity_search
        - similarity_search_batch
        - keyword_search
        - hybrid_search
        - max_marginal_relevance_search
