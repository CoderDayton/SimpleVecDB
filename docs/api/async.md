# Async API

SimpleVecDB provides async wrappers for use in async/await contexts. These are thin wrappers around the synchronous API using `ThreadPoolExecutor`.

## Quick Start

```python
import asyncio
from simplevecdb import AsyncVectorDB

async def main():
    db = AsyncVectorDB("vectors.db")
    collection = db.collection("docs")

    # Add documents asynchronously
    ids = await collection.add_texts(
        ["Hello world", "Async is great"],
        embeddings=[[0.1] * 384, [0.2] * 384]
    )

    # Search asynchronously
    results = await collection.similarity_search([0.1] * 384, k=5)
    return results

results = asyncio.run(main())
```

## Configuration

The async wrappers use a `ThreadPoolExecutor` for concurrent operations. You can configure the number of workers:

```python
# Default: 4 workers
db = AsyncVectorDB("vectors.db")

# Custom worker count
db = AsyncVectorDB("vectors.db", max_workers=8)
```

## Available Methods

`AsyncVectorCollection` provides async versions of all search and modification methods:

| Sync Method                       | Async Method                                       |
| --------------------------------- | -------------------------------------------------- |
| `add_texts()`                     | `await collection.add_texts()`                     |
| `similarity_search()`             | `await collection.similarity_search()`             |
| `keyword_search()`                | `await collection.keyword_search()`                |
| `hybrid_search()`                 | `await collection.hybrid_search()`                 |
| `max_marginal_relevance_search()` | `await collection.max_marginal_relevance_search()` |
| `delete_by_ids()`                 | `await collection.delete_by_ids()`                 |
| `remove_texts()`                  | `await collection.remove_texts()`                  |

Synchronous properties remain unchanged:

- `collection.name` - Collection name

## Concurrent Operations

Run multiple searches in parallel with `asyncio.gather`:

```python
async def concurrent_search():
    db = AsyncVectorDB("vectors.db")
    collection = db.collection("docs")

    queries = [[0.1] * 384, [0.2] * 384, [0.3] * 384]

    # Run all searches concurrently
    results = await asyncio.gather(*[
        collection.similarity_search(q, k=5)
        for q in queries
    ])
    return results
```

## When to Use

**Use Async API when:**

- Building async web servers (FastAPI, aiohttp)
- Running concurrent searches
- Integrating with async frameworks

**Use Sync API when:**

- Simple scripts and notebooks
- Single-threaded applications
- Maximum simplicity is needed

## API Reference

::: simplevecdb.async_core.AsyncVectorDB

::: simplevecdb.async_core.AsyncVectorCollection
