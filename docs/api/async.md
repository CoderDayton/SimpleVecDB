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
| `reserve_ids()`                   | `await collection.reserve_ids()`                   |
| `similarity_search()`             | `await collection.similarity_search()`             |
| `similarity_search_batch()`       | `await collection.similarity_search_batch()`       |
| `keyword_search()`                | `await collection.keyword_search()`                |
| `hybrid_search()`                 | `await collection.hybrid_search()`                 |
| `max_marginal_relevance_search()` | `await collection.max_marginal_relevance_search()` |
| `delete_by_ids()`                 | `await collection.delete_by_ids()`                 |
| `remove_texts()`                  | `await collection.remove_texts()`                  |

Synchronous properties remain unchanged:

- `collection.name` - Collection name

## Sub-namespaces

Every sync sub-namespace is mirrored on the async collection, name for name:

```python
await collection.edges.upsert(src, dst, kind="cites", weight=0.9)
await collection.counters.increment(doc_id, {"hits": 1})
await collection.ttl.sweep()
await collection.pending.flush()

async for event in collection.events.subscribe(since=0):
    ...
```

## Transactions

`async with collection.tx()` mirrors the sync context manager. Catalog writes
and vector writes commit or roll back together:

```python
async with collection.tx() as coll:
    await coll.delete_by_ids([1])
    await coll.add_texts(["replacement"], embeddings=[[0.1] * 384])
```

A transaction holds a `threading.RLock` for its lifetime, so its enter and
exit must happen on one thread — the shared pool cannot promise that, and
releasing an RLock from a thread that never acquired it raises. Each
transaction therefore gets a private single-worker executor.

**Operate through the yielded handle.** It is bound to that pinned thread;
the outer collection is not. Awaiting work on the outer handle inside the
block sends it to the shared pool, where it blocks on the DB lock this
transaction holds — and that lock is not released until the block exits,
which cannot happen while it is awaiting.

`atomic()` makes that mistake unrepresentable: the callback is synchronous
and cannot await at all.

```python
def move(coll):
    coll.delete_by_ids([1])
    return coll.add_texts(["replacement"], embeddings=[[0.1] * 384])

new_ids = await collection.atomic(move)
```

`AsyncVectorDB.transaction(fn)` is the database-wide equivalent, spanning
collections via `tx["name"]`.

## Concurrent Operations

Run multiple searches in parallel with `asyncio.gather` or use batch search for better performance:

```python
async def concurrent_search():
    db = AsyncVectorDB("vectors.db")
    collection = db.collection("docs")

    queries = [[0.1] * 384, [0.2] * 384, [0.3] * 384]

    # Option 1: Batch search (recommended, ~10x faster)
    results = await collection.similarity_search_batch(queries, k=5)

    # Option 2: Concurrent individual searches
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
