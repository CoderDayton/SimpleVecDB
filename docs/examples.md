# Examples

## RAG notebooks

End-to-end RAG pipelines against a real LLM:

- [LangChain](https://github.com/coderdayton/simplevecdb/blob/main/examples/rag/langchain_rag.ipynb)
- [LlamaIndex](https://github.com/coderdayton/simplevecdb/blob/main/examples/rag/llama_rag.ipynb)
- [Ollama (fully local)](https://github.com/coderdayton/simplevecdb/blob/main/examples/rag/ollama_rag.ipynb)

## Storage & search

### Basic usage

```python
from simplevecdb import VectorDB, Quantization

db = VectorDB("notes.db")
notes = db.collection("personal", quantization=Quantization.FLOAT16)

notes.add_texts(
    [
        "Cherry MX silent reds bottom out around 45g",
        "Sourdough hydration sweet spot is ~75% with this flour",
    ],
    embeddings=[your_embedder(t) for t in texts],
)

for doc, score in notes.similarity_search(query_vec, k=5):
    print(f"{score:.3f}  {doc.page_content}")
```

### Keyword + hybrid search

```python
notes.keyword_search("postgres", k=3)              # BM25 over FTS5
notes.hybrid_search("slow query", k=3)             # BM25 + vector RRF
notes.hybrid_search(
    "slow query",
    k=3,
    query_vector=v,
    vector_k=20,
    keyword_k=20,
    rrf_k=60,
)
```

### Batch search

```python
results = notes.similarity_search_batch([v1, v2, v3], k=10)
for q, hits in zip(("a", "b", "c"), results):
    print(q, len(hits))
```

### Adaptive vs exact

```python
notes.similarity_search(q, k=10)                 # adaptive — brute < 10k, HNSW above
notes.similarity_search(q, k=10, exact=True)     # always brute (perfect recall)
notes.similarity_search(q, k=10, exact=False)    # always HNSW
```

### Metadata + range filters (v2.6.1)

Equality and `$in` work everywhere; the v2.6.1 operators add range and
existence checks. Filters apply to `similarity_search`,
`keyword_search`, `hybrid_search`, `edges.get_edges`, and
`events.read`.

```python
notes.add_texts(
    [...],
    embeddings=[...],
    metadatas=[
        {"tag": "work",   "year": 2026, "score": 0.91},
        {"tag": "baking", "year": 2024, "score": 0.40},
    ],
)

notes.similarity_search(q, k=10, filter={"tag": "work"})

notes.similarity_search(q, k=10, filter={
    "score":    {"$gt": 0.5, "$lte": 0.95},
    "tag":      {"$in": ["work", "research"]},
    "archived": {"$exists": False},
})

# Tuple shorthand normalises to the operator-dict form
notes.similarity_search(q, k=10, filter={"score": (">", 0.5)})
notes.similarity_search(q, k=10, filter={"year":  ("range", 2024, 2026)})
```

### Encrypted database

```bash
pip install "simplevecdb[encryption]"
```

```python
from simplevecdb import VectorDB

db = VectorDB("secure.db", encryption_key="your-secret-key")
db.collection("confidential").add_texts(
    ["sensitive note"],
    embeddings=[[0.1] * 384],
)
db.close()

VectorDB("secure.db", encryption_key="your-secret-key")  # ok
VectorDB("secure.db", encryption_key="wrong-key")        # raises
```

### Streaming insert

```python
import json
from simplevecdb import VectorDB

db = VectorDB("dump.db")
notes = db.collection("dump")

def feed():
    with open("dump.jsonl") as f:
        for line in f:
            row = json.loads(line)
            yield row["text"], row.get("metadata", {}), row["embedding"]

for progress in notes.add_texts_streaming(feed(), batch_size=1000):
    print(f"batch {progress['batch_num']}: {progress['docs_processed']} docs")
```

### Document hierarchies

```python
parents = collection.add_texts(
    ["Chapter 1", "Chapter 2"],
    embeddings=[[0.1] * 384, [0.2] * 384],
)
children = collection.add_texts(
    ["1.1 intro", "1.2 history", "2.1 perceptrons"],
    embeddings=[[0.11] * 384, [0.12] * 384, [0.21] * 384],
    parent_ids=[parents[0], parents[0], parents[1]],
)

collection.get_children(parents[0])
collection.get_parent(children[0])
collection.get_descendants(parents[0])
collection.get_ancestors(children[0])
collection.set_parent(children[2], parents[0])  # reparent
```

### Async usage

```python
import asyncio
from simplevecdb import AsyncVectorDB

async def main():
    async with AsyncVectorDB("notes.db") as db:
        notes = db.collection("personal")
        await notes.add_texts(texts, embeddings=embeddings)
        return await notes.similarity_search_batch(queries, k=10)

asyncio.run(main())
```

### Cross-collection search

```python
from simplevecdb import VectorDB

db = VectorDB("app.db")
db.collection("users").add_texts([...],    embeddings=[...])
db.collection("products").add_texts([...], embeddings=[...])
db.collection("tickets").add_texts([...],  embeddings=[...])

for doc, score, name in db.search_collections(query, k=5):
    print(f"[{name}] {doc.page_content}  {score:.3f}")

db.search_collections(query, collections=["users", "products"], k=3)
db.search_collections(query, k=10, filter={"category": "software"})
db.list_collections()
```

## Memory primitives (v2.6.1)

The 2.6.1 release adds primitives for retrieval-with-memory systems —
in-place vector updates, atomic counters, edges, expiry, and an
append-only change feed. Full reference:
[Features](Features.md).

### Pending vector buffer

`update_embedding` writes to a per-collection overlay; the new vector
becomes visible to reads immediately and is promoted into HNSW on
`pending.flush()`. Removes the HNSW remove+re-add churn previously
required for in-place updates.

```python
collection.update_embedding(doc_id, new_vector, source="recompute")
collection.update_embedding(other_id, vec2)
collection.pending.flush(max_batch=512)        # promote both into HNSW

collection.pending.update_many([(id1, v1), (id2, v2)])
collection.pending.blend_toward([id1, id2, id3], centroid=c, alpha=0.1)
```

### Atomic counters

`increment_metadata` applies a dict-of-deltas to JSON metadata in one
SQL statement — WAL-atomic and safe under concurrent writers.

```python
collection.increment_metadata(doc_id, {"hits": 1, "drift": 0.02})
```

### Weighted directed edges

```python
collection.edges.add_edge(src, dst, kind="cites", weight=0.8, hits=1)
collection.edges.update_edge(src, dst, kind="cites", dweight=+0.05, dhits=+1)

collection.edges.get_edges(
    src=src,
    filter={"weight": {"$gt": 0.5}, "hits": ("range", 1, 10)},
)
collection.edges.delete_edge(src, dst, kind="cites")

# Bulk threshold prune
collection.edges.prune(kind="cites", max_weight=0.1, idle_before=cutoff_ts)
```

### TTL / expiry

```python
import time

collection.ttl.set(doc_id, seconds=3600,             on_expire="delete")
collection.ttl.set(other,  expires_at=time.time()+5, on_expire="callback")

deleted, callbacks = collection.ttl.sweep()         # one-shot
collection.ttl.start_background(interval=60.0)      # daemon thread
collection.ttl.clear(doc_id)
```

### Append-only event feed

Every mutating call appends one row (kind, doc_id, payload, monotonic
seq). Useful for change feeds, replication, and audit trails:

```python
seq = collection.events.last_seq()
# … do some writes …
for ev in collection.events.read(since=seq, kind="edge_add", limit=200):
    print(ev.seq, ev.kind, ev.doc_id, ev.payload)

for ev in collection.events.subscribe(since=seq, poll_interval=0.5):
    handle(ev)                                       # blocking generator

collection.events.prune(before_seq=seq - 100_000)
```

### Transactions

`db.transaction()` and `collection.tx()` wrap a SAVEPOINT around
catalog writes (metadata, counters, edges, events, TTL, and the
pending overlay). A raised exception rolls all SQL writes back. Coarse
vector mutations (`add_texts`, `delete_by_ids`) are *not* rolled back —
use `update_embedding` + `pending.flush()` for vector changes that
must be commit-gated.

```python
with db.transaction() as tx:
    tx["personal"].increment_metadata(1, {"hits": 1})
    tx["personal"].edges.add_edge(1, 2, kind="cites", weight=0.6)
    # any exception below rolls both writes back
```

## Benchmark scripts

```bash
python examples/backend_benchmark.py         # HNSW vs brute-force
python examples/quant_benchmark.py           # quantization tradeoffs
python examples/embeddings/perf_benchmark.py # local embedding throughput
```
