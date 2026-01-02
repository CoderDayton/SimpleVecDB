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
        - add_texts_streaming
        - similarity_search
        - similarity_search_batch
        - keyword_search
        - hybrid_search
        - max_marginal_relevance_search
        - delete_by_ids
        - remove_texts
        - rebuild_index
        - get_children
        - get_parent
        - get_descendants
        - get_ancestors
        - set_parent

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

### Streaming Insert

For large-scale ingestion without memory pressure:

```python
# From generator/iterator
def load_documents():
    for line in open("large_file.jsonl"):
        doc = json.loads(line)
        yield (doc["text"], doc.get("metadata"), doc.get("embedding"))

for progress in collection.add_texts_streaming(load_documents()):
    print(f"Batch {progress['batch_num']}: {progress['docs_processed']} total")

# With progress callback
def log_progress(p):
    print(f"{p['docs_processed']} docs, batch {p['batch_num']}")

list(collection.add_texts_streaming(items, batch_size=500, on_progress=log_progress))
```

### Hierarchical Relationships

Organize documents in parent-child hierarchies for chunked documents, threaded conversations, or nested content:

```python
# Add documents with parent relationships
parent_ids = collection.add_texts(["Main document"], metadatas=[{"type": "parent"}])
parent_id = parent_ids[0]

# Add children referencing the parent
child_ids = collection.add_texts(
    ["Chunk 1", "Chunk 2", "Chunk 3"],
    parent_ids=[parent_id, parent_id, parent_id]
)

# Navigate the hierarchy
children = collection.get_children(parent_id)         # Direct children
parent = collection.get_parent(child_ids[0])          # Get parent document
descendants = collection.get_descendants(parent_id)   # All nested children
ancestors = collection.get_ancestors(child_ids[0])    # Path to root

# Reparent or orphan documents
collection.set_parent(child_ids[0], new_parent_id)    # Move to new parent
collection.set_parent(child_ids[0], None)             # Make root document

# Search within a subtree
results = collection.similarity_search(
    query_embedding,
    k=5,
    filter={"parent_id": parent_id}  # Only search children
)
```

| Method | Description |
|--------|-------------|
| `get_children(doc_id)` | Direct children of a document |
| `get_parent(doc_id)` | Parent document (or None if root) |
| `get_descendants(doc_id, max_depth)` | All nested children recursively |
| `get_ancestors(doc_id)` | Path from document to root |
| `set_parent(doc_id, parent_id)` | Move document to new parent (or None to orphan) |
