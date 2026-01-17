# Core API

## VectorDB

The main database class for managing vector collections.

::: simplevecdb.core.VectorDB
    options:
      members:
        - collection
        - list_collections
        - search_collections
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
        - cluster
        - auto_tag
        - assign_cluster_metadata
        - get_cluster_members
        - save_cluster
        - load_cluster
        - list_clusters
        - delete_cluster
        - assign_to_cluster

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

### Cross-Collection Search

Search across multiple collections with unified, ranked results:

```python
from simplevecdb import VectorDB

db = VectorDB("app.db")

# Initialize collections
users = db.collection("users")
products = db.collection("products")
docs = db.collection("docs")

# Add data to each collection
users.add_texts(["Alice likes hiking"], embeddings=[[0.1]*384])
products.add_texts(["Hiking boots", "Trail map"], embeddings=[[0.2]*384, [0.15]*384])
docs.add_texts(["Mountain hiking guide"], embeddings=[[0.12]*384])

# List initialized collections
print(db.list_collections())  # ['users', 'products', 'docs']

# Search across ALL collections
results = db.search_collections([0.1]*384, k=5)
for doc, score, collection_name in results:
    print(f"[{collection_name}] {doc.page_content} (score: {score:.3f})")

# Search specific collections only
results = db.search_collections(
    [0.1]*384,
    collections=["users", "products"],  # Exclude 'docs'
    k=3
)

# With metadata filtering (applies to all collections)
results = db.search_collections(
    [0.1]*384,
    k=10,
    filter={"category": "outdoor"}
)

# Disable score normalization (returns inverted distances)
results = db.search_collections([0.1]*384, normalize_scores=False)

# Sequential search (disable parallelism)
results = db.search_collections([0.1]*384, parallel=False)
```

| Method | Description |
|--------|-------------|
| `list_collections()` | Names of all initialized collections |
| `search_collections(query, collections, k, filter, normalize_scores, parallel)` | Search across multiple collections with merged results |

<a id="clustering-auto-tagging"></a>

### Clustering & Auto-Tagging

Group similar documents and generate descriptive tags:

```python
from simplevecdb import VectorDB

db = VectorDB("app.db")
collection = db.collection("docs")

# Add documents with embeddings
collection.add_texts(texts, embeddings=embeddings)

# Cluster documents into groups
result = collection.cluster(
    n_clusters=5,
    algorithm="minibatch_kmeans",  # or "kmeans", "hdbscan"
    random_state=42
)
print(result.summary())  # {0: 42, 1: 38, 2: 15, 3: 3, 4: 2}

# Generate keyword tags for each cluster
tags = collection.auto_tag(result, n_keywords=5)
# {0: 'machine learning, neural network, deep', 1: 'database, sql, query', ...}

# Persist cluster assignments to metadata
collection.assign_cluster_metadata(result, tags)

# Query documents by cluster
ml_docs = collection.get_cluster_members(0)
db_docs = collection.similarity_search(query, filter={"cluster": 1})

# Custom tagging callback
def summarize_cluster(texts: list[str]) -> str:
    return f"Group of {len(texts)} docs about {texts[0][:20]}..."

custom_tags = collection.auto_tag(result, method="custom", custom_callback=summarize_cluster)
```

| Method | Description |
|--------|-------------|
| `cluster(n_clusters, algorithm, filter, sample_size)` | Cluster documents by embedding similarity |
| `auto_tag(result, method, n_keywords, custom_callback)` | Generate descriptive tags for clusters |
| `assign_cluster_metadata(result, tags, metadata_key)` | Persist cluster IDs to document metadata |
| `get_cluster_members(cluster_id, metadata_key)` | Retrieve all documents in a cluster |
| `save_cluster(name, result, metadata)` | Save cluster centroids for later assignment |
| `load_cluster(name)` | Load saved cluster configuration |
| `list_clusters()` | List all saved cluster configurations |
| `delete_cluster(name)` | Delete a saved cluster configuration |
| `assign_to_cluster(name, doc_ids, metadata_key)` | Assign documents to saved clusters |

**Algorithms:**

| Algorithm | Best For | Requires n_clusters |
|-----------|----------|-------------------|
| `minibatch_kmeans` | Large datasets (default) | Yes |
| `kmeans` | Small datasets, precise centroids | Yes |
| `hdbscan` | Unknown cluster count, density-based | No |

Clustering is included in the standard installation (no extras needed).

### Cluster Metrics

Access clustering quality metrics to evaluate results:

```python
result = collection.cluster(n_clusters=5, random_state=42)

# Inertia (K-means only): sum of squared distances to centroids
# Lower is better; indicates tighter clusters
print(f"Inertia: {result.inertia}")

# Silhouette score: measure of cluster separation (-1 to 1)
# Higher is better; >0.5 indicates good clustering
print(f"Silhouette: {result.silhouette_score}")

# Get all metrics as dict
metrics = result.metrics()
# {'inertia': 1523.45, 'silhouette_score': 0.62}
```

### Cluster Persistence

Save cluster configurations for fast assignment of new documents:

```python
# 1. Cluster your documents
result = collection.cluster(n_clusters=5, random_state=42)
tags = collection.auto_tag(result)

# 2. Save cluster state (centroids + metadata)
collection.save_cluster(
    "product_categories",
    result,
    metadata={"tags": tags, "version": 1}
)

# 3. Later: assign new documents without re-clustering
new_ids = collection.add_texts(new_texts, embeddings=new_embeddings)
collection.assign_to_cluster("product_categories", new_ids)

# List saved clusters
clusters = collection.list_clusters()
# [{'name': 'product_categories', 'n_clusters': 5, 'algorithm': 'minibatch_kmeans', ...}]

# Load cluster for inspection
saved = collection.load_cluster("product_categories")
if saved:
    result, meta = saved
    print(f"Loaded {result.n_clusters} clusters, tags: {meta['tags']}")

# Delete when no longer needed
collection.delete_cluster("product_categories")
```
