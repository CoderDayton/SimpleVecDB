# Vector Clustering Guide

Vector clustering discovers natural groupings in your embeddings, enabling automatic categorization, topic discovery, and semantic organization of documents.

## Installation

Clustering is included in the standard installation:

```bash
pip install simplevecdb
```

**Dependencies included:**
- `scikit-learn>=1.3.0` — K-means and MiniBatch K-means algorithms
- `hdbscan>=0.8.33` — Density-based clustering

No extra installation steps required!

## Quick Start

```python
from simplevecdb import VectorDB

db = VectorDB("products.db")
collection = db.collection("items")

# Add documents with embeddings
collection.add_texts(texts=descriptions, embeddings=embeddings)

# Cluster into 5 groups
result = collection.cluster(n_clusters=5)

# Auto-generate descriptive tags
tags = collection.auto_tag(result, method="tfidf", n_keywords=3)
# {'0': ['electronics', 'wireless', 'bluetooth'], '1': ['clothing', ...], ...}

# Persist cluster IDs to document metadata
collection.assign_cluster_metadata(result, tags)

# Retrieve all documents in cluster 0
docs = collection.get_cluster_members(0)
```

## Algorithms

### K-Means (`kmeans`)

Classic centroid-based clustering. Best for balanced, spherical clusters.

```python
result = collection.cluster(
    n_clusters=5,
    algorithm="kmeans",
    random_state=42  # Reproducible results
)
```

**Pros:**
- Fast and deterministic
- Works well with balanced clusters
- Provides cluster centroids for assignment

**Cons:**
- Requires specifying `n_clusters` upfront
- Sensitive to outliers
- Assumes spherical cluster shapes

**Best for:** Product categorization, customer segmentation, content organization

### MiniBatch K-Means (`minibatch_kmeans`, default)

Scalable variant of K-means using mini-batches. 3-10x faster on large datasets.

```python
result = collection.cluster(
    n_clusters=10,
    algorithm="minibatch_kmeans",
    sample_size=5000,  # Use subset for speed
    random_state=42
)
```

**Pros:**
- Scales to millions of documents
- Memory-efficient
- Nearly identical quality to K-means

**Cons:**
- Slightly less stable than K-means
- Still requires `n_clusters`

**Best for:** Large-scale document clustering, real-time categorization

### HDBSCAN (`hdbscan`)

Density-based clustering that automatically discovers cluster count and handles noise.

```python
result = collection.cluster(
    algorithm="hdbscan",
    min_cluster_size=10  # Minimum documents per cluster
)
```

**Pros:**
- Automatically determines optimal cluster count
- Handles noise (assigns label `-1` to outliers)
- Discovers non-spherical clusters

**Cons:**
- Slower than K-means variants
- No centroids (cannot assign new documents)
- Requires tuning `min_cluster_size`

**Best for:** Exploratory analysis, topic discovery, anomaly detection

## Cluster Quality Metrics

Evaluate clustering quality with built-in metrics:

```python
result = collection.cluster(n_clusters=5)

# Silhouette Score: -1 to 1 (higher is better)
# Measures how well-separated clusters are
print(f"Silhouette: {result.silhouette_score:.2f}")
# > 0.7: Strong clustering
# 0.5-0.7: Reasonable clustering
# < 0.5: Weak clustering

# Inertia: Sum of squared distances to centroids (K-means only)
# Lower is better (indicates tighter clusters)
print(f"Inertia: {result.inertia:.2f}")

# Get all metrics as dict
metrics = result.metrics()
# {'inertia': 1523.45, 'silhouette_score': 0.62}
```

## Auto-Tagging

Generate human-readable labels for clusters:

### TF-IDF Method (default)

Extracts keywords with highest TF-IDF scores per cluster.

```python
tags = collection.auto_tag(result, method="tfidf", n_keywords=5)
# {'0': ['machine', 'learning', 'neural', 'network', 'deep'], ...}
```

**Best for:** Text documents with distinct vocabulary per cluster

### Frequency Method

Extracts most common words per cluster.

```python
tags = collection.auto_tag(result, method="frequency", n_keywords=3)
```

**Best for:** Short documents, social media posts

### Custom Callback

Implement custom tagging logic:

```python
def custom_tagger(cluster_id: int, texts: list[str]) -> list[str]:
    # Your logic here (e.g., LLM-based summarization)
    return ["tag1", "tag2", "tag3"]

tags = collection.auto_tag(result, custom_callback=custom_tagger)
```

## Cluster Persistence

Save cluster configurations for fast assignment of new documents without re-clustering.

### Save Cluster State

```python
result = collection.cluster(n_clusters=5)
tags = collection.auto_tag(result)

collection.save_cluster(
    "product_categories",
    result,
    metadata={"tags": tags, "version": 1, "created_at": "2026-01-17"}
)
```

### Load Cluster State

```python
loaded = collection.load_cluster("product_categories")
if loaded:
    result, metadata = loaded
    print(f"Loaded {result.n_clusters} clusters")
    print(f"Tags: {metadata['tags']}")
```

### Assign New Documents

```python
# Add new documents
new_ids = collection.add_texts(new_texts, embeddings=new_embeddings)

# Assign to nearest cluster centroids
assigned_count = collection.assign_to_cluster("product_categories", new_ids)
print(f"Assigned {assigned_count} documents")

# Retrieve assigned documents
docs = collection.get_cluster_members(0)
```

### List and Delete

```python
# List all saved clusters
clusters = collection.list_clusters()
for c in clusters:
    print(f"{c['name']}: {c['n_clusters']} clusters, {c['algorithm']}")

# Delete when no longer needed
collection.delete_cluster("product_categories")
```

## Filtering and Sampling

Cluster subsets of your collection:

```python
# Cluster only verified documents
result = collection.cluster(
    n_clusters=3,
    filter={"verified": True}
)

# Use random sample for speed (large collections)
result = collection.cluster(
    n_clusters=10,
    sample_size=10000  # Cluster 10k random documents
)
```

## Async Support

All clustering methods have async equivalents:

```python
from simplevecdb import AsyncVectorDB

async with AsyncVectorDB("products.db") as db:
    collection = db.collection("items")
    
    result = await collection.cluster(n_clusters=5)
    tags = await collection.auto_tag(result)
    await collection.save_cluster("categories", result)
    
    new_ids = await collection.add_texts(texts, embeddings=embeddings)
    await collection.assign_to_cluster("categories", new_ids)
```

## Use Cases

### Product Categorization

```python
# Cluster products by description embeddings
result = collection.cluster(n_clusters=20, algorithm="minibatch_kmeans")
tags = collection.auto_tag(result, n_keywords=5)
collection.assign_cluster_metadata(result, tags)

# Save for new products
collection.save_cluster("product_taxonomy", result, metadata={"tags": tags})
```

### Topic Discovery

```python
# Let HDBSCAN discover natural topics
result = collection.cluster(algorithm="hdbscan", min_cluster_size=50)
tags = collection.auto_tag(result, method="tfidf", n_keywords=10)

# Analyze cluster sizes
for cluster_id in range(result.n_clusters):
    docs = collection.get_cluster_members(cluster_id)
    print(f"Topic {cluster_id}: {len(docs)} docs - {tags[str(cluster_id)]}")
```

### Customer Segmentation

```python
# Cluster customer profiles
result = collection.cluster(n_clusters=8, random_state=42)
collection.assign_cluster_metadata(result)

# Target marketing campaigns per segment
segment_0_customers = collection.get_cluster_members(0)
```

### Duplicate Detection

```python
# Use high cluster count to find near-duplicates
result = collection.cluster(n_clusters=1000, algorithm="minibatch_kmeans")
collection.assign_cluster_metadata(result)

# Find potential duplicates in same cluster
for cluster_id in range(result.n_clusters):
    docs = collection.get_cluster_members(cluster_id)
    if len(docs) > 1:
        print(f"Potential duplicates in cluster {cluster_id}: {len(docs)} docs")
```

## Best Practices

### Choosing Cluster Count

1. **Elbow Method**: Plot inertia vs. `n_clusters`, look for "elbow"
2. **Silhouette Analysis**: Maximize silhouette score
3. **Domain Knowledge**: Use business requirements (e.g., 10 product categories)
4. **HDBSCAN**: Let algorithm decide

```python
# Elbow method
inertias = []
for k in range(2, 20):
    result = collection.cluster(n_clusters=k)
    inertias.append(result.inertia)
# Plot and find elbow
```

### Performance Optimization

```python
# Large collections: use sampling + MiniBatch K-means
result = collection.cluster(
    n_clusters=50,
    algorithm="minibatch_kmeans",
    sample_size=50000
)

# Small collections: use K-means for stability
result = collection.cluster(
    n_clusters=5,
    algorithm="kmeans",
    random_state=42
)
```

### Reproducibility

Always set `random_state` for deterministic results:

```python
result = collection.cluster(n_clusters=5, random_state=42)
```

### Metadata Organization

Use consistent metadata keys:

```python
collection.assign_cluster_metadata(result, tags, metadata_key="category")
collection.assign_cluster_metadata(result, tags, metadata_key="topic")
```

## API Reference

See [VectorCollection API](../api/core.md#clustering-auto-tagging) for complete method signatures and parameters.

## Troubleshooting

**ValueError: n_clusters must be >= 2**

K-means requires at least 2 clusters. Use HDBSCAN for single-cluster detection.

**Silhouette score is None**

Occurs when clustering produces only 1 cluster or all documents in separate clusters. Adjust `n_clusters` or `min_cluster_size`.

**HDBSCAN assigns all documents to noise (cluster -1)**

Decrease `min_cluster_size` or increase document count. HDBSCAN needs sufficient density.

**Slow clustering on large collections**

Use `sample_size` parameter or switch to `minibatch_kmeans`:

```python
result = collection.cluster(n_clusters=10, sample_size=10000)
```

**Cluster assignments change between runs**

Set `random_state` for reproducibility:

```python
result = collection.cluster(n_clusters=5, random_state=42)
```
