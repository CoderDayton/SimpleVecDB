# Types Reference

Core type definitions and data classes used throughout SimpleVecDB.

## Document

::: simplevecdb.types.Document
    options:
      show_root_heading: true
      show_source: false

Represents a document with text content, embeddings, and metadata.

**Example:**
```python
from simplevecdb.types import Document

doc = Document(
    id=1,
    page_content="Paris is the capital of France.",
    metadata={"category": "geography", "verified": True},
    embedding=[0.1, 0.2, 0.3, ...]
)
```

## ClusterResult

::: simplevecdb.types.ClusterResult
    options:
      show_root_heading: true
      show_source: false
      members:
        - n_clusters
        - labels
        - centroids
        - algorithm
        - inertia
        - silhouette_score
        - summary
        - metrics

Result of clustering operation with quality metrics.

**Example:**
```python
result = collection.cluster(n_clusters=5)

print(f"Clusters: {result.n_clusters}")
print(f"Algorithm: {result.algorithm}")
print(f"Silhouette: {result.silhouette_score:.2f}")
print(f"Inertia: {result.inertia:.2f}")

# Cluster size distribution
print(result.summary())
# {0: 42, 1: 38, 2: 15, 3: 3, 4: 2}

# All metrics
metrics = result.metrics()
# {'inertia': 1523.45, 'silhouette_score': 0.62}
```

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `n_clusters` | `int` | Number of clusters discovered |
| `labels` | `np.ndarray` | Cluster ID for each document (shape: `[n_docs]`) |
| `centroids` | `np.ndarray \| None` | Cluster centroids (shape: `[n_clusters, dim]`). `None` for HDBSCAN. |
| `algorithm` | `ClusterAlgorithm` | Algorithm used: `"kmeans"`, `"minibatch_kmeans"`, or `"hdbscan"` |
| `inertia` | `float \| None` | Sum of squared distances to centroids (K-means only, lower is better) |
| `silhouette_score` | `float \| None` | Cluster separation metric (-1 to 1, higher is better) |

### Methods

#### `summary() -> dict[int, int]`

Returns cluster size distribution.

```python
result = collection.cluster(n_clusters=3)
sizes = result.summary()
# {0: 50, 1: 30, 2: 20}
```

#### `metrics() -> dict[str, float | None]`

Returns all quality metrics as a dictionary.

```python
metrics = result.metrics()
# {'inertia': 1523.45, 'silhouette_score': 0.62}
```

## Enums

### DistanceStrategy

::: simplevecdb.types.DistanceStrategy
    options:
      show_root_heading: true
      show_source: false

Distance metrics for vector similarity.

| Value | Description | Use Case |
|-------|-------------|----------|
| `COSINE` | Cosine similarity (default) | Text embeddings, normalized vectors |
| `EUCLIDEAN` | L2 distance | Image embeddings, spatial data |
| `INNER_PRODUCT` | Dot product | Pre-normalized embeddings |

**Example:**
```python
from simplevecdb import VectorDB, DistanceStrategy

collection = db.collection("docs", distance_strategy=DistanceStrategy.COSINE)
```

### Quantization

::: simplevecdb.types.Quantization
    options:
      show_root_heading: true
      show_source: false

Vector compression strategies.

| Value | Precision | Compression | Speed | Use Case |
|-------|-----------|-------------|-------|----------|
| `FLOAT` | 32-bit | 1x | Baseline | High precision required |
| `FLOAT16` | 16-bit | 2x | Fast | Recommended default |
| `INT8` | 8-bit | 4x | Faster | Large collections |
| `BIT` | 1-bit | 32x | Fastest | Massive scale, approximate search |

**Example:**
```python
from simplevecdb import VectorDB, Quantization

# 2x memory savings, minimal precision loss
collection = db.collection("docs", quantization=Quantization.FLOAT16)

# 32x compression for massive scale
collection = db.collection("docs", quantization=Quantization.BIT)
```

### ClusterAlgorithm

Clustering algorithms (string literals accepted by `VectorCollection.cluster`).

| Value | Description | Requires n_clusters | Provides Centroids |
|-------|-------------|--------------------|--------------------|
| `kmeans` | Classic K-means | Yes | Yes |
| `minibatch_kmeans` | Scalable K-means (default) | Yes | Yes |
| `hdbscan` | Density-based clustering | No | No |

**Example:**
```python
# Auto-discover cluster count
result = collection.cluster(algorithm="hdbscan", min_cluster_size=10)

# Fixed cluster count with centroids
result = collection.cluster(n_clusters=5, algorithm="minibatch_kmeans")
```

## Type Aliases

### EmbeddingVector

```python
EmbeddingVector = list[float] | np.ndarray
```

Represents a single embedding vector.

### MetadataDict

```python
MetadataDict = dict[str, Any]
```

Document metadata with arbitrary key-value pairs.

### FilterDict

```python
FilterDict = dict[str, Any]
```

Metadata filter for search operations. Supports equality matching.

**Example:**
```python
results = collection.similarity_search(
    query_vector,
    k=10,
    filter={"category": "tech", "verified": True}
)
```
