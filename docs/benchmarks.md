# Benchmarks

## Test Environment

**Hardware**: Intel i9-13900K, 64GB RAM  
**Backend**: usearch HNSW v2.12+  
**Vectors**: 384 dimensions (Snowflake/snowflake-arctic-embed-xs)

## Search Speed (k=10)

Comparison of usearch HNSW approximate search vs brute-force exact search:

| Collection Size | usearch (ms) | Brute-force (ms) | Speedup | Recall@10 |
| :-------------- | :----------- | :--------------- | :------ | :-------- |
| **1,000**       | 0.11         | 0.11             | 1.0x    | 1.00      |
| **5,000**       | 0.33         | 0.34             | 1.0x    | 1.00      |
| **10,000**      | 0.23         | 0.82             | 3.6x    | 1.00      |
| **50,000**      | 0.26         | 4.67             | 18x     | 0.70      |
| **100,000**     | 0.27         | 9.18             | 34x     | 0.60      |

> **Note**: Collections <10k use adaptive brute-force for perfect recall by default.

## Storage Efficiency (Quantization)

10,000 vectors at 384 dimensions:

| Quantization  | Total Size | Query Latency | Vector Compression | Notes                           |
| :------------ | :--------- | :------------ | :----------------- | :------------------------------ |
| **FLOAT32**   | 36.0 MB    | 0.20 ms       | 1x (baseline)      | Full precision                  |
| **FLOAT16**   | 28.7 MB    | 0.20 ms       | 2x                 | Best balance for most use cases |
| **INT8**      | 25.0 MB    | 0.16 ms       | 4x                 | Good for memory-constrained     |
| **BIT**       | 21.8 MB    | 0.08 ms       | 32x                | Fastest, Hamming distance       |

> **Note**: Total size includes SQLite metadata + HNSW graph structure (~21MB overhead). Pure vector storage follows theoretical compression ratios.

## Adaptive Search Behavior

SimpleVecDB v2.0 automatically selects the optimal search strategy:

| Collection Size | Default Strategy | Rationale                              |
| --------------- | ---------------- | -------------------------------------- |
| < 10,000        | Brute-force      | Perfect recall, HNSW overhead not worth it |
| ≥ 10,000        | HNSW             | 3-34x faster, tunable recall           |

Override with the `exact` parameter:

```python
# Force brute-force (perfect recall)
results = collection.similarity_search(query, k=10, exact=True)

# Force HNSW (faster)
results = collection.similarity_search(query, k=10, exact=False)
```

## Batch Search Performance

Use `similarity_search_batch()` for multi-query workloads:

| Batch Size | Sequential Time | Batch Time | Speedup |
| ---------- | --------------- | ---------- | ------- |
| 10         | 2.0 ms          | 0.5 ms     | 4x      |
| 100        | 20.0 ms         | 3.0 ms     | 7x      |
| 1000       | 200.0 ms        | 25.0 ms    | 8x      |

```python
queries = [query1, query2, query3]  # List of embedding vectors
batch_results = collection.similarity_search_batch(queries, k=10)
```

## Memory-Mapping Behavior

Large indexes automatically use memory-mapped mode for instant startup:

| Collection Size | Mode | Startup Time | Memory Usage |
| --------------- | ---- | ------------ | ------------ |
| < 100,000       | Load | ~50ms        | Full index   |
| ≥ 100,000       | Mmap | ~1ms         | On-demand    |

## Quantization Guide

| Quantization | Bits/Dim | Memory | Speed | Recall | Best For                     |
| ------------ | -------- | ------ | ----- | ------ | ---------------------------- |
| FLOAT32      | 32       | 1x     | 1x    | 100%   | Maximum precision            |
| FLOAT16      | 16       | 0.5x   | 1x    | ~99%   | Balanced (recommended)       |
| INT8         | 8        | 0.25x  | 1.2x  | ~97%   | Memory-constrained           |
| BIT          | 1        | 0.03x  | 2.5x  | ~85%   | Massive scale, fast filtering|

## Legacy Benchmarks (v1.x with sqlite-vec)

For historical reference, here are the v1.x benchmarks using sqlite-vec brute-force search.

### 10,000 Vectors (sqlite-vec v0.1.6)

| Quantization | Storage | Insert Speed | Query (k=10) |
| ------------ | ------- | ------------ | ------------ |
| FLOAT32      | 15.5 MB | 15,585 vec/s | 3.55 ms      |
| INT8         | 4.2 MB  | 27,893 vec/s | 3.93 ms      |
| BIT          | 0.95 MB | 32,321 vec/s | 0.27 ms      |

### 100,000 Vectors (sqlite-vec v0.1.6)

| Quantization | Storage  | Insert Speed | Query (k=10) |
| ------------ | -------- | ------------ | ------------ |
| FLOAT32      | 151.8 MB | 9,513 vec/s  | 38.73 ms     |
| INT8         | 41.4 MB  | 13,213 vec/s | 39.08 ms     |
| BIT          | 9.3 MB   | 14,334 vec/s | 1.96 ms      |

## Running Your Own Benchmarks

```bash
# Install with server extras for embedding generation
pip install "simplevecdb[server]"

# Run the backend benchmark
python examples/backend_benchmark.py

# Run quantization benchmark  
python examples/quant_benchmark.py

# Run full performance benchmark
python examples/embeddings/perf_benchmark.py
```
