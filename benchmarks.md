# Benchmarks

## v2.0 HNSW Benchmarks (usearch v2.12+)

**Model**: Snowflake/snowflake-arctic-embed-xs  
**Hardware**: Intel i9-13900K, 64GB RAM  
**Backend**: usearch HNSW

### 100,000 Vectors (384 dimensions)

| Quantization | Storage Size | Insert Speed | Query (k=10) | vs Brute-Force | Recall@10 |
| ------------ | ------------ | ------------ | ------------ | -------------- | --------- |
| FLOAT32      | 153 MB       | 45,000 vec/s | 0.8 ms       | 48x faster     | 99.1%     |
| FLOAT16      | 77 MB        | 52,000 vec/s | 0.5 ms       | 78x faster     | 98.9%     |
| INT8         | 42 MB        | 58,000 vec/s | 0.4 ms       | 98x faster     | 97.5%     |
| BIT          | 9 MB         | 65,000 vec/s | 0.2 ms       | 10x faster     | 89.2%     |

### 1,000,000 Vectors (384 dimensions)

| Quantization | Storage Size | Insert Speed | Query (k=10) | vs Brute-Force | Recall@10 |
| ------------ | ------------ | ------------ | ------------ | -------------- | --------- |
| FLOAT32      | 1.5 GB       | 38,000 vec/s | 1.2 ms       | 320x faster    | 98.8%     |
| FLOAT16      | 768 MB       | 44,000 vec/s | 0.8 ms       | 480x faster    | 98.5%     |
| INT8         | 416 MB       | 50,000 vec/s | 0.6 ms       | 640x faster    | 96.8%     |
| BIT          | 86 MB        | 58,000 vec/s | 0.3 ms       | 30x faster     | 85.1%     |

### Batch Search Performance

| Batch Size | Sequential Time | Batch Time | Speedup |
| ---------- | --------------- | ---------- | ------- |
| 10         | 8.0 ms          | 1.2 ms     | 6.7x    |
| 100        | 80.0 ms         | 8.5 ms     | 9.4x    |
| 1000       | 800.0 ms        | 72.0 ms    | 11.1x   |

Use `similarity_search_batch()` for multi-query workloads.

## Adaptive Search Behavior

SimpleVecDB v2.0 automatically selects the optimal search strategy:

| Collection Size | Default Strategy | Rationale |
| --------------- | ---------------- | --------- |
| < 10,000        | Brute-force      | Perfect recall, HNSW overhead not worth it |
| ≥ 10,000        | HNSW             | 10-100x faster, >97% recall |

Override with the `exact` parameter:
```python
# Force brute-force (perfect recall)
results = collection.similarity_search(query, k=10, exact=True)

# Force HNSW (faster)
results = collection.similarity_search(query, k=10, exact=False)
```

## Memory-Mapping Behavior

Large indexes automatically use memory-mapped mode:

| Collection Size | Mode | Startup Time | Memory Usage |
| --------------- | ---- | ------------ | ------------ |
| < 100,000       | Load | ~50ms        | Full index   |
| ≥ 100,000       | Mmap | ~1ms         | On-demand    |

Memory-mapped indexes load instantly and only read pages as needed.

## Quantization Guide

| Quantization | Bits/Dim | Memory | Speed | Recall | Best For |
| ------------ | -------- | ------ | ----- | ------ | -------- |
| FLOAT32      | 32       | 1x     | 1x    | 100%   | Maximum precision |
| FLOAT16      | 16       | 0.5x   | 1.5x  | ~99%   | Balanced (recommended) |
| INT8         | 8        | 0.25x  | 2x    | ~97%   | Memory-constrained |
| BIT          | 1        | 0.03x  | 3x    | ~85%   | Massive scale, initial filtering |

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
```
