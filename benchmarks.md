# Benchmarks

## 13900k & 4090, sqlite-vec v0.1.6

**Model**: Snowflake/snowflake-arctic-embed-xs<br>
**Batch Size**: Auto (512 on GPU)<br>
**Vector Count**: 10,000

| Quantization | Vectors | Dimensions | File Size | Insert Speed | Avg Query (k=10) |
| ------------ | ------- | ---------- | --------- | ------------ | ---------------- |
| FLOAT        | 10,000  | 384        | 15.50 MB  | 15,585 vec/s | 3.55 ms          |
| INT8         | 10,000  | 384        | 4.23 MB   | 27,893 vec/s | 3.93 ms          |
| BIT          | 10,000  | 384        | 0.95 MB   | 32,321 vec/s | 0.27 ms          |
| FLOAT        | 10,000  | 1536       | 60.55 MB  | 2,537 vec/s  | 15.71 ms         |

## 13900k & 4090, sqlite-vec v0.1.6

**Model**: Snowflake/snowflake-arctic-embed-xs<br>
**Batch Size**: Auto (512 on GPU)<br>
**Vector Count**: 100,000

| Quantization | Vectors | Dimensions | File Size | Insert Speed | Avg Query (k=10) |
| ------------ | ------- | ---------- | --------- | ------------ | ---------------- |
| FLOAT        | 100,000 | 384        | 151.83 MB | 9,513 vec/s  | 38.73 ms         |
| INT8         | 100,000 | 384        | 41.44 MB  | 13,213 vec/s | 39.08 ms         |
| BIT          | 100,000 | 384        | 9.28 MB   | 14,334 vec/s | 1.96 ms          |
| FLOAT        | 100,000 | 1536       | 593.40 MB | 2,258 vec/s  | 146.67 ms        |

## M2 MacBook Pro, sqlite-vec v0.1.2

| Quantization | Vectors | Dimensions | File Size | Avg Query (k=10) |
| ------------ | ------- | ---------- | --------- | ---------------- |
| FLOAT        | 10,000  | 384        | 14.7 MB   | 1.8 ms           |
| INT8         | 10,000  | 384        | 3.7 MB    | 1.9 ms           |
| BIT          | 10,000  | 384        | 0.5 MB    | 1.7 ms           |
| FLOAT        | 10,000  | 1536       | 59.2 MB   | 2.4 ms           |
