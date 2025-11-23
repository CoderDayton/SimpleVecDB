# Benchmarks

## M2 MacBook Pro, sqlite-vec v0.1.2

| Quantization | Vectors | Dimensions | File Size | Avg Query (k=10) |
| ------------ | ------- | ---------- | --------- | ---------------- |
| FLOAT        | 10,000  | 384        | 14.7 MB   | 1.8 ms           |
| INT8         | 10,000  | 384        | 3.7 MB    | 1.9 ms           |
| BIT          | 10,000  | 384        | 0.5 MB    | 1.7 ms           |
| FLOAT        | 10,000  | 1536       | 59.2 MB   | 2.4 ms           |

## 13900k & 4090, batch_size=512, sqlite-vec v0.1.2

| Quantization | Vectors | Dimensions | File Size | Insert Speed | Avg Query (k=10) |
| ------------ | ------- | ---------- | --------- | ------------ | ---------------- |
| FLOAT        | 10,000  | 384        | 15.50 MB  | 13,241 vec/s | 4.29 ms          |
| INT8         | 10,000  | 384        | 4.23 MB   | 23,472 vec/s | 4.33 ms          |
| BIT          | 10,000  | 384        | 0.95 MB   | 25,299 vec/s | 0.30 ms          |
| FLOAT        | 10,000  | 1536       | 60.55 MB  | 2,276 vec/s  | 16.74 ms         |
