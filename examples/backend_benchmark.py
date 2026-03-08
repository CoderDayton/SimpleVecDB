#!/usr/bin/env python3
"""
Benchmark: usearch HNSW vs brute-force baseline.

Demonstrates the performance characteristics of usearch's HNSW algorithm
for vector similarity search.

IMPORTANT NOTES:
- HNSW has overhead; it's SLOWER than brute-force for small collections (<50k vectors)
- Speedup increases with collection size: 2x at 50k, 10x at 500k, 100x at 5M+
- Random vectors are pathologically hard for HNSW (similarities are very close)
- Real embeddings (from models) typically show better recall
- Increase --ef for higher recall at cost of speed (default: 64, try 256 for ~90% recall)

Usage:
    python examples/backend_benchmark.py
    python examples/backend_benchmark.py --n 100000 --dim 768
    python examples/backend_benchmark.py --n 50000 --clustered  # More realistic
    python examples/backend_benchmark.py --ef 256  # Higher recall
"""

import argparse
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from simplevecdb import VectorDB, Quantization


def generate_vectors(
    n: int, dim: int, seed: int = 42, clustered: bool = False
) -> np.ndarray:
    """Generate normalized random vectors.

    Args:
        n: Number of vectors
        dim: Vector dimension
        seed: Random seed
        clustered: If True, generate clustered data (more realistic for embeddings)
    """
    rng = np.random.default_rng(seed)

    if clustered:
        # Generate clustered data - more realistic for real embeddings
        n_clusters = max(10, n // 100)
        centroids = rng.standard_normal((n_clusters, dim)).astype(np.float32)
        centroids /= np.linalg.norm(centroids, axis=1, keepdims=True)

        # Assign each vector to a random cluster with noise
        cluster_ids = rng.integers(0, n_clusters, n)
        noise_scale = 0.3  # Controls cluster tightness
        vectors = centroids[cluster_ids] + noise_scale * rng.standard_normal(
            (n, dim)
        ).astype(np.float32)
    else:
        vectors = rng.standard_normal((n, dim)).astype(np.float32)

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / norms


def brute_force_search(
    vectors: np.ndarray, query: np.ndarray, k: int
) -> tuple[np.ndarray, np.ndarray]:
    """Baseline brute-force cosine similarity search."""
    # Cosine similarity = dot product of normalized vectors
    similarities = vectors @ query
    # Top-k indices (highest similarity = lowest distance)
    top_k_idx = np.argpartition(-similarities, k)[:k]
    top_k_idx = top_k_idx[np.argsort(-similarities[top_k_idx])]
    # Convert similarity to distance: distance = 1 - similarity (for [0,1] range)
    distances = 1.0 - similarities[top_k_idx]
    return top_k_idx, distances


def compute_recall(ground_truth: np.ndarray, predicted: np.ndarray) -> float:
    """Compute recall@k: fraction of ground truth found in predictions."""
    gt_set = set(ground_truth.tolist())
    pred_set = set(predicted.tolist())
    return len(gt_set & pred_set) / len(gt_set)


def get_storage_size(path: str) -> float:
    """Get total storage size in MB including usearch files."""
    total = 0
    base = Path(path)

    # SQLite files
    for suffix in ["", "-wal", "-shm"]:
        p = Path(str(path) + suffix)
        if p.exists():
            total += p.stat().st_size

    # Usearch index files
    for usearch_file in base.parent.glob(f"{base.name}.*.usearch"):
        total += usearch_file.stat().st_size

    return total / (1024 * 1024)


def benchmark(
    n_vectors: int,
    dim: int,
    k: int = 10,
    n_queries: int = 100,
    quantization: Quantization = Quantization.FLOAT,
    clustered: bool = False,
    expansion_search: int = 64,
) -> dict:
    """Run benchmark comparing usearch to brute-force baseline."""

    print(f"\n{'=' * 60}")
    print(
        f"Benchmark: {n_vectors:,} vectors, {dim}D, k={k}, {quantization.value}, ef={expansion_search}"
    )
    if clustered:
        print("(using clustered data - more realistic)")
    print(f"{'=' * 60}")

    # Generate test data
    print("Generating vectors...", end=" ", flush=True)
    t0 = time.perf_counter()
    vectors = generate_vectors(n_vectors, dim, clustered=clustered)
    queries = generate_vectors(n_queries, dim, seed=123, clustered=clustered)
    print(f"done ({time.perf_counter() - t0:.2f}s)")

    # Setup temp directory
    tmp_dir = tempfile.mkdtemp(prefix="simplevecdb_bench_")
    db_path = os.path.join(tmp_dir, "bench.db")

    try:
        # ----------------------------------------------------------------
        # Usearch HNSW benchmark
        # ----------------------------------------------------------------
        print("\n[Usearch HNSW]")

        # Insert
        print("  Inserting...", end=" ", flush=True)
        t0 = time.perf_counter()
        db = VectorDB(db_path, quantization=quantization)
        collection = db.collection("default")

        # Set custom expansion_search if specified
        collection._index._expansion_search = expansion_search

        # Batch insert
        batch_size = 10000
        texts = [f"doc_{i}" for i in range(n_vectors)]
        for i in range(0, n_vectors, batch_size):
            end = min(i + batch_size, n_vectors)
            collection.add_texts(
                texts[i:end],
                embeddings=vectors[i:end].tolist(),
            )

        insert_time = time.perf_counter() - t0
        print(f"{insert_time:.2f}s ({n_vectors / insert_time:.0f} vec/s)")

        # Force checkpoint for accurate size
        db.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        storage_mb = get_storage_size(db_path)
        print(f"  Storage: {storage_mb:.2f} MB")

        # Search
        print("  Searching...", end=" ", flush=True)
        usearch_results = []
        t0 = time.perf_counter()
        for query in queries:
            results = collection.similarity_search(query.tolist(), k=k)
            # Extract IDs from results (doc content is "doc_{id}")
            doc_ids = [int(doc.page_content.split("_")[1]) for doc, _ in results]
            usearch_results.append(np.array(doc_ids))
        search_time = time.perf_counter() - t0
        avg_ms = (search_time / n_queries) * 1000
        qps = n_queries / search_time
        print(f"{avg_ms:.3f} ms/query ({qps:.0f} QPS)")

        db.close()

        # ----------------------------------------------------------------
        # Brute-force baseline
        # ----------------------------------------------------------------

        print("\n[Brute-force baseline]")

        print("  Searching...", end=" ", flush=True)
        bf_results: list[np.ndarray] = []
        t0 = time.perf_counter()
        for query in queries:
            bf_ids, _ = brute_force_search(vectors, query, k)
            bf_results.append(bf_ids)
        bf_time = time.perf_counter() - t0
        bf_avg_ms = (bf_time / n_queries) * 1000
        bf_qps = n_queries / bf_time
        print(f"{bf_avg_ms:.3f} ms/query ({bf_qps:.0f} QPS)")

        # ----------------------------------------------------------------
        # Recall computation
        # ----------------------------------------------------------------
        recalls = [
            compute_recall(bf, us) for bf, us in zip(bf_results, usearch_results)
        ]
        avg_recall = np.mean(recalls)

        # ----------------------------------------------------------------
        # Summary
        # ----------------------------------------------------------------
        speedup = bf_avg_ms / avg_ms if avg_ms > 0 else float("inf")

        print("\n[Summary]")
        print(f"  Speedup: {speedup:.1f}x faster than brute-force")
        print(f"  Recall@{k}: {avg_recall:.3f} (1.0 = perfect)")
        print(
            f"  Storage: {storage_mb:.2f} MB ({storage_mb * 1024 * 1024 / n_vectors:.1f} bytes/vector)"
        )

        return {
            "n_vectors": n_vectors,
            "dim": dim,
            "k": k,
            "quantization": quantization.value,
            "usearch_ms": avg_ms,
            "usearch_qps": qps,
            "bruteforce_ms": bf_avg_ms,
            "bruteforce_qps": bf_qps,
            "speedup": speedup,
            "recall": avg_recall,
            "storage_mb": storage_mb,
            "insert_time_s": insert_time,
        }

    finally:
        # Cleanup
        shutil.rmtree(tmp_dir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark usearch HNSW vs brute-force search"
    )
    parser.add_argument(
        "--n", type=int, default=10000, help="Number of vectors (default: 10000)"
    )
    parser.add_argument(
        "--dim", type=int, default=384, help="Vector dimension (default: 384)"
    )
    parser.add_argument(
        "--k", type=int, default=10, help="Number of nearest neighbors (default: 10)"
    )
    parser.add_argument(
        "--queries", type=int, default=100, help="Number of test queries (default: 100)"
    )
    parser.add_argument(
        "--all-quant", action="store_true", help="Test all quantization levels"
    )
    parser.add_argument(
        "--clustered",
        action="store_true",
        help="Use clustered data (more realistic, better recall)",
    )
    parser.add_argument(
        "--ef",
        type=int,
        default=64,
        help="HNSW expansion_search (default: 64, try 256 for ~90%% recall)",
    )
    args = parser.parse_args()

    print("SimpleVecDB Backend Benchmark")
    print("=" * 60)
    print(f"Testing with {args.n:,} vectors, {args.dim}D embeddings")
    if args.n < 50000:
        print(
            "\nNOTE: HNSW has overhead - expect brute-force to be faster at <50k vectors."
        )
        print("      Try --n 100000 to see HNSW speedup.\n")

    if args.all_quant:
        quantizations = [Quantization.FLOAT, Quantization.INT8, Quantization.BIT]
    else:
        quantizations = [Quantization.FLOAT]

    results = []
    for quant in quantizations:
        result = benchmark(
            n_vectors=args.n,
            dim=args.dim,
            k=args.k,
            n_queries=args.queries,
            quantization=quant,
            clustered=args.clustered,
            expansion_search=args.ef,
        )
        results.append(result)

    # Print comparison table
    if len(results) > 1:
        print("\n" + "=" * 60)
        print("COMPARISON TABLE")
        print("=" * 60)
        print(
            f"{'Quant':<8} {'Search(ms)':<12} {'Speedup':<10} {'Recall':<10} {'Size(MB)':<10}"
        )
        print("-" * 60)
        for r in results:
            print(
                f"{r['quantization']:<8} "
                f"{r['usearch_ms']:<12.3f} "
                f"{r['speedup']:<10.1f}x "
                f"{r['recall']:<10.3f} "
                f"{r['storage_mb']:<10.2f}"
            )

    print("\nBenchmark complete!")


if __name__ == "__main__":
    main()
