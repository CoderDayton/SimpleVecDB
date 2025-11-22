"""
Performance Benchmark for TinyVecDB
Generates metrics matching the README benchmark table.
"""

import time
import platform
import numpy as np
from pathlib import Path
from tinyvecdb import VectorDB, Quantization
from tinyvecdb.config import config as EmbeddingConfig


def format_size_mb(bytes_size: int) -> str:
    """Format bytes to Megabytes (MB)."""
    return f"{bytes_size / (1024 * 1024):.2f} MB"


def benchmark_config(
    vectors: int,
    dimensions: int,
    quantization: Quantization,
    iterations: int = 100,
) -> dict:
    """
    Run benchmark for a specific configuration.

    Args:
        vectors: Number of vectors to insert.
        dimensions: Vector dimensions.
        quantization: Quantization mode (FLOAT, INT8, BIT).
        iterations: Number of query iterations for timing.

    Returns:
        Dictionary with benchmark results.
    """
    # Create temporary database file
    db_path = Path(f"bench_{quantization.value}_{dimensions}d.db")

    # Clean up any existing file
    if db_path.exists():
        db_path.unlink()

    try:
        # Initialize database
        db = VectorDB(str(db_path), quantization=quantization)

        # Generate random normalized vectors
        print(f"  Generating {vectors:,} random {dimensions}d vectors...")
        embeddings = np.random.randn(vectors, dimensions).astype(np.float32)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        # Insert vectors
        print(f"  Inserting with {quantization.value} quantization...")
        texts = [f"document_{i}" for i in range(vectors)]

        insert_start = time.perf_counter()
        db.add_texts(texts, embeddings=embeddings.tolist())
        insert_time = time.perf_counter() - insert_start
        insert_speed = vectors / insert_time if insert_time > 0 else 0

        # Close and get file size
        db.close()
        file_size = db_path.stat().st_size

        # Reopen for queries
        db = VectorDB(str(db_path), quantization=quantization)

        # Generate random query vector
        query = np.random.randn(dimensions).astype(np.float32)
        query = query / np.linalg.norm(query)
        query_list = query.tolist()

        # Warm up
        db.similarity_search(query_list, k=10)

        # Benchmark queries
        print(f"  Running {iterations} query iterations...")
        query_times = []
        for _ in range(iterations):
            query_start = time.perf_counter()
            _ = db.similarity_search(query_list, k=10)
            query_times.append(
                (time.perf_counter() - query_start) * 1000
            )  # Convert to ms

        avg_query_time = np.mean(query_times)

        # Cleanup
        db.close()

        return {
            "quantization": str(quantization.value).upper(),
            "vectors": vectors,
            "dimensions": dimensions,
            "file_size_mb": format_size_mb(file_size),
            "insert_time": insert_time,
            "insert_speed": insert_speed,
            "avg_query_ms": avg_query_time,
        }

    finally:
        # Cleanup database files
        if db_path.exists():
            db_path.unlink()
        for ext in ["-shm", "-wal"]:
            wal_path = Path(str(db_path) + ext)
            if wal_path.exists():
                wal_path.unlink()


def run_benchmarks():
    """Run all benchmark configurations and print results table."""

    print("\n" + "=" * 90)
    print("TinyVecDB Performance Benchmark")
    print("=" * 90 + "\n")

    # Benchmark configurations matching README table
    configs = [
        {"vectors": 10_000, "dimensions": 384, "quantization": Quantization.FLOAT},
        {"vectors": 10_000, "dimensions": 384, "quantization": Quantization.INT8},
        {"vectors": 10_000, "dimensions": 384, "quantization": Quantization.BIT},
        {"vectors": 10_000, "dimensions": 1536, "quantization": Quantization.FLOAT},
    ]

    results = []

    for i, config in enumerate(configs, 1):
        print(
            f"[{i}/{len(configs)}] Benchmarking: {config['quantization'].value.upper()} "
            f"({config['vectors']:,} vectors × {config['dimensions']} dims)"
        )

        result = benchmark_config(**config)
        results.append(result)

        print(f"  ✓ File size: {result['file_size_mb']}")
        print(f"  ✓ Avg query time: {result['avg_query_ms']:.2f} ms")
        print(f"  ✓ Insert speed: {result['insert_speed']:,.0f} vec/s\n")

    # Print results table
    print("\n" + "=" * 90)
    print("BENCHMARK RESULTS")
    print("=" * 90 + "\n")

    print(f"Model: {EmbeddingConfig.EMBEDDING_MODEL}\n")
    print(f"Batch Size: {EmbeddingConfig.EMBEDDING_BATCH_SIZE}\n")

    # Table header
    print(
        f"{'Quantization':<15} {'Vectors':<10} {'Dimensions':<12} {'File Size':<12} {'Insert Speed':<18} {'Avg Query (k=10)':<18}"
    )
    print("-" * 90)

    # Table rows
    for r in results:
        print(
            f"{r['quantization']:<15} {r['vectors']:<10,} {r['dimensions']:<12} "
            f"{r['file_size_mb']:<12} {r['insert_speed']:<18,.0f} {r['avg_query_ms']:.2f} ms"
        )

    print("\n" + "=" * 90)
    print(f"Platform: {platform.system()} {platform.machine()} ({platform.release()})")
    print("sqlite-vec version: (check with `pip show sqlite-vec`)")
    print("=" * 90 + "\n")


if __name__ == "__main__":
    run_benchmarks()
