# tests/perf/test_benchmark.py
import time
import numpy as np
import pytest
from simplevecdb import VectorDB

N = 10_000  # Reduced from 100k for CI/Dev environment safety
DIM = 384


@pytest.fixture(scope="module")
def populated_benchmark_db():
    """Fixture for a populated database to test query performance."""
    db = VectorDB(":memory:")
    collection = db.collection("default")
    # Generate random normalized vectors
    embs = np.random.randn(N, DIM).astype(np.float32)
    embs /= np.linalg.norm(embs, axis=1, keepdims=True)

    collection.add_texts(["text"] * N, embeddings=embs.tolist())
    return db


@pytest.mark.perf
def test_insert_performance():
    """Benchmark insertion speed."""
    db = VectorDB(":memory:")
    collection = db.collection("default")
    embs = np.random.randn(N, DIM).astype(np.float32)
    embs /= np.linalg.norm(embs, axis=1, keepdims=True)

    t0 = time.time()
    collection.add_texts(["text"] * N, embeddings=embs.tolist())
    duration = time.time() - t0

    # 10k items should take < 10s
    assert duration < 10
    print(f"\n{N} inserts: {duration:.2f}s ({N / duration:.0f} vec/s)")


@pytest.mark.perf
def test_query_performance(populated_benchmark_db):
    """Benchmark query speed on a populated database."""
    collection = populated_benchmark_db.collection("default")
    query = np.random.randn(DIM).astype(np.float32)
    query /= np.linalg.norm(query)

    t0 = time.time()
    iterations = 100
    for _ in range(iterations):
        collection.similarity_search(query, k=10)

    total_time = time.time() - t0
    avg_ms = (total_time / iterations) * 1000

    # Assert reasonable query time (< 10ms for 10k items in memory)
    assert avg_ms < 10
    print(f"\nAvg query time (N={N}): {avg_ms:.2f} ms")
