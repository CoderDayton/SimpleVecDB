"""Similarity search and query tests."""

from unittest.mock import patch
import pytest

from simplevecdb import VectorDB
from simplevecdb.core import Quantization


def test_similarity_search_text_query_error():
    """Test similarity search with text query when embeddings unavailable."""
    import sys

    db = VectorDB(":memory:")
    collection = db.collection("default")
    collection.add_texts(["test"], embeddings=[[0.1, 0.2]])

    # Mock embeddings module to be unavailable
    with patch.dict(sys.modules, {"simplevecdb.embeddings.models": None}):
        # Try text query - should raise ValueError
        with pytest.raises((ValueError, AttributeError)):
            collection.similarity_search("query text", k=1)


def test_similarity_search_with_int8_quantization():
    """Test similarity search uses vec_int8 placeholder for INT8."""
    db = VectorDB(":memory:", quantization=Quantization.INT8)
    collection = db.collection("default")
    texts = ["a", "b"]
    embs = [[0.1, 0.2], [0.3, 0.4]]
    collection.add_texts(texts, embeddings=embs)

    results = collection.similarity_search([0.1, 0.2], k=1)
    assert len(results) == 1


def test_similarity_search_with_bit_quantization():
    """Test similarity search uses vec_bit placeholder for BIT."""
    db = VectorDB(":memory:", quantization=Quantization.BIT)
    collection = db.collection("default")
    texts = ["a", "b"]
    # BIT requires dimensions divisible by 8
    embs = [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] for _ in range(2)]
    collection.add_texts(texts, embeddings=embs)

    results = collection.similarity_search([0.1] * 8, k=1)
    assert len(results) == 1


def test_similarity_search_exact_parameter():
    """Test that exact parameter forces brute-force or HNSW search."""
    db = VectorDB(":memory:")
    collection = db.collection("default")

    # Add some vectors
    texts = [f"doc_{i}" for i in range(100)]
    embs = [[float(i) / 100] * 8 for i in range(100)]
    collection.add_texts(texts, embeddings=embs)

    query = [0.5] * 8

    # Default (adaptive) - should use brute-force for small collection
    results_adaptive = collection.similarity_search(query, k=5)
    assert len(results_adaptive) == 5

    # Force exact=True (brute-force)
    results_exact = collection.similarity_search(query, k=5, exact=True)
    assert len(results_exact) == 5

    # Force exact=False (HNSW approximate)
    results_hnsw = collection.similarity_search(query, k=5, exact=False)
    assert len(results_hnsw) == 5

    # All should return results (may differ slightly due to HNSW approximation)
    # Exact search should find the same results
    exact_texts = {doc.page_content for doc, _ in results_exact}
    adaptive_texts = {doc.page_content for doc, _ in results_adaptive}
    # For small collections, adaptive uses exact, so should match
    assert exact_texts == adaptive_texts
