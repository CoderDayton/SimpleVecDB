"""Quantization and vector encoding/decoding tests."""

import numpy as np
import pytest

from simplevecdb.types import Quantization
from simplevecdb.engine.quantization import QuantizationStrategy, normalize_l2


def test_dequantize_bit_vector():
    """Test BIT vector dequantization edge case."""
    # BIT quantization with proper dimension
    blob = np.packbits(np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.uint8)).tobytes()
    strategy = QuantizationStrategy(Quantization.BIT)
    result = strategy.deserialize(blob, dim=8)
    expected = np.array([1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0], dtype=np.float32)
    assert np.allclose(result, expected)


def test_dequantize_unsupported():
    """Test dequantize with unsupported quantization type."""
    blob = b"\x00\x01\x02\x03"
    # Use a mock object that's not a valid Quantization
    strategy = QuantizationStrategy("invalid")  # type: ignore
    with pytest.raises(ValueError, match="Unsupported quantization"):
        strategy.deserialize(blob, dim=4)


def test_normalize_l2_zero_vector():
    """Test L2 normalization with zero vector."""
    zero_vec = np.array([0.0, 0.0, 0.0])
    result = normalize_l2(zero_vec)
    # Should return the same zero vector
    assert np.allclose(result, zero_vec)


def test_int8_quantization_search():
    """Test INT8 quantization works for search."""
    from simplevecdb import VectorDB

    db = VectorDB(":memory:", quantization=Quantization.INT8)
    collection = db.collection("default")
    emb = np.random.randn(10, 128).tolist()
    collection.add_texts([f"t{i}" for i in range(10)], embeddings=emb)

    # Search should work with INT8 quantization
    results = collection.similarity_search(emb[0], k=1)
    assert len(results) == 1
    assert results[0][0].page_content == "t0"
    db.close()


def test_float16_quantization_search():
    """Test FLOAT16 (half-precision) quantization works for search."""
    from simplevecdb import VectorDB

    db = VectorDB(":memory:", quantization=Quantization.FLOAT16)
    collection = db.collection("default")
    emb = np.random.randn(10, 128).tolist()
    collection.add_texts([f"t{i}" for i in range(10)], embeddings=emb)

    # Search should work with FLOAT16 quantization
    results = collection.similarity_search(emb[0], k=1)
    assert len(results) == 1
    assert results[0][0].page_content == "t0"
    db.close()


def test_threads_parameter():
    """Test threads parameter for parallel add/search."""
    from simplevecdb import VectorDB

    db = VectorDB(":memory:")
    collection = db.collection("default")

    # Add with explicit threads
    emb = np.random.randn(100, 64).tolist()
    texts = [f"doc_{i}" for i in range(100)]
    ids = collection.add_texts(texts, embeddings=emb, threads=4)
    assert len(ids) == 100

    # Search with explicit threads
    results = collection.similarity_search(emb[0], k=5, threads=4)
    assert len(results) == 5
    assert results[0][0].page_content == "doc_0"
    db.close()
