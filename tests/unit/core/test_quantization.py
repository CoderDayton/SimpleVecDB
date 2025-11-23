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


def test_quantization_storage():
    """Test INT8 quantization storage uses 1 byte per dimension."""
    from simplevecdb import VectorDB

    db = VectorDB(":memory:", quantization=Quantization.INT8)
    collection = db.collection("default")
    emb = np.random.randn(100, 128).tolist()
    collection.add_texts(["t"] * 100, embeddings=emb)

    # Manual check serialized is int8
    blob = db.conn.execute(
        f"SELECT embedding FROM {collection._vec_table_name} LIMIT 1"
    ).fetchone()[0]
    assert len(blob) == 128  # 1 byte/dim
    db.close()
