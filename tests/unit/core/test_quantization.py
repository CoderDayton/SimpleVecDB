"""Quantization and vector encoding/decoding tests."""

import numpy as np
import pytest

from tinyvecdb.core import Quantization, _dequantize_vector, _normalize_l2


def test_dequantize_bit_vector():
    """Test BIT vector dequantization edge case."""
    # BIT quantization with proper dimension
    blob = np.packbits(np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.uint8)).tobytes()
    result = _dequantize_vector(blob, dim=8, quant=Quantization.BIT)
    expected = np.array([1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0], dtype=np.float32)
    assert np.allclose(result, expected)


def test_dequantize_unsupported():
    """Test dequantize with unsupported quantization type."""
    blob = b"\x00\x01\x02\x03"
    # Use a mock object that's not a valid Quantization
    with pytest.raises(ValueError, match="Unsupported quantization"):
        # Type ignore for intentional test of error handling
        _dequantize_vector(blob, dim=4, quant="invalid")  # type: ignore


def test_normalize_l2_zero_vector():
    """Test L2 normalization with zero vector."""
    zero_vec = np.array([0.0, 0.0, 0.0])
    result = _normalize_l2(zero_vec)
    # Should return the same zero vector
    assert np.allclose(result, zero_vec)


def test_quantization_storage():
    """Test INT8 quantization storage uses 1 byte per dimension."""
    from tinyvecdb import VectorDB

    db = VectorDB(":memory:", quantization=Quantization.INT8)
    emb = np.random.randn(100, 128).tolist()
    db.add_texts(["t"] * 100, embeddings=emb)

    # Manual check serialized is int8
    blob = db.conn.execute("SELECT embedding FROM vec_index LIMIT 1").fetchone()[0]
    assert len(blob) == 128  # 1 byte/dim
    db.close()
