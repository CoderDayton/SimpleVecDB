"""Core VectorDB initialization and dimension recovery tests."""

import pytest
from unittest.mock import patch
import sys

from simplevecdb import VectorDB


def test_recover_dim_no_match(tmp_path):
    """Test dimension recovery when no vectors exist."""
    db_path = tmp_path / "test_recover.db"
    db = VectorDB(str(db_path))
    collection = db.collection("default")

    # New collection should have _dim as None until vectors are added
    assert collection._dim is None or isinstance(collection._dim, int)


def test_recover_dim_none(tmp_path):
    """Test dimension is None when no vectors exist."""
    db_path = tmp_path / "test_recover_none.db"
    db = VectorDB(str(db_path))
    collection = db.collection("default")

    # New DB should have _dim as None initially
    assert collection._dim is None


def test_dimension_mismatch_on_add():
    """Test adding vectors with mismatched dimensions raises error."""
    db = VectorDB(":memory:")
    collection = db.collection("default")
    collection.add_texts(["test1"], embeddings=[[0.1, 0.2]])  # 2D

    # Try to add 3D vector - should raise
    with pytest.raises(ValueError, match="dimension"):
        collection.add_texts(["test2"], embeddings=[[0.1, 0.2, 0.3]])


def test_add_texts_empty_list():
    """Test add_texts with empty list returns empty."""
    db = VectorDB(":memory:")
    collection = db.collection("default")
    result = collection.add_texts([])
    assert result == []


def test_add_texts_no_embeddings_no_model():
    """Test add_texts without embeddings when model unavailable."""
    db = VectorDB(":memory:")
    collection = db.collection("default")

    # Mock the import to fail
    with patch.dict(sys.modules, {"simplevecdb.embeddings.models": None}):
        with pytest.raises(ValueError, match="No embeddings provided"):
            collection.add_texts(["test"])


def test_close_exception_handling():
    """Test that __del__ handles exceptions gracefully."""
    db = VectorDB(":memory:")
    collection = db.collection("default")
    collection.add_texts(["test"], embeddings=[[0.1, 0.2]])

    # Close once
    db.close()

    # __del__ should handle the exception when connection is already closed
    try:
        db.__del__()
    except Exception:
        pytest.fail("__del__ should not raise exceptions")
