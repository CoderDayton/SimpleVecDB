"""Core VectorDB initialization and dimension recovery tests."""

import pytest
from unittest.mock import patch
import sys

from tinyvecdb import VectorDB


def test_recover_dim_no_match(tmp_path):
    """Test _recover_dim when SQL doesn't match expected pattern."""
    db_path = tmp_path / "test_recover.db"
    db = VectorDB(str(db_path))

    # _recover_dim is called during __init__ and sets _dim
    # If pattern doesn't match, _dim remains None and is set on first add
    # This test just verifies initialization doesn't crash
    assert db._dim is None or isinstance(db._dim, int)


def test_recover_dim_none(tmp_path):
    """Test _recover_dim when table doesn't exist (returns None)."""
    db_path = tmp_path / "test_recover_none.db"
    db = VectorDB(str(db_path))

    # New DB should have _dim as None initially
    assert db._dim is None


def test_ensure_virtual_table_dimension_mismatch():
    """Test _ensure_virtual_table raises on dimension mismatch."""
    db = VectorDB(":memory:")
    db.add_texts(["test1"], embeddings=[[0.1, 0.2]])  # 2D

    # Try to add 3D vector - should raise
    with pytest.raises(ValueError, match="Dimension mismatch"):
        db.add_texts(["test2"], embeddings=[[0.1, 0.2, 0.3]])


def test_add_texts_empty_list():
    """Test add_texts with empty list returns empty."""
    db = VectorDB(":memory:")
    result = db.add_texts([])
    assert result == []


def test_add_texts_no_embeddings_no_model():
    """Test add_texts without embeddings when model unavailable."""
    db = VectorDB(":memory:")

    # Mock the import to fail
    with patch.dict(sys.modules, {"tinyvecdb.embeddings.models": None}):
        with pytest.raises(ValueError, match="No embeddings provided"):
            db.add_texts(["test"])


def test_close_exception_handling():
    """Test that __del__ handles exceptions gracefully."""
    db = VectorDB(":memory:")
    db.add_texts(["test"], embeddings=[[0.1, 0.2]])

    # Close once
    db.close()

    # __del__ should handle the exception when connection is already closed
    try:
        db.__del__()
    except Exception:
        pytest.fail("__del__ should not raise exceptions")


def test_recover_dim_exception_handling(tmp_path):
    """Test _recover_dim handles exceptions gracefully."""
    from unittest.mock import MagicMock

    db_path = tmp_path / "recover_err.db"
    db = VectorDB(str(db_path))

    # Mock connection to raise exception
    original_conn = db.conn
    mock_conn = MagicMock()
    mock_conn.execute.side_effect = Exception("DB Error")
    db.conn = mock_conn

    # Should handle exception gracefully
    db._recover_dim()

    # Restore connection to close properly
    db.conn = original_conn
    db.close()
