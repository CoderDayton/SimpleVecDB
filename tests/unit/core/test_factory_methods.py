"""Factory method and sqlite_vec extension loading tests."""

import sqlite3
from unittest.mock import patch

from tinyvecdb import VectorDB


def test_as_langchain_factory(tmp_path):
    """Test as_langchain factory method."""
    db_path = tmp_path / "factory.db"
    db = VectorDB(str(db_path))

    lc = db.as_langchain()
    assert lc is not None
    assert hasattr(lc, "add_texts")
    assert hasattr(lc, "similarity_search")
    db.close()


def test_as_llama_index_factory(tmp_path):
    """Test as_llama_index factory method."""
    db_path = tmp_path / "factory.db"
    db = VectorDB(str(db_path))

    li = db.as_llama_index()
    assert li is not None
    assert hasattr(li, "add")
    assert hasattr(li, "query")
    db.close()


def test_sqlite_vec_load_failure(tmp_path):
    """Test VectorDB handles sqlite_vec load failure gracefully."""
    db_path = tmp_path / "load_fail.db"

    # Mock sqlite_vec.load to raise
    with patch("sqlite_vec.load", side_effect=sqlite3.OperationalError("Load failed")):
        db = VectorDB(str(db_path))
        # Should set _extension_available to False
        assert db._extension_available is False
        db.close()
