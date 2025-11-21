# tests/unit/test_core.py
import pytest
import numpy as np
import json
import sqlite3
from tinyvecdb import VectorDB


def test_init(empty_db):
    """Verify that the database initializes with correct default values."""
    assert empty_db._dim is None
    assert empty_db.quantization == "float"  # Ensure default configuration values are set correctly.
    assert empty_db.distance_strategy == "cosine"


def test_add_texts_basic(empty_db):
    """Test adding texts with embeddings and verify storage integrity."""
    texts = ["test1", "test2"]
    embs = [[0.1, 0.2], [0.3, 0.4]]
    ids = empty_db.add_texts(texts, embeddings=embs)
    assert len(ids) == 2
    assert empty_db._dim == 2

    # Verify that the text content is persisted in the main table.
    rows = empty_db.conn.execute(
        "SELECT text FROM tinyvec_items ORDER BY id"
    ).fetchall()
    assert rows[0][0] == "test1"

    # Verify that the embedding vector is stored in the virtual table.
    vec_row = empty_db.conn.execute(
        "SELECT embedding FROM vec_index WHERE rowid = ?", (ids[0],)
    ).fetchone()

    # Ensure vectors are normalized when using Cosine distance strategy.
    expected = np.array([0.1, 0.2], dtype=np.float32)
    expected /= np.linalg.norm(expected)
    assert np.allclose(np.frombuffer(vec_row[0], dtype=np.float32), expected)


def test_add_with_metadata(populated_db):
    """Verify that metadata is correctly stored and retrievable."""
    row = populated_db.conn.execute(
        "SELECT metadata FROM tinyvec_items WHERE id=1"
    ).fetchone()[0]
    meta = json.loads(row)
    assert meta["color"] == "red"
    assert meta["likes"] == 10


def test_upsert(populated_db):
    """Test the upsert functionality (update existing records)."""
    new_emb = [0.5, 0.5, 0.5, 0.5]
    populated_db.add_texts(
        ["updated apple"], embeddings=[new_emb], ids=[1], metadatas=[{"color": "green"}]
    )

    updated = populated_db.conn.execute(
        "SELECT text, metadata FROM tinyvec_items WHERE id=1"
    ).fetchone()
    assert updated[0] == "updated apple"
    assert json.loads(updated[1])["color"] == "green"


def test_delete_by_ids(populated_db):
    """Test deletion of records by their IDs."""
    populated_db.delete_by_ids([1, 2])
    remaining = populated_db.conn.execute(
        "SELECT COUNT(*) FROM tinyvec_items"
    ).fetchone()[0]
    assert remaining == 2
    vec_count = populated_db.conn.execute("SELECT COUNT(*) FROM vec_index").fetchone()[
        0
    ]
    assert vec_count == 2  # Ensure the virtual table row count matches the main table.


def test_add_no_embeddings_raises(empty_db, monkeypatch):
    """Ensure ValueError is raised when no embeddings are provided and local embedder fails."""
    # Mock the module where embed_texts comes from
    import sys
    from unittest.mock import MagicMock

    # Create a mock module that raises ImportError when accessed or imported
    mock_module = MagicMock()
    mock_module.embed_texts.side_effect = ImportError("Mocked import error")

    # Mock the embedding module to simulate an import error.
    with monkeypatch.context() as m:
        m.setitem(sys.modules, "tinyvecdb.embeddings.models", mock_module)

        with pytest.raises(ValueError):
            empty_db.add_texts(["test"])


def test_close_and_del():
    """Test explicit closing of the database connection and resource cleanup."""
    db = VectorDB(":memory:")
    conn = db.conn
    db.close()

    # Verify connection is closed by attempting an operation
    with pytest.raises(sqlite3.ProgrammingError):
        conn.execute("SELECT 1")
